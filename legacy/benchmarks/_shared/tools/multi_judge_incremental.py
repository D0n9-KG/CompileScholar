# -*- coding: utf-8 -*-
"""Multi-108 incremental judge loop (user directive 2026-09-23: answer one,
judge one — never wait for a full arm to finish).

TWO tracks per answered question, both merged into one scores row
(user: "和判引用的弄在一块，每题答完直接判"):

  Track 1  Citation F1 — deterministic, official extract_citations verbatim
           (the [0]/[2,3] zero-based form), gold = official gold output.
  Track 2  AutoAIS-LLM — official citation_correctness_eval.py protocol
           VERBATIM (sent_tokenize; >=50-char filter; per-sentence citation
           inheritance; out-of-range=0; joint passage = Title+text of cited
           ctxs; at_most_citations=3; multi-cite per-doc necessity checks),
           with the validator swapped from the OSU NLI model to GLM-5.3 via
           Paratera using the OFFICIAL input_prompt verbatim ("As an
           Attribution Validator..." — the official LLM form of AutoAIS;
           NLI local model deferred per user ruling 2026-09-23).

Judge calls go to a SEPARATE ledger (prereg discipline: judging must not
mix into arm ledgers). Verdict disk-cache dedups identical (passage, claim)
pairs across arms.

Scoring rows append to answers/<arm>.scores.jsonl (resume-safe). Backfill
mode recomputes rows scored before Track 2 existed (append full rows; last
row per qid wins downstream).

Usage: python multi_judge_incremental.py [--once] [--interval 120] [--backfill]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "..", "..", "src")
sys.path.insert(0, _SRC)

# judge ledger MUST be set before the first kb_infra call logs
_MULTI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "scholarqa_multi")
os.environ["LLM_CALL_LOG"] = os.path.join(_MULTI, "judge", "ledger_judge.jsonl")
os.environ.setdefault("LLM_RUN_ID", "multi-judge")

from kb_infra.llm import call_paratera  # noqa: E402
from multi_baseline_common import load_questions  # noqa: E402

ARMS = {
    "ours": os.path.join(_MULTI, "baselines", "ours", "answers_ours.json"),
    "lightrag": os.path.join(_MULTI, "baselines", "lightrag", "answers_lightrag.json"),
    "paperqa": os.path.join(_MULTI, "baselines", "paperqa", "answers_paperqa.json"),
}
QFILE = os.path.join(_MULTI, "data", "scholarqa_multi.json")
VERDICT_CACHE = os.path.join(_MULTI, "judge", "autoais_verdict_cache.jsonl")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "GLM-5.3")
JUDGE_WORKERS = int(os.environ.get("JUDGE_WORKERS", "16"))

# ---- official extract_citations / remove_citations (verbatim) ----
_citation_pattern = re.compile(r'\[(\d+(?:,\s*\d+)*)\]')
_citation_rm = re.compile(r'\[\d+(?:,\s*\d+)*\]')


def extract_citations(text):
    matches = _citation_pattern.findall(text)
    citations = []
    for match in matches:
        citations.extend([int(num.strip()) for num in match.split(',')])
    return citations


def remove_citations(text):
    cleaned = re.sub(_citation_rm, '', text)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    cleaned = cleaned.replace(" .", ".").replace(" ,", ",")
    return cleaned


# ---- official AutoAIS-LLM prompt (verbatim from citation_correctness_eval) ----
INPUT_PROMPT = "As an Attribution Validator, your task is to verify whether a given reference can support the given claim. A claim can be either a plain sentence or a question followed by its answer. Specifically, your response should clearly indicate the relationship: Attributable, Contradictory or Extrapolatory. A contradictory error occurs when you can infer that the answer contradicts the fact presented in the context, while an extrapolatory error means that you cannot infer the correctness of the answer based on the information provided in the context. \n\nClaim: {claim}\n Reference: {output}"

_VERDICT_RE = re.compile(r'attributable|contradictory|extrapolatory', re.I)

_cache_lock = threading.Lock()
_verdict_cache: dict[str, float] = {}
_unparsed = {"n": 0}


def _cache_key(passage: str, claim: str) -> str:
    return hashlib.md5((passage + "\x00" + claim).encode("utf-8")).hexdigest()


def _load_cache():
    os.makedirs(os.path.dirname(VERDICT_CACHE), exist_ok=True)
    if os.path.exists(VERDICT_CACHE):
        for l in open(VERDICT_CACHE, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    r = json.loads(l)
                    _verdict_cache[r["k"]] = r["v"]
                except Exception:
                    pass


def llm_verdict(passage: str, claim: str) -> float:
    """Official AutoAIS-LLM judgement: Attributable -> 1.0 else 0.0.
    Disk-cached (identical passage+claim pairs recur across arms).

    N3/N4 (carpet-audit 2026-09-23): the old version (a) let GLM-5.3's
    thinking chain eat the 3000-token budget — 522 calls truncated with the
    verdict never emitted, silently scored 0 and PERMANENTLY cached; (b) took
    the FIRST regex match — "not attributable" scored 1.0, verdict-first
    prose-then-conclusion output scored 0.0. Now: thinking disabled, budget
    raised, LAST match wins (verdicts conclude responses), negated matches
    are unparsed, unparsed pairs return None and are NEVER cached (retryable
    in a later cycle), and the raw response is stored for audit."""
    k = _cache_key(passage, claim)
    with _cache_lock:
        if k in _verdict_cache:
            return _verdict_cache[k]
    prompt = INPUT_PROMPT.replace("{claim}", claim).replace("{output}", passage)
    v = None
    raw_resp = ""
    for _ in range(2):
        try:
            raw = call_paratera(prompt, model=JUDGE_MODEL, max_tokens=12000,
                                temperature=0.0, enable_thinking=False)
        except Exception:
            raw = None
        if raw:
            raw_resp = raw
            # N4: LAST match wins — official semantics is the final verdict
            # label; negated forms ("not attributable") are NOT verdicts.
            matches = list(_VERDICT_RE.finditer(raw))
            m = matches[-1] if matches else None
            if m:
                word = m.group(0).lower()
                # negation guard: char before match must not be part of 'not '
                neg = raw[max(0, m.start() - 4):m.start()].lower().rstrip()
                if neg.endswith("not") or neg.endswith("n't"):
                    continue  # negated verdict — treat as unparsed, retry
                v = 1.0 if word == "attributable" else 0.0
                break
    if v is None:
        with _cache_lock:
            _unparsed["n"] += 1
        return 0.0  # NOT cached — a later cycle re-attempts this pair
    with _cache_lock:
        if k not in _verdict_cache:
            _verdict_cache[k] = v
            with open(VERDICT_CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "v": v, "resp": raw_resp[:600]})
                        + "\n")
    return v


def citation_f1(answer_official: str, gold_output: str) -> dict:
    preds = sorted(set(extract_citations(answer_official or "")))
    gold = sorted(set(extract_citations(gold_output or "")))
    inter = len(set(preds) & set(gold))
    prec = inter / len(preds) if preds else 0.0
    rec = inter / len(gold) if gold else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"n_pred": len(preds), "n_gold": len(gold), "n_correct": inter,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4)}


def _fmt_doc(ctx: dict) -> str:
    """Official _format_document: 'Title: %s\\n%s' (NaN texts -> empty,
    disclosed; the official evaluator would hard-crash on those 2 ctxs)."""
    text = ctx.get("text")
    if not isinstance(text, str):
        text = ""
    return f"Title: {ctx.get('title', '')}\n{text}"


def autoais_scores(answer_official: str, ctxs: list) -> dict | None:
    """Official compute_autoais per-item logic VERBATIM (recall + precision),
    sentence-level calls parallelized; order preserved (inheritance is
    sequential, verdict calls are not)."""
    from nltk import sent_tokenize
    output = answer_official or ""
    sents = sent_tokenize(output)
    if not sents:
        return None
    target_sents = [remove_citations(s).strip() for s in sents]

    entail = 0
    entail_prec = 0
    total_citations = 0
    total_sents = 0
    previous_citations = None
    n_multi = 0
    n_overcite = 0
    # (kind, passage, claim) jobs: 'joint' or ('doc', rid) / ('subset', rid)
    jobs = []          # sequential skeleton, parallel verdict filling

    for sent_id, sent in enumerate(sents):
        if len(sent) < 50:
            continue
        total_sents += 1
        target_sent = target_sents[sent_id]
        joint_entail = -1
        ref = [int(r[1:]) for r in re.findall(r"\[\d+", sent)]
        if len(ref) == 0 and previous_citations is not None:
            ref = previous_citations
        if len(ref) == 0:
            joint_entail = 0
        elif any(rid >= len(ctxs) for rid in ref):
            joint_entail = 0
        else:
            previous_citations = ref
            ref = ref[:3]  # official default at_most_citations=3
            total_citations += len(ref)
            joint_passage = "\n".join(_fmt_doc(ctxs[i]) for i in ref if i >= 0)
        if joint_entail == -1:
            jobs.append(("joint", joint_passage, target_sent, sent_id, ref))
        else:
            jobs.append(("zero", "", target_sent, sent_id, []))

    # parallel verdict resolution (cache + GLM-5.3)
    def _run(job):
        kind, passage, claim, sid, ref = job
        if kind == "zero":
            return job, 0.0, []
        v = llm_verdict(passage, claim)
        extras = []
        if v and len(ref) > 1:
            # precision: per-doc necessity (official condition A then B)
            for rid in ref:
                p_a = _fmt_doc(ctxs[rid])
                v_a = llm_verdict(p_a, claim)
                if v_a:
                    extras.append((rid, 1.0))
                else:
                    subset = [i for i in ref if i != rid]
                    p_b = "\n".join(_fmt_doc(ctxs[i]) for i in subset)
                    v_b = llm_verdict(p_b, claim)
                    extras.append((rid, 1.0 if not v_b else 0.0))
        return job, v, extras

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
        results = list(ex.map(_run, jobs))

    for (kind, _p, _c, sid, ref), v, extras in results:
        entail += v
        if v and len(ref) > 1:
            n_multi += 1
            entail_prec += sum(1 for _rid, e in extras if e)
            n_overcite += sum(1 for _rid, e in extras if not e)
        else:
            entail_prec += v

    return {"ais_rec": round(entail / total_sents, 4) if total_sents else 0.0,
            "ais_prec": round(entail_prec / total_citations, 4)
            if total_citations else 0.0,
            "ais_sents": total_sents, "ais_citations": total_citations,
            "ais_multi_supported": n_multi, "ais_overcite": n_overcite}


def load_gold() -> dict:
    return {q["id"]: q for q in load_questions(QFILE)}


def judge_arm(arm: str, path: str, gold: dict, backfill: bool = False) -> list[str]:
    out_path = path.rsplit(".", 1)[0] + ".scores.jsonl"
    if not os.path.exists(path):
        return []
    # P0-4 (carpet-audit 2026-09-23): SCORE-DATA VERSION LOCK. The old
    # append-only scoring trusted a mutable input file forever — a resident
    # postpass rewrote answers mid-judging and scores silently diverged from
    # data (the voided 0.5165). Now: every score row records the input file's
    # sha256; a changed hash invalidates ALL prior rows of that arm (they are
    # quarantined to .stale and scoring restarts from the current data).
    import hashlib as _hl
    _data_hash = _hl.sha256(open(path, "rb").read()).hexdigest()[:16]
    scored: dict[str, dict] = {}
    stale = False
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                if r.get("data_hash") and r["data_hash"] != _data_hash:
                    stale = True
                    break
                scored[r["qid"]] = r   # last row per qid wins
        if stale:
            os.replace(out_path, out_path + f".stale.{int(time.time())}")
            print(f"[judge] {arm}: input data changed ({_data_hash}) — prior "
                  f"scores quarantined, rescoring from current data", flush=True)
            scored = {}
    rows = json.load(open(path, encoding="utf-8"))
    newly = []
    for r in rows:
        qid = r.get("qid")
        if not qid or qid in scored:
            continue
        if r.get("err") is not None and r.get("err") != "":
            continue
        ans = (r.get("answer_official_all")
               or r.get("answer_official_first") or "")
        if not ans.strip():
            continue
        g = gold.get(qid)
        if g is None:
            continue
        row = _score_row(arm, r, qid, ans, g, data_hash=_data_hash)
        newly.append(row)
    if backfill:
        # rows scored before Track 2 existed: recompute with AutoAIS
        for qid, old in scored.items():
            if "ais_rec" in old:
                continue
            match = next((r for r in rows if r.get("qid") == qid), None)
            g = gold.get(qid)
            if match is None or g is None:
                continue
            ans = (match.get("answer_official_all")
                   or match.get("answer_official_first") or "")
            if not ans.strip():
                continue
            newly.append(_score_row(arm, match, qid, ans, g, data_hash=_data_hash))
    if newly:
        with open(out_path, "a", encoding="utf-8") as f:
            for row in newly:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return [r["qid"] for r in newly]


def _score_row(arm, r, qid, ans, g, data_hash: str = "") -> dict:
    s = citation_f1(ans, g.get("output") or "")
    ais = autoais_scores(ans, g.get("ctxs") or []) or {}
    return {"qid": qid, "arm": arm, "policy": "all", "data_hash": data_hash,
            "n_cited_pids": len(r.get("cited_pids") or []),
            "citation_dropped": r.get("citation_dropped"),
            "citation_mapped": r.get("citation_mapped"),
            "wall_s": r.get("wall_s"), **s, **ais}


def summarize() -> dict:
    out = {}
    for arm, path in ARMS.items():
        sp = path.rsplit(".", 1)[0] + ".scores.jsonl"
        rows: dict[str, dict] = {}
        if os.path.exists(sp):
            for l in open(sp, encoding="utf-8"):
                l = l.strip()
                if l:
                    try:
                        r = json.loads(l)
                        rows[r["qid"]] = r
                    except Exception:
                        pass
        if rows:
            rs = list(rows.values())
            f1 = [r["f1"] for r in rs]
            rec = [r["ais_rec"] for r in rs if "ais_rec" in r]
            prec = [r["ais_prec"] for r in rs if "ais_prec" in r]
            out[arm] = {"n": len(rs),
                        "f1_mean": round(sum(f1) / len(f1), 4),
                        "ais_rec_mean": round(sum(rec) / len(rec), 4) if rec else None,
                        "ais_prec_mean": round(sum(prec) / len(prec), 4) if prec else None,
                        "ais_n": len(rec),
                        "zero_cite_rows": sum(1 for r in rs if r.get("n_pred") == 0),
                        # N3: the unparsed-verdict counter is now actually
                        # disclosed (the old code counted it but never read it)
                        "verdict_unparsed": _unparsed["n"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--backfill", action="store_true",
                    help="also compute AutoAIS for rows scored before Track 2")
    ap.add_argument("--snapshot",
                    help="judge from a frozen input snapshot made by "
                         "multi_judge_freeze.py (P2-9) instead of the live "
                         "answers files")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    global QFILE
    if args.snapshot:
        mani_p = os.path.join(args.snapshot, "MANIFEST.json")
        mani = json.load(open(mani_p, encoding="utf-8"))
        for arm in list(ARMS):
            entry = mani["files"].get(arm)
            if not entry:
                print(f"[judge] snapshot missing arm {arm} — skipping it",
                      flush=True)
                ARMS.pop(arm, None)
                continue
            ARMS[arm] = os.path.join(args.snapshot, entry["path"])
        qe = mani["files"].get("questions")
        if qe:
            QFILE = os.path.join(args.snapshot, qe["path"])
        print(f"[judge] frozen inputs: {args.snapshot} "
              f"(arms={sorted(ARMS)})", flush=True)

    _load_cache()
    gold = load_gold()
    print(f"[judge] gold {len(gold)}q | model={JUDGE_MODEL} workers={JUDGE_WORKERS} "
          f"| verdict cache {len(_verdict_cache)} entries", flush=True)
    while True:
        total_new = 0
        for arm, path in ARMS.items():
            try:
                new = judge_arm(arm, path, gold, backfill=args.backfill)
            except json.JSONDecodeError as e:
                print(f"[judge] {arm}: answers file mid-write ({e}) — retry "
                      f"next cycle", flush=True)
                continue
            if new:
                total_new += len(new)
                s = summarize().get(arm) or {}
                print(f"[judge] {arm}: +{len(new)} scored "
                      f"(n={s.get('n')} f1={s.get('f1_mean')} "
                      f"ais_rec={s.get('ais_rec_mean')} "
                      f"ais_prec={s.get('ais_prec_mean')})", flush=True)
        if args.once:
            break
        # backfill only on the first pass
        args.backfill = False
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
