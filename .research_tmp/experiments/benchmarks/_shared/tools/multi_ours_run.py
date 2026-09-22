# -*- coding: utf-8 -*-
"""Multi-108 OUR-arm runner: the frozen gate2r answer stack over the Multi KB.

Same stack as PS53 (ps53r_run.py): harness aliasing + F18 resolver + F21b
projections + F31V2 notes spec + R-A no-global-cap + scalable/dynamic step
caps. Deltas for Multi-108 (2026-09-23):

  - KB products from scholarqa_multi/kb: postcheck records_checked.json,
    views.json, registry_v2.json (grown), dim_vocab_v1.json, cards.json;
    manifest from corpus/manifest.json
  - qfile = data/scholarqa_multi.json converted GOLD-BLIND (gold_hint="",
    type=aggregation: Multi questions are multi-doc by construction, so the
    broad + corpus-scaled step cap applies uniformly — disclosed in the run
    report)
  - MODEL = local:Qwen3.8-27B — the SAME answering model both baselines use
    (harness _chat routes local: specs to call_local)
  - citations: answers KEEP [paper_id] markers (no deinternalize) — the
    official bridge (answer row build below) translates [pid] -> [ctx_idx]
    via CitationTranslator, same dual all/first policies as the baselines
  - output rows byte-shape-compatible with the baseline answer files so the
    incremental judge loop can treat all three arms uniformly

Resume: answers file rows are keyed by qid; completed qids are skipped.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

SRC = r"C:\Users\D0n9\Desktop\CompileScholar\src"
STAGEB = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\stageB"
G2DIR = os.path.join(STAGEB, "rebuild27b_2026-09-11")
MULTI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "scholarqa_multi")
KB = os.path.join(MULTI, "kb")
ARM = os.path.join(MULTI, "baselines", "ours")
RB = ARM  # harness aliasing convention (D.R2)

sys.path.insert(0, SRC)
sys.path.insert(0, STAGEB)
sys.path.insert(0, G2DIR)
sys.path.insert(0, RB)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = os.environ.get("OURS_MODEL", "local:Qwen3.8-27B")
TAG = os.environ.get("OURS_TAG", "multi")
os.environ.setdefault("LLM_CALL_LOG", os.path.join(ARM, f"ledger_ours_{TAG}.jsonl"))
os.environ.setdefault("LLM_RUN_ID", f"ours-{TAG}")
os.environ.setdefault("LLM_SOCK_TIMEOUT", "900")
os.environ.setdefault("LLM_WALL_TIMEOUT", "1200")
os.environ.setdefault("LOCAL_SOCK_TIMEOUT", "900")
os.environ.setdefault("KB_EMBED_PROVIDER", "local")   # arm purity: embeddings local
os.environ.setdefault("PS53_TEXT_INDEX", os.path.join(ARM, "text_index"))

import evidence_gate2r_harness as _HF  # noqa: E402  (shared _shared/tools copy)
sys.modules["evidence_pilot_a1r2_harness"] = _HF

import evidence_pilot_a1r2 as D  # noqa: E402
from f18_resolver import F18Resolver, wrap_f18  # noqa: E402
assert D.H is _HF, "harness aliasing failed"

D.R2 = RB
D.TOKEN_CAP_R3 = int(os.environ.get("G2_CAP", "999999999999"))  # R-A: no global cap
D.H.MODEL = MODEL

QFILE_OUT = os.path.join(ARM, "questions_multi.json")
ANSWERS = os.path.join(ARM, "answers_ours.json")


# ---------------- qfile conversion (gold-blind) ----------------

def build_arm_qfile() -> str:
    """Official 108-question file -> arm qfile in the PS shape
    ({questions: [{qid, prompt_type, question, gold_answer}]}), gold emptied."""
    src = os.path.join(MULTI, "data", "scholarqa_multi.json")
    qs = json.load(open(src, encoding="utf-8"))
    out = [{"qid": q["id"], "prompt_type": "aggregation",
            "question": q["input"], "gold_answer": ""}
           for q in qs]
    os.makedirs(ARM, exist_ok=True)
    json.dump({"questions": out}, open(QFILE_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return QFILE_OUT


# ---------------- KB build override ----------------

def build_tools_multi():
    from kb_compiler.views.tools import KBTools

    records = json.load(open(os.path.join(KB, "postcheck", "records_checked.json"),
                             encoding="utf-8"))
    views = json.load(open(os.path.join(KB, "views.json"), encoding="utf-8"))
    registry = json.load(open(os.path.join(KB, "registry_v2.json"),
                              encoding="utf-8"))
    vocab = json.load(open(os.path.join(KB, "dim_vocab_v1.json"),
                           encoding="utf-8"))
    manifest = {r["paper_id"]: r for r in
                json.load(open(os.path.join(MULTI, "corpus", "manifest.json"),
                               encoding="utf-8"))}
    # R3-F11 port: star = entity mentioned by a corpus paper (mention_papers
    # ∩ manifest); strip stale in_corpus_paper_id otherwise
    pids = set(manifest)
    n_star = n_unstar = 0
    for e in registry.get("entities", []):
        m = next((p for p in (e.get("mention_papers") or []) if p in pids), None)
        if m:
            if e.get("in_corpus_paper_id") != m:
                e["in_corpus_paper_id"] = m
                n_star += 1
        elif e.get("in_corpus_paper_id"):
            e["in_corpus_paper_id"] = None
            n_unstar += 1
    print(f"[registry] stars rebuilt: {n_star} set from Multi mentions, "
          f"{n_unstar} unstarred", flush=True)
    kb = KBTools(views, registry, vocab, manifest, records,
                 emb_cache_path=os.path.join(ARM, "emb_cache_records.bin"))
    return kb, records, views, manifest


def wrap_f21(kb):
    """F21b projections (ported from ps53r_run.py; texts dir -> Multi corpus)."""
    _orig_findings, _orig_card, _orig_compare = kb.findings, kb.card, kb.compare
    _TEXTS = os.path.join(MULTI, "corpus", "texts")

    def findings_cx(entity=None, claim_type=None, contains=None, paper_id=None, k=40):
        res = _orig_findings(entity=entity, claim_type=claim_type,
                             contains=contains, paper_id=paper_id, k=k)
        entries = res.get("entries")
        if not entries:
            return res
        if len(entries) > 8:
            def _frow(e):
                return {"record_id": str(e.get("record_id") or "")[:14],
                        "claim_type": str(e.get("claim_type") or "")[:12],
                        "paper": str(e.get("paper_id") or "")[:10],
                        "claim": str(e.get("claim") or "")[:70],
                        "quote60": str(e.get("quote") or "")[:60]}
            ents = entries
            full = ents[:3]
            rest = [_frow(e) for e in ents[3:15]]
            res["entries"] = full + rest
            res["compact_row_columns"] = ["record_id", "claim_type", "paper",
                                          "claim", "quote60"]
            res["projection"] = (f"hybrid: first {len(full)} entries full, then "
                                 f"{len(rest)} as compact rows (columns above), "
                                 f"of n={res.get('n')}; narrow with "
                                 "claim_type/entity/contains to page deeper")
        return res

    def card_cx(entity):
        res = _orig_card(entity)
        c = res.get("configs")
        if c:
            res["configs"] = [{"item": str(x.get("item") or "")[:40],
                                "value": str(x.get("value") or "")[:30],
                                "role": x.get("role"),
                                "record_id": x.get("record_id"),
                                "quote": (x.get("quote") or "")[:60]}
                               for x in c[:20]]
            res["note_fy2"] = f"{len(c)} configs compacted to 20"
        f = res.get("findings")
        if f:
            res["findings"] = [{"claim_type": x.get("claim_type"),
                                 "claim": str(x.get("claim") or "")[:100],
                                 "record_id": x.get("record_id")}
                                for x in f[:20]]
            res["note_fy2"] = (f"{len(f)} findings compacted to 20"
                               + ("; " + res["note_fy2"] if res.get("note_fy2") else ""))
        return res

    def compare_cx(subject=None, metric=None, entities=None, band=None):
        res = _orig_compare(subject=subject, metric=metric,
                            entities=entities, band=band)
        rows = res.get("rows")
        if rows and len(rows) > 24:
            for r in rows[24:]:
                r.pop("quote", None)
            res["note_fy1"] = f"{len(rows)} rows: first 24 full quotes kept"
        if not entities and rows:
            from collections import Counter
            ent_counts = Counter(r.get("entity") for r in rows if r.get("entity"))
            if ent_counts:
                top = ", ".join(f"{e} ({n} rows)" for e, n in ent_counts.most_common(6))
                res["note_rb"] = (f"entities present in these rows: {top} — "
                                  "pass entities=[...] to focus on specific methods")
        return res

    def fetch_chunk(record_id=None, window=1500):
        if not record_id:
            return {"tool": "fetch_chunk", "error": "record_id required (from a note anchor or tool row)"}
        rec = None
        for pid, payload in (kb.records or {}).items():
            for r in (payload.get("records") or []):
                if r.get("id") == record_id:
                    rec = r
                    break
            if rec:
                break
        if rec is None:
            return {"tool": "fetch_chunk", "error": f"record_id {record_id} not found in KB"}
        pid = rec.get("paper_id")
        tf = os.path.join(_TEXTS, pid + ".md")
        if not os.path.exists(tf):
            tf = os.path.join(_TEXTS, pid + ".txt")
        if not os.path.exists(tf):
            return {"tool": "fetch_chunk", "error": f"text file for {pid} not on disk"}
        txt = open(tf, encoding="utf-8", errors="replace").read()
        start = rec.get("chunk_char_start") or 0
        lo = max(0, start - window // 2)
        hi = min(len(txt), start + window)
        return {"tool": "fetch_chunk", "record_id": record_id, "paper_id": pid,
                "quote": rec.get("quote"), "chunk_id": rec.get("chunk_id"),
                "text_window": txt[lo:hi]}

    kb.findings, kb.card, kb.compare = findings_cx, card_cx, compare_cx
    kb.fetch_chunk = fetch_chunk
    return kb


# ---------------- official-format answer rows ----------------

_PID_MARK = re.compile(r"\[([A-Za-z0-9_\-]{6,80})\]")


def to_official_row(r: dict, id_mapping: dict) -> dict:
    """Native result row -> baseline-compatible official answer row.
    [paper_id] markers translate 1:1 to stems (paper_id == corpus text stem)."""
    from multi_baseline_common import CitationTranslator
    qid = r["id"]
    raw = r.get("answer") or ""
    tr = CitationTranslator(qid, id_mapping)
    pairs, seen = [], set()
    for m in _PID_MARK.finditer(raw):
        pid = m.group(1)
        if pid in seen:
            continue
        seen.add(pid)
        pairs.append((m.group(0), pid))
    off_all, cites = tr.translate(raw, pairs, policy="all")
    off_first, _ = tr.translate(raw, pairs, policy="first")
    return {"qid": qid, "question": r.get("question"),
            "raw_answer": raw, "references": [],
            "answer_official_all": off_all, "citations_all": cites,
            "answer_official_first": off_first,
            "citation_mapped": tr.mapped, "citation_dropped": tr.dropped,
            "unresolved_markers": sum(1 for _, s in pairs
                                      if not (id_mapping.get(qid) or {}).get(s)),
            "steps": r.get("steps"), "gate": r.get("gate"),
            "cited_pids": sorted(seen)}


def main():
    os.makedirs(ARM, exist_ok=True)
    qfile = build_arm_qfile()
    argv = sys.argv[1:]
    sys.argv = ["multi", "--run", TAG, "--qfile", qfile] + argv

    _orig_build = D.build_tools_ps16

    def build_tools_f21():
        kb, records, views, manifest = build_tools_multi()
        wrap_f21(kb)
        cards = json.load(open(os.path.join(KB, "cards.json"), encoding="utf-8"))
        resolver = F18Resolver(kb, manifest, kb.registry, views, cards=cards)
        wrap_f18(kb, resolver, log=F18_LOG)
        n_cards = len(cards)
        print(f"[F21] armed: resolver {len(resolver.pid2entity)} papers, "
              f"cards {n_cards}", flush=True)
        return kb, records, views, manifest

    F18_LOG = []
    D.build_tools_ps16 = build_tools_f21
    try:
        D.main()   # runs stage_answer -> answers_{tag}.json in RB
    finally:
        json.dump(F18_LOG, open(os.path.join(ARM, f"f18_events_{TAG}.json"), "w",
                                encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[F21] {len(F18_LOG)} resolver events", flush=True)

    # post-pass: native results -> official-format answer rows
    native = os.path.join(RB, f"answers_pilot_{TAG}.json")
    if os.path.exists(native):
        from multi_baseline_common import load_id_mapping
        id_mapping = load_id_mapping()
        rows = json.load(open(native, encoding="utf-8"))
        done = {}
        if os.path.exists(ANSWERS):
            done = {r["qid"]: r for r in json.load(open(ANSWERS, encoding="utf-8"))}
        for r in rows:
            if r.get("answer"):
                done[r["id"]] = to_official_row(r, id_mapping)
        json.dump(list(done.values()), open(ANSWERS, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"[ours] official rows: {len(done)} -> {ANSWERS}", flush=True)


if __name__ == "__main__":
    import sys as _s
    _s.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
