# -*- coding: utf-8 -*-
"""PaperScope pilot A1 round-2 driver (IL-P6 fix package).

Harness = evidence_pilot_a1r2_harness.py (translated/genericized fork of the
frozen B3 loop; F2 fallback-compile + F5 cap-12 inside).
This driver adds:
- runtime {kb_stats} injection (generic self-description, zero hardcoding)
- dynamic NAME_NUMS (KB stats replace home constants)
- F3 answer de-internalization (post-gate presentation pass: [paper_id] ->
  (Author et al., year); strip raw hex record ids; raw kept in answer_raw)
- PS question bindings + output answers_pilot_a1r2.json

Round-3 (R3 fix package F6-F10 per PROMPT-AUDIT-R2 / DIAGNOSIS-GAP-R2):
- --run tag (default a1r3) -> answers_pilot_<tag>.json + <tag>_cost.json
- --smoke: 2-question verification run, tight cap, separate output
- F8: find_gap instance wrapper adds paper_id scoping (home tools.py frozen);
  grounding lists normalized to half-width parens; kb_stats injected into
  CONTROL_SYSTEM as well (F10 translation)

Usage: python evidence_pilot_a1r2.py [--run a1r3] [--smoke] [--qids a,b | --limit N]
"""
import argparse, json, os, re, sys

SB = "C:/Users/D0n9/Desktop/LogicKG/.research_tmp/experiments/stageB"
R2 = f"{SB}/paperscope_r2"
sys.path.insert(0, "C:/Users/D0n9/Desktop/LogicKG/src")
sys.path.insert(0, SB)

import evidence_pilot_a1r2_harness as H
from kb_compiler.views.tools import KBTools

H.BROAD_TYPES = H.BROAD_TYPES | {"trend", "gap", "results_comparison"}
# IL-P8 lesson applied: caps from MEASURED unit cost, not band estimates.
# r2 measured: 15q -> 1.98M (first run), 15q -> 2.12M (gapfill) = ~132-141k/q
# at cap-20 with substantive notes. r3 dev30: 30 x 141k = 4.23M -> cap 4.5M.
TOKEN_CAP_R3 = 4_900_000   # 30q x 141k measured + F9 recompile allowance + margin
TOKEN_CAP_SMOKE = 700_000


def wrap_compare(kb):
    """F14 (card-fix smoke finding): compare(entities=[...]) returned zero rows
    because PS16 matrix rows carry DESCRIPTOR-style entity strings (e.g.
    "BPJDet (anchor-based body-to-part)") that canonical-name args never match,
    while bare compare(metric=...) returned the rows fine — the model's natural
    call shape (entities + metric) was silently dead, resu answer ended with 0
    numerics. Fork-side wrapper: native call first; if entities given and n==0,
    retry wide (no entities) and filter rows fork-side by name containment
    (canonical + aliases + requested string, case-insensitive). Deterministic,
    corpus-generic, home tools.py untouched."""
    orig = kb.compare

    def compare_scoped(subject=None, metric=None, entities=None, band=None):
        res = orig(subject=subject, metric=metric, entities=entities, band=band)
        if entities and isinstance(res, dict) and not res.get("n"):
            wide = orig(subject=subject, metric=metric, entities=None, band=band)
            rows = wide.get("rows", []) if isinstance(wide, dict) else []
            if rows:
                names = set()
                for e in entities:
                    s0 = re.sub(r"\s+", " ", str(e).strip().lower())
                    if s0:
                        names.add(s0)
                    try:
                        ent = kb.resolve(e)
                    except Exception:
                        ent = None
                    if ent:
                        names.add(str(ent.get("canonical", "")).lower())
                        names.update(str(a).lower() for a in ent.get("aliases", []))
                names = {n for n in names if len(n) >= 2}
                keep = [r for r in rows
                        if any(n in (str(r.get("entity", "")) + " "
                                     + str(r.get("subject", ""))).lower()
                               for n in names)]
                if keep:
                    res = dict(wide)
                    res["rows"] = keep
                    res["n"] = len(keep)
                    res["entities_filter"] = "applied fork-side (descriptor rows)"
        if isinstance(res, dict) and not res.get("n") and (subject or metric or entities):
            res["note"] = ("zero rows — matrix rows may carry descriptive entity/"
                           "subject strings; try compare(metric=<exact metric from "
                           "the Metric list>) with no other args and read the rows")
        return res

    kb.compare = compare_scoped
    return kb


def wrap_find_gap(kb):
    """F8: add paper_id scoping to find_gap WITHOUT touching the frozen home
    module — instance-level wrapper; absences_extracted entries carry paper_id
    (verified in views_ps16 coverage)."""
    orig = kb.find_gap

    def find_gap_scoped(entity=None, subject_family=None, paper_id=None):
        res = orig(entity=entity, subject_family=subject_family)
        if paper_id and isinstance(res, dict):
            res["absences_extracted"] = [a for a in res.get("absences_extracted", [])
                                         if a.get("paper_id") == paper_id]
            res["paper_id_scope"] = paper_id
        return res

    kb.find_gap = find_gap_scoped
    return kb


def build_tools_ps16():
    records = json.load(open(f"{R2}/ps16_postcheck/records_checked.json", encoding="utf-8"))
    records.pop("canary", None)
    # R3-F13: views_ps16.json's cards slot was HOME residue (37 DRL dossiers,
    # zero PS16) — card() errored on every PS16 entity in r1/r2 (80 dead calls
    # in r2) and kb_stats told the model "37 method dossiers" existed.
    # views_ps16_r3.json = same file with cards rebuilt from PS16 records
    # (329 dossiers) + stats corrected. Original kept for r2 reproducibility.
    views = json.load(open(f"{R2}/views_ps16_r3.json", encoding="utf-8"))
    registry = json.load(open(f"{R2}/registry_v32_pilot.json", encoding="utf-8"))
    # R3-F11 (found during fix verification): the governance pass seeded from the
    # home v32 registry — ALL 40 starred entities were home DRL residue pointing
    # at rl40 paper ids that do not exist in PS16 (false "full text available"
    # markers; the ground_lists sparse-fill even injected Agent57/Ape-X/Rainbow*
    # into PS prompts), while genuine PS16 entities carried NO star. Rebuild the
    # flag from PS16 evidence: star = entity is mentioned by at least one PS16
    # paper (mention_papers ∩ manifest). Cited-reference entities (e.g. GPT-4)
    # get starred under this rule with thinner dossiers — handbook wording says
    # "records grounded in the corpus", which stays true.
    ps_pids = {r["paper_id"] for r in
               json.load(open(f"{R2}/manifest_ps16.json", encoding="utf-8"))}
    n_star = n_unstar = 0
    for e in registry.get("entities", []):
        ps_mention = next((p for p in (e.get("mention_papers") or []) if p in ps_pids), None)
        if ps_mention:
            if e.get("in_corpus_paper_id") != ps_mention:
                e["in_corpus_paper_id"] = ps_mention
                n_star += 1
        elif e.get("in_corpus_paper_id"):
            e["in_corpus_paper_id"] = None
            n_unstar += 1
    print(f"[registry] stars rebuilt: {n_star} set from PS16 mentions, "
          f"{n_unstar} home-residue unstarred", flush=True)
    vocab = json.load(open(f"{SB}/dim_vocab_v1.json", encoding="utf-8"))
    manifest = {r["paper_id"]: r for r in
                json.load(open(f"{R2}/manifest_ps16.json", encoding="utf-8"))}
    kb = KBTools(views, registry, vocab, manifest, records,
                 emb_cache_path=f"{R2}/emb_cache_ps16.bin")
    return kb, records, views, manifest


QFILE = f"{R2}/pilot_questions30.json"   # --qfile overrides (fresh-16终判)


def load_questions_ps16(qids=None, limit=None):
    qs = json.load(open(QFILE, encoding="utf-8"))["questions"]
    out = [{"id": q["qid"], "type": q["prompt_type"], "question": q["question"],
            "gold_hint": q["gold_answer"][:4000]} for q in qs]
    if qids:
        keep = set(qids.split(","))
        out = [q for q in out if q["id"] in keep]
    if limit:
        out = out[:limit]
    return out


def deinternalize(answer, manifest):
    """F3: presentation-layer pass AFTER in-loop gates. [pid] -> (Author et al., year);
    strip leftover raw hex record ids and note-line markers."""
    if not answer:
        return answer
    out = answer
    for pid, m in manifest.items():
        auth = ""
        if m.get("authors"):
            a0 = m["authors"][0] if isinstance(m["authors"], list) else str(m["authors"])
            auth = str(a0).split()[-1].rstrip(",")  # surname
        yr = m.get("venue_year") or m.get("arxiv_year") or ""
        if auth and yr:
            cite = f"{auth} et al., {yr}"
        elif auth:
            cite = auth
        else:
            title = (m.get("title") or pid)
            words = [w for w in re.split(r"[\s:]+", title) if w and w[0].isalnum()][:4]
            cite = " ".join(words) if words else pid  # short-title fallback (never raw pid unless titleless)
        out = out.replace(f"[{pid}]", f"({cite})")
        out = re.sub(rf"\b{re.escape(pid)}\b", cite, out)
    out = re.sub(r"\[[0-9a-fA-F]{8,}\]", "", out)          # raw hex record ids
    # IL-P9: record ids leaked inside compound brackets, e.g. [recid, 24] or
    # [Author et al., 2024, recid] — bare-hex sweep + bracket residue cleanup
    out = re.sub(r"\b[0-9a-f]{12,}\b", "", out)
    out = re.sub(r"\[\s*[0-9,\s]*\]", "", out)
    out = re.sub(r"\[\s*,\s*", "[", out)
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r",\s*\]", "]", out)
    out = re.sub(r"\[\s*\]", "", out)
    out = re.sub(r"(?m)^\s*[NXG]\d+\.\s*", "", out)          # note-line markers
    out = re.sub(r"\[unsourced\]", "", out)
    return out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--run", default="a1r3")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--qfile", default="")
    args = ap.parse_args()

    if args.qfile:
        global QFILE
        QFILE = args.qfile
    tag = "smoke_r3" if args.smoke else args.run
    if args.smoke and not args.qids:
        # verification pair: the NEFTune gap loss (F6/F8/F9 targets) + the
        # worst resu loss (F7 comparison-structure target)
        args.qids = "PS-gap-imp-13a60827ac,PS-resu-imp-d070cc4638"
    qs = load_questions_ps16(args.qids or None, args.limit or None)
    caps = {"a1f16": 2_600_000}   # fresh-16终判: 16q x ~145k 实测单价 + margin
    H.TOKEN_CAP_TOTAL = (TOKEN_CAP_SMOKE if args.smoke
                         else caps.get(args.run, TOKEN_CAP_R3))
    print(f"[{tag}] questions: {len(qs)} cap={H.TOKEN_CAP_TOTAL}", flush=True)
    outp = f"{R2}/answers_{tag if args.smoke else 'pilot_' + tag}.json"
    glog = []

    kb, records, views, manifest = build_tools_ps16()
    wrap_find_gap(kb)
    wrap_compare(kb)
    n_rec = views.get("stats", {}).get("n_records", 0)
    n_pap = len(manifest)
    stats = views.get("stats", {})
    kb_stats = (f"The knowledge base was compiled from a corpus of {n_pap} scientific "
                f"papers into {n_rec} typed records (kinds: finding, config, result, "
                f"absence, lineage, shift, notation), plus precompiled views "
                f"({stats.get('matrix_tables', 0)} comparison matrices, "
                f"{stats.get('genealogy_edges', 0)} lineage edges, "
                f"{stats.get('cards', 0)} method dossiers, "
                f"{stats.get('coverage_entities', 0)} coverage entities).")
    assert "{kb_stats}" in H.MAIN_SYSTEM
    H.MAIN_SYSTEM = H.MAIN_SYSTEM.replace("{kb_stats}", kb_stats)
    if "{kb_stats}" in H.CONTROL_SYSTEM:   # F10: control arm same treatment
        H.CONTROL_SYSTEM = H.CONTROL_SYSTEM.replace("{kb_stats}", kb_stats)
    print(f"[{tag}] kb_stats injected: {kb_stats[:120]}...", flush=True)

    for pid, payload in records.items():
        recs = payload.get("records", payload) if isinstance(payload, dict) else payload
        for r in recs:
            H.REC_INDEX[r.get("id")] = r
    grounding = H.build_grounding(kb)
    # F8: alias lists render with full-width parens in the shared home module —
    # normalize so the injected entity list is uniformly English typography
    for k in list(grounding):
        if isinstance(grounding[k], str):
            grounding[k] = grounding[k].replace("（", "(").replace("）", ")")
    # R3-F11: dedupe entity list by canonical name (registry carries repeated
    # canonical entries, e.g. DoReMi x7 — pure injection noise). Split on ", "
    # is safe here: aliases render inside () separated by |.
    seen, ded = set(), []
    for part in grounding.get("entities", "").split(", "):
        key = part.split("(")[0].strip().lower()
        if key and key not in seen:
            seen.add(key)
            ded.append(part)
    grounding["entities"] = ", ".join(ded)
    H.NAME_NUMS.update(H.build_name_nums(kb))
    H.NAME_NUMS.update({str(n_rec), str(n_pap)})

    H.stage_answer("main", qs, outp, kb, None, grounding, glog)

    # F3 post-pass (presentation only; gates already ran in-loop)
    # SKIP_DEINTERNALIZE (Multi runner, 2026-09-23 N1): the official citation
    # bridge consumes raw [pid] markers — keep `answer` unmodified there.
    rows = json.load(open(outp, encoding="utf-8"))
    if not getattr(sys.modules[__name__], "SKIP_DEINTERNALIZE", False):
        for r in rows:
            if r.get("answer") and "answer_raw" not in r:
                r["answer_raw"] = r["answer"]
                r["answer"] = deinternalize(r["answer"], manifest)
    _tmp = outp + ".tmp"
    with open(_tmp, "w", encoding="utf-8") as _f:
        json.dump(rows, _f, ensure_ascii=False, indent=1)
    os.replace(_tmp, outp)  # N11 atomic write
    n_fb = sum(1 for r in rows if r.get("gate", {}).get("forced_notes_submit"))
    n_fc = sum(1 for r in rows if r.get("gate", {}).get("fallback_compiled"))
    fg = {k: sum(1 for r in rows if r.get("gate", {}).get(k))
          for k in ("absence_repair", "absence_recompile",
                    "false_absence_unrepaired", "false_absence_forced")}
    json.dump({"calls": H.COST["calls"], "est_tokens": H.COST["est_tokens"],
               "by_stage": dict(H.COST["by_stage"]), "backfills": glog[:50],
               "model": H.MODEL, "fallback_submit": n_fb, "fallback_compiled": n_fc,
               "f9_flags": fg},
              open(f"{R2}/{tag}_cost.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[{tag}] done: {len(rows)} answers, fallback_submit={n_fb} (compiled={n_fc}), F9={fg}",
          flush=True)
    print("cost:", json.dumps(dict(H.COST["by_stage"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
