# -*- coding: utf-8 -*-
"""PS53 arm runner (PS53-SCALE-PREREG v1.1, 2026-09-18): frozen gate2f stack
over the D2 hard-negative staging.

Derived from psfix_2026-09-11/psv5/psv_run.py; deltas: RB = arm staging dir
(from env PS53_ARM_DIR), TAG from env, qfile from staging. Everything else
byte-identical to the frozen gate2f wrapper (harness aliasing + F18 +
COACH 10-14 + F21b projections + F31V2 case-A + G2_F31 default ON).
"""
from __future__ import annotations

import json
import os
import sys

SRC = r"C:\Users\D0n9\Desktop\CompileScholar\src"
STAGEB = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\stageB"
RB = os.environ["PS53_ARM_DIR"]
G2DIR = os.path.join(STAGEB, "rebuild27b_2026-09-11")
sys.path.insert(0, SRC)
sys.path.insert(0, STAGEB)
sys.path.insert(0, G2DIR)
sys.path.insert(0, RB)

MODEL = os.environ.get("G2_MODEL", "Qwen3.8-Max")
TAG = os.environ.get("G2_TAG", "ps53")
os.environ["LLM_CALL_LOG"] = os.path.join(RB, f"ledger_gate2_{TAG}.jsonl")
os.environ["LLM_RUN_ID"] = f"gate2-{TAG}"
os.environ.setdefault("LLM_SOCK_TIMEOUT", "200")
os.environ.setdefault("LLM_WALL_TIMEOUT", "420")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # co-located _shared/tools copy (verified byte-identical to d2/repair_stack original pre-rename)
import evidence_gate2r_harness as _HF  # noqa: E402  (PS53 repair fork: R-E)
sys.modules["evidence_pilot_a1r2_harness"] = _HF

import evidence_pilot_a1r2 as D  # noqa: E402
from f18_resolver import F18Resolver, wrap_f18  # noqa: E402
assert D.H is _HF, "harness aliasing failed"

D.R2 = RB
# R-A (PS53 repair, 2026-09-18, user ruling): NO global run-level token cap —
# budget-abort at step 0 canned 3 questions in D53 (systematic tail-question
# crippling). Discipline moves to: per-step caps stay; loop safety via
# STEP_CAP ladder + anti-loop breakers; cost accounting post-hoc via ledger.
D.TOKEN_CAP_R3 = int(os.environ.get("G2_CAP", "999999999999"))
D.H.MODEL = MODEL

COACH = (
    "\n\n10. Entity-argument discipline: entity arguments to card()/findings() "
    "MUST be entity names from the grounding list or from previous tool outputs. "
    "Paper titles, paper ids, matrix row labels and free-text descriptions are "
    "NOT entity names. When a tool response carries a resolver note, follow it: "
    "reuse the suggested canonical name verbatim, or switch to "
    "findings(paper_id=...) for per-paper search. Do not retry the same "
    "malformed name in a different form — each retry burns a step."
    "\n11. Comparison-question budget: lock the entity set first "
    "(entities()/card()), then pull numbers claim-by-claim (findings/compare), "
    "and stop tool use with 4 steps remaining to write final notes. Every "
    "number written into notes MUST carry its metric name and [record_id] on "
    "the same line — a number without metric+id is unusable in the final answer."
    "\n12. Absence-claim exhaustion discipline: before stating that any item "
    "is 'not found', 'missing', or 'not retrievable', you MUST have completed "
    "a per-paper sweep for EVERY relevant paper: "
    "findings(claim_type=criticism, paper_id=X), find_gap(paper_id=X), and "
    "findings(paper_id=X, contains=<2-3 alternative phrasings>). Defect and "
    "limitation information may live in criticism findings, absence records, "
    "or plain stated findings — sweep all three channels. A negative claim "
    "without a completed per-paper sweep is a protocol violation: report the "
    "sweep you actually ran, per paper."
    "\n13. Request targeting and numeric density: in scenario-framed "
    "questions, first identify the actual request (the comparison, enumeration "
    "or analysis being asked for); the scenario's stack details are context, "
    "not the question. For comparison requests, the final answer MUST carry "
    "the concrete values (numbers with metric names and per-paper attribution) "
    "retrieved via compare()/findings(); a purely qualitative comparison when "
    "numeric records exist is incomplete. Content retrieved from one paper's "
    "records must never be attributed to another paper."
    "\n14. Multi-paper comparison playbook: for questions comparing N papers, "
    "spend at least 60% of steps on retrieval; final notes must anchor every "
    "one of the N papers with at least one record id."
)


def wrap_f21(kb, views):
    """F21b compact projections (copied verbatim from gate2f_run.py)."""
    def _c(t, r, cap):
        return r

    def findings_cx(entity=None, claim_type=None, contains=None, paper_id=None, k=40):
        # FINDINGS-WINDOW WIDENING (2026-09-20, 批5 queue item 2): port of psv5
        # IL-G2e-1 slim hybrid (was queued in REPAIR-LOG GOLDCOV round as
        # "findings 可见窗口加宽——独立变量独立轮"). Old form: 4 full + 14
        # 3-column skeleton rows (no quote/claim_type) — the model lost the
        # verbatim anchor and the genre filter when paging. Slim form: 3 full
        # + 12 five-column rows sized to fit the 4300 findings cap WITHOUT
        # shrink (all 15 entries visible; rows are DICTS — the PSV-IL-4/
        # GOLD-COV lesson: compact() does {**e} per entry, list/str rows die).
        res = _orig_findings(entity=entity, claim_type=claim_type,
                             contains=contains, paper_id=paper_id, k=k)
        entries = res.get("entries")
        if not entries:
            return res
        if len(entries) > 8:
            def _frow(e):
                # slim row: scope dropped (usually empty), claim 70 / quote 60
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
        # GOLD-COV FIX (2026-09-19, GOLDCOV-VERDICT.md §三): the string/list
        # rows below crashed evidence_b2_tools.compact()'s card branch
        # ({**x} on a str -> TypeError -> 99-char "bad args" stub) — 110/140
        # card calls in the terminal run returned nothing, ALL rich cards died
        # (perfect inversion: 101/110 crashed were rich, 30/30 survivors were
        # empty shells). Port of psv5's PSV-IL-5 fix: emit DICT rows, which
        # compact() then slices safely ({**x, "quote": ...} works on dicts).
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
                                 "claim": (x.get("claim") or "")[:100],
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
        # R-B (PS53 repair): entity-suggestion injection. Deep-read evidence:
        # compare usage fell 36% on the 4.5x-larger KB (models retreat to
        # card/findings when unsure which entities to pass). Deterministic
        # assist: when the model called compare WITHOUT entities (the
        # hesitation signature) and the result has rows, surface the
        # entities that actually appear — visible next-step material for
        # the refine/band calls. KB-structure-only input; zero gold.
        if not entities and rows:
            from collections import Counter
            ent_counts = Counter(r.get("entity") for r in rows if r.get("entity"))
            if ent_counts:
                top = ", ".join(f"{e} ({n} rows)" for e, n in ent_counts.most_common(6))
                res["note_rb"] = (f"entities present in these rows: {top} — "
                                  "pass entities=[...] to focus on specific methods")
        return res

    # R-C (PS53 repair): fetch_chunk(record_id) -> original text window around
    # the record's chunk. Closes the granularity gap the deep-read exposed:
    # record quotes anchor values in <=40 words, but comparison questions
    # sometimes need surrounding context (table rows, neighboring sentences).
    # A9 reads raw text; we now can too — deterministically, record-anchored.
    _D2_TEXTS = os.path.join(os.path.dirname(RB), "..", "..", "d2", "texts")

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
        tf = os.path.join(_D2_TEXTS, pid + ".md")
        if not os.path.exists(tf):
            return {"tool": "fetch_chunk", "error": f"text file for {pid} not on disk"}
        txt = open(tf, encoding="utf-8", errors="replace").read()
        start = rec.get("chunk_char_start") or 0
        lo = max(0, start - window // 2)
        hi = min(len(txt), start + window)
        out = {"tool": "fetch_chunk", "record_id": record_id, "paper_id": pid,
               "quote": rec.get("quote"), "chunk_id": rec.get("chunk_id"),
               "text_window": txt[lo:hi]}
        return out

    _orig_findings, _orig_card, _orig_compare = kb.findings, kb.card, kb.compare
    kb.findings, kb.card, kb.compare = findings_cx, card_cx, compare_cx
    kb.fetch_chunk = fetch_chunk
    return kb


if __name__ == "__main__":
    argv = sys.argv[1:]
    sys.argv = ["ps53", "--run", TAG] + argv
    if not any(a.startswith("--qfile") for a in argv):
        sys.argv += ["--qfile", os.path.join(RB, "pilot_questions30.json")]

    _orig_build = D.build_tools_ps16

    def build_tools_f21():
        kb, records, views, manifest = _orig_build()
        wrap_f21(kb, views)
        cards = json.load(open(os.path.join(G2DIR, "cards_rebuild.json"), encoding="utf-8"))
        resolver = F18Resolver(kb, manifest, kb.registry, views, cards=cards)
        wrap_f18(kb, resolver, log=F18_LOG)
        n_cards = len((views.get("cards") or {}).get("cards") or {})
        print(f"[F21] armed: resolver {len(resolver.pid2entity)} papers, "
              f"cards {n_cards}", flush=True)
        return kb, records, views, manifest

    F18_LOG = []
    D.build_tools_ps16 = build_tools_f21
    try:
        D.main()
    finally:
        json.dump(F18_LOG, open(os.path.join(RB, f"f18_events_{TAG}.json"), "w",
                  encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[F21] {len(F18_LOG)} resolver events", flush=True)
