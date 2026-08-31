"""Case-library regression runner (2026-08-26) — the 10-minute exam.

Re-runs the REAL extraction stages (plan -> execute -> gate -> verify -> fix)
on each stored case chunk with a FROZEN schema (case_library/schema_frozen.json),
then scores the kept edges against the case's expectation. No map_structure,
no evolution, no A-box accumulation — the unit under test is extraction-stack
behavior on the failing sentence, so a fix round is ~minutes, not 1.5h.

Usage (repo root):
  python src/run_cases.py --cases .research_tmp/case_library/cases_main.jsonl \
      [--classes negation,routing,polarity,lawinput,setup] [--limit N] \
      [--out .research_tmp/case_library/runs/<name>.jsonl] [--times 1]

Outcomes per case kind:
  fail: STILL_WRONG (banned binding reappeared) > FIXED_CORRECT (good binding /
        alt pattern present) > FIXED_DROP (no watched edge — acceptable fix)
  pass: RETAINED / REGRESSED (soft — single-run miss may be variance; use
        --times 2+ before calling a regression real)
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LIB = os.path.join(".research_tmp", "case_library")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]{3,}", _norm(s)))


def _watches(edge_ev: str, watch: str, thr: float = 0.5) -> bool:
    """Does this edge's evidence span sit on the watched sentence?"""
    wt = _tokens(watch)
    if not wt:
        return False
    return len(wt & _tokens(edge_ev)) / len(wt) >= thr


def _role_ok(he, spec: dict, nid2surface: dict) -> bool:
    """role_contains: every named role has SOME node in that role whose surface
    contains one of the keywords. surface_any: some node surface hits a keyword."""
    roles = he.node_roles
    for role, kws in (spec.get("role_contains") or {}).items():
        hits = [nid2surface.get(n, "") for n, r in zip(he.node_ids, roles)
                if r == role]
        if not any(any(_norm(k) in _norm(s) for k in kws) for s in hits if s):
            return False
    kws = spec.get("surface_any")
    if kws:
        surfaces = [nid2surface.get(n, "") for n in he.node_ids]
        if not any(any(_norm(k) in _norm(s) for k in kws) for s in surfaces if s):
            return False
    return True


def evaluate(case: dict, kept, nodes) -> dict:
    exp = case["expect"]
    nid2surface = {n.nid: n.surface for n in nodes}
    watched = [he for he in kept if _watches(he.evidence_span, exp["watch"])]
    w_summ = [{"pattern": he.pattern_type,
               "roles": {r: nid2surface.get(n, "") for n, r in
                         zip(he.node_ids, he.node_roles)},
               "evidence": he.evidence_span[:120]} for he in watched]

    if case["kind"] == "pass":
        gb = exp.get("good_binding") or {}
        ok = any(he.pattern_type == gb.get("pattern") and
                 _role_ok(he, gb, nid2surface) for he in watched)
        return {"outcome": "RETAINED" if ok else "REGRESSED",
                "watched": w_summ}

    bad = exp.get("bad") or {}
    still = any(he.pattern_type == bad.get("pattern") and
                _role_ok(he, bad, nid2surface) for he in watched)
    if still:
        return {"outcome": "STILL_WRONG", "watched": w_summ}
    gb = exp.get("good_binding") or {}
    alts = set(exp.get("good_alt") or [])
    if gb and any(he.pattern_type == gb.get("pattern") and
                  _role_ok(he, gb, nid2surface) for he in watched):
        return {"outcome": "FIXED_CORRECT", "watched": w_summ}
    if alts and any(he.pattern_type in alts for he in watched):
        return {"outcome": "FIXED_CORRECT", "watched": w_summ}
    return {"outcome": "FIXED_DROP", "watched": w_summ}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--classes", default="", help="comma filter on failure_class")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--times", type=int, default=1,
                    help="repeats per case (variance guard; outcome = majority)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from granular_agent.hypergraph_schema import MetaHypergraph
    from granular_agent.concept_graph import ConceptGraph
    from granular_agent.knowledge_base import KnowledgeBase
    from granular_agent.extraction_agent import ExtractionAgent
    from granular_agent.llm_client import call_paratera, call_llm

    schema = json.load(open(os.path.join(LIB, "schema_frozen.json"), encoding="utf-8"))
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8")]
    if args.classes:
        keep = set(args.classes.split(","))
        cases = [c for c in cases if c.get("failure_class") in keep]
    if args.limit:
        cases = cases[:args.limit]
    print(f"[cases] {len(cases)} cases x{args.times} | SOFT_ROUTING={os.environ.get('SOFT_ROUTING','(unset)')}")

    def _mk_agent() -> ExtractionAgent:
        tbox = MetaHypergraph.from_dict(json.loads(json.dumps(schema)))
        kb = KnowledgeBase(tbox=tbox, abox=ConceptGraph(),
                           domain_ns={"global": True, "granular": True,
                                      "ml": True, "molecular": True})
        return ExtractionAgent(
            kb,
            llm_extract=lambda p, mt=8000: call_paratera(
                p, model="DeepSeek-V4-Flash", max_tokens=mt, enable_thinking=False),
            llm_verify=lambda p, mt=4000: call_llm(
                p, model="deepseek-chat", max_tokens=mt),
            domain_default="granular flow physics",
            executor_model="DeepSeek-V4-Flash")

    def run_one(case: dict) -> dict:
        outs = []
        for t in range(args.times):
            ext = _mk_agent()
            try:
                plan = ext.plan(case["chunk"], case.get("section") or "CASE")
                nodes, edges = ext.execute(case["chunk"], plan, case["case_id"])
                edges, _gd = ext._deterministic_gate(edges, nodes, case["chunk"])
                verdicts = ext.verify(edges, nodes, case["chunk"], plan.domain)
                kept, _fstats, _dropped = ext.fix(
                    edges, verdicts, nodes, case["chunk"], plan)
                outs.append(evaluate(case, kept, nodes))
            except Exception as e:
                outs.append({"outcome": "HARNESS_ERROR", "error": repr(e)})
        # majority vote across repeats
        counts = {}
        for o in outs:
            counts[o["outcome"]] = counts.get(o["outcome"], 0) + 1
        best = max(counts.items(), key=lambda kv: kv[1])
        rec = {"case_id": case["case_id"], "class": case["failure_class"],
               "kind": case["kind"], "domain": case["domain"],
               "outcome": best[0], "votes": counts,
               "watched": outs[0].get("watched", [])[:4]}
        print(f"[case] {case['case_id']} {case['failure_class']:<16} "
              f"-> {rec['outcome']} {counts if args.times > 1 else ''}", flush=True)
        return rec

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, c): c["case_id"] for c in cases}
        for f in as_completed(futs):
            results.append(f.result())

    # aggregate
    agg = {}
    for r in results:
        k = r["class"]
        agg.setdefault(k, {}).setdefault(r["outcome"], 0)
        agg[k][r["outcome"]] += 1
    print(f"\n[cases] === aggregate ({len(results)} cases, {time.time()-t0:.0f}s) ===")
    for k in sorted(agg):
        print(f"  {k:<18} {agg[k]}")
    n_sw = sum(v.get("STILL_WRONG", 0) for v in agg.values())
    n_reg = sum(v.get("REGRESSED", 0) for v in agg.values())
    n_err = sum(v.get("HARNESS_ERROR", 0) for v in agg.values())
    scored = [r for r in results if r["class"] != "misc"]
    fixed = sum(1 for r in scored if r["outcome"].startswith("FIXED"))
    n_fail = sum(1 for r in scored if r["kind"] == "fail")
    if n_fail:
        print(f"  fail-case fix rate: {fixed}/{n_fail} "
              f"({fixed/n_fail:.0%}) | STILL_WRONG {n_sw} | REGRESSED {n_reg} "
              f"| ERRORS {n_err}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in sorted(results, key=lambda x: x["case_id"]):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[cases] results -> {args.out}")


if __name__ == "__main__":
    main()
