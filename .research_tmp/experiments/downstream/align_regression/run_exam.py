# -*- coding: utf-8 -*-
"""Alignment regression exam (2026-08-27) — run the FULL aligner on the final
A2G_EVO2 10-paper concept graph (544 concepts) and score it against the
audit's failure inventory (align_first_test/REPORT.md).

Exam cases (from the audit, hand-verified):
  SHOULD-MERGE families (audit §2/§4 real dups):
    kinetic theory family (11 METHODs -> expect collapse to <=2)
    Chapman-Enskog within-955A (6 -> expect 1)
    restitution coefficient (3 PARAMETERs -> expect 1)
    granular temperature (4 -> expect <=2)
    steady state / steady-state (2 -> 1)
    inertial number pair (CP0122/CP0320 -> 1)
    molecular dynamics pair (CM0377/CM0418 -> 1)
  SHOULD-NOT-MERGE (audit §3 bad merges — the fixed aligner must not create
  NEW such merges; the already-merged ones are baked in the data):
    generic "model"/"theory" swallowing named methods
    density segregation vs size segregation (different mechanisms)
    single-letter params across papers with different meanings

Usage: python .research_tmp/align_regression/run_exam.py --tag <name>
Writes .research_tmp/align_regression/<tag>_result.json + prints scorecard.
"""
import argparse, json, os, re, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GRAPH = os.path.join(ROOT, ".research_tmp", "runs", "kernel_v2",
                     "A2G_EVO2_PPR_866D169A4642", "concept_graph.json")

def _tok(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# cid lists from the audit (fixed exam key — do not edit per run)
FAMILIES = {
    "kinetic_theory": ["CM0120", "CM0383", "CM0433", "CM0613", "CM0005",
                       "CM0002", "CM0023", "CM0057",
                       # + surface-matched members found at runtime
                       ],
    "chapman_enskog_955A": ["CM0003", "CM0006", "CM0020", "CM0024", "CM0064", "CM0066"],
    "molecular_dynamics": ["CM0377", "CM0418"],
    "restitution": ["CP0009", "CP0081", "CP0378"],
    "inertial_number": ["CP0122", "CP0320"],
    "steady_state": ["CR0178", "CR0264"],
}
FAMILY_SURFACE_HINTS = {
    "kinetic_theory": ("kinetic theor", "METHOD"),
    "granular_temperature": ("granular temperature", None),
}

def load():
    from granular_agent.concept_graph import ConceptGraph
    with open(GRAPH, encoding="utf-8") as f:
        return ConceptGraph.from_dict(json.load(f))

def cid2concept(cg):
    return cg.concepts

def family_members(cg, name):
    """Resolve a family to live cids: audit cids + surface-hint matches."""
    conc = cg.concepts
    members = set(FAMILIES.get(name, []))
    hint = FAMILY_SURFACE_HINTS.get(name)
    if hint:
        frag, typ = hint
        for cid, c in conc.items():
            if typ and c.type != typ:
                continue
            if any(frag in s.lower() for s in c.surfaces()):
                members.add(cid)
    return sorted(m for m in members if m in conc)

def n_distinct(cg, cids):
    """How many DISTINCT concepts do these cids currently map to?
    (merged ones share the same id or are deprecated with redirected id —
    from_dict keeps them as-is; after a merge pass, look up by canonical id
    via _surface2concept is unreliable, so we count live concept ids.)"""
    live = [c for c in cids if c in cg.concepts and not cg.concepts[c].deprecated]
    return len(set(live))

def run(tag):
    from granular_agent.knowledge_base import KnowledgeBase
    from granular_agent.alignment_agent import AlignmentAgent
    from granular_agent.hypergraph_schema import seed_meta_hypergraph
    from granular_agent.llm_client import call_llm

    cg = load()
    before = {name: sorted(family_members(cg, name)) for name in
              set(FAMILIES) | set(FAMILY_SURFACE_HINTS)}
    n_before = len(cg.concepts)
    print(f"[exam] loaded: {n_before} concepts; families:",
          {k: len(v) for k, v in before.items()})

    kb = KnowledgeBase(tbox=seed_meta_hypergraph(), abox=cg,
                       domain_ns={"global": True, "granular": True})
    agent = AlignmentAgent(kb,
                           llm_define=lambda p, mt=4000: call_llm(p, model="deepseek-chat", max_tokens=mt),
                           llm_judge=lambda p, mt=3000: call_llm(p, model="deepseek-chat", max_tokens=mt),
                           domain_default="granular")
    t0 = time.time()
    rep = agent.align(domain="granular")
    dt = time.time() - t0
    print(f"[exam] align done in {dt:.0f}s: candidates={rep.n_candidates} "
          f"merges={rep.n_merge} reject={rep.n_reject}")

    # scorecard
    score = {"tag": tag, "n_concepts_before": n_before,
             "n_concepts_after": len(cg.concepts),
             "n_merge": rep.n_merge, "seconds": round(dt),
             "merges": rep.merges, "families": {}}
    for name, cids in before.items():
        distinct_after = n_distinct(cg, cids)
        score["families"][name] = {
            "members": cids, "distinct_after": distinct_after,
            "collapsed": distinct_after <= max(1, len(cids) // 2)}
    print(f"\n[exam] === {tag} scorecard ===")
    for name, s in score["families"].items():
        flag = "OK " if s["collapsed"] else "MISS"
        print(f"  {flag} {name}: {len(s['members'])} members -> {s['distinct_after']} distinct")
    out = os.path.join(os.path.dirname(__file__), f"{tag}_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(score, f, ensure_ascii=False, indent=1)
    # also dump the merged graph for inspection
    with open(os.path.join(os.path.dirname(__file__), f"{tag}_graph.json"), "w",
              encoding="utf-8") as f:
        json.dump(cg.to_dict(), f, ensure_ascii=False, indent=1)
    print(f"[exam] -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    run(ap.parse_args().tag)
