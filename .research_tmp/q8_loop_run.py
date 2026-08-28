# -*- coding: utf-8 -*-
"""q8 closed-loop runner: build graph on 3 seeds, then two arms (graph vs
LLM-rewrite control) recall the 5 missing gold. Pre-registered verdict rule:
graph arm must recall >=2 more gold than control at pool level.

Phase 1 (this script, ~25min): incremental build — 3 papers through
process_paper_via_kernel (frozen arm + citation stage), ML schema loaded.
Phase 2 (q8_loop_round2.py): arms.
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from granular_agent.agent import GranularFlowAgent

SEEDS = [
    {"arxiv": "1805.02343", "pid": "PPR_ED4F1C5E2AE2",
     "title": "Deep reinforcement learning for page-wise recommendations",
     "surnames": ["chen"], "years": ["2018"]},
    {"arxiv": "2101.06286", "pid": "PPR_E0FDD4E670F6",
     "title": "Reinforcement Learning based Recommender Systems: A Survey",
     "surnames": ["afsar"], "years": ["2021", "2022"]},
    {"arxiv": "2006.05779", "pid": "PPR_963C5A4E29ED",
     "title": "Self-Supervised Reinforcement Learning for Recommender Systems",
     "surnames": ["chen"], "years": ["2020", "2021"]},
]


def main():
    agent = GranularFlowAgent(llms=["DeepSeek-V4-Flash"], domain="ml")
    kb = agent._get_kernel()

    # register the 3 seeds as PAPER nodes (intra-corpus cites edges)
    from granular_agent.citation_stage import register_corpus_papers
    register_corpus_papers(kb, [
        {"pid": s["pid"], "title": s["title"],
         "surnames": s["surnames"], "years": s["years"]} for s in SEEDS])

    for s in SEEDS:
        print(f"\n=== {s['title'][:60]} ===", flush=True)
        rep = agent.process_paper_via_kernel(s["pid"], arm="add_only")
        ci = rep.get("citation_intents") or {}
        print(f"  concepts={rep['n_concepts']} edges={rep['n_hyperedges']} "
              f"cit={ci.get('n_mentions')}/{ci.get('n_committed')}", flush=True)

    # export the grown graph for the arms
    out = {"concepts": {}, "hyperedges": []}
    for cid, c in kb.abox.concepts.items():
        out["concepts"][cid] = {
            "type": c.type,
            "surfaces": list({v.surface for v in c.surface_variants}),
            "canonical": c.canonical_name,
            "central": c.central,
        }
    for h in kb.abox.hyperedges:
        out["hyperedges"].append({
            "nodes": h.node_ids, "roles": h.node_roles,
            "kind": h.kind, "pattern_type": h.pattern_type,
            "evidence": (h.provenance[0].get("evidence", "") if h.provenance else "")[:200],
        })
    json.dump(out, open(".research_tmp/_q8_loop_graph.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n_cites = sum(1 for h in out["hyperedges"] if h["kind"] == "cites")
    print(f"\ngraph: {len(out['concepts'])} concepts, "
          f"{len(out['hyperedges'])} hyperedges, {n_cites} cites edges")
    print("-> .research_tmp/_q8_loop_graph.json")


if __name__ == "__main__":
    main()
