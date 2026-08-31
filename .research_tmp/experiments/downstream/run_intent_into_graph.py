# -*- coding: utf-8 -*-
"""DRL-6 corpus through the kernel WITH the citation stage (2026-08-28).

Registers the 6 papers as PAPER nodes, runs process_paper_via_kernel (frozen
arm — we already have ml_schema_v1; this run is about the citation stage, not
evolution) in succession order, and reports the intra-corpus 'cites' edges
that landed in the graph. Target: the 16-edge citation graph from the bypass
run, now in the SAME graph as the concepts.
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from granular_agent.agent import GranularFlowAgent

CORPUS = [
    ("PPR_24493BE6E8C2", "Human-level control through deep reinforcement learning",
     ["mnih"], ["2015"], "2015"),
    ("PPR_DD48410E18B3", "Deep Reinforcement Learning with Double Q-learning",
     ["van hasselt", "hasselt"], ["2015", "2016"], "2016"),
    ("PPR_746E68D2E93B", "Prioritized Experience Replay",
     ["schaul"], ["2015", "2016"], "2016"),
    ("PPR_F8D5D4C3C2B1", "Asynchronous Methods for Deep Reinforcement Learning",
     ["mnih"], ["2016"], "2016"),
    ("PPR_FC0F6B04EEBF", "Dueling Network Architectures for Deep Reinforcement Learning",
     ["wang"], ["2015", "2016"], "2016"),
    ("PPR_65EB3FEB4B1B", "Rainbow: Combining Improvements in Deep Reinforcement Learning",
     ["hessel"], ["2017", "2018"], "2018"),
]


def main():
    agent = GranularFlowAgent(llms=["DeepSeek-V4-Flash"], domain="ml")
    kb = agent._get_kernel()

    # register corpus PAPER nodes (before any paper runs: later papers cite
    # earlier ones, and the registry must know both ends at commit time)
    from granular_agent.citation_stage import register_corpus_papers
    register_corpus_papers(kb, [
        {"pid": pid, "title": title, "surnames": surnames,
         "years": years, "year": year}
        for pid, title, surnames, years, year in CORPUS])

    for pid, title, _, _, year in CORPUS:
        print(f"\n=== {pid} ({title[:50]}, {year}) ===", flush=True)
        rep = agent.process_paper_via_kernel(pid, arm="frozen")
        ci = rep.get("citation_intents") or {}
        print(f"  concepts={rep['n_concepts']} edges={rep['n_hyperedges']} "
              f"cit_mentions={ci.get('n_mentions')} cit_committed={ci.get('n_committed')}",
              flush=True)

    # final: the intra-corpus citation graph in the A-box
    from granular_agent.citation_stage import paper_concept_id, CORPUS_REGISTRY
    pid2short = {pid: title.split(":")[0][:28] for pid, title, _, _, _ in CORPUS}
    cid2pid = {paper_concept_id(kb, pid): pid for pid, _, _, _, _ in CORPUS}
    cites = [h for h in kb.abox.hyperedges if h.kind == "cites"]
    print(f"\n=== intra-corpus 'cites' edges in graph: {len(cites)} ===")
    for h in cites:
        src, dst = h.node_ids[0], h.node_ids[-1]
        s = pid2short.get(cid2pid.get(src, "?"), "?")
        d = pid2short.get(cid2pid.get(dst, "?"), "?")
        ev = (h.provenance[0].get("evidence", "") if h.provenance else "")[:60]
        print(f"  {s[:26]:26s} -{h.pattern_type:<9s}-> {d[:26]:26s} | {ev}")

    out = {"n_cites_edges": len(cites),
           "edges": [{"from": cid2pid.get(h.node_ids[0]),
                      "to": cid2pid.get(h.node_ids[-1]),
                      "intent": h.pattern_type,
                      "evidence": (h.provenance[0].get("evidence", "")
                                   if h.provenance else "")[:200]}
                     for h in cites]}
    json.dump(out, open(".research_tmp/citation_stage_drl6_result.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-> .research_tmp/citation_stage_drl6_result.json")


if __name__ == "__main__":
    main()
