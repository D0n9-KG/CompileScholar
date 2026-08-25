"""诊断：verifier 为什么放行 Run 2 失败的 defines/influences 边。
重建失败边 → 跑真 verify()（deepseek-chat）→ 看 verdict 的三必检字段。
用法: python .research_tmp/diagnose_verifier.py
"""
import json, sys, os
sys.path.insert(0, 'src')

d = '.research_tmp/runs/kernel_v2/REWRITE_PPR_24493BE6E8C2'
cg = json.load(open(os.path.join(d, 'concept_graph.json'), encoding='utf-8'))
jc = json.load(open(os.path.join(d, 'judge_cross.json'), encoding='utf-8'))
concepts = cg['concepts']

# failing edges (either-fail by judges), grouped: defines + influences
targets = [e for a, b, e in zip(jc['glm'], jc['qwen'], jc['edges'])
           if e['pt'] in ('defines', 'influences') and not (a['pass'] or b['pass'])]
print(f"re-verifying {len(targets)} failing edges (verifier: deepseek-chat)")

from granular_agent.knowledge_base import KnowledgeBase
from granular_agent.hypergraph_schema import seed_meta_hypergraph, Hyperedge, HGNode
from granular_agent.extraction_agent import ExtractionAgent
from granular_agent.llm_client import call_llm, call_paratera

kb = KnowledgeBase(tbox=seed_meta_hypergraph())
agent = ExtractionAgent(kb, llm_extract=lambda p, m: "", llm_verify=call_llm,
                        domain_default="ml")

# rebuild nodes+edges in the format verify() expects; section text = the fulltext
fulltext = open('.research_tmp/probe_dqn_fulltext.txt', encoding='utf-8').read()
nodes, edges = [], []
for i, e in enumerate(targets):
    nids, roles = [], []
    for role, surf in e['nodes']:
        nid = f"n{i}_{len(nids)}"
        nodes.append(HGNode(nid=nid, labels=["METHOD"], surface=surf, evidence_span=""))
        nids.append(nid); roles.append(role)
    edges.append(Hyperedge(eid=f"e{i}", pattern_type=e['pt'], node_ids=nids,
                           node_roles=roles, evidence_span=e['ev']))

verdicts = agent.verify(edges, nodes, fulltext, "ml")
print("\n=== verifier verdicts on judge-failing edges ===")
for v, e in zip(verdicts, targets):
    print(f"[{v.edge_id}] {e['pt']:12s} fix={v.fix:20s} "
          f"rel={v.relation_exists} type={v.type_correct} role={v.role_correct} "
          f"| note: {v.note[:60]}")
    # show the raw fields the new prompt asks for (may be absent in parse)
    print(f"    ev: {e['ev'][:90]}")
