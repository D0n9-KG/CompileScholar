"""Kernel pipeline smoke: process_paper_via_kernel wires the 4 builder agents
through the KnowledgeBase on a paper's section DAG.

Mocks map_structure (returns a fixed 1-section DAG) + load_paper_blocks +
the LLM (extractor multistep + verifier + aligner). Verifies the KB-backed
pipeline runs end-to-end and the ledger records the writes (transactional,
not direct mutate).

Not a real-LLM test (the section text is synthetic; the extractor's multistep
core is exercised with a mock LLM returning canned JSON).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import granular_agent.agent as agent_mod
from granular_agent.agent import GranularFlowAgent

fail = []

def check(name, cond):
    print(("OK  " if cond else "FAIL") + "  " + name)
    if not cond:
        fail.append(name)


# --- mock the section-DAG + blocks + LLM ---
SECTION_TEXT = ("We extend the μ(I) rheology to nonlocal flows. "
                "The model improves the local rheology which fails near boundaries. "
                "We define the cooperativity length ξ as a parameter. ")


def _mock_load_blocks(paper_id, corpus_dir=""):
    return [{"index": 0, "text": SECTION_TEXT}]


def _mock_map_structure(paper_id, blocks, llm="deepseek", domain=""):
    return {"dag": {"nodes": [{"id": "s1", "section": "intro"}]},
            "sections": [{"name": "intro", "block_range": [0, 1]}]}


agent_mod.load_paper_blocks = _mock_load_blocks
agent_mod.map_structure = _mock_map_structure

# mock the joint-extraction LLM (B+ rewrite: ONE call returns nodes+types+edges
# together). Output satisfies the binding-locality contract (every participant
# surface inside the edge's evidence sentence).
def _mock_extract_llm(prompt, max_tokens=8000):
    p = prompt.lower()
    if "knowledge hypergraph" in p:
        return ('{"nodes":[{"nid":"n1","surface":"μ(I) rheology","type":"METHOD","evidence_span":"the μ(I) rheology"},'
                '{"nid":"n2","surface":"nonlocal flows","type":"PHENOMENON","evidence_span":"nonlocal flows"}],'
                '"hyperedges":[{"eid":"e1","pattern_type":"extends",'
                '"node_ids":["n1","n2"],"node_roles":["from","to"],'
                '"evidence_span":"We extend the μ(I) rheology to nonlocal flows","qualifiers":{}}],'
                '"summary":"sec"}')
    # planner / verifier / aligner / define — return minimal valid JSON
    if "planning hypergraph" in p:
        return ('{"domain":"granular","central_entities":[{"surface":"μ(I) rheology","type":"METHOD"}],'
                '"expected_patterns":["extends"],"relation_outline":[],"paper_anchor":"x"}')
    if "critiquing" in p:
        # verdict for EVERY edge in the batch (fail-closed verifier drops
        # edges without a verdict — a partial mock would drop the rest)
        import re as _re
        ids = _re.findall(r'"edge_id":\s*"([^"]+)"', p)
        vs = ",".join(f'{{"edge_id":"{i}","verbatim_in_source":true,"type_correct":true,'
                      f'"relation_exists":true,"fix":"keep"}}' for i in ids)
        return '{"verdicts":[' + vs + ']}'
    if "definitions" in p:
        return '{"definitions":[]}'
    if "same concept" in p or "groups" in p:
        return '{"groups":[[0,1]]}'
    return ""


agent = GranularFlowAgent(domain="granular", llms=["DeepSeek-V4-Flash"])
# patch the LLM methods to the mock
agent._kernel_llm_extract = _mock_extract_llm
agent._kernel_llm_verify = _mock_extract_llm

result = agent.process_paper_via_kernel("test_paper_2020")

check("process_paper_via_kernel runs without error", "error" not in result or result.get("error") is None)
check("result has pipeline=kernel tag", result.get("pipeline") == "kernel")
check("KB ledger has entries (transactional writes, not direct mutate)",
      len(agent._get_kernel().ledger) >= 1)
# the extractor's add_edge writes should be in the ledger
from granular_agent.knowledge_base import Op
kb = agent._get_kernel()
has_add_edge = any(m.get("op") == Op.ADD_EDGE for e in kb.ledger for m in e.get("mutations", []))
check("ledger has add_edge writes (extractor went through KB)", has_add_edge)
# the KB shares meta_hg / concept_graph (same objects, not copies)
check("KB shares agent's meta_hg (same object)", kb.tbox is agent.meta_hg)
check("KB shares agent's concept_graph (same object)", kb.abox is agent.concept_graph)
# 单管线原则 (B+ rebuild 2026-08-25): legacy paths physically archived under
# legacy/ — they must NOT exist on the agent anymore.
check("legacy paths removed (单管线原则)",
      not any(hasattr(agent, m) for m in
              ["process_paper", "process_batch", "process_paper_hypergraph",
               "process_batch_hypergraph", "save_hypergraph_results", "save_results"]))

print()
print(f"{'ALL PASS' if not fail else 'FAILURES: ' + str(fail)}  ({len(fail)} fail)")
sys.exit(1 if fail else 0)
