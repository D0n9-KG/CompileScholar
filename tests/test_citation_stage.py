# -*- coding: utf-8 -*-
"""Citation stage kernel tests (mock LLM — structural contract only)."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from granular_agent.knowledge_base import KnowledgeBase
from granular_agent.citation_stage import (register_corpus_papers,
                                           extract_citation_intents,
                                           paper_concept_id)

fails = 0
def check(name, cond):
    global fails
    print(("OK   " if cond else "FAIL ") + name)
    if not cond:
        fails += 1

kb = KnowledgeBase()
register_corpus_papers(kb, [
    {"pid": "PPR_A", "title": "Human-level control through deep reinforcement learning",
     "surnames": ["mnih"], "years": ["2015"], "year": "2015"},
    {"pid": "PPR_B", "title": "Deep Reinforcement Learning with Double Q-learning",
     "surnames": ["van hasselt", "hasselt"], "years": ["2015", "2016"], "year": "2016"},
])

# PAPER nodes registered
a_cid = paper_concept_id(kb, "PPR_A")
b_cid = paper_concept_id(kb, "PPR_B")
check("PAPER node A registered", a_cid is not None and kb.abox.concepts[a_cid].type == "PAPER")
check("PAPER node B registered", b_cid is not None and kb.abox.concepts[b_cid].type == "PAPER")

# fulltext of B citing A with an 'improves' trigger
fulltext = ("In this paper we address the overestimation bias of the DQN algorithm. "
            "The recent DQN algorithm (Mnih et al., 2015) suffers from substantial "
            "overestimations in some games. We propose a specific adaptation to the "
            "DQN algorithm and show it leads to much better performance. "
            "Background noise about other things. Sutton and Barto (1998) wrote a "
            "textbook about reinforcement learning fundamentals.")

def mock_llm(prompt, max_tokens):
    # deterministic mock: 'mnih 2015' -> improves; others -> background
    if "mnih 2015" in prompt or "[Mnih" in prompt:
        return ('{"intent": "improves", "confidence": 0.9, '
                '"note": "specific adaptation to DQN improves performance"}')
    return '{"intent": "background", "confidence": 0.9, "note": "prior context"}'

rep = extract_citation_intents(kb, "PPR_B", fulltext, mock_llm)
check("report counts mentions", rep["n_author_year"] >= 2)
check("one intra-corpus edge committed", rep.get("n_committed") == 1)

# the edge is in the A-box with kind='cites' + intent pattern_type
cites_edges = [h for h in kb.abox.hyperedges if h.kind == "cites"]
check("'cites' hyperedge in A-box", len(cites_edges) == 1)
if cites_edges:
    e = cites_edges[0]
    check("edge connects B -> A", set(e.node_ids) == {a_cid, b_cid})
    check("intent stored as pattern_type", e.pattern_type == "improves")
    check("evidence carries the mention window", "Mnih" in (e.provenance[0].get("evidence") if e.provenance else ""))

# ledger recorded the commit (auditability)
check("ledger has the citation mutation", any(
    "cites" in str(md.get("payload", {})) or
    "citation-intent" in str(md.get("rationale", ""))
    for entry in kb.ledger
    for md in entry.get("mutations", [])))

# idempotence: re-registering corpus papers doesn't duplicate
register_corpus_papers(kb, [
    {"pid": "PPR_A", "title": "Human-level control through deep reinforcement learning",
     "surnames": ["mnih"], "years": ["2015"], "year": "2015"}])
n_paper = sum(1 for c in kb.abox.concepts.values() if c.type == "PAPER")
check("re-registration idempotent", n_paper == 2)

# post-check: mention must be in evidence (structural rule) — an edge whose
# window lacks the surname token is NOT committed
fulltext2 = "Some unrelated text with no citations at all here."
rep2 = extract_citation_intents(kb, "PPR_B", fulltext2, mock_llm)
check("no mentions -> no edges", rep2.get("n_committed") == 0)

print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAIL'}  ({fails} fail)")
sys.exit(1 if fails else 0)
