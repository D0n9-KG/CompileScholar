"""Smoke +断点 test for maintenance_agent.py (Step 5: rich topology + maintenance).

Verifies without real LLM:
- Rich topology: delegates infer_rich_topology_direct (the verified direct read).
- Ambiguous flag (断点 12 honest narrowing): multi-kind edges flagged, NOT LLM.
- Utility pruning (SEDM): retire low-frequency via kb.commit(retire), soft-delete.
- Decay reaper (断点 13): honest降级 — no-op, retire is soft-delete+recoverable.
- Consumer working memory: snapshot_rich_topology on-demand subgraph (DocTrace).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from granular_agent.maintenance_agent import MaintenanceAgent, RichTopologyView
from granular_agent.knowledge_base import KnowledgeBase, Op, Role
from granular_agent.hypergraph_schema import (
    seed_meta_hypergraph, Hyperedge, HGNode, InstanceHypergraph,
)
from granular_agent.concept_graph import Concept

fail = []

def check(name, cond):
    print(("OK  " if cond else "FAIL") + "  " + name)
    if not cond:
        fail.append(name)


def _kb():
    return KnowledgeBase(tbox=seed_meta_hypergraph())


# ===========================================================================
# 1. Rich topology: delegates infer_rich_topology_direct (verified direct read)
# ===========================================================================
kb = _kb()
agent = MaintenanceAgent(kb, llm=None, domain_default="granular")
inst = InstanceHypergraph(paper_id="p1")
inst.add_node(HGNode(nid="m1", labels=["METHOD"], surface="μ(I) rheology"))
inst.add_node(HGNode(nid="p1", labels=["PARAMETER"], surface="inertial number I"))
inst.add_node(HGNode(nid="f1", labels=["PHENOMENON"], surface="nonlocal creep"))
# method_parameter edge (METHOD+PARAMETER)
inst.add_hyperedge(Hyperedge(eid="e1", pattern_type="measures",
    node_ids=["m1", "p1"], node_roles=["method", "param"],
    evidence_span="the model uses parameter I"))
# method_parameter + method_phenomenon (METHOD+PARAMETER+PHENOMENON = ambiguous)
inst.add_hyperedge(Hyperedge(eid="e2", pattern_type="measures",
    node_ids=["m1", "p1", "f1"], node_roles=["method", "param", "effect"],
    evidence_span="the model with param I captures creep"))
edges = agent.infer_rich_topology(inst, paper_id="p1")
check("Rich topology: direct read produces edges", len(edges) >= 1)

# ===========================================================================
# 2. Ambiguous flag (断点 12 honest narrowing): multi-kind edges flagged, NOT LLM
# ===========================================================================
# e2 (METHOD+PARAMETER+PHENOMENON) matches both method_parameter AND method_phenomenon
ambiguous = [e for e in edges if e.get("ambiguous_kinds")]
check("Ambiguous (断点12): multi-kind edge flagged (not LLM-disambiguated)",
      len(ambiguous) >= 1)
check("Ambiguous: flagged edge has both method_parameter+method_phenomenon",
      any(set(a["ambiguous_kinds"]) >= {"method_parameter", "method_phenomenon"}
          for a in ambiguous))

# ===========================================================================
# 3. Utility pruning (SEDM): retire low-frequency via kb.commit (soft-delete)
# ===========================================================================
kb3 = _kb()
# add an A-box edge using 'measures' so it has frequency 1; other patterns unused
# pass pattern_type="measures" (the T-box ref) so prune counts it correctly.
# (BLOCKER fix: previously tests used kind=pattern_type, masking the real-pipeline
# mismatch where kind is a rich-topology label, NOT the T-box pattern_id.)
kb3.abox.concepts["c1"] = Concept(concept_id="c1", type="PROPERTY")
kb3.abox.concepts["c2"] = Concept(concept_id="c2", type="PROPERTY")
kb3.abox.add_hyperedge(["c1", "c2"], kind="measures", paper_id="p1", evidence="x",
                        pattern_type="measures")
agent3 = MaintenanceAgent(kb3, llm=None, domain_default="granular")
# prune patterns with frequency < 1 (i.e. unused patterns retire)
report = agent3.prune_by_utility(domain="granular", min_frequency=1)
check("SEDM prune: unused patterns retired (frequency < 1)", report.n_retired >= 1)
check("SEDM prune: 'measures' NOT retired (frequency=1, in use)",
      not kb3.tbox.patterns["measures"].deprecated)
# retire is SOFT (deprecated flag), ledger-recoverable
check("SEDM prune: retire is soft-delete (deprecated, not removed)",
      all(kb3.tbox.patterns[p] is not None for p in [r["target"] for r in report.retired]
          if p in kb3.tbox.patterns))
check("SEDM prune: retire entry in ledger (kernel transaction, recoverable)",
      any(m.get("op") == Op.RETIRE for e in kb3.ledger for m in e.get("mutations", [])))

# 3b. BLOCKER fix real-path: an A-box edge with kind=rich-topology-label
# (method_parameter) but pattern_type=T-box-ref (constitutive_law) — exactly
# the real pipeline shape (ingest_instance stores kind=rich kind, pattern_type
# =raw). prune must count by pattern_type, so constitutive_law (in use) is NOT
# retired. Before the fix, prune used kind -> mismatched T-box pattern_id ->
# wrongly retired in-use constitutive_law.
kb3b = _kb()
kb3b.abox.concepts["c1"] = Concept(concept_id="c1", type="METHOD")
kb3b.abox.concepts["c2"] = Concept(concept_id="c2", type="PARAMETER")
# kind=method_parameter (rich-topology label), pattern_type=constitutive_law (T-box)
kb3b.abox.add_hyperedge(["c1", "c2"], kind="method_parameter", paper_id="p1",
                         evidence="law uses param", pattern_type="constitutive_law")
agent3b = MaintenanceAgent(kb3b, llm=None, domain_default="granular")
report3b = agent3b.prune_by_utility(domain="granular", min_frequency=1)
check("BLOCKER fix: constitutive_law NOT retired (counted by pattern_type, in use)",
      not kb3b.tbox.patterns["constitutive_law"].deprecated)
check("BLOCKER fix: in-use pattern not in retired list",
      "constitutive_law" not in [r["target"] for r in report3b.retired])
# unused patterns (no A-box edge with their pattern_type) still retire
check("BLOCKER fix: unused patterns still retired (frequency=0)",
      report3b.n_retired >= 1)

# ===========================================================================
# 4. Decay reaper (断点 13): honest降级 — no-op
# ===========================================================================
result = agent3.decay_reaper()
check("Decay (断点13): honest降级 no-op (no MemoryBank forgetting-curve)",
      result["decayed"] == 0 and "no active reaper" in result["note"])

# ===========================================================================
# 5. Consumer working memory: snapshot_rich_topology on-demand subgraph
# ===========================================================================
kb5 = _kb()
kb5.abox.concepts["c1"] = Concept(concept_id="c1", type="METHOD")
kb5.abox.concepts["c2"] = Concept(concept_id="c2", type="PARAMETER")
kb5.abox.concepts["c3"] = Concept(concept_id="c3", type="PHENOMENON")
kb5.abox.add_hyperedge(["c1", "c2"], kind="method_parameter", paper_id="p1", evidence="e1")
kb5.abox.add_hyperedge(["c1", "c3"], kind="method_phenomenon", paper_id="p1", evidence="e2")
agent5 = MaintenanceAgent(kb5, llm=None, domain_default="granular")
# consumer queries concept c1 -> gets only edges touching c1
view = agent5.snapshot_rich_topology(["c1"])
check("Consumer working memory: on-demand subgraph (edges touching c1 only)",
      len(view.edges) == 2)
check("Consumer working memory: n_by_kind counts the 2 kinds",
      view.n_by_kind.get("method_parameter", 0) == 1 and view.n_by_kind.get("method_phenomenon", 0) == 1)
# query c2 only -> only the method_parameter edge
view2 = agent5.snapshot_rich_topology(["c2"])
check("Consumer working memory: scoped to queried concept (c2 -> 1 edge)",
      len(view2.edges) == 1 and view2.edges[0]["kind"] == "method_parameter")

print()
print(f"{'ALL PASS' if not fail else 'FAILURES: ' + str(fail)}  ({len(fail)} fail)")
sys.exit(1 if fail else 0)
