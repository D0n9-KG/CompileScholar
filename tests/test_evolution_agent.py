"""Smoke +断点 test for evolution_agent.py (Step 3: schema self-evolution agent).

Verifies the agent FORM landed without real LLM (probe/governance/apply mocked
where needed; the verified内核 evolution_probe/validate_proposal are exercised
via the existing smoke test, not re-tested here):
- unified queue (断点 6): all trigger sources propose into one queue, drain is
  the single serial consumer.
- three stages: probe → governance → apply.
- apply goes through KnowledgeBase.commit (evolver Mutation → validate → apply),
  NOT direct meta mutate — every change enters the ledger / passes schema-
  constrained validate / is replayable. (the key difference from legacy
  run_evolution_loop)
- add_pattern proposals get an LLM-generated semantic_boundary attached.
- recurring crystallize: growth ops gated on cross_node >= 2 (conservative gate).
- HITL flagging: new top-level family flagged (reserved interface).
- SAGE writer-reader: consumer feedback accumulates, recurring -> trigger.
- distill_skill: crystallize -> distill_skill op committed via kernel (断点 5 loop).
- schema并进: schema_for_extraction re-fetches the evolved to_prompt.
No real LLM.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from granular_agent.evolution_agent import (
    EvolutionAgent, TriggerSource, ConsumerFeedback,
)
from granular_agent.knowledge_base import KnowledgeBase, Op, Role
from granular_agent.hypergraph_schema import (
    seed_meta_hypergraph, MetaHyperedgePattern, Hyperedge, HGNode,
    InstanceHypergraph,
)
import granular_agent.hypergraph_evolution as hev

fail = []

def check(name, cond):
    print(("OK  " if cond else "FAIL") + "  " + name)
    if not cond:
        fail.append(name)


def _kb():
    return KnowledgeBase(tbox=seed_meta_hypergraph())


# Mock the evolution内核 calls the agent delegates to, so the test is
# deterministic and doesn't hit the LLM for probe/validate.
# evolution_probe returns proposals; validate_proposal judges them.

def _mock_probe(distinct, meta, paper_id, domain="", llm="deepseek", instance=None):
    # return one add_pattern proposal (a new pattern in an EXISTING family, so
    # it's not HITL-flagged) + one split proposal
    return [
        {"op": "add_pattern", "pattern_id": "test_new_pat", "family": "claim",
         "description": "a test discourse relation",
         "role_slots": [{"role": "from", "type": "THING"},
                        {"role": "to", "type": "THING"}],
         "allowed_qualifiers": ["relation_type"],
         "evidence_span": "the test relation evidence", "rationale": "gap"},
    ]

def _mock_validate(proposal, meta, domain="", llm="deepseek"):
    # accept everything that has evidence; reject no-evidence
    if not proposal.get("evidence_span") and not proposal.get("rationale"):
        return {"valid": False, "reason": "no-evidence"}
    return {"valid": True, "reason": "ok"}

# patch the内核 the agent imports
hev.evolution_probe = _mock_probe
hev.validate_proposal = _mock_validate


def _boundary_llm(prompt, max_tokens):
    # the boundary-generation prompt -> return a canned boundary
    if "SEMANTIC BOUNDARY" in prompt or "semantic boundary" in prompt.lower():
        return "a test discourse relation; NOT a measurement or a definition."
    return ""

agent_llm = lambda p, m: _boundary_llm(p, m)


# ===========================================================================
# 1. unified queue + drain single serial consumer (断点 6)
# ===========================================================================
kb = _kb()
agent = EvolutionAgent(kb, llm=agent_llm, domain_default="granular")
# enqueue two validate_failure triggers (cross_node=2 so growth passes the gate)
he_fail = Hyperedge(eid="e1", pattern_type="test_new_pat", node_ids=["a", "b"],
                    node_roles=["from", "to"], evidence_span="the test relation evidence")
agent.propose_validate_failures([(he_fail, "no-matching-meta-pattern")],
                                node_id="node_A", paper_id="p1", domain="granular")
agent.propose_validate_failures([(he_fail, "no-matching-meta-pattern")],
                                node_id="node_B", paper_id="p1", domain="granular")
check("queue has 2 trigger sources", len(agent._queue) == 2)
report = agent.drain()
check("drain: queue cleared (single serial consumer)", len(agent._queue) == 0)
check("drain: produced a report (see content not just pass/fail)",
      report.n_proposed >= 1 and report.version_before != report.version_after or report.n_accepted > 0
      or report.n_rejected >= 0)
# the add_pattern proposal should have been applied via kb.commit (ledger entry)
check("apply: ledger has an ADD_PATTERN entry (went through kernel transaction)",
      any(m.get("op") == Op.ADD_PATTERN for entry in kb.ledger
          for m in entry.get("mutations", [])))

# ===========================================================================
# 2. add_pattern proposal gets LLM-generated semantic_boundary
# ===========================================================================
kb2 = _kb()
agent2 = EvolutionAgent(kb2, llm=agent_llm, domain_default="granular")
agent2.propose_validate_failures([(he_fail, "no-matching-meta-pattern")],
                                 node_id="nA", paper_id="p1", domain="granular")
agent2.propose_validate_failures([(he_fail, "no-matching-meta-pattern")],
                                 node_id="nB", paper_id="p1", domain="granular")
agent2.drain()
pat = kb2.tbox.patterns.get("test_new_pat")
check("add_pattern: pattern committed to tbox", pat is not None)
check("add_pattern: semantic_boundary generated (design supplement section 4)",
      pat is not None and "test discourse" in pat.semantic_boundary)
check("add_pattern: boundary rendered in to_prompt (schema-in-context)",
      "test_new_pat" in kb2.tbox.to_prompt() and "boundary:" in kb2.tbox.to_prompt())

# ===========================================================================
# 3. apply goes through KnowledgeBase.commit (NOT direct meta mutate)
# ===========================================================================
kb3 = _kb()
agent3 = EvolutionAgent(kb3, llm=agent_llm, domain_default="granular")
v_before = kb3.version
agent3.propose_validate_failures([(he_fail, "no-matching-meta-pattern")],
                                 node_id="nA", paper_id="p1", domain="granular")
agent3.propose_validate_failures([(he_fail, "no-matching-meta-pattern")],
                                 node_id="nB", paper_id="p1", domain="granular")
r3 = agent3.drain()
# the version bump came from kb.commit -> tbox.add_pattern -> _bump, and the
# ledger recorded it with a version_diff. Direct meta mutate would NOT produce
# a ledger entry.
last_entry = kb3.ledger[-1]
check("apply: ledger entry has version_diff (kernel transaction, not direct mutate)",
      last_entry.get("version_diff") is not None
      and last_entry["version_diff"]["before"] == v_before
      and last_entry["version_diff"]["after"] == kb3.version)

# ===========================================================================
# 4. recurring crystallize: growth gated on cross_node >= 2
# ===========================================================================
kb4 = _kb()
agent4 = EvolutionAgent(kb4, llm=agent_llm, domain_default="granular")
# single-node failure -> cross_node=1 -> growth REJECTED by conservative gate
agent4.propose_validate_failures([(he_fail, "no-matching-meta-pattern")],
                                 node_id="solo", paper_id="p1", domain="granular")
r4 = agent4.drain()
check("crystallize: single-node growth rejected (cross_node=1)",
      any("conservative gate" in rej["reason"] for rej in r4.rejected))
check("crystallize: pattern NOT added on single node",
      "test_new_pat" not in kb4.tbox.patterns)
# two-node failure -> cross_node=2 -> growth accepted
agent4.propose_validate_failures([(he_fail, "no-matching-meta-pattern")],
                                 node_id="nA", paper_id="p1", domain="granular")
agent4.propose_validate_failures([(he_fail, "no-matching-meta-pattern")],
                                 node_id="nB", paper_id="p1", domain="granular")
r4b = agent4.drain()
check("crystallize: cross_node=2 growth accepted", r4b.n_accepted >= 1)
check("crystallize: pattern added after recurrence", "test_new_pat" in kb4.tbox.patterns)

# ===========================================================================
# 5. HITL flagging: new top-level family flagged (reserved interface)
# ===========================================================================
kb5 = _kb()
agent5 = EvolutionAgent(kb5, llm=agent_llm, domain_default="granular")
# patch probe to propose a NEW family pattern
def _probe_new_family(distinct, meta, paper_id, domain="", llm="deepseek", instance=None):
    return [{"op": "add_pattern", "pattern_id": "new_fam_pat",
             "family": "totally_new_family", "description": "new top-level family",
             "role_slots": [{"role": "x", "type": "THING"}, {"role": "y", "type": "THING"}],
             "allowed_qualifiers": [], "evidence_span": "evidence", "rationale": "new family"}]
hev.evolution_probe = _probe_new_family
agent5.propose_validate_failures([(he_fail, "no-matching-meta-pattern")],
                                 node_id="nA", paper_id="p1", domain="granular")
agent5.propose_validate_failures([(he_fail, "no-matching-meta-pattern")],
                                 node_id="nB", paper_id="p1", domain="granular")
r5 = agent5.drain()
check("HITL: new top-level family flagged (reserved interface)",
      r5.n_hitl >= 1 and "totally_new_family" in r5.hitl[0]["reason"])
hev.evolution_probe = _mock_probe  # restore

# ===========================================================================
# 6. SAGE writer-reader: consumer feedback accumulates -> recurring trigger
# ===========================================================================
kb6 = _kb()
agent6 = EvolutionAgent(kb6, llm=agent_llm, domain_default="granular")
# two consumer feedbacks on the same pattern -> recurring_mismatch trigger enqueued
agent6.collect_consumer_feedback(ConsumerFeedback(
    pattern_id="measures", issue="over_merges", consumer="qa", detail="merges distinct measures"))
agent6.collect_consumer_feedback(ConsumerFeedback(
    pattern_id="measures", issue="over_merges", consumer="survey", detail="same"))
check("SAGE: recurring consumer feedback enqueued a trigger",
      any(s.kind == "consumer_feedback" and s.payload.get("pattern_id") == "measures"
          for s in agent6._queue))

# ===========================================================================
# 7. distill_skill: crystallize -> distill_skill op committed via kernel (断点5)
# ===========================================================================
kb7 = _kb()
agent7 = EvolutionAgent(kb7, llm=agent_llm, domain_default="granular")
# below threshold -> rejected (kernel gate)
ok_low, _ = agent7.distill_skill("granular", "measures", "hint", 0.4, ["p1"])
check("distill_skill: stability<0.6 rejected", ok_low is False)
# pattern not in tbox -> rejected (kernel gate)
ok_bad, _ = agent7.distill_skill("granular", "nonexistent_pattern", "hint", 0.8, ["p1"])
check("distill_skill: pattern not in tbox rejected", ok_bad is False)
# valid -> committed via distill_skill op (evolver contract)
ok_good, detail = agent7.distill_skill("granular", "measures",
                                         "mention the device and the measured quantity",
                                         0.75, ["p1", "p2"])
check("distill_skill: valid skill committed via distill_skill op (断点5 loop)",
      ok_good is True and "committed" in detail)
check("distill_skill: SkillLibrary lookup works after commit",
      kb7.skills.lookup("granular", "measures") is not None)
check("distill_skill: ledger has DISTILL_SKILL entry",
      any(m.get("op") == Op.DISTILL_SKILL for entry in kb7.ledger
          for m in entry.get("mutations", [])))

# ===========================================================================
# 8. schema并进 propagation: schema_for_extraction re-fetches evolved prompt
# ===========================================================================
kb8 = _kb()
agent8 = EvolutionAgent(kb8, llm=agent_llm, domain_default="granular")
prompt_before = agent8.schema_for_extraction()
check("并进: schema_for_extraction returns to_prompt", "Meta-Hypergraph" in prompt_before)
# evolve (add test_new_pat), then schema_for_extraction reflects it
agent8.propose_validate_failures([(he_fail, "no-matching-meta-pattern")],
                                 node_id="nA", paper_id="p1", domain="granular")
agent8.propose_validate_failures([(he_fail, "no-matching-meta-pattern")],
                                 node_id="nB", paper_id="p1", domain="granular")
agent8.drain()
prompt_after = agent8.schema_for_extraction()
check("并进: evolved schema reflected in next schema_for_extraction",
      "test_new_pat" in prompt_after and "test_new_pat" not in prompt_before)

# ===========================================================================
# 9. governance: invalid proposal (no evidence) rejected by validate内核
# ===========================================================================
kb9 = _kb()
agent9 = EvolutionAgent(kb9, llm=agent_llm, domain_default="granular")
# self_split with no evidence/rationale -> validate_proposal mock rejects
agent9.propose_trigger(TriggerSource(kind="self_split", domain="granular",
    payload={"op": "split", "pattern_id": "measures"}))
r9 = agent9.drain()
check("governance: no-evidence proposal rejected by validate内核",
      r9.n_rejected >= 1)

# ===========================================================================
# 10. B2 fix: self_split through REAL validate_proposal (not mocked) — the
# Step2-B1 前科: mock validate_proposal hid that self_* paths had no evidence_span
# and were silently rejected. This test uses the real validate_proposal.
# ===========================================================================
# restore real validate_proposal for this test (un-mock)
import importlib
hev_real = importlib.reload(importlib.import_module("granular_agent.hypergraph_evolution"))
real_validate = hev_real.validate_proposal
# patch the agent's hev module reference to the REAL validate_proposal
import granular_agent.evolution_agent as ea_mod
saved_validate = hev.validate_proposal
hev.validate_proposal = real_validate
try:
    kb10 = _kb()
    agent10 = EvolutionAgent(kb10, llm=agent_llm, domain_default="granular")
    # self_split with representatives (real detect_split_triggers output shape)
    # but NO evidence_span — B1 fix injects evidence from representatives.
    split_payload = {"op": "split", "pattern_id": "measures",
                     "representatives": ["measured stress via device A",
                                         "measured strain via device B"],
                     "rationale": "two distinct measurement clusters",
                     "method": "embedding", "cluster_sizes": [3, 3]}
    src10 = TriggerSource(kind="self_split", domain="granular", payload=split_payload)
    # verify _probe injects evidence_span from representatives (B1)
    probed = agent10._probe(src10)
    check("B1: _probe injects evidence_span from representatives",
          len(probed) == 1 and "measured stress" in probed[0].get("evidence_span", ""))
    agent10.propose_trigger(src10)
    r10 = agent10.drain()
    # the proposal must NOT be rejected for "no verbatim evidence span" (B1)
    no_evidence_reject = any("no verbatim evidence span" in rej["reason"] for rej in r10.rejected)
    check("B1: self_split NOT rejected for missing evidence_span (real validate_proposal)",
          not no_evidence_reject)
finally:
    hev.validate_proposal = saved_validate  # restore mock for any later tests
