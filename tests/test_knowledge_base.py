"""Smoke +断点 test for knowledge_base.py (the 底座内核).

Verifies the v2 design mechanisms landed honestly (no downgrade):
- Write Contract: role -> op enforced (extractor can't retire, etc.)
- Two-phase commit: validate ATOMIC (any failure rolls back the batch);
  route NON-atomic per-edge, ALIGNER ONLY (5-outcome).
- 5-outcome n-ary subset-relation merge semantics (Challenge B):
    identical -> MERGE; subset/superset -> MERGE to superset (union nodes);
    partial overlap non-subset -> RELATE; no overlap -> INSERT (or judge conflict).
- A-box referential integrity: dangling concept refs rejected (断点 11).
- schema-constrained rewrite (Challenge C, narrowed to IS-A/family/invariant):
    split: no new role; merge: survivor covers union of roles; retire: no active
    abox edge references the pattern; add_pattern: slot types known.
- Skill Library (4th layer) + distill_skill op + stability threshold 0.6 (断点 5).
- CAS narrowed: base_version recorded, NOT enforced (serial builders) (断点 10).
- domain namespace gate (断点 4).
- Ledger: audit / replay_to time-travel; soft-delete only (retire, not hard-del).
- candidate-evidence binding: empty evidence rejected.
No real LLM (judge_fn is a mock when exercised).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from granular_agent.knowledge_base import (
    KnowledgeBase, Mutation, Skill, SkillLibrary, Op, Role, Outcome,
    WRITE_CONTRACT,
)
from granular_agent.hypergraph_schema import (
    seed_meta_hypergraph, MetaHyperedgePattern, MetaNode, MetaEdge,
)

fail = []

def check(name, cond):
    print(("OK  " if cond else "FAIL") + "  " + name)
    if not cond:
        fail.append(name)


def _kb():
    """Fresh KB on a seeded meta-hypergraph."""
    return KnowledgeBase(tbox=seed_meta_hypergraph())


# ===========================================================================
# 1. Write Contract: role -> op enforced
# ===========================================================================
kb = _kb()
# extractor may add_edge but NOT retire / split (evolver ops)
r = kb.commit([Mutation(op=Op.RETIRE, target="measures", proposer_role=Role.EXTRACTOR,
                        evidence="x")])
check("extractor cannot retire (write contract)", r.ok is False
      and "op-not-allowed-for-role" in r.rejected[0]["reason"])
r = kb.commit([Mutation(op=Op.SPLIT, target="measures", proposer_role=Role.EXTRACTOR,
                        payload={"sub_patterns": []}, evidence="x")])
check("extractor cannot split (write contract)", r.ok is False)
# aligner may align_merge but NOT add_pattern
r = kb.commit([Mutation(op=Op.ADD_PATTERN, target="x", proposer_role=Role.ALIGNER,
                        payload={"pattern": MetaHyperedgePattern(pattern_id="zz",
                                  role_slots=[{"role":"a","type":"THING"}])},
                        evidence="x")])
check("aligner cannot add_pattern (write contract)", r.ok is False)
# maintainer may retire but not add_pattern
r = kb.commit([Mutation(op=Op.ADD_PATTERN, target="x", proposer_role=Role.MAINTAINER,
                        payload={"pattern": MetaHyperedgePattern(pattern_id="zz2",
                                  role_slots=[{"role":"a","type":"THING"}])},
                        evidence="x")])
check("maintainer cannot add_pattern (write contract)", r.ok is False)
# consumer may propose nothing directly (empty contract)
check("consumer write contract is empty (read-only)",
      WRITE_CONTRACT[Role.CONSUMER] == set())

# ===========================================================================
# 2. candidate-evidence binding: empty evidence rejected (except relabel)
# ===========================================================================
kb = _kb()
r = kb.commit([Mutation(op=Op.ADD_PATTERN, target="new_pat",
                        proposer_role=Role.EVOLVER,
                        payload={"pattern": MetaHyperedgePattern(
                            pattern_id="new_pat", family="claim",
                            role_slots=[{"role":"from","type":"THING"},
                                        {"role":"to","type":"THING"}])},
                        evidence="")])
check("missing candidate-evidence rejected", r.ok is False
      and "missing-candidate-evidence" in r.rejected[0]["reason"])
# relabel does NOT require evidence (utility op; rationale suffices)
kb2 = _kb()
# add a concept first (via extractor add_edge carrying concepts)
r = kb2.commit([Mutation(op=Op.ADD_EDGE, target="e1", proposer_role=Role.EXTRACTOR,
        domain="granular", evidence="the cell measured stress",
        payload={"kind":"measures", "roles":["object","instrument"],
                 "concepts":[
                     {"surface":"stress","type":"PROPERTY","evidence":"stress"},
                     {"surface":"shear cell","type":"MATERIAL","evidence":"cell"}],
                 "provenance":{"paper_id":"p1"}})])
check("extractor add_edge carries concept instances inline (断点7)", r.ok is True)
cid = list(kb2.abox.concepts.keys())[0]
r = kb2.commit([Mutation(op=Op.RELABEL, target=cid, proposer_role=Role.MAINTAINER,
                         evidence="", payload={"canonical_name":"stress(renamed)"})])
check("relabel allowed without evidence (utility op)", r.ok is True)

# ===========================================================================
# 3. domain namespace gate (断点 4)
# ===========================================================================
kb = _kb()
r = kb.commit([Mutation(op=Op.ADD_PATTERN, target="zz", proposer_role=Role.EVOLVER,
        domain="nonexistent_domain", evidence="x",
        payload={"pattern": MetaHyperedgePattern(pattern_id="zz",
                  role_slots=[{"role":"a","type":"THING"}])})])
check("unregistered domain rejected (断点4)", r.ok is False
      and "domain-not-registered" in r.rejected[0]["reason"])

# ===========================================================================
# 4. Atomic validate: any failure rolls back the WHOLE batch
# ===========================================================================
kb = _kb()
v_pre = kb.version
# batch: one good add_pattern + one bad (colliding id) -> whole batch rolled back
good = Mutation(op=Op.ADD_PATTERN, target="good_pat", proposer_role=Role.EVOLVER,
                evidence="evidence-good",
                payload={"pattern": MetaHyperedgePattern(pattern_id="good_pat",
                          family="claim",
                          role_slots=[{"role":"from","type":"THING"},
                                      {"role":"to","type":"THING"}])})
# bad: colliding pattern_id (already exists in seed) -> schema constraint reject
bad = Mutation(op=Op.ADD_PATTERN, target="measures", proposer_role=Role.EVOLVER,
               evidence="evidence-bad",
               payload={"pattern": MetaHyperedgePattern(pattern_id="measures",
                         family="measure",
                         role_slots=[{"role":"object","type":"THING"}])})
r = kb.commit([good, bad])
check("atomic rollback: batch with one failure -> ok=False", r.ok is False)
check("atomic rollback: version unchanged", kb.version == v_pre)
check("atomic rollback: good mutation NOT applied (pattern absent)",
      "good_pat" not in kb.tbox.patterns)

# ===========================================================================
# 5. A-box referential integrity (断点 11): dangling concept refs rejected
# ===========================================================================
kb = _kb()
r = kb.commit([Mutation(op=Op.ALIGN_MERGE, target="am1", proposer_role=Role.ALIGNER,
        domain="granular", evidence="align",
        payload={"new_edge": {"node_ids": ["CNOPE0001","CNOPE0002"],
                              "kind":"method_parameter",
                              "roles":["method","param"],
                              "provenance":{"paper_id":"p1"}}})])
check("align_merge dangling concept rejected (断点11)", r.ok is False
      and "dangling-concept" in r.rejected[0]["reason"])
# add_edge with <2 concepts rejected
r = kb.commit([Mutation(op=Op.ADD_EDGE, target="e1", proposer_role=Role.EXTRACTOR,
        domain="granular", evidence="only one concept",
        payload={"kind":"measures", "roles":["object"],
                 "concepts":[{"surface":"stress","type":"PROPERTY","evidence":"s"}],
                 "provenance":{"paper_id":"p1"}})])
check("add_edge with <2 concepts rejected", r.ok is False
      and "needs->=2-concepts" in r.rejected[0]["reason"])

# ===========================================================================
# 6. schema-constrained rewrite — Challenge C (narrowed IS-A/family/invariant)
# ===========================================================================
# 6a. split: sub-pattern introducing a NEW role rejected
kb = _kb()
parent = kb.tbox.patterns["measures"]
ok_sub = MetaHyperedgePattern(pattern_id="meas_a", family="measure",
    role_slots=[dict(s) for s in parent.role_slots],
    allowed_qualifiers=list(parent.allowed_qualifiers))
bad_sub = MetaHyperedgePattern(pattern_id="meas_b", family="measure",
    role_slots=[{"role":"NEW_ROLE","type":"THING"}],   # introduces a new role
    allowed_qualifiers=list(parent.allowed_qualifiers))
r = kb.commit([Mutation(op=Op.SPLIT, target="measures", proposer_role=Role.EVOLVER,
                evidence="split evidence",
                payload={"sub_patterns":[ok_sub, bad_sub]})])
check("split: sub introducing new role rejected (Challenge C)", r.ok is False
      and "sub-introduces-new-role" in r.rejected[0]["reason"])
# 6b. split: subs that keep parent roles (subset) accepted
kb = _kb()
ok_sub2 = MetaHyperedgePattern(pattern_id="meas_c", family="measure",
    role_slots=[dict(s) for s in parent.role_slots],
    allowed_qualifiers=list(parent.allowed_qualifiers))
ok_sub3 = MetaHyperedgePattern(pattern_id="meas_d", family="measure",
    role_slots=[dict(s) for s in parent.role_slots],
    allowed_qualifiers=list(parent.allowed_qualifiers))
r = kb.commit([Mutation(op=Op.SPLIT, target="measures", proposer_role=Role.EVOLVER,
                evidence="split evidence",
                payload={"sub_patterns":[ok_sub2, ok_sub3]})])
check("split: subs keeping parent roles accepted (Challenge C)", r.ok is True)
check("split: parent becomes abstract (IS-A taxonomy kept)",
      kb.tbox.patterns["measures"].is_abstract is True)

# 6c. merge: survivor missing roles after union rejected
kb = _kb()
# give two patterns with different roles, merge into a third that lacks both
p1 = MetaHyperedgePattern(pattern_id="m_a", family="claim",
    role_slots=[{"role":"from","type":"THING"},{"role":"to","type":"THING"}],
    allowed_qualifiers=["relation_type"])
p2 = MetaHyperedgePattern(pattern_id="m_b", family="claim",
    role_slots=[{"role":"from","type":"THING"},{"role":"parameter","type":"THING"}],
    allowed_qualifiers=["relation_type"])
kb.tbox.add_pattern(p1); kb.tbox.add_pattern(p2)
# survivor m_c lacks both 'to' and 'parameter'
p3 = MetaHyperedgePattern(pattern_id="m_c", family="claim",
    role_slots=[{"role":"from","type":"THING"}],
    allowed_qualifiers=["relation_type"])
kb.tbox.add_pattern(p3)
r = kb.commit([Mutation(op=Op.MERGE, target="m_c", proposer_role=Role.EVOLVER,
                evidence="merge evidence",
                payload={"pattern_ids":["m_a","m_b"], "into":"m_c"})])
check("merge: survivor missing roles after union rejected (Challenge C)", r.ok is False
      and "survivor-missing-roles-after-union" in r.rejected[0]["reason"])

# 6d. retire: active A-box edge referencing pattern -> rejected (T-box<->A-box RI)
kb = _kb()
from granular_agent.concept_graph import Concept
kb.abox.concepts["c1"] = Concept(concept_id="c1", type="PROPERTY")
kb.abox.concepts["c2"] = Concept(concept_id="c2", type="PROPERTY")
# add an abox edge with kind == a seed pattern_id (e.g. 'measures')
kb.abox.add_hyperedge(["c1","c2"], kind="measures", paper_id="p1", evidence="x")
r = kb.commit([Mutation(op=Op.RETIRE, target="measures", proposer_role=Role.MAINTAINER,
                        evidence="retire evidence")])
check("retire: active abox edge references pattern -> rejected (Challenge C)", r.ok is False
      and "active-abox-edge-references-pattern" in r.rejected[0]["reason"])

# 6e. add_pattern: unknown slot type rejected
kb = _kb()
r = kb.commit([Mutation(op=Op.ADD_PATTERN, target="bad_pat", proposer_role=Role.EVOLVER,
        evidence="x",
        payload={"pattern": MetaHyperedgePattern(pattern_id="bad_pat", family="claim",
                  role_slots=[{"role":"a","type":"NONEXISTENT_TYPE"}])})])
check("add_pattern: unknown slot type rejected", r.ok is False
      and "unknown-slot-type" in r.rejected[0]["reason"])

# ===========================================================================
# 7. 5-outcome n-ary subset-relation merge (Challenge B) — ALIGNER ONLY
# ===========================================================================
from granular_agent.concept_graph import Concept


def _am_mut(node_ids, kind, roles, ev, paper="p2", domain="granular"):
    """Build an align_merge Mutation with an explicit new_edge dict (no brace hell)."""
    ne = {"node_ids": list(node_ids), "kind": kind, "roles": list(roles),
          "provenance": {"paper_id": paper, "evidence": ev}}
    return Mutation(op=Op.ALIGN_MERGE, target="am", proposer_role=Role.ALIGNER,
                    domain=domain, evidence=ev, payload={"new_edge": ne})


def _seed_concepts(kb, cids):
    for cid in cids:
        kb.abox.concepts[cid] = Concept(
            concept_id=cid, type="METHOD" if cid.startswith("M") else "PARAMETER")


# 7a. identical node-set -> MERGE
kb = _kb()
_seed_concepts(kb, ["M1", "P1", "P2", "P3", "M2", "Q1"])
kb.abox.add_hyperedge(["M1", "P1", "P2"], kind="method_parameter", paper_id="p1", evidence="orig")
r = kb.commit([_am_mut(["M1", "P1", "P2"], "method_parameter", ["method", "param", "param"], "second")])
check("5-outcome: identical node-set -> MERGE", r.ok and r.routed[0]["outcome"] == Outcome.MERGE)
check("MERGE unions provenance (2 papers)", r.ok and
      len([p for p in kb.abox.hyperedges[0].provenance if p.get("paper_id") in ("p1", "p2")]) == 2)

# 7b. subset -> MERGE to superset (union nodes, arity preserved)
kb2 = _kb()
_seed_concepts(kb2, ["M1", "P1", "P2", "P3"])
kb2.abox.add_hyperedge(["M1", "P1"], kind="method_parameter", paper_id="p1", evidence="small")
r = kb2.commit([_am_mut(["M1", "P1", "P2", "P3"], "method_parameter",
                        ["method", "param", "param", "param"], "big")])
check("5-outcome: subset -> MERGE to superset (Challenge B)", r.ok and
      r.routed[0]["outcome"] == Outcome.MERGE)
check("MERGE to superset unions nodes (arity preserved, not binary collapse)",
      r.ok and set(kb2.abox.hyperedges[0].node_ids) == {"M1", "P1", "P2", "P3"})

# 7c. partial overlap non-subset -> RELATE
kb3 = _kb()
_seed_concepts(kb3, ["M1", "P1", "P2", "P3"])
kb3.abox.add_hyperedge(["M1", "P1"], kind="method_parameter", paper_id="p1", evidence="a")
r = kb3.commit([_am_mut(["M1", "P2", "P3"], "method_parameter",
                        ["method", "param", "param"], "partial")])
check("5-outcome: partial overlap non-subset -> RELATE (Challenge B)", r.ok and
      r.routed[0]["outcome"] == Outcome.RELATE)
check("RELATE adds a 'relates' cross-link (does NOT collapse the two edges)",
      r.ok and any(he.kind == "relates" for he in kb3.abox.hyperedges))
check("RELATE keeps both original edges distinct",
      r.ok and len([he for he in kb3.abox.hyperedges if he.kind == "method_parameter"]) >= 2)

# 7d. no overlap same kind, no judge -> INSERT (data-preserving) + flagged
kb4 = _kb()
_seed_concepts(kb4, ["M1", "P1", "M2", "Q1"])
kb4.abox.add_hyperedge(["M1", "P1"], kind="method_parameter", paper_id="p1", evidence="x")
r = kb4.commit([_am_mut(["M2", "Q1"], "method_parameter", ["method", "param"], "disjoint")])
check("5-outcome: no-overlap no-judge -> INSERT (data-preserving)", r.ok and
      r.routed[0]["outcome"] == Outcome.INSERT)
check("INSERT flagged no-judge ambiguous (honest)", r.ok and
      "no-judge" in r.routed[0]["detail"])

# 7e. no overlap same kind WITH judge -> judge can return CONFLICT
kb5 = _kb()
_seed_concepts(kb5, ["M1", "P1", "M2", "Q1"])
kb5.abox.add_hyperedge(["M1", "P1"], kind="method_parameter", paper_id="p1", evidence="x")
kb5.set_judge(lambda ctx: Outcome.CONFLICT)  # judge: semantically same but inconsistent
r = kb5.commit([_am_mut(["M2", "Q1"], "method_parameter", ["method", "param"], "disjoint2")])
check("5-outcome: judge can decide CONFLICT on no-overlap same-kind", r.ok and
      r.routed[0]["outcome"] == Outcome.CONFLICT)

# 7f. extractor add_edge does NOT enter 5-outcome (断点 2)
kb6 = _kb()
r = kb6.commit([Mutation(op=Op.ADD_EDGE, target="e1", proposer_role=Role.EXTRACTOR,
        domain="granular", evidence="raw extraction",
        payload={"kind":"measures","roles":["object","instrument"],
                 "concepts":[{"surface":"stress","type":"PROPERTY","evidence":"s"},
                             {"surface":"cell","type":"MATERIAL","evidence":"c"}],
                 "provenance":{"paper_id":"p1"}})])
check("extractor add_edge NOT routed (断点2: no 5-outcome for extractor)", r.ok and
      len(r.routed) == 0)
check("extractor add_edge created concept + hyperedge", r.ok and
      len(kb6.abox.concepts) == 2 and len(kb6.abox.hyperedges) == 1)

# ===========================================================================
# 8. Skill Library (4th layer) + distill_skill + stability threshold (断点 5)
# ===========================================================================
kb = _kb()
# below threshold -> rejected
low_skill = Skill(skill_id="s_low", domain="granular", pattern="measures",
                  extraction_hint="hint", stability_score=0.4)
r = kb.commit([Mutation(op=Op.DISTILL_SKILL, target="s_low", proposer_role=Role.EVOLVER,
                        evidence="distill", payload={"skill": low_skill})])
check("distill_skill: stability < 0.6 rejected (crystallize threshold)", r.ok is False
      and "stability" in r.rejected[0]["reason"])
# pattern not in tbox -> rejected
bad_pat_skill = Skill(skill_id="s_bad", domain="granular", pattern="nonexistent_pattern",
                      extraction_hint="hint", stability_score=0.8)
r = kb.commit([Mutation(op=Op.DISTILL_SKILL, target="s_bad", proposer_role=Role.EVOLVER,
                        evidence="distill", payload={"skill": bad_pat_skill})])
check("distill_skill: pattern not in tbox rejected", r.ok is False
      and "pattern-not-in-tbox" in r.rejected[0]["reason"])
# valid -> accepted, lookup works
good_skill = Skill(skill_id="s_good", domain="granular", pattern="measures",
                   extraction_hint="mention the device and the measured quantity",
                   stability_score=0.75, provenance_papers=["p1","p2"])
r = kb.commit([Mutation(op=Op.DISTILL_SKILL, target="s_good", proposer_role=Role.EVOLVER,
                        evidence="distill", payload={"skill": good_skill})])
check("distill_skill: valid skill accepted (4th layer)", r.ok is True)
check("SkillLibrary lookup by (domain,pattern) works",
      kb.skills.lookup("granular","measures").skill_id == "s_good")
check("distill_skill is evolver op (NOT extractor)",
      Op.DISTILL_SKILL in WRITE_CONTRACT[Role.EVOLVER] and
      Op.DISTILL_SKILL not in WRITE_CONTRACT[Role.EXTRACTOR])

# ===========================================================================
# 9. CAS narrowed (断点 10): base_version recorded, NOT enforced
# ===========================================================================
kb = _kb()
# propose with a STALE base_version; serial builders -> NOT rejected for staleness
mut = Mutation(op=Op.ADD_PATTERN, target="cas_pat", proposer_role=Role.EVOLVER,
              base_version="0.0_stale", evidence="x",
              payload={"pattern": MetaHyperedgePattern(pattern_id="cas_pat", family="claim",
                        role_slots=[{"role":"from","type":"THING"},
                                    {"role":"to","type":"THING"}])})
mid = kb.propose(mut)
check("propose returns mutation_id", mid.startswith("mut_"))
r = kb.commit()
check("CAS narrowed: stale base_version NOT enforced (serial builders, 断点10)", r.ok is True)
check("CAS narrowed: base_version recorded in ledger for audit",
      kb.ledger[-1]["mutations"][0]["base_version"] == "0.0_stale")

# ===========================================================================
# 10. Ledger: audit + replay_to time-travel; soft-delete only
# ===========================================================================
kb = _kb()
# evolve: add a pattern, then retire it (soft)
kb.commit([Mutation(op=Op.ADD_PATTERN, target="tmp_pat", proposer_role=Role.EVOLVER,
        evidence="tmp",
        payload={"pattern": MetaHyperedgePattern(pattern_id="tmp_pat", family="claim",
                  role_slots=[{"role":"from","type":"THING"},{"role":"to","type":"THING"}])})])
v_after_add = kb.version
kb.commit([Mutation(op=Op.RETIRE, target="tmp_pat", proposer_role=Role.MAINTAINER,
                    evidence="retiring tmp")])
check("retire is SOFT delete (pattern kept, deprecated flag set)",
      "tmp_pat" in kb.tbox.patterns and kb.tbox.patterns["tmp_pat"].deprecated is True)
# audit returns ledger entries
audit = kb.audit()
check("audit returns ledger entries", len(audit) >= 2)
# replay to the version BEFORE retire -> tmp_pat not deprecated there
kb_replay = kb.replay_to(v_after_add)
check("replay_to time-travel: at pre-retire version, pattern not deprecated",
      kb_replay.tbox.patterns.get("tmp_pat") and
      kb_replay.tbox.patterns["tmp_pat"].deprecated is False)
check("replay_to: version matches target", kb_replay.version == v_after_add)

# ===========================================================================
# 11. snapshot (consumer read, version-stamped)
# ===========================================================================
kb = _kb()
kb.commit([Mutation(op=Op.ADD_EDGE, target="e1", proposer_role=Role.EXTRACTOR,
        domain="granular", evidence="snap",
        payload={"kind":"measures","roles":["object","instrument"],
                 "concepts":[{"surface":"s","type":"PROPERTY","evidence":"s"},
                             {"surface":"c","type":"MATERIAL","evidence":"c"}],
                 "provenance":{"paper_id":"p1"}})])
snap = kb.snapshot("granular")
check("snapshot carries version stamp", snap.version == kb.version)
check("snapshot domain matches request", snap.domain == "granular")
check("snapshot tbox is full (not summary)", "patterns" in snap.tbox_snapshot)
check("snapshot abox is summary (on-demand fetch)", "n_concepts" in snap.abox_summary
      and snap.abox_summary["n_concepts"] == 2)

# ===========================================================================
# 12. persistence round-trip
# ===========================================================================
kb = _kb()
kb.commit([Mutation(op=Op.ADD_PATTERN, target="rt_pat", proposer_role=Role.EVOLVER,
        evidence="rt",
        payload={"pattern": MetaHyperedgePattern(pattern_id="rt_pat", family="claim",
                  role_slots=[{"role":"from","type":"THING"},{"role":"to","type":"THING"}])})])
good_skill = Skill(skill_id="rt_skill", domain="granular", pattern="rt_pat",
                   extraction_hint="h", stability_score=0.7)
kb.commit([Mutation(op=Op.DISTILL_SKILL, target="rt_skill", proposer_role=Role.EVOLVER,
                    evidence="rt", payload={"skill": good_skill})])
d = kb.to_dict()
kb2 = KnowledgeBase.from_dict(d)
check("round-trip: version preserved", kb.version == kb2.version)
check("round-trip: pattern preserved", "rt_pat" in kb2.tbox.patterns)
check("round-trip: skill preserved (4th layer)", kb2.skills.lookup("granular","rt_pat") is not None)
check("round-trip: ledger preserved", len(kb2.ledger) == len(kb.ledger))
check("round-trip: mutation id counter restored (no collision)",
      kb2._mid_counter == kb._mid_counter)

print()
print(f"{'ALL PASS' if not fail else 'FAILURES: ' + str(fail)}  ({len(fail)} fail)")
sys.exit(1 if fail else 0)
