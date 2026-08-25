"""Smoke +断点 test for extraction_agent.py (Step 2: plan-execute-verify+skill).

Verifies the agent FORM landed (no downgrade) without real LLM calls:
- planner: produces Plan{domain, central_entities, expected_patterns,
  relation_outline}; domain is the injection point (断点 4); schema-in-context
  includes semantic_boundary.
- executor: delegates to the verified core (mocked here); skill hints injected.
- verifier: ≠ extraction model (different mock); per-edge verdict fix
  {keep/retype/reextract/drop}; re-type INTERNALIZED (type_correct).
- fixer: keep keeps; retype:X changes pattern_type; drop removes; reextract
  honestly drops-with-note.
- commit_edges: add_edge Mutation, domain carried (断点 4), concept instances
  inline (断点 7), validate only NO 5-outcome route (断点 2).
- skill injection: SkillLibrary hint looked up by (domain, pattern) and injected.
- semantic_boundary: rendered in to_prompt (schema-in-context fix).
No real LLM. No real _run_hg_node_multistep (executor core mocked).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from granular_agent.extraction_agent import ExtractionAgent, Plan, Verdict
from granular_agent.knowledge_base import KnowledgeBase, Mutation, Op, Role, Skill
from granular_agent.hypergraph_schema import (
    seed_meta_hypergraph, Hyperedge, HGNode,
)

fail = []

def check(name, cond):
    print(("OK  " if cond else "FAIL") + "  " + name)
    if not cond:
        fail.append(name)


def _kb():
    return KnowledgeBase(tbox=seed_meta_hypergraph())


SECTION = ("We extend the μ(I) rheology (a local rheology) to nonlocal flows. "
          "The nonlocal creep phenomenon is captured by introducing a cooperativity "
          "length ξ. The model improves the local rheology, which fails near "
          "regime boundaries. We compare our nonlocal model with the gradient "
          "model.")


# ===========================================================================
# 1. planner: produces a Plan, domain is the injection point (断点 4)
# ===========================================================================
kb = _kb()


def planner_llm(prompt, max_tokens):
    # the planner prompt must include the schema with semantic_boundary
    return ('{"domain":"granular","central_entities":'
            '[{"surface":"μ(I) rheology","type":"METHOD","role_hint":"from"},'
            '{"surface":"nonlocal creep","type":"PHENOMENON","role_hint":"effect"}],'
            '"secondary_entities":[{"surface":"cooperativity length ξ","type":"PARAMETER"}],'
            '"expected_patterns":["extends","improves","compares","background"],'
            '"relation_outline":[{"pattern_type":"extends","expected_nodes":["nonlocal model","μ(I) rheology"],"evidence_hint":"extend the μ(I) rheology"},'
            '{"pattern_type":"improves","expected_nodes":["nonlocal model","local rheology"],"evidence_hint":"improves the local rheology"}],'
            '"paper_anchor":"nonlocal extension of μ(I)"}')


agent = ExtractionAgent(kb, llm_extract=planner_llm, llm_verify=lambda p, m: "{}",
                        domain_default="global")
plan = agent.plan(SECTION, "method")
check("planner: domain extracted (断点4 injection point)", plan.domain == "granular")
check("planner: central_entities populated", len(plan.central_entities) == 2)
check("planner: expected_patterns populated", "extends" in plan.expected_patterns)
check("planner: relation_outline populated", len(plan.relation_outline) == 2)
check("planner: paper_anchor populated", "μ(I)" in plan.paper_anchor)
check("to_prompt renders semantic_boundary for seed patterns (schema-in-context fix)",
      "boundary:" in kb.tbox.to_prompt())
# M3 fix: ALL 12 seed patterns (incl. the 6 evolution verbs) have a boundary —
# improves/compares confusion was the worst type-confusion source in P0.
for _eid in ("extends", "improves", "compares", "replaces", "adapts", "background"):
    check(f"M3: evolution pattern '{_eid}' has semantic_boundary",
          kb.tbox.patterns[_eid].semantic_boundary != "")

# ===========================================================================
# 2. executor: delegates to core (mocked), skill hint injected
# ===========================================================================
kb2 = _kb()
# install a skill for (granular, extends) — should be injected into executor
kb2.skills.add(Skill(skill_id="s_ext", domain="granular", pattern="extends",
                     extraction_hint="connect the generalizing method as 'from', the base as 'to'",
                     stability_score=0.75))

injected_skill = {"seen": False}


def exec_llm(prompt, max_tokens):
    return "{}"


class _MockAgent(ExtractionAgent):
    """Override _execute_core to capture the schema_prompt (verify skill injection)
    and return canned nodes/edges without calling the real extractor."""
    def _execute_core(self, section_text, plan, node_id, schema_prompt, predecessor_summary):
        # verify the skill hint was injected into the schema_prompt
        if "Extraction skills" in schema_prompt and "s_ext" not in schema_prompt:
            # the hint text (not the id) should be present
            if "generalizing method" in schema_prompt:
                injected_skill["seen"] = True
        # verify plan.relation_outline was injected as schema-in-context
        injected_skill["outline_seen"] = ("Planner's expected relations" in schema_prompt)
        # canned output: 2 nodes + 2 edges (one good, one with wrong type)
        nodes = [
            HGNode(nid="n1", labels=["METHOD"], surface="nonlocal model", evidence_span="our nonlocal"),
            HGNode(nid="n2", labels=["METHOD"], surface="μ(I) rheology", evidence_span="the μ(I) rheology"),
            HGNode(nid="n3", labels=["METHOD"], surface="local rheology", evidence_span="local rheology"),
        ]
        edges = [
            Hyperedge(eid="e1", pattern_type="extends", node_ids=["n1", "n2"],
                      node_roles=["from", "to"], evidence_span="We extend the μ(I) rheology"),
            # mistyped: should be 'improves' not 'compares'
            Hyperedge(eid="e2", pattern_type="compares", node_ids=["n1", "n3"],
                      node_roles=["from", "to"], evidence_span="improves the local rheology"),
        ]
        return nodes, edges


agent2 = _MockAgent(kb2, llm_extract=planner_llm, llm_verify=lambda p, m: "{}",
                    domain_default="granular")
plan2 = agent2.plan(SECTION, "method")
nodes, edges = agent2.execute(SECTION, plan2, "sec1")
check("executor: skill hint injected when skill exists (断点5 use)", injected_skill["seen"])
check("executor: plan relation_outline injected as schema-in-context",
      injected_skill.get("outline_seen"))
check("executor: returns nodes + edges (delegated core)", len(nodes) == 3 and len(edges) == 2)

# ===========================================================================
# 3. verifier: ≠ extraction model, per-edge verdict, re-type internalized
# ===========================================================================
kb3 = _kb()
verify_calls = {"prompts": []}


def verify_llm(prompt, max_tokens):
    verify_calls["prompts"].append(prompt)
    # verifier catches: e2 evidence is real ("improves the local rheology") but
    # mistyped as 'compares' -> fix retype:improves. e1 keep.
    return ('{"verdicts":['
            '{"edge_id":"e1","verbatim_in_source":true,"type_correct":true,'
            '"relation_exists":true,"fix":"keep","note":"ok"},'
            '{"edge_id":"e2","verbatim_in_source":true,"type_correct":false,'
            '"relation_exists":true,"fix":"retype:improves","note":"should be improves not compares"}'
            ']}')


agent3 = _MockAgent(kb3, llm_extract=exec_llm, llm_verify=verify_llm,
                    domain_default="granular")
# section text is the ground truth the verifier checks verbatim against
verdicts = agent3.verify(edges, nodes, SECTION, "granular")
check("verifier: one verdict per edge", len(verdicts) == 2)
check("verifier: e1 keep", verdicts[0].fix == "keep")
check("verifier: e2 retype:improves (re-type INTERNALIZED)", verdicts[1].fix == "retype:improves")
check("verifier: type_correct flag captures mistype", verdicts[1].type_correct is False)
# verifier prompt includes the section text (ground truth for verbatim check)
check("verifier: prompt includes section text (verbatim ground truth)",
      "μ(I) rheology" in verify_calls["prompts"][0])

# ===========================================================================
# 4. fixer: keep/retype/drop/reextract
# ===========================================================================
kb4 = _kb()
agent4 = _MockAgent(kb4, llm_extract=exec_llm, llm_verify=verify_llm,
                     domain_default="granular")
v4 = [
    Verdict(edge_id="e1", fix="keep"),
    Verdict(edge_id="e2", fix="retype:improves"),
    Verdict(edge_id="e3", fix="drop"),
    Verdict(edge_id="e4", fix="reextract"),
]
edges4 = [
    Hyperedge(eid="e1", pattern_type="extends", node_ids=["n1", "n2"], node_roles=["from", "to"]),
    Hyperedge(eid="e2", pattern_type="compares", node_ids=["n1", "n3"], node_roles=["from", "to"]),
    Hyperedge(eid="e3", pattern_type="background", node_ids=["n1", "n4"], node_roles=["from", "to"]),
    Hyperedge(eid="e4", pattern_type="defines", node_ids=["n5", "n6"], node_roles=["subject", "definition"]),
]
kept, stats, _ = agent4.fix(edges4, v4, [], SECTION, plan)
check("fixer: keep kept", stats["keep"] == 1 and kept[0].eid == "e1")
check("fixer: retype changes pattern_type", stats["retype"] == 1
      and kept[1].pattern_type == "improves")
check("fixer: drop removed", stats["drop"] == 1 and not any(he.eid == "e3" for he in kept))
check("fixer: reextract honestly drops-with-note (not unbounded loop)",
      stats["reextract_as_drop"] == 1 and not any(he.eid == "e4" for he in kept))
check("fixer: kept only keep+retype (quality bar)", len(kept) == 2)

# 4b. DETERMINISTIC verbatim post-check (rule, overrides LLM verifier): even
# if the LLM says "keep", an evidence_span NOT in the section text is dropped.
# (real-run failure: verifier kept a non-verbatim evidence)
v4b = [Verdict(edge_id="e1", fix="keep")]  # LLM says keep
edges4b = [Hyperedge(eid="e1", pattern_type="extends", node_ids=["n1", "n2"],
                     node_roles=["from", "to"],
                     evidence_span="this phrase is NOT in the section text")]
kept_b, stats_b, _ = agent4.fix(edges4b, v4b, [], SECTION, plan)
check("fixer: deterministic verbatim gate drops non-substring evidence (rule over LLM)",
      stats_b["dropped_nonverbatim"] == 1 and len(kept_b) == 0)
# a truly verbatim evidence is kept
edges4c = [Hyperedge(eid="e1", pattern_type="extends", node_ids=["n1", "n2"],
                     node_roles=["from", "to"],
                     evidence_span="We extend the μ(I) rheology")]
kept_c, stats_c, _ = agent4.fix(edges4c, v4b, [], SECTION, plan)
check("fixer: verbatim substring evidence kept", stats_c["keep"] == 1 and len(kept_c) == 1)

# 4c. retype to a pattern NOT in tbox is dropped (B3 safety in fixer)
v4d = [Verdict(edge_id="e1", fix="retype:nonexistent_pattern")]
edges4d = [Hyperedge(eid="e1", pattern_type="compares", node_ids=["n1", "n2"],
                     node_roles=["from", "to"],
                     evidence_span="We extend the μ(I) rheology")]
kept_d, stats_d, _ = agent4.fix(edges4d, v4d, [], SECTION, plan)
check("fixer: retype to non-tbox pattern -> drop (B3 safety)",
      stats_d["drop"] == 1 and len(kept_d) == 0)

# 4d. DETERMINISTIC role gate (rule, no downgrade/remap): a 'defines' edge
# carrying from/to roles (composed_of/extends roles, NOT defines' subject/
# definition) is DROPPED — wrong role = wrong structure = drop, NOT remapped to
# correct roles (remap would smuggle bad structure in = downgrade). The real
# run showed the executor emits from/to for every pattern; this gate drops it.
v4e = [Verdict(edge_id="e1", fix="keep")]  # LLM says keep
edges4e = [Hyperedge(eid="e1", pattern_type="defines", node_ids=["n1", "n2"],
                     node_roles=["from", "to"],
                     evidence_span="We extend the μ(I) rheology")]
kept_e, stats_e, _ = agent4.fix(edges4e, v4e, [], SECTION, plan)
check("role gate: defines with from/to roles DROPPED (no remap, no downgrade)",
      stats_e["dropped_bad_role"] == 1 and len(kept_e) == 0)
# correct roles -> kept
edges4f = [Hyperedge(eid="e1", pattern_type="defines", node_ids=["n1", "n2"],
                     node_roles=["subject", "definition"],
                     evidence_span="We extend the μ(I) rheology")]
kept_f, stats_f, _ = agent4.fix(edges4f, v4e, [], SECTION, plan)
check("role gate: defines with correct subject/definition roles kept",
      stats_f["keep"] == 1 and len(kept_f) == 1)

# 4d-bis. ROLE-FIX (bad-role cure): a 'measures' edge with WRONG roles (from/to
# instead of object/instrument) — without rolefix the rule gate drops it whole
# (coverage hole). WITH rolefix, the LLM verifier gives corrected roles; fixer
# re-points them; the rule gate RE-CHECKS (passes, declared roles); edge survives.
# rule守 structure (no downgrade: bad roles never committed); LLM守 semantic
# (judges correct roles). cures the 12-edge coverage loss on ML_DQN.
v4rf = [Verdict(edge_id="e1", fix="rolefix:object,instrument", role_correct=False)]
edges4rf = [Hyperedge(eid="e1", pattern_type="measures", node_ids=["n1", "n2"],
                      node_roles=["from", "to"],  # wrong roles (measures wants object/instrument)
                      evidence_span="We extend the μ(I) rheology")]
kept_rf, stats_rf, _ = agent4.fix(edges4rf, v4rf, [], SECTION, plan)
check("rolefix: LLM-repaired roles survive the rule gate (bad-role cured)",
      len(kept_rf) == 1 and kept_rf[0].node_roles == ["object", "instrument"])
check("rolefix: counted in stats", stats_rf.get("rolefix", 0) == 1)
# LLM gives an INVALID rolefix (role not in pattern) -> rule gate still drops (no downgrade)
v4rf_bad = [Verdict(edge_id="e1", fix="rolefix:bogus_role,another_fake", role_correct=False)]
kept_rf2, stats_rf2, _ = agent4.fix(edges4rf, v4rf_bad, [], SECTION, plan)
check("rolefix: invalid roles still dropped (rule gate守 structure, no downgrade)",
      len(kept_rf2) == 0 and stats_rf2.get("drop", 0) >= 1)
# LLM doesn't issue rolefix (missed it) but roles wrong -> rule gate drops whole
# (auditable as verifier-missed-bad-role — a verifier-quality signal)
v4miss = [Verdict(edge_id="e1", fix="keep", role_correct=True)]  # verifier missed the bad role
edges4miss = [Hyperedge(eid="e1", pattern_type="measures", node_ids=["n1", "n2"],
                        node_roles=["from", "to"],  # wrong roles (fresh object — rolefix test mutates)
                        evidence_span="We extend the μ(I) rheology")]
kept_miss, stats_miss, dropped_miss = agent4.fix(edges4miss, v4miss, [], SECTION, plan)
check("role gate: verifier-missed-bad-role dropped (rule守 structure, not silently kept)",
      len(kept_miss) == 0 and stats_miss.get("dropped_bad_role", 0) >= 1
      and len(dropped_miss) >= 1)

# ===========================================================================
# 5. commit_edges: add_edge Mutation, domain carried, concepts inline, NO route
# ===========================================================================
kb5 = _kb()
agent5 = _MockAgent(kb5, llm_extract=exec_llm, llm_verify=verify_llm,
                    domain_default="granular")
nodes5 = [
    HGNode(nid="n1", labels=["METHOD"], surface="nonlocal model", evidence_span="our nonlocal"),
    HGNode(nid="n2", labels=["METHOD"], surface="μ(I) rheology", evidence_span="the μ(I) rheology"),
]
edges5 = [
    Hyperedge(eid="e1", pattern_type="extends", node_ids=["n1", "n2"],
              node_roles=["from", "to"], evidence_span="We extend the μ(I) rheology"),
]
plan5 = Plan(domain="granular", expected_patterns=["extends"])
n_comm, rejected = agent5.commit_edges(edges5, nodes5, plan5, paper_id="p1", year="2020")
check("commit_edges: edge committed (n_committed=1)", n_comm == 1)
check("commit_edges: concept created in A-box", len(kb5.abox.concepts) == 2)
check("commit_edges: hyperedge in A-box", len(kb5.abox.hyperedges) == 1)
# domain must be recorded on the mutation (断点 4)
last_mut = kb5.ledger[-1]["mutations"][0]
check("commit_edges: domain carried on Mutation (断点4)", last_mut["domain"] == "granular")
check("commit_edges: evidence carried (candidate-evidence binding)",
      "extend the μ(I)" in last_mut["evidence"])
# extractor edges must NOT be routed (断点 2: no 5-outcome for extractor)
last_result = kb5.ledger[-1]["result"]
check("commit_edges: extractor NOT routed (断点2, no 5-outcome)", len(last_result["routed"]) == 0)

# commit with <2 concepts rejected (referential integrity)
edges_bad = [Hyperedge(eid="e9", pattern_type="extends", node_ids=["n1"], node_roles=["from"])]
n_comm_bad, rej_bad = agent5.commit_edges(edges_bad, nodes5, plan5, "p1")
check("commit_edges: <2 concepts -> not committed (referential integrity)", n_comm_bad == 0)

# 5b. B3 fix: add_edge with a pattern_type NOT in the tbox -> kernel rejects
# (a retype to a non-existent pattern must be caught at the kernel, not
# silently committed as bad data). DECISION had claimed this; B3 makes it true.
edges_unknown = [Hyperedge(eid="e_unk", pattern_type="analogy_not_in_schema",
                           node_ids=["n1", "n2"], node_roles=["from", "to"],
                           evidence_span="We extend the μ(I) rheology")]
n_comm_unk, rej_unk = agent5.commit_edges(edges_unknown, nodes5, plan5, "p1")
check("B3: add_edge unknown pattern_type -> kernel rejects (bad data blocked)",
      n_comm_unk == 0 and "unknown-pattern-type" in rej_unk[0]["reason"])

# 5b2. B3 role-check: a 'defines' edge carrying composed_of's roles (whole/
# component) is rejected — roles must match the pattern's declared role_slots.
# (real-run failure: defines edge had whole/component roles)
edges_wrong_role = [Hyperedge(eid="e_wr", pattern_type="defines",
                              node_ids=["n1", "n2"], node_roles=["whole", "component"],
                              evidence_span="We extend the μ(I) rheology")]
n_comm_wr, rej_wr = agent5.commit_edges(edges_wrong_role, nodes5, plan5, "p1")
check("B3: add_edge with roles not in pattern -> kernel rejects",
      n_comm_wr == 0 and "role-not-in-pattern" in rej_wr[0]["reason"])

# 5c. B2 fix: central entities from the plan are marked central on the Concept
kb5c = _kb()
agent5c = _MockAgent(kb5c, llm_extract=planner_llm, llm_verify=verify_llm,
                     domain_default="granular")
# planner returns a plan whose central_entities include "μ(I) rheology"
plan5c = agent5c.plan(SECTION, "method")
nodes5c = [HGNode(nid="n1", labels=["METHOD"], surface="μ(I) rheology",
                  evidence_span="the μ(I) rheology"),
           HGNode(nid="n2", labels=["METHOD"], surface="local rheology",
                  evidence_span="local rheology")]
# only commit edges so central-matching runs
agent5c.commit_edges(edges5, nodes5c, plan5c, "p1", "2020")
central_concepts = [c for c in kb5c.abox.concepts.values() if c.central]
non_central = [c for c in kb5c.abox.concepts.values() if not c.central]
check("B2: central entity from plan marked central=True on Concept",
      len(central_concepts) == 1 and central_concepts[0].surfaces()[0].lower().startswith("μ(i) rheology"))
check("B2: non-central concept stays central=False", len(non_central) == 1)

# 5d. M4 fix: provenance first-class fields (cited_from/method/evidence_strength)
# lifted from qualifiers onto ConceptHyperedge.provenance
kb5d = _kb()
agent5d = _MockAgent(kb5d, llm_extract=planner_llm, llm_verify=verify_llm,
                     domain_default="granular")
plan5d = Plan(domain="granular", expected_patterns=["extends"])
edges5d = [Hyperedge(eid="e1", pattern_type="extends", node_ids=["n1", "n2"],
                     node_roles=["from", "to"],
                     evidence_span="We extend the μ(I) rheology",
                     qualifiers={"cited_from": "this_work", "method": "theory",
                                 "evidence_strength": "derived",
                                 "relation_type": "generalization"})]
agent5d.commit_edges(edges5d, nodes5, plan5d, "p1", "2020")
he5d = kb5d.abox.hyperedges[0]
prov_entry = he5d.provenance[-1] if he5d.provenance else {}
check("M4: cited_from lifted to provenance (first-class)", prov_entry.get("cited_from") == "this_work")
check("M4: method lifted to provenance (first-class)", prov_entry.get("method") == "theory")
check("M4: evidence_strength lifted to provenance (first-class)",
      prov_entry.get("evidence_strength") == "derived")
check("M4: business qualifier relation_type stays in qualifiers (not provenance)",
      he5d.kind == "extends")  # kind unchanged; qualifiers retained on edge via add_hyperedge

# ===========================================================================
# 6. skill injection NOT triggered when no skill for (domain, pattern)
# ===========================================================================
kb6 = _kb()  # no skills installed
injected2 = {"seen": False}


class _MockAgentNoSkill(ExtractionAgent):
    def _execute_core(self, section_text, plan, node_id, schema_prompt, predecessor_summary):
        injected2["seen"] = "Extraction skills" in schema_prompt
        return [], []


agent6 = _MockAgentNoSkill(kb6, llm_extract=exec_llm, llm_verify=lambda p, m: "{}",
                           domain_default="granular")
plan6 = agent6.plan(SECTION, "method")
agent6.execute(SECTION, plan6, "sec1")
check("executor: NO skill block when SkillLibrary has no matching skill",
      injected2["seen"] is False)

# ===========================================================================
# 7. end-to-end extract_section returns a report (see results, not just pass/fail)
# ===========================================================================
kb7 = _kb()
agent7 = _MockAgent(kb7, llm_extract=exec_llm, llm_verify=verify_llm,
                    domain_default="granular")
# planner returns a granular plan
agent7.llm_extract = planner_llm
report = agent7.extract_section(SECTION, "method", "sec1", "p1", year="2020")
check("e2e: report has domain", report["domain"] == "granular")
check("e2e: report has raw/kept edge counts", report["n_raw_edges"] == 2)
check("e2e: report has fix_stats", "keep" in report["fix_stats"])
check("e2e: report has committed count", report["n_committed"] >= 1)
check("e2e: report has kept_edges detail (see content not just number)",
      len(report["kept_edges"]) >= 1 and "pattern_type" in report["kept_edges"][0])

print()
print(f"{'ALL PASS' if not fail else 'FAILURES: ' + str(fail)}  ({len(fail)} fail)")
sys.exit(1 if fail else 0)
