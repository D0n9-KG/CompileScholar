"""Extraction agent (Step 2): plan-execute-verify + skill evolution, wired to
the KnowledgeBase kernel (Step 1).

Design (see .research_tmp/plan_total_design_v2.md "Builder 群 / 抽取 agent" +
plan_detailed_design_supplement.md section 4):

  planner (LLM)
    └─ section + schema-in-context (patterns + semantic_boundary + examples
       marked "illustrative not rules") → Plan{domain, central_entities,
       secondary_entities, expected_patterns, relation_outline}. planner also
       SPECIFIES the domain (断点 4: domain injection point).
  executor (LLM, single-task, structured output)
    └─ per relation_outline item: extract one hyperedge, pattern_type ∈ seed ∪
       {new}, role ∈ allowed. Mutation MUST carry domain. Injects the Skill
       Library's extraction_hint for (domain, pattern) if a skill exists.
  verifier/critic (LLM ≠ extraction model)
    └─ per edge: verbatim_in_source? type_correct? (re-type INTERNALIZED here,
       not a separate step) relation_exists? (existence-gate internalized) →
       fix ∈ {keep, retype:X, reextract, drop}. Does NOT enter 5-outcome
       (extractor edges go candidate-evidence + schema, 断点 2).
  fixer
    └─ applies the fix list (keep / retype / reextract re-runs executor once /
       drop removes).
  skill distiller (occasional, stability > 0.6)
    └─ crystallizes a recurring (domain, pattern) extraction mode into a Skill
       via the distill_skill op (evolver write contract — proposed to the KB,
       committed by the evolver's transaction).

Then commits every kept edge to the KB as an add_edge Mutation (validate phase
only, no route — 断点 2).

Honest scope / no-downgrade:
- This does NOT throw away the verified multi-step extraction core
  (_run_hg_node_multistep: entities→types→relations). That core is the
  executor's internal implementation and is泛化-verified (4 cross-domain
  papers, GRANULAR LEAKAGE=0, commit 38f1ff24). "Redesign the extraction agent"
  = re-organize into plan-execute-verify FORM + add the missing pieces (plan,
  verify, skill, KB), NOT rewrite the verified multi-step core. re-type
  internalization = the verifier checks type correctness (retype fix), not
  deleting the type-labeling step. (DECISION-extraction-agent-form.md)
- verifier uses a DIFFERENT model than the extractor (avoid self-endorsement).
- Every edge carries domain + verbatim evidence (candidate-evidence binding).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from granular_agent.hypergraph_schema import (
    MetaHypergraph, Hyperedge, HGNode, InstanceHypergraph,
)
from granular_agent.knowledge_base import (
    KnowledgeBase, Mutation, Op, Role, Skill,
)

# The verified multi-step core lives in hypergraph_extractor; we reuse it as
# the executor's internal implementation (surgical: not rewriting a verified
# component). Imported lazily inside _execute to avoid a hard import cycle and
# keep this module importable standalone for the unit test (which mocks it).


# ---------------------------------------------------------------------------
# Plan + Verdict (I/O schemas, plan_detailed_design_supplement section 4)
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    """planner output. domain is the injection point (断点 4). central_entities
    drive ConceptGraph.get_or_create(central=True); expected_patterns +
    relation_outline give the executor schema-in-context (not blind extraction)."""
    domain: str = "global"
    central_entities: list[dict] = field(default_factory=list)
    # [{surface, type, role_hint}]
    secondary_entities: list[dict] = field(default_factory=list)
    expected_patterns: list[str] = field(default_factory=list)   # pattern_ids
    relation_outline: list[dict] = field(default_factory=list)
    # [{pattern_type, expected_nodes:[surface...], evidence_hint}]
    paper_anchor: str = ""   # paper-level anchor (title/doi for provenance)


@dataclass
class Verdict:
    """verifier output per edge. fix drives the fixer. re-type is INTERNALIZED
    here (type_correct + fix=retype:X), not a separate step."""
    edge_id: str
    verbatim_in_source: bool = True
    type_correct: bool = True
    relation_exists: bool = True
    fix: str = "keep"   # keep | retype:<pattern_id> | reextract | drop
    note: str = ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLAN_PROMPT = """You are PLANNING hypergraph extraction from ONE section of a {domain} paper.

{schema_prompt}

Section discourse role: {discourse_role}
Section text (first {cap} chars):
{text_preview}

Task: produce an EXTRACTION PLAN — what entities and relations this section likely
contains, so the executor extracts with schema-in-context (not blind).
- central_entities: the 2-6 entities this section is ABOUT (methods/phenomena/quantities
  it centers on). Each: {{"surface","type","role_hint"}}.
- secondary_entities: other entities mentioned (params, materials, regimes).
- expected_patterns: which schema pattern_ids (from above) you expect edges of here.
- relation_outline: 0-8 expected relations, each
  {{"pattern_type","expected_nodes":[surfaces],"evidence_hint":"short phrase"}}.
- domain: the paper's domain (granular / ml / molecular / molecular / global). PICK ONE.

The schema patterns above include a [boundary: ...] note for each — use it to pick
the RIGHT pattern (e.g. influences = functional dependence, NOT co-listing).

Output JSON: {{"domain":"...","central_entities":[{{"surface":"...","type":"...","role_hint":"..."}}],
"secondary_entities":[...],"expected_patterns":["..."],"relation_outline":[{{"pattern_type":"...","expected_nodes":["..."],"evidence_hint":"..."}}],
"paper_anchor":"..."}}
"""


_VERIFY_PROMPT = """You are CRITIQUING extracted hyperedges from a {domain} paper section.

{schema_prompt}

Section text (the ground truth):
{section_text}

Edges to verify (each has pattern_type, node surfaces+roles, qualifiers, evidence_span):
{edges_json}

For EACH edge judge (be strict — a wrong edge is worse than a dropped one):
- verbatim_in_source: is the evidence_span an EXACT substring of the section text?
  (paraphrase / summary / invented = false)
- type_correct: is pattern_type the RIGHT one per the schema patterns + their
  [boundary: ...] notes ABOVE? Use the boundary notes to disambiguate the
  confusable families (e.g. influences = functional dependence NOT co-listing;
  composed_of = structural parts NOT classification; defines = definitional identity
  NOT structural part-of; extends/improves/compares = method-to-method evolution NOT
  a discourse claim). If the relation is real but the pattern_type is wrong, fix=retype.
- relation_exists: does the section actually state this relation (not inferred)?
- fix: "keep" if all good; "retype:<correct_pattern_id>" if the relation is real but
  mistyped — the <correct_pattern_id> MUST be one of the schema patterns listed above
  (do NOT retype to a pattern not in the schema); "reextract" if evidence is real but
  the edge structure is wrong; "drop" if the relation is not in the text (inferred /
  hallucinated / paraphrase evidence).

Output JSON: {{"verdicts":[{{"edge_id":"...","verbatim_in_source":true,"type_correct":true,
"relation_exists":true,"fix":"keep","note":"..."}}]}}
"""


# ---------------------------------------------------------------------------
# ExtractionAgent
# ---------------------------------------------------------------------------

class ExtractionAgent:
    """plan-execute-verify + skill, writing to the KnowledgeBase kernel.

    llm_extract: the extraction model (planner + executor + fixer reextract).
    llm_verify: a DIFFERENT model for the verifier (avoid self-endorsement).
    Both are call(prompt, max_tokens)->str functions (so the unit test can mock).
    """

    def __init__(self, kb: KnowledgeBase,
                 llm_extract: Callable[[str, int], str | None],
                 llm_verify: Callable[[str, int], str | None],
                 domain_default: str = "global"):
        self.kb = kb
        self.llm_extract = llm_extract
        self.llm_verify = llm_verify
        self.domain_default = domain_default

    # ---- planner ----
    def plan(self, section_text: str, discourse_role: str,
             text_preview_cap: int = 4000) -> Plan:
        schema_prompt = self.kb.tbox.to_prompt()
        p = _PLAN_PROMPT.format(
            domain=self.domain_default, schema_prompt=schema_prompt,
            discourse_role=discourse_role, cap=text_preview_cap,
            text_preview=section_text[:text_preview_cap])
        raw = self.llm_extract(p, 4000)
        obj = _parse_json(raw) or {}
        return Plan(
            domain=obj.get("domain", self.domain_default) or self.domain_default,
            central_entities=obj.get("central_entities", []) or [],
            secondary_entities=obj.get("secondary_entities", []) or [],
            expected_patterns=obj.get("expected_patterns", []) or [],
            relation_outline=obj.get("relation_outline", []) or [],
            paper_anchor=obj.get("paper_anchor", ""))

    # ---- executor ----
    def execute(self, section_text: str, plan: Plan, node_id: str,
                predecessor_summary: str = "") -> tuple[list[HGNode], list[Hyperedge]]:
        """Run the verified multi-step core (entities→types→relations) as the
        executor's internal implementation. Injects the Skill Library hint for
        each expected pattern. Returns (nodes, edges).

        Surgical: delegates to hypergraph_extractor._run_hg_node_multistep (the
        verified core), NOT a rewrite. The plan drives schema-in-context via the
        skill hints appended to the schema prompt."""
        # skill injection: gather extraction_hints for the plan's expected patterns
        skill_hints = []
        for pid in plan.expected_patterns:
            sk = self.kb.skills.lookup(plan.domain, pid)
            if sk:
                skill_hints.append(f"- {pid}: {sk.extraction_hint}")
        skill_block = ""
        if skill_hints:
            skill_block = ("\n\nExtraction skills (crystallized from prior extractions "
                           "of this domain+pattern — follow these hints):\n"
                           + "\n".join(skill_hints))
        # build the schema prompt WITH the skill hints + plan's relation outline
        # so the executor sees schema-in-context (not blind)
        schema_prompt = self.kb.tbox.to_prompt() + skill_block
        if plan.relation_outline:
            schema_prompt += ("\n\nPlanner's expected relations (use as a guide, "
                              "extract what the text actually supports):\n"
                              + json.dumps(plan.relation_outline, ensure_ascii=False))
        # delegate to the verified core (lazy import avoids cycle + lets the
        # unit test mock _execute without importing the heavy extractor)
        return self._execute_core(section_text, plan, node_id, schema_prompt,
                                  predecessor_summary)

    def _execute_core(self, section_text: str, plan: Plan, node_id: str,
                      schema_prompt: str, predecessor_summary: str
                      ) -> tuple[list[HGNode], list[Hyperedge]]:
        """Default: delegate to hypergraph_extractor._run_hg_node_multistep
        (the verified multi-step core), constructing the sections/blocks in the
        EXACT format section_text_for_node expects: sections=[{name, block_range}],
        blocks=[{index, text}], node={id, section=<name>}. (BLOCKER B1 fix —
        the prior {id,text}+empty-blocks construction made section_text_for_node
        return "" and the real LLM extracted 0 edges; mock tests masked it.)

        Override (mock) in tests."""
        from granular_agent.hypergraph_extractor import _run_hg_node_multistep, HGBlackboard
        bb = HGBlackboard()
        if predecessor_summary:
            bb.add(node_id, predecessor_summary)
        sec_name = node_id   # section name == node id (one section per call)
        node = {"id": node_id, "section": sec_name}
        sections = [{"name": sec_name, "block_range": [0, 1]}]
        blocks = [{"index": 0, "text": section_text}]
        nodes, edges, _summary = _run_hg_node_multistep(
            node, sections, blocks, schema_prompt, bb,
            llm="deepseek", domain=plan.domain, meta=self.kb.tbox)
        return nodes, edges

    # ---- verifier ----
    def verify(self, edges: list[Hyperedge], nodes: list[HGNode],
               section_text: str, domain: str) -> list[Verdict]:
        if not edges:
            return []
        nid2surface = {n.nid: n.surface for n in nodes}
        edges_for_prompt = []
        for he in edges:
            edges_for_prompt.append({
                "edge_id": he.eid,
                "pattern_type": he.pattern_type,
                "nodes": [{"surface": nid2surface.get(nid, nid),
                           "role": r}
                          for nid, r in zip(he.node_ids, he.node_roles)],
                "qualifiers": he.qualifiers,
                "evidence_span": he.evidence_span,
            })
        p = _VERIFY_PROMPT.format(
            domain=domain, schema_prompt=self.kb.tbox.to_prompt(),
            section_text=section_text,
            edges_json=json.dumps(edges_for_prompt, ensure_ascii=False))
        raw = self.llm_verify(p, 4000)
        obj = _parse_json(raw) or {}
        verdicts = []
        by_id = {v.get("edge_id"): v for v in obj.get("verdicts", []) if isinstance(v, dict)}
        for he in edges:
            v = by_id.get(he.eid, {})
            fix = str(v.get("fix", "keep")).strip() or "keep"
            verdicts.append(Verdict(
                edge_id=he.eid,
                verbatim_in_source=bool(v.get("verbatim_in_source", True)),
                type_correct=bool(v.get("type_correct", True)),
                relation_exists=bool(v.get("relation_exists", True)),
                fix=fix,
                note=str(v.get("note", ""))))
        return verdicts

    # ---- fixer ----
    def fix(self, edges: list[Hyperedge], verdicts: list[Verdict],
            nodes: list[HGNode], section_text: str, plan: Plan
            ) -> tuple[list[Hyperedge], dict, list[dict]]:
        """Apply verdicts. keep -> keep; retype:X -> change pattern_type;
        reextract -> drop (one-shot reextract is a future refinement; for now
        reextract is treated as drop-with-note to avoid unbounded re-extraction
        loops — honest scope, not a downgrade of the keep/retype/drop path);
        drop -> remove.

        DETERMINISTIC VERBATIM POST-CHECK (铁律: structure/deterministic ->
        rule): the LLM verifier can be lenient on verbatim (real-run showed it
        kept an evidence that was NOT a source substring). So after the LLM
        verdict, we RE-CHECK verbatim_in_source as an exact substring rule: if
        the evidence_span is not a substring of section_text, force fix=drop
        (overriding keep/retype). This is the structural gate the LLM can't be
        trusted with. The LLM still judges type_correct/relation_exists (real
        semantic judgment); verbatim is a string check = rule, not LLM.

        DETERMINISTIC ROLE POST-CHECK (same 铁律): a kept edge's node_roles
        must be declared by its pattern_type's role_slots (a 'defines' edge
        carrying composed_of's roles whole/component = wrong structure). The
        real-run showed the executor sometimes emits from/to for every pattern
        regardless of the pattern's declared roles. We do NOT remap wrong roles
        to right ones (that would be a downgrade — smuggling bad structure in).
        We DROP the edge (bad role = bad structure, won't commit). This is the
        structural role gate; the LLM still judges type_correct semantically.
        The executor is expected to emit correct roles (per _STEP3_PROMPT); when
        it doesn't, the edge is dropped here rather than committed wrong.

        Returns (kept, stats, dropped_edges). dropped_edges carries the REAL
        failed edges (pattern_type + evidence + node_ids/roles + reason) so the
        evolver (Step 3/7) can feed them as schema-gap triggers — NOT a synthetic
        trigger. (MA1 fix: the evolver's failure feed was a合成 Hyperedge; real
        dropped edges are the true signal for schema evolution.)"""
        stats = {"keep": 0, "retype": 0, "reextract_as_drop": 0, "drop": 0,
                 "dropped_nonverbatim": 0, "dropped_bad_role": 0}
        kept: list[Hyperedge] = []
        dropped_edges: list[dict] = []
        vmap = {v.edge_id: v for v in verdicts}

        def _record_drop(he: Hyperedge, reason: str, v: Verdict | None):
            """Record a dropped edge for the evolver's failure feed (real
            schema-gap signal, not synthetic). Carries node SURFACES + labels
            so the evolver can build an instance for evolution_probe (P0 audit
            fix: probe instance=None made LLM see only role:nid, weakening
            proposals to 0)."""
            nid2node = {n.nid: n for n in nodes}
            node_surfaces = []
            for nid in he.node_ids:
                n = nid2node.get(nid)
                if n:
                    node_surfaces.append({"surface": n.surface, "labels": list(n.labels)})
                else:
                    node_surfaces.append({"surface": "", "labels": []})
            dropped_edges.append({
                "edge_id": he.eid,
                "pattern_type": he.pattern_type,
                "evidence_span": he.evidence_span,
                "node_ids": list(he.node_ids),
                "node_roles": list(he.node_roles),
                "node_surfaces": node_surfaces,
                "reason": reason,
                "verifier_note": (v.note if v else ""),
            })

        for he in edges:
            v = vmap.get(he.eid)
            # DETERMINISTIC verbatim gate (rule, overrides LLM verdict)
            ev = (he.evidence_span or "").strip()
            if ev and ev not in section_text:
                # evidence is paraphrase / hallucinated / truncated -> drop,
                # regardless of what the LLM verifier said.
                stats["dropped_nonverbatim"] += 1
                _record_drop(he, "non-verbatim-evidence", v)
                continue
            if v is None:
                pass  # keep
            elif v.fix == "keep":
                pass  # keep
            elif v.fix.startswith("retype:"):
                new_pt = v.fix.split(":", 1)[1].strip()
                # B3 safety: retype to a pattern NOT in the tbox is invalid —
                # the kernel would reject it at commit anyway; drop here instead
                # so it's flagged in stats (don't silently smuggle bad data).
                if new_pt not in self.kb.tbox.patterns:
                    stats["drop"] += 1
                    _record_drop(he, f"retype-to-unknown-pattern:{new_pt}", v)
                    continue
                he.pattern_type = new_pt
            elif v.fix == "reextract":
                # honest: one-shot reextract not implemented (would risk loops);
                # drop with note. The kept-edge quality bar is preserved (only
                # keep + retype survive).
                stats["reextract_as_drop"] += 1
                _record_drop(he, "verifier-reextract", v)
                continue
            else:  # drop
                stats["drop"] += 1
                _record_drop(he, f"verifier-drop:{v.fix}", v)
                continue
            # ---- passed verdict; now DETERMINISTIC role gate (rule) ----
            if not self._role_ok(he):
                stats["dropped_bad_role"] += 1
                _record_drop(he, "bad-role-not-in-pattern", v)
                continue
            kept.append(he)
            if v is None or v.fix == "keep":
                stats["keep"] += 1
            else:
                stats["retype"] += 1
        return kept, stats, dropped_edges

    def _role_ok(self, he: Hyperedge) -> bool:
        """Deterministic role gate (rule, not LLM): every role the edge uses
        must be declared by its pattern_type's role_slots. A pattern not in the
        tbox -> fail (will be caught by kernel B3 anyway, but fail here too so
        it's flagged in stats). We do NOT remap — wrong roles = wrong structure
        = drop, not commit-and-remap (that's a downgrade)."""
        pat = self.kb.tbox.patterns.get(he.pattern_type)
        if pat is None:
            return False
        declared = {s.get("role") for s in pat.role_slots}
        used = {r for r in he.node_roles}
        return used.issubset(declared)

    # ---- commit to KB (add_edge Mutation, validate only, no route 断点 2) ----
    def commit_edges(self, edges: list[Hyperedge], nodes: list[HGNode],
                     plan: Plan, paper_id: str, year: str = "",
                     section: str = "") -> tuple[int, list[dict]]:
        """Commit kept edges to the KB as add_edge Mutations. Each carries
        domain (断点 4) + concept instances inline (断点 7). extractor edges go
        through validate only — NO 5-outcome route (断点 2). Returns
        (n_committed, rejected)."""
        nid2node = {n.nid: n for n in nodes}
        # central surfaces from the plan — concepts matching these get central=True
        central_surfaces = {_norm(c.get("surface", ""))
                            for c in plan.central_entities if isinstance(c, dict)}
        mutations: list[Mutation] = []
        for he in edges:
            concepts = []
            for nid, role in zip(he.node_ids, he.node_roles):
                n = nid2node.get(nid)
                if n is None:
                    continue
                concepts.append({"surface": n.surface, "type": (n.labels[0] if n.labels else "PROPERTY"),
                                 "role": role, "evidence": n.evidence_span,
                                 "central": _norm(n.surface) in central_surfaces})
            if len(concepts) < 2:
                continue
            # provenance first-class fields (M4 fix): lift cited_from /
            # method / evidence_strength OUT of qualifiers into the provenance
            # sub-structure (design行17: Hyperedge.provenance 一等公民). The
            # remaining qualifiers carry only relation-specific business keys
            # (relation_type/dependency_type/applies_in_regime/function_form/...).
            quals = dict(he.qualifiers)
            prov_extra = {}
            for fk in ("cited_from", "method", "evidence_strength"):
                if fk in quals:
                    prov_extra[fk] = quals.pop(fk)
            mutations.append(Mutation(
                op=Op.ADD_EDGE, target=he.eid, proposer_role=Role.EXTRACTOR,
                domain=plan.domain, evidence=he.evidence_span,
                rationale=f"extracted {he.pattern_type}",
                payload={"kind": he.pattern_type,
                         "roles": list(he.node_roles),
                         "concepts": concepts,
                         "qualifiers": quals,
                         "provenance": {"paper_id": paper_id, "year": year,
                                        "section": section, **prov_extra}}))
        if not mutations:
            return 0, []
        result = self.kb.commit(mutations)
        return len(result.validated), result.rejected

    # ---- end-to-end on one section ----
    def extract_section(self, section_text: str, discourse_role: str,
                        node_id: str, paper_id: str, year: str = "",
                        section: str = "",
                        predecessor_summary: str = "") -> dict:
        """Full plan-execute-verify-fix-commit on one section. Returns a report
        dict with the plan, raw/kept edge counts, verdict stats, and commit
        result — so the caller (and the subagent reviewer) can actually SEE the
        extraction result, not just a pass/fail number."""
        plan = self.plan(section_text, discourse_role)
        nodes, edges = self.execute(section_text, plan, node_id, predecessor_summary)
        verdicts = self.verify(edges, nodes, section_text, plan.domain)
        kept, fstats, dropped_edges = self.fix(edges, verdicts, nodes, section_text, plan)
        n_committed, rejected = self.commit_edges(kept, nodes, plan, paper_id, year, section)
        return {
            "node_id": node_id, "domain": plan.domain,
            "n_central_entities": len(plan.central_entities),
            "n_expected_patterns": len(plan.expected_patterns),
            "n_relation_outline": len(plan.relation_outline),
            "n_raw_edges": len(edges),
            "n_kept_edges": len(kept),
            "fix_stats": fstats,
            "n_committed": n_committed,
            "n_rejected": len(rejected),
            "rejected": rejected,
            "plan": plan,
            # REAL dropped edges (pattern_type + evidence + reason) for the
            # evolver's failure feed — NOT a synthetic trigger. (MA1 fix)
            "dropped_edges": dropped_edges,
            "kept_edges": [{"pattern_type": he.pattern_type,
                            "roles": list(he.node_roles),
                            "evidence": he.evidence_span[:80]}
                           for he in kept],
        }


def _parse_json(raw: str | None) -> dict:
    """Best-effort JSON parse (tolerant of code fences / leading prose)."""
    if not raw:
        return {}
    from granular_agent.llm_client import parse_json_response
    return parse_json_response(raw) or {}


def _norm(s: str) -> str:
    """Normalize a surface for central-matching: NFKC + lowercase + collapse."""
    if not s:
        return ""
    import unicodedata, re
    s = unicodedata.normalize("NFKC", s.lower())
    return re.sub(r"[\s\-_]+", " ", s).strip(" .,;:()")
