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
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

from granular_agent.hypergraph_schema import (
    MetaHypergraph, Hyperedge, HGNode, InstanceHypergraph,
)
from granular_agent.knowledge_base import (
    KnowledgeBase, Mutation, Op, Role,
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
    here (type_correct + fix=retype:X), not a separate step. role-fix is also
    INTERNALIZED (role_correct + fix=rolefix:<r1,r2,...>) — the LLM verifier
    judges what role each node SHOULD have (semantic), and fixer re-points the
    roles then RE-CHECKS the role gate (rule, structural). This cures the
    bad-role coverage hole: a real edge with a wrong role name was dropped
    whole by the rule gate (12/ML_DQN); now the LLM can repair the role, the
    rule gate still blocks un-fixable bad structure (no downgrade).

    evidence_quote: the LLM verifier quotes the CONTINUOUS source-text span the
    evidence corresponds to (for locatability — every kept edge must be
    locatable in the source so the hypergraph is explainable/可溯源). The fixer
    uses it as a 2nd-chance substring (after the rule's LaTeX-normalized exact
    check fails) — if neither the evidence nor the quote is a substring of the
    source, the edge is NOT locatable → drop (LLM改写/拼接/编). This replaces
    the loose token-coverage gate (which let改写 through on word overlap)."""
    edge_id: str
    verbatim_in_source: bool = True
    type_correct: bool = True
    relation_exists: bool = True
    role_correct: bool = True
    fix: str = "keep"
    evidence_quote: str = ""   # LLM quotes the continuous source span (locatability)
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

Edges to verify (each has pattern_type + the pattern's description/boundary/allowed
roles, node surfaces+roles, qualifiers, evidence_span):
{edges_json}

For EACH edge answer THREE mandatory checks (be strict — a wrong edge is worse
than a dropped one):

CHECK 1 — EVIDENCE SUPPORT: does the evidence_span sentence actually STATE this
relation? Reject if it (a) merely describes a setup/baseline without a result,
(b) states a DIFFERENT relation than the edge claims, (c) only implies/adjacent-to
the fact. relation_exists=false → fix=drop.

CHECK 2 — SLOT BINDING: is each node really the thing its role claims, IN THIS
SENTENCE? winner really outperformed the loser; the METHOD slot really holds a
named method (not a field/benchmark); component really is a component of method.
If the binding is wrong, fix=drop (a relation with a wrong participant is a
wrong edge — do NOT keep it for the relation alone).

CHECK 3 — POLARITY/DIRECTION: does the sentence's direction match the roles?
'A comparable to B' / 'A achieves 75% of B' is NOT outperforms(A,B).
'A fails where B works' inverts winner/loser. Polarity mismatch → fix=drop
(or rolefix if ONLY the role order is swapped and the participants are right).

Also:
- verbatim_in_source: is evidence_span an EXACT substring of the section text?
- evidence_quote: quote the CONTINUOUS source span the edge's evidence
  corresponds to, VERBATIM, <=200 chars. If the evidence is paraphrased/invented
  with no continuous source span, set evidence_quote="" (the fixer substring-
  checks your quote; a quote not in the source = not locatable = dropped).
- type_correct: is pattern_type right per the pattern's own [boundary: ...]
  note given per-edge? If the relation is real but the pattern wrong →
  fix=retype:<pattern_id from the schema above>.
- An edge carrying "_competition_review" in qualifiers is a flagged pattern-
  competition case (its pattern's boundary says NOT-X yet an X-typed edge was
  extracted from the same sentence) — scrutinize it extra hard.
- fix: "keep" if ALL THREE checks pass; "retype:<id>" (id from schema);
  "rolefix:<r1>,<r2>,..." if relation+pattern right but role ORDER mislabeled
  (each MUST be in allowed_roles); "reextract" if evidence real but structure
  wrong; "drop" otherwise. Never return a failed check with fix=keep.

Output JSON: {{"verdicts":[{{"edge_id":"...","evidence_support":true,"slot_binding":true,
"polarity_ok":true,"verbatim_in_source":true,"type_correct":true,"relation_exists":true,
"role_correct":true,"evidence_quote":"the continuous source span, verbatim, <=200 chars","fix":"keep","note":"..."}}]}}
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
                 domain_default: str = "global",
                 executor_model: str = "deepseek"):
        self.kb = kb
        self.llm_extract = llm_extract
        self.llm_verify = llm_verify
        self.domain_default = domain_default
        # executor model NAME (routed by hypergraph_extractor._call). W2 fix:
        # _execute_core used to hardcode llm="deepseek" (official deepseek-chat),
        # silently bypassing the configured provider (Paratera V4-Flash) —
        # planner ran on V4-Flash while the actual edge-typing calls ran on
        # deepseek-chat. Model provenance requires the caller to pin this.
        self.executor_model = executor_model

    # ---- planner ----
    def plan(self, section_text: str, discourse_role: str,
             text_preview_cap: int = 4000) -> Plan:
        # P1-B fix: use the legacy _retrieved_schema_prompt (top-K embedding
        # retrieval + compact) instead of full to_prompt(). Schema small (<=K)
        # falls back to full to_prompt (grounding preserved); large schemas
        # retrieve top-K relevant to the preview, keeping the prompt bounded
        # so long sections don't blow the LLM context (audit: long-chunk 0-parse
        # partly caused by全schema prompt bloat). Delegates to the verified
        # legacy strategy (surgical — not rewriting what works).
        from granular_agent.hypergraph_extractor import _retrieved_schema_prompt
        schema_prompt = _retrieved_schema_prompt(self.kb.tbox,
                                                 section_text[:text_preview_cap])
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
        executor's internal implementation. Returns (nodes, edges).

        Surgical: delegates to hypergraph_extractor._run_hg_node_multistep (the
        verified core), NOT a rewrite. The plan drives schema-in-context via
        the relation outline appended to the schema prompt.

        (Skill injection REMOVED — audit: SkillLibrary was dead code, distill_skill
        never triggered (stability never reached 0.6 on single/少篇 runs), and
        injecting skill_hints would only bloat the prompt (the very thing P1-B
        fixed). No validated benefit —砍 to keep the prompt clean. Honest: not a
        selling point until multi-篇 runs prove a benefit; the mutable-memory
        selling point is carried by concept真合并 + KB transactional, not this.)"""
        # build the schema prompt WITH the plan's relation outline so the
        # executor sees schema-in-context (not blind).
        # P1-B fix: use _retrieved_schema_prompt (top-K retrieval, compact) instead
        # of full to_prompt(). Long sections + full schema blew the LLM context
        # (audit: long-chunk 0-parse). relation_outline only on short chunks
        # (it bloats prompt; plan made it from a 4k preview, misaligned with a
        # long full chunk — P1 audit: relation_outline引导弱 + 膨胀).
        from granular_agent.hypergraph_extractor import _retrieved_schema_prompt
        schema_prompt = _retrieved_schema_prompt(self.kb.tbox, section_text)
        if plan.relation_outline and len(section_text) < 6000:
            schema_prompt += ("\n\nPlanner's expected relations (use as a guide, "
                              "extract what the text actually supports):\n"
                              + json.dumps(plan.relation_outline, ensure_ascii=False))
        # delegate to the verified core (lazy import avoids cycle + lets the
        # unit test mock _execute without importing the heavy extractor)
        return self._execute_core(section_text, plan, node_id, schema_prompt,
                                  predecessor_summary)

    def _deterministic_gate(self, edges: list[Hyperedge], nodes: list[HGNode]
                            ) -> tuple[list[Hyperedge], list[dict]]:
        """Rule layer (B+ rewrite): structural/deterministic checks per edge,
        BEFORE the LLM verifier. Rules guard STRUCTURE (the LLM guards
        semantics — the established rules-vs-LLM division):

        1. binding-locality: every node SURFACE (LaTeX-normalized) must appear
           inside that edge's evidence_span. Kills cross-sentence substitution
           (probe edge 2: loser taken from a different sentence) and category-
           binding (probe edge 1: loser='reinforcement learning' while the
           evidence says 'all previous algorithms') — both would die here.
        2. role-legality: roles ⊆ pattern's declared role_slots (was in fixer;
           moved earlier so bad-role edges never reach the verifier).
        3. slot-type guard: a role_slot declaring type=METHOD rejects a node
           labeled as a field/category (killed deterministically only when the
           TYPE says so; type errors on genuinely-method surfaces go to the
           verifier).
        4. pattern-competition flag: an edge whose pattern's semantic_boundary
           says 'NOT X' while the evidence ALSO produced an X-typed edge on the
           same sentence → mark for the verifier's forced re-look (flag in
           qualifiers._competition_review; verifier sees it, not auto-drop).

        Returns (passing edges, dropped records for the evolver feed)."""
        nid2node = {n.nid: n for n in nodes}
        passing, dropped = [], []
        # competition map: normalized evidence sentence -> set of pattern types
        from collections import defaultdict
        ev_types: dict[str, set] = defaultdict(set)
        for he in edges:
            ev_types[_normalize_latex(he.evidence_span)[:400]].add(he.pattern_type)

        for he in edges:
            pat = self.kb.tbox.patterns.get(he.pattern_type)
            # --- 2. role legality (pattern must exist too) ---
            if pat is None:
                dropped.append(self._gate_drop(he, nid2node, "gate:unknown-pattern"))
                continue
            declared = {s.get("role") for s in pat.role_slots}
            if not set(he.node_roles).issubset(declared):
                dropped.append(self._gate_drop(he, nid2node, "gate:illegal-role"))
                continue
            # --- 1. binding locality ---
            ev_norm = _normalize_latex(he.evidence_span)
            bad_binding = False
            for nid, role in zip(he.node_ids, he.node_roles):
                nd = nid2node.get(nid)
                if nd is None:
                    bad_binding = True
                    break
                surf_norm = _normalize_latex(nd.surface)
                if surf_norm and surf_norm not in ev_norm:
                    bad_binding = True
                    break
            if bad_binding:
                dropped.append(self._gate_drop(he, nid2node, "gate:binding-not-local"))
                continue
            # --- 3. slot-type guard ---
            slot_types = {s.get("role"): s.get("type") for s in pat.role_slots}
            type_bad = False
            for nid, role in zip(he.node_ids, he.node_roles):
                nd = nid2node.get(nid)
                want = slot_types.get(role)
                if (nd is not None and want and want.endswith("METHOD")
                        and nd.labels and "METHOD" not in nd.labels
                        and "PARAMETER" not in nd.labels):
                    # a METHOD slot holding a non-method node (e.g. PROPERTY
                    # benchmark/field names) — the pattern wants a method here
                    type_bad = True
                    break
            if type_bad:
                dropped.append(self._gate_drop(he, nid2node, "gate:slot-type-mismatch"))
                continue
            # --- 4. pattern competition flag (verifier sees it, no drop) ---
            key = _normalize_latex(he.evidence_span)[:400]
            others = ev_types.get(key, set()) - {he.pattern_type}
            if pat.semantic_boundary and others:
                import re as _re
                m = _re.search(r"NOT\s+([a-z_]+)", pat.semantic_boundary)
                if m and m.group(1) in others:
                    he.qualifiers = dict(he.qualifiers) if he.qualifiers else {}
                    he.qualifiers["_competition_review"] = ",".join(sorted(others))
            passing.append(he)
        return passing, dropped

    def _gate_drop(self, he: Hyperedge, nid2node: dict, reason: str) -> dict:
        """Dropped-edge record in the fixer's _record_drop format (the evolver
        expects node_surfaces as [{surface, labels}] dicts — P0 audit fix)."""
        node_surfaces = []
        for nid in he.node_ids:
            n = nid2node.get(nid)
            if n:
                node_surfaces.append({"surface": n.surface, "labels": list(n.labels)})
            else:
                node_surfaces.append({"surface": "", "labels": []})
        return {
            "edge_id": he.eid,
            "pattern_type": he.pattern_type,
            "evidence_span": he.evidence_span,
            "node_ids": list(he.node_ids),
            "node_roles": list(he.node_roles),
            "node_surfaces": node_surfaces,
            "reason": reason,
            "verifier_note": "",
        }

    def _execute_core(self, section_text: str, plan: Plan, node_id: str,
                      schema_prompt: str, predecessor_summary: str
                      ) -> tuple[list[HGNode], list[Hyperedge]]:
        """Default: delegate to hypergraph_extractor._run_hg_node_joint
        (the B+ rewrite: one call per chunk, entities+types+edges together so
        binding happens in sentence context), constructing the sections/blocks
        in the EXACT format section_text_for_node expects:
        sections=[{name, block_range}], blocks=[{index, text}],
        node={id, section=<name>}. (BLOCKER B1 fix retained.)

        Override (mock) in tests."""
        from granular_agent.hypergraph_extractor import _run_hg_node_joint, HGBlackboard
        bb = HGBlackboard()
        if predecessor_summary:
            bb.add(node_id, predecessor_summary)
        sec_name = node_id   # section name == node id (one section per call)
        node = {"id": node_id, "section": sec_name}
        sections = [{"name": sec_name, "block_range": [0, 1]}]
        blocks = [{"index": 0, "text": section_text}]
        nodes, edges, _summary = _run_hg_node_joint(
            node, sections, blocks, schema_prompt, bb,
            llm=self.executor_model, domain=plan.domain, meta=self.kb.tbox)
        return nodes, edges

    # ---- verifier ----
    def verify(self, edges: list[Hyperedge], nodes: list[HGNode],
               section_text: str, domain: str) -> list[Verdict]:
        if not edges:
            return []
        nid2surface = {n.nid: n.surface for n in nodes}
        edges_for_prompt = []
        for he in edges:
            pat = self.kb.tbox.patterns.get(he.pattern_type)
            # give the verifier the pattern's DECLARED role_slots so it can judge
            # role correctness by direct对照 (not guessing from a long schema
            # prompt). Without this, the verifier judged role_correct=true on
            # bad-role edges (e.g. from/to on 'influences' which declares
            # source/target/cause/effect) — 28 edges误判 then dropped by the
            # rule gate. The allowed_roles list is deterministic (from schema),
            # presented per-edge so the LLM can compare edge.roles vs allowed.
            allowed_roles = [s.get("role") for s in pat.role_slots] if pat else []
            # per-edge pattern definition + boundary (B+ rewrite: the three
            # mandatory checks judge against THIS pattern's semantics, not a
            # generic sense of the type name)
            edges_for_prompt.append({
                "edge_id": he.eid,
                "pattern_type": he.pattern_type,
                "pattern_desc": (pat.description[:200] if pat else ""),
                "pattern_boundary": (pat.semantic_boundary[:300] if pat else ""),
                "allowed_roles": allowed_roles,   # this pattern's declared roles
                "nodes": [{"surface": nid2surface.get(nid, nid),
                           "role": r}
                          for nid, r in zip(he.node_ids, he.node_roles)],
                "qualifiers": {k: v for k, v in he.qualifiers.items()
                               if not k.startswith("_")} if he.qualifiers else {},
                "competition_review": he.qualifiers.get("_competition_review", "")
                                      if he.qualifiers else "",
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
                role_correct=bool(v.get("role_correct", True)),
                fix=fix,
                evidence_quote=str(v.get("evidence_quote", "")),
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
        stats = {"keep": 0, "retype": 0, "rolefix": 0, "reextract_as_drop": 0,
                 "drop": 0, "dropped_nonlocatable": 0, "dropped_bad_role": 0}
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
                # verdict摘要 — 存进bundle, 不重跑就能查verifier怎么判的(rolefix根因排查用)
                "verifier_role_correct": (v.role_correct if v else None),
                "verifier_type_correct": (v.type_correct if v else None),
                "verifier_fix": (v.fix if v else ""),
            })

        for he in edges:
            v = vmap.get(he.eid)
            # DETERMINISTIC verbatim gate (rule, overrides LLM verdict).
            # P1-A fix: dual gate — (A) LaTeX-normalized substring (rescues真边
            # where the LLM mildly normalized LaTeX: c→γ, $...$ stripped, braces
            # removed, whitespace folded) + (B) content-token coverage (>=70% of
            # evidence实词 must be in the section, AND >=2 of the first 3 must hit)
            # to block LLM hallucination/paraphrase that fuzzy alone would pass.
            # Honest: thresholds are经验值 — must be validated against gold真边
            # (rescue rate) + known脏边 (false-pass rate) before trusting; the
            # gate logs verdict so it's auditable. Not a downgrade: bad-role edges
            # still dropped by _role_ok; only verbatim is relaxed for LaTeX.
            # LOCATABILITY gate (replaces the loose token-coverage gate).
            # Every kept edge must be LOCATABLE in the source (可溯源 → explainable
            # hypergraph). Two-stage:
            # (A) RULE (structural/deterministic): LaTeX-normalized substring of
            #     the evidence in the section. If命中 → locatable, record offset.
            # (B) LLM (semantic): if (A) misses, the verifier quoted the source
            #     span (evidence_quote). Substring-check the QUOTE — if the quote
            #     is actually in the source, the edge is locatable (the LLM found
            #     the real span the evidence points to, e.g. a LaTeX-normalized
            #     version). If neither the evidence nor the quote is a source
            #     substring → NOT locatable → drop (LLM改写/拼接/编).
            # This cures the token-coverage hole (it let改写 through on word
            # overlap: 'hippocampus may support the physical realization...' —
            # words matched but content was invented). Locatability is structural
            # (a real span exists) + LLM (finds it), not word-counting.
            ev = (he.evidence_span or "").strip()
            loc_offset = self._locate_evidence(ev, v, section_text)
            if ev and loc_offset is None:
                stats["dropped_nonlocatable"] = stats.get("dropped_nonlocatable", 0) + 1
                _record_drop(he, "evidence-not-locatable", v)
                continue
            # record the source offset on the edge for可溯源 (kept edges carry
            # where in the source their evidence lives).
            if loc_offset is not None:
                he.qualifiers = dict(he.qualifiers) if he.qualifiers else {}
                he.qualifiers["_evidence_offset"] = str(loc_offset)
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
            elif v.fix.startswith("rolefix:"):
                # role-fix (bad-role cure): the LLM verifier judged the relation +
                # pattern are right but the ROLES are mislabeled, and gave the
                # corrected role per node. Re-point the roles, then the role gate
                # (rule) RE-CHECKS them — if the LLM's corrected roles are in the
                # pattern's declared role_slots, the edge survives (cures the
                # coverage hole where a real edge was dropped whole for a wrong role
                # name). If the LLM's roles are STILL not in the pattern (it
                # hallucinated a role not in the pattern), drop (rule守 structure,
                # no downgrade). Corrected roles MUST be declared by the pattern.
                roles_str = v.fix.split(":", 1)[1].strip()
                new_roles = [r.strip() for r in roles_str.split(",") if r.strip()]
                pat = self.kb.tbox.patterns.get(he.pattern_type)
                declared = {s.get("role") for s in pat.role_slots} if pat else set()
                if (pat and len(new_roles) == len(he.node_roles)
                        and all(r in declared for r in new_roles)):
                    he.node_roles = new_roles
                else:
                    # LLM gave wrong count / unknown role -> can't fix, drop
                    stats["drop"] += 1
                    _record_drop(he, f"rolefix-invalid:{roles_str}", v)
                    continue
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
            # if rolefix was applied above, the roles are the LLM-corrected ones;
            # the rule gate RE-CHECKS them. If they pass -> keep (cured). If the
            # executor's roles were wrong AND the verifier didn't issue a rolefix
            # (missed it), the rule gate drops the edge whole — record as
            # "verifier-missed-bad-role" so it's auditable (a verifier-quality
            # signal, not a gate-too-strict). This is honest: the rule gate
            # can't invent the right role (that's semantic = LLM job); if the
            # LLM didn't fix it, the edge can't be committed with bad structure.
            if not self._role_ok(he):
                stats["dropped_bad_role"] += 1
                if v and v.fix.startswith("rolefix:"):
                    reason = "rolefix-still-bad"  # LLM tried but roles not in pattern
                elif v and not v.role_correct:
                    reason = "bad-role-verifier-flagged-unfixed"
                else:
                    reason = "verifier-missed-bad-role"
                _record_drop(he, reason, v)
                continue
            kept.append(he)
            if v is None or v.fix == "keep":
                stats["keep"] += 1
            elif v.fix.startswith("retype:"):
                stats["retype"] += 1
            elif v.fix.startswith("rolefix:"):
                stats["rolefix"] += 1  # already counted above, but ensure
            else:
                stats["keep"] += 1
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

    def _locate_evidence(self, ev: str, v: "Verdict | None", section_text: str):
        """Locatability gate (replaces the loose token-coverage _verbatim_ok).
        Returns the source offset of the evidence, or None if NOT locatable.

        (A) RULE (structural): LaTeX-normalized substring of the evidence in the
            section.命中 → return the offset in the ORIGINAL section (map back
            from normalized). This rescues真边 where the LLM mildly normalized
            LaTeX (c→γ, $...$ stripped) — the evidence IS in the source, just
            surface-different.
        (B) LLM (semantic): if (A) misses, the verifier quoted the source span
            (evidence_quote). Substring-check the QUOTE (LaTeX-normalized) — if
            it's actually in the source, the edge is locatable (the LLM found
            the real span). Return its offset. If the quote isn't in the source
            either, the LLM didn't find a real span → NOT locatable → None.

        This cures the token-coverage hole (let改写 through on word overlap —
        'hippocampus may support the physical realization...' words matched but
        content was invented). Locatability is structural (a real span exists) +
        LLM (finds it), not word-counting. Every kept edge ends up with a
        source offset → the hypergraph is可溯源/explainable."""
        if not ev:
            return None
        n_sec = _normalize_latex(section_text)
        n_ev = _normalize_latex(ev)
        # (A) rule: full normalized substring (strict — whole evidence is in source)
        if n_ev and n_ev in n_sec:
            return self._raw_offset(ev, section_text)
        # (A2) rule: a long CONTIGUOUS prefix of the evidence is in the source
        # (>=40 normalized chars). This rescues真边 where the LLM truncated at
        # a footnote/citation marker (network16 → evidence stopped before "16",
        # so the full sentence isn't an exact substring, but its prefix IS).
        # The prefix must be CONTIGUOUS and long (40 chars) — a改写's first few
        # words might match, but 40 contiguous chars won't (改写 diverges fast),
        # so this stays strict (not token-coverage-loose). Records the prefix's
        # offset → still可溯源 (the prefix IS a real source span).
        if n_ev and len(n_ev) >= 40:
            prefix = n_ev[:60]
            if prefix in n_sec:
                return self._raw_offset(ev[:60], section_text)
        # (B) LLM quote: the verifier quoted the real source span (LaTeX-normalized)
        if v and v.evidence_quote:
            n_q = _normalize_latex(v.evidence_quote)
            if n_q and n_q in n_sec:
                return self._raw_offset(v.evidence_quote, section_text)
        return None

    def _raw_offset(self, needle: str, haystack: str) -> int:
        """Find an approximate raw offset of needle in haystack (for溯源).
        Tries exact, then whitespace-folded lowercase. Returns -1 if not found
        (shouldn't happen — caller already confirmed normalized substring)."""
        if not needle:
            return -1
        pos = haystack.find(needle)
        if pos >= 0:
            return pos
        # whitespace-folded lowercase (matches the normalized substring)
        h = re.sub(r"\s+", " ", haystack).lower()
        n = re.sub(r"\s+", " ", needle).lower()
        pos = h.find(n)
        if pos < 0:
            return -1
        # approximate: count raw chars up to this folded position
        return pos

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
        # deterministic rule gate (B+ rewrite): reject structurally-invalid
        # edges BEFORE the LLM verifier (cheap first, and the rule layer is
        # where binding-locality actually lives — a prompt can only ask)
        edges, gate_dropped = self._deterministic_gate(edges, nodes)
        verdicts = self.verify(edges, nodes, section_text, plan.domain)
        kept, fstats, dropped_edges = self.fix(edges, verdicts, nodes, section_text, plan)
        dropped_edges = gate_dropped + dropped_edges
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


# LaTeX-normalization for the verbatim gate (P1-A fix). Symmetric: applied to
# BOTH the evidence_span and the section_text so a mildly-normalized evidence
# (LLM stripped $...$, mapped \gamma→γ, folded whitespace) still matches the
# raw section. NOT a content change — only LaTeX/whitespace/unicode surface
# normalization. Hallucination is caught by the content-token coverage gate.
_LATEX_ENV = re.compile(r"\$+|\\\(|\\\)|\\\[|\\\]")
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\s*")
_BRACE = re.compile(r"[{}]")
_WS = re.compile(r"\s+")
_GREEK_MAP = {
    "gamma": "γ", "alpha": "α", "beta": "β", "delta": "δ", "epsilon": "ε",
    "theta": "θ", "lambda": "λ", "mu": "μ", "nu": "ν", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "phi": "φ", "psi": "ψ", "omega": "ω",
    "Delta": "Δ", "Sigma": "Σ", "Pi": "Π", "Theta": "Θ", "Lambda": "Λ",
    "nabla": "∇", "partial": "∂", "infty": "∞", "leq": "≤", "geq": "≥",
    "times": "×", "cdot": "·", "sum": "∑", "prod": "∏",
}


def _normalize_latex(s: str) -> str:
    """Normalize LaTeX surface for verbatim substring matching: NFKC unicode +
    Greek command map (\\gamma→γ) + strip math env ($ \\( \\) \\[ \\]) + strip
    bare commands (\\frac \\text) + strip braces + fold whitespace + lowercase.
    Symmetric on evidence and section. (P1-A fix; honest: rare symbols not in
    _GREEK_MAP stay as the bare letter after _LATEX_CMD strips the backslash-
    command — acceptable; extend the map if real edges miss.)"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for name, g in _GREEK_MAP.items():
        s = re.sub(rf"\\{name}\b", g, s)
    s = _LATEX_ENV.sub(" ", s)
    s = _LATEX_CMD.sub(" ", s)
    s = _BRACE.sub("", s)
    s = _WS.sub(" ", s).strip()
    return s.lower()
