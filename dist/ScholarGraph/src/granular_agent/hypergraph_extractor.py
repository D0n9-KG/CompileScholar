"""Phase 1 (hypergraph): chained extraction producing hyperedges, with the
deep self-evolution closed loop wired in.

Mirrors chained_extractor.py's DAG/blackboard topology but emits hyperedges
(not flat atoms). Each extracted hyperedge is validated against the
meta-hypergraph; structural mismatches feed
hypergraph_evolution.run_evolution_loop, which mutates the meta-hypergraph
in place. Downstream DAG nodes then re-fetch meta.to_prompt() — this is
P4 forward propagation (the schema a later node sees reflects evolution
that happened at an earlier node).

This is the ENTRY of the closed loop: without real hyperedges there is
nothing to validate, so the trigger never fires.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from granular_agent.llm_client import call_llm, call_paratera, call_cst, _is_cst_model, parse_json_response
from granular_agent.structure_mapper import topo_order, section_text_for_node
from granular_agent.hypergraph_schema import (
    MetaHypergraph, Hyperedge, HGNode, InstanceHypergraph,
)
from granular_agent.hypergraph_evolution import EvolutionTrigger, run_evolution_loop

SECTION_TEXT_CAP = None  # no truncation — DAG splits full text into sections;
# each node handles its own section's COMPLETE text (the core DAG卖点, see
# structure_mapper). An 8000-char cap here previously re-broke full-text
# coverage on long Method/Results sections — removed.


def _norm_surface(s: str) -> str:
    """Normalize a node surface for cross-section dedup: NFKC, lowercase,
    collapse whitespace/punctuation. "Granular Materials" == "granular
    materials" == "granular  materials"."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s.lower())
    s = re.sub(r"[\s\-_]+", " ", s).strip(" .,;:()")
    return s


EXTRACT_HG_PROMPT = """You are extracting a knowledge HYPERGRAPH from ONE section of a {domain} paper.

{schema_prompt}

This section's discourse role is: {discourse_role}.

Extract:
1. NODES: the entities/concepts/methods/quantities/results mentioned in this section. Each node has >=1 label from the schema's node types above, a surface (the mention text), and a verbatim evidence_span copied exactly from the section.
2. HYPEREDGES: the N-ARY relations connecting those nodes. A hyperedge connects N nodes (N>=2, and N>=3 whenever the relation is genuinely n-ary — see below). Each hyperedge has a pattern_type, node_ids (the nodes it connects, in order), node_roles, qualifiers, and a verbatim evidence_span.

EXTRACT DIVERSE RELATION TYPES (critical — do NOT default everything to "influences").
A section usually expresses several DISTINCT kinds of relations; capture each with the
matching pattern_type from the schema. The common families:
  * definition   (defines) — "X is defined as Y", "we refer to Z as ...", "X denotes ...",
    "X is among the most ...", "X is a type of Y", "X is classified into A, B, C" (classification = defines, NOT composed_of).
    "X can be viewed as Y" or "X is analogous to Y" is a CLAIM (analogy), NOT defines.
  * composition  (composed_of) — "X consists of Y and Z", "X is composed of ...",
    "the model is A + B + C" — ONLY structural/constitutive composition (parts that make up a whole).
    "X is divided into A, B, C" or "X includes types A, B, C" is classification → use defines, NOT composed_of.
    "X has control parameters W, Q" is NOT composition (control params are not structural parts) → use influences.
    "we test X by doing experiments" is NOT composition (testing is not structural) → use measures or claim_relation.
  * dependency   (influences) — "X affects/depends on/scales with Y", "X improves Y"
    (a genuine causal/functional dependence, NOT a mere co-listing or classification — those are composed_of / defines)
  * measure      (uses_method / measures) — "we use method M to evaluate X", "M is applied to X", "X was measured by M"
  * claim        (reports / claim_relation) — "we find that ...", "results show ...", "X outperforms Y", "X is associated with Y"
  * constitutive_law — a quantitative/formal law "output = f(input1, input2, ...)" (use when a formula or formal relation is stated)
  * evolution    (extends / improves / compares / replaces / adapts / background) — a
    METHOD→METHOD development relation: how one method/theory builds on another. This is
    ONE of a method's relation types (alongside uses_parameter/captures/composed_of), not
    a special category. Extract BOTH what the text STATES and what it IMPLIES (you read
    the source text; a relation that is expressed indirectly still counts as extraction,
    not inference). Examples:
    EXPLICIT: "X extends/generalizes Y", "X improves Y's accuracy", "X compared with Y".
    IMPLIED (still extract — the text points to it, just not with the verb directly):
      "X resolves/overcomes Y's limitation" or "X provides the solution to [issues
      that Y has]" → improves (X METHOD, Y METHOD)
      "X is built on / inspired by / based on Y", "we extend Y to ..." → extends
      "X deviates from / as an alternative to Y" → compares
      "X replaces the [Y] approach" → replaces
      "X adapts Y to [new scenario]" → adapts
      "drawing on Y as background / motivated by Y" → background
    Do NOT invent a relation that the text gives NO evidence for at all (not even
    implied) — THAT is inference (a downstream agent's job). But if the text's
    wording points to an evolution relation (even implicitly), extract it with the
    matching type and the verbatim wording as evidence_span. Connect the METHODS
    (n-ary if >2 are compared); pattern_type = the verb (extends/improves/compares/
    replaces/adapts/background). "X uses Y as background/motivation" → background.
    DECISION PATHS (distinguish extends/improves/compares — the easy-to-confuse ones;
    the principle is domain-general, the named examples are illustrative only):
      path1 - does B fail/have a limitation in some regime/situation, and A handles it?
              → improves (e.g. a local model that fails near a regime boundary, and a
              nonlocal/higher-order model that resolves it → improves)
      path1b - does A derive a more accurate parameter/coefficient from first principles,
              improving B's empirical/phenomenological parameter? → improves
              (first-principles vs phenomenological = improves, NOT compares)
      path2 - is A a direct extension/generalization of B (adds a term/gradient/new
              parameter, or generalizes B to a new regime)? → extends
              (a higher-order/nonlocal extension of a base model → extends)
      path3 - do A and B have DIFFERENT modeling forms/origins but overlapping scope
              (both model the same phenomenon, different mechanisms, each pros/cons)?
              → compares (two parallel mechanisms for the same phenomenon, differing in
              what they assume)
              (ONLY compares if A,B are PARALLEL different-mechanism; if A improves B's
              accuracy/scope → improves NOT compares)
      path4 - no modeling-form/scope relation at all → do NOT extract (not evolution).
    When unsure between improves/extends: improves = solves a limitation; extends = generalizes scope.
METHOD-PARAMETER N-ARY EDGES (critical for rich topology): when a section discusses
a method/model AND the parameters/quantities it uses, connect them in ONE n-ary hyperedge.
E.g. "the model M uses parameters p1 and p2" → ONE arity-3 edge:
[M(METHOD) — uses → p1(PARAMETER), p2(PARAMETER)].
MANDATORY for named laws: a constitutive_law edge whose law is a NAMED MODEL (a rheology,
a kinetics, a neural architecture, a scaling law, etc.) MUST include that METHOD node as
a member of the hyperedge alongside its parameters — NOT a parameters-only edge. E.g. a law
"y = f(x1, x2, x3)" belonging to a named model M is ONE edge:
[M(METHOD) — defines → y(PARAM), x1(PARAM), x2(PARAM), x3(PARAM)].
Do NOT split method and its parameters into separate edges — the method→parameter
connection is what the rich topology needs. An all-PARAMETER constitutive_law edge with no
METHOD node is WRONG (it severs the law from the method that owns it).
Pick the pattern_type that matches the RELATION SEMANTICS, not the surface verb. A paper's
core contribution usually appears as a DEFINITION (what they propose), a COMPOSITION (what
it's made of), and CLAIMS (what they show) — extract all three, not just "influences".
Self-check before output: if >70% of your edges are the SAME pattern_type (especially all
"influences"), you are collapsing distinct relations — re-read and re-classify at least a
third of them into definition / composed_of / measures / claim_relation as the semantics
warrant. A sentence like "X includes A, B, and C" is composed_of, NOT influences. A sentence
like "X is among the most heritable Y" is defines, NOT influences.

CORE-ENTITY SHARING (critical for schema topology): papers have a few CENTRAL entities
(the proposed method/model/concept, the main metric, the key dataset) that RECUR across many
relations. These central entities MUST be emitted ONCE (reused by nid) and connected to
MULTIPLE hyperedges of DIFFERENT pattern_types. E.g. if "Informer" is the proposed model:
  - defines: Informer is defined as a transformer-based forecasting model
  - composed_of: Informer is composed of ProbSparse attention + distilling + decoder
  - reports: experiments show Informer outperforms DeepAR on MSE
  -> "Informer" is ONE node shared across 3 hyperedges (different pattern_types).
Do NOT invent a new node per relation; reuse the central entity so the schema can link
relations through it. This is what makes the schema a connected graph, not isolated edges.

Rules:
- node_roles = the FUNCTIONAL ROLE of each node IN the relation, NOT the node's name/entity. A role describes the node's position in this relation. Use ONLY these role kinds (pick the best fit; pluralize/repeat for multiple same-role nodes):
    * output / input — for laws/dependencies (what's computed vs what feeds it)
    * cause / effect — for causal relations
    * subject / object — for general relations
    * from / to — for directional/claim relations
    * whole / component — for compositional/part-of relations
    * source / target — for influence/flow relations
    * instrument / object — for measurement (what measures vs what's measured)
    * exponent / parameter / coefficient — for numeric roles in equations
  Do NOT use entity names or concept names as roles. The role of a method M in "we use M to evaluate X" is "instrument", NOT "M" or the method's name. Using an entity name as a role is a category error and breaks schema reuse.
  Keep roles from this small set; reuse the SAME role name across hyperedges of the same pattern (e.g. all influences edges use source/target). This lets the schema recognize the same relation structure across papers.
- evidence_span MUST be a verbatim phrase copied from the section (exact string, including symbols/units). Paraphrasing causes rejection.
- Use existing patterns when they fit. Only propose a NEW pattern_type when the section expresses a relation none of the existing patterns can hold (the system will validate it).
- A node can carry multiple labels (e.g. a metric that is both a RESULT and a numeric quantity).
- HYPERGRAPH = N-ARY (the core point, do NOT degrade to binary):
    * A constitutive_law / governing relation MUST be ONE hyperedge connecting its output PLUS every input it depends on PLUS every named constant/parameter. E.g. "loss = -sum(y log p)" -> ONE arity-3 edge: [loss(output) <- y(input) <- p(input)]. "accuracy improves with both dataset size and model capacity" -> ONE arity-3 edge [accuracy <- dataset_size <- model_capacity].
    * If a relation has 1 output and >=2 inputs, the hyperedge MUST connect all of them (arity = 1 + n_inputs + n_params).
    * Only use arity 2 for genuinely binary relations (A depends on B alone, nothing else).
- COORDINATED ENTITIES MUST SHARE ONE EDGE (the most common arity-loss bug): when
  a sentence lists PARALLEL entities all in the same relation to the same
  other entity, they MUST all be nodes of ONE hyperedge, NOT split into
  separate binary edges and NOT dropped. Examples:
    "we compare against DeepAR, ARIMA, and Prophet" ->
       ONE arity-4 edge [DeepAR, ARIMA, Prophet] <- (baseline comparison), arity 4.
    "the model is composed of an encoder, a decoder, and a distilling layer" ->
       ONE arity-4 edge model <- [encoder, decoder, distilling_layer], arity 4.
  Do NOT extract only the first listed entity and drop the rest — that loses
  information and orphans the dropped nodes. If the schema's pattern has a
  repeatable input/component role, use it; if not, propose a new pattern that
  does (the system will validate variadic patterns).
- NUMERIC NODES (REQUIRED): every number, constant, coefficient, exponent, and measured
  value in the section MUST be emitted as a NUMERIC node (with properties.value set).
  Parameters of laws/formulas (learning rates, exponents, thresholds, metric values)
  MUST be nodes AND wired into the relevant hyperedge. A section stating quantitative
  results with zero NUMERIC nodes is WRONG.
- NODE LABELING (critical for rich topology — do NOT default everything to PROPERTY):
    * METHOD — ONLY a named scientific modeling approach/theory/law/model that the paper
      PROPOSES or REVIEWS as a model of some phenomenon (a constitutive relation / rheology /
      continuum theory / neural architecture / kinetic scheme / estimator, etc.).
      METHOD is for the MODEL itself, not the procedure of using it.
      NOT a research group/consortium name → PROPERTY.
      NOT a measurement tool/procedure: "numerical simulation", "experiments", "finite
      difference", "particle-image velocimetry", "MRI", "X-ray tomography", "PIV",
      "spectrometry", "assay", "benchmark" → these are TOOLS/PROCEDURES, label as
      PROPERTY (they are not models). Only a specific named model counts as METHOD.
      NOT an experimental setup/geometry/configuration: "plane shear", "annular shear",
      "rotating drum", "silo", "chute", "hopper", "petri dish", "reactor", "test set" →
      these are SETUPS, label as PROPERTY or REGIME (the regime part), never METHOD.
      NOT a generic phrase: "constitutive equations", "unified framework", "future model",
      "a model for X", "the equations" → do NOT emit as a METHOD node at all; if a real
      named model is meant, emit THAT named model instead.
      LAW-NAME DISAMBIGUATION (principle, any domain): a model named by a formula or proper
      noun — "μ(I) rheology", "local rheology", "inertial rheology", "Arrhenius kinetics",
      "Transformer" — is a METHOD (it names a model). A bare symbol that model CONTAINS
      ("μ", "μ_s", "E_a") is a PARAMETER. When you see a law-name, treat it as the METHOD
      (the model), NOT as its bare parameter. A paper's central model MUST be labeled
      METHOD — if it is not, the extraction is WRONG.
      If unsure whether something is a METHOD vs PROPERTY: it is METHOD only if it names a
      mathematical/scientific MODEL of behavior (constitutive relation, flow rule, scaling
      law, network architecture, kinetic scheme, estimator).
      METHOD NAMING (critical for cross-paper alignment — vague names break it): emit the
      CANONICAL name a field uses for the method, NOT a descriptive phrase or sentence.
      GOOD (illustrative, not rules): "μ(I) rheology", "kinetic theory", "Transformer",
      "BERT", "Arrhenius kinetics", "maximum-likelihood phylogenetics".
      BAD (descriptive phrases that won't align across papers): "nonlocal model",
      "a better architecture", "a model for dense flows", "the theory".
      If the paper gives a named method (acronym or proper noun), use that name. If it only
      describes, emit the SHORTEST noun phrase that names the method (not a full clause).
    * PARAMETER — named quantities/constants/coefficients that appear in equations
      (e.g. μ, I, d, P, τ, learning rate, rate constant k, activation energy E_a). Distinguish
      from generic PROPERTY: a PARAMETER is a specific named symbol/constant in a law/model,
      while PROPERTY is a generic quantity (stress, velocity, loss, accuracy).
    * PHENOMENON — an effect/behavior that methods aim to capture or fail in
      (e.g. nonlocal creep, secondary rheology, segregation, clogging, overfitting,
      allosteric regulation). Not a method, not a parameter.
    * REGIME — an operational regime/condition (quasi-static, dense, gaseous/collisional for
      physics; training/inference for ML; aerobic/anaerobic for bio).
    * MATERIAL — a material or substance (a granular material, a chemical reagent, a cell line).
    * NUMERIC — a numeric value.
    * PROPERTY — ONLY use for generic quantities not fitting the above.
  A node can carry MULTIPLE labels (e.g. a method that is also a parameter: ["METHOD","PARAMETER"]).
  This labeling is what lets the schema link method↔parameter↔phenomenon relations directly
  from hyperedge structure — without it, all entities collapse to PROPERTY and the rich
  topology is lost.
- NO ORPHAN NODES: every node you emit MUST participate in >=1 hyperedge. If you would emit a node that no edge connects, either (a) find the edge it belongs to and add it, or (b) do NOT emit that node. Dangling mentions are noise.
- NO DUPLICATE NODES: before emitting a node, check if an existing node has the SAME surface (case/punctuation-insensitive). If so, reuse its nid; do NOT create a second node for "attention mechanism" when "Attention Mechanism" exists. This is ESPECIALLY important for central entities — they must be one node, reused across all their relations.
- Equations/laws: when the section states a quantitative or formal relation (output computed from inputs + parameters), emit ONE n-ary hyperedge wiring the output + every input + every named constant/parameter as nodes. Carry the equation text in a qualifier if a function-form key is among the pattern's allowed_qualifiers. Pick the pattern_type from the schema's existing patterns (shown above); if the schema lacks a fitting pattern, use the relation's natural name as pattern_type (the system will validate + evolve the schema to accommodate it).
- Be exhaustive but wired: extract every distinct entity and relation, and ensure every node is connected. A section typically yields 8-20 nodes and 5-12 hyperedges, with most hyperedges arity>=3.
- node_ids must reference node nid values you defined in THIS output.
- CONTROLLED-ENUM relation kind (REQUIRED for any dependency/relates pattern): the
  `dependency_type` (or `relation_kind`) qualifier MUST take one of these enum
  values, picked by the SEMANTICS of the relation (not the surface verb):
    * "monotonic"   — one quantity varies monotonically with another (increases/decreases with, scales as, ratio)
    * "derivation"  — one relation is derived/follows/depends on another (derived from, follows from, depends on)
    * "analogy"     — one relation is proposed as analogous/representative of another (possible representation, corresponds to)
    * "composition" — one relation is composed of / accounts for / is a measure of another (divided by, accounts for, is a measure of)
  Do NOT write free-text values like "increases with" or "possible representation" — write the enum value. This constraint is what lets the schema-refinement loop detect over-wide patterns and split them.
- applies_in_regime (when the pattern declares it as an allowed qualifier):
  carry a SHORT tag for the operational regime/condition the relation holds under (e.g. dense/quasi-static/inertial/flow for physics, or train/test/online for ML, or any short domain-appropriate tag). If the section does not specify, use "unknown".
- cited_from provenance (REQUIRED): every hyperedge carries cited_from, one of:
    * "this_work"   — the relation is asserted by THIS paper's own experiments/analysis
    * "prior_art"   — the relation is reported as another's result being cited/built on (the evidence span will name the cited work or use citation markers)
    * "definition"  — the relation is a definitional identity (e.g. "X is defined as Y")
  This distinguishes the paper's own claims from work it cites (a missing
  provenance qualifier causes silent mis-attribution).
- method (REQUIRED when the source specifies it): how the relation was established, one of:
    * "experiment"  — measured in an experiment / apparatus
    * "simulation"  — from a numerical simulation
    * "theory"      — derived theoretically / analytically
    * "review"      — surveyed / asserted in a review without derivation
  Pick by how THIS paper establishes the relation, not by the field. Omit only if the section genuinely does not say.
- evidence_strength (REQUIRED): the epistemic status of the relation, one of:
    * "measured"     — directly measured / observed
    * "derived"      — derived from other quantities / computed
    * "hypothesized"  — proposed as a hypothesis / assumption
    * "assumed"       — taken as a modeling assumption
  This + cited_from + method together give the hyper-relational provenance (P-E2 attribution).
- qualifier keys are FIXED: only use keys from {{condition, method, evidence_strength,
  cited_from, applies_in_regime, dependency_type, relation_type, function_form,
  parameters}}. An ad-hoc key is rejected. Do not invent qualifier names. For enum
  keys (method/evidence_strength/dependency_type/applies_in_regime/cited_from) the
  VALUE must be exactly one of the enum values listed above — free-text values like
  "seminar discussion" or "qualitative" are rejected.

{predecessor_context}

SECTION TEXT ({section_name}):
{section_text}

Output a JSON object (note: hyperedges are N-ARY — connect >=3 nodes when the relation involves multiple inputs/dependents/components):
{{"nodes":[{{"nid":"n1","labels":["METHOD"],"surface":"Informer","evidence_span":"we propose Informer for long-sequence forecasting","properties":{{}}}},
          {{"nid":"n2","labels":["METHOD"],"surface":"ProbSparse self-attention","evidence_span":"ProbSparse self-attention mechanism","properties":{{}}}},
          {{"nid":"n3","labels":["RESULT"],"surface":"MSE","evidence_span":"MSE decrease of 26.8%","properties":{{}}}},
          {{"nid":"n4","labels":["NUMERIC"],"surface":"26.8%","evidence_span":"MSE decrease of 26.8%","properties":{{"value":26.8}}}}],"hyperedges":[{{"eid":"e1","pattern_type":"composed_of","node_ids":["n1","n2"],"node_roles":["whole","component"],"qualifiers":{{"relation_type":"composition","method":"theory","evidence_strength":"derived","cited_from":"this_work"}},"evidence_span":"Informer is composed of ProbSparse self-attention"}},
          {{"eid":"e2","pattern_type":"reports","node_ids":["n1","n3","n4"],"node_roles":["from","to","parameter"],"qualifiers":{{"relation_type":"monotonic","method":"experiment","evidence_strength":"measured","cited_from":"this_work"}},"evidence_span":"Informer achieves MSE decrease of 26.8%"}}],"summary":"<=150 token compact summary for the next section"}}
Output ONLY the JSON object."""


class HGBlackboard:
    """JSON carrier: per-node summaries for chained context."""

    def __init__(self):
        self.summaries: dict[str, str] = {}

    def add(self, node_id: str, summary: str):
        self.summaries[node_id] = summary

    def predecessor_summary(self, deps: list[str]) -> str:
        if not deps:
            return ""
        parts = [f"[{d}] {self.summaries[d]}" for d in deps if d in self.summaries]
        return "Predecessor extracts:\n" + "\n".join(parts) if parts else ""


def _call(prompt: str, llm: str, max_tokens: int = 16384) -> str | None:
    if llm == "deepseek":
        # deepseek-chat (official API) — fast, decent quality
        return call_llm(prompt, model="deepseek-chat", max_tokens=max_tokens)
    if "V4-Flash" in llm or "R1" in llm or "Thinking" in llm:
        # DeepSeek-V4-Flash via Paratera, thinking.type=disabled (correct param,
        # verified: reasoning_tokens=0, 1s response, content normal JSON)
        return call_paratera(prompt, model=llm, max_tokens=max_tokens,
                             enable_thinking=False)
    if _is_cst_model(llm):
        return call_cst(prompt, model=llm, max_tokens=max_tokens)
    return call_paratera(prompt, model=llm, max_tokens=max_tokens)


# ---- retrieval-based schema injection (DIAL-KG style) ----
# When the schema grows, meta.to_prompt() bloats the extraction prompt and the
# LLM starts missing basic seed-pattern edges (measured: frozen 57 -> full 30
# influences on one paper as schema went 6 -> 237 patterns). Fix: for each
# chunk, retrieve the top-K most relevant patterns (by embedding cosine) and
# render ONLY those + always-include seed patterns, so the schema can grow
# without degrading extraction. Mirrors DIAL-KG's "retrieve top-K=30 relevant
# schemas from S_{k-1}".
RETRIEVAL_K = 30
SEED_PATTERN_IDS = ("influences", "constitutive_law", "defines",
                     "composed_of", "measures", "claim_relation")
_pat_emb_cache = {}  # meta_id -> {pid: emb}  (cleared when meta version changes)


def _retrieved_schema_prompt(meta, chunk_text, k=RETRIEVAL_K):
    """Build a compact schema prompt with the top-k patterns relevant to this
    chunk, EXPANDED to preserve topology (IS-A ancestors/descendants + dep/con/
    comp neighbors), so the extractor sees connected structure, not isolated
    patterns. Falls back to full meta.to_prompt() if embedding unavailable or
    schema small. Seed patterns always included (basic relations extractable)."""
    active = meta.active_patterns()
    if len(active) <= k:
        return meta.to_prompt()
    try:
        from granular_agent.hypergraph_evolution import _ensure_pattern_embeds, _embed_texts_robust
        from granular_agent.llm_client import cosine_sim
        pat_embs = _ensure_pattern_embeds(meta)
        if not pat_embs:
            return meta.to_prompt()
        chunk_emb = _embed_texts_robust([chunk_text[:2000]])
        if not chunk_emb:
            return meta.to_prompt()
        chunk_emb = chunk_emb[0]
        scored = [(pid, cosine_sim(emb, chunk_emb))
                  for pid, emb in pat_embs.items() if pid in active]
        scored.sort(key=lambda x: -x[1])
        top = {pid for pid, _ in scored[:k]}
        # always include seed patterns
        keep = set(top) | {p for p in SEED_PATTERN_IDS if p in active}
        # topology-preserving expansion: IS-A ancestors/descendants + dep/con/comp neighbors
        keep = _expand_topology(meta, keep)
        return _render_patterns_compact(meta, keep)
    except Exception:
        return meta.to_prompt()


def _expand_topology(meta, keep):
    """Expand the kept set so the schema stays CONNECTED:
    (1) IS-A: add ancestors (so a sub-pattern's parent generalization is
        visible — the LLM can fall back to the abstract parent) and descendants
        (so a retrieved parent's concrete specializations are pickable).
    (2) dep/con/comp: add patterns directly connected by a schema topology edge
        to a kept pattern (so 'constitutive_law constrains measures' isn't
        severed when only one endpoint is retrieved).
    One hop is enough — multi-hop would re-bloat the prompt."""
    # IS-A edges (subclass_of): src is child, dst is parent
    isa_parent_of = {}   # child -> parent
    isa_children_of = {} # parent -> [children]
    topo_neighbors = {}  # pattern -> [patterns via dep/con/comp]
    for e in meta.meta_edges:
        if e.relation == "subclass_of":
            isa_parent_of.setdefault(e.src, []).append(e.dst)
            isa_children_of.setdefault(e.dst, []).append(e.src)
        elif e.relation in ("depends_on", "constrains", "composes"):
            topo_neighbors.setdefault(e.src, []).append(e.dst)
            topo_neighbors.setdefault(e.dst, []).append(e.src)
    expanded = set(keep)
    for pid in list(keep):
        # IS-A ancestors (walk up)
        stack = list(isa_parent_of.get(pid, []))
        while stack:
            p = stack.pop()
            if p in expanded:
                continue
            expanded.add(p)
            stack.extend(isa_parent_of.get(p, []))
        # IS-A direct descendants (one hop down)
        for child in isa_children_of.get(pid, []):
            expanded.add(child)
        # dep/con/comp direct neighbors (one hop)
        for nb in topo_neighbors.get(pid, []):
            expanded.add(nb)
    return expanded


def _render_patterns_compact(meta, keep_ids):
    """Render kept patterns + the topology edges AMONG them (IS-A + dep/con/comp),
    so the extractor sees connected structure. Compact: id + description +
    family (role slots omitted to stay bounded)."""
    keep = {p for p in keep_ids if p in meta.patterns and not meta.patterns[p].deprecated}
    lines = ["Schema (topology-preserving retrieved subset; seed patterns always present):"]
    ordered = [p for p in SEED_PATTERN_IDS if p in keep]
    ordered += sorted(p for p in keep if p not in SEED_PATTERN_IDS)
    for pid in ordered:
        pat = meta.patterns.get(pid)
        if not pat or pat.deprecated:
            continue
        fam = f"<{pat.family}>" if pat.family else ""
        abs_tag = " (abstract)" if pat.is_abstract else ""
        desc = (pat.description or "")[:120]
        # role slots MUST render in compact mode (A2 late-production-root-cause:
        # once the schema grows past the retrieval-K threshold the compact
        # renderer omitted roles, the LLM never saw the declared role names,
        # invented its own (predictor/influencer/...), and the role gate killed
        # the whole edge — 282 edges lost across 4 papers, 87% of late-run
        # production). Roles are short; the bounded-budget concern that
        # justified omitting them is nothing next to this.
        slots = ",".join(s.get("role") for s in pat.role_slots if s.get("role"))
        slot_tag = f"({slots})" if slots else ""
        line = f"- {pid}{fam}{abs_tag}{slot_tag}: {desc}"
        # P1-B fix: render semantic_boundary in compact mode too (truncated) so
        # the extractor/verifier sees the disambiguation hint for confusable
        # patterns even under retrieval+compact. Without it, retrieval severs
        # the schema-in-context signal Step 2 added (boundary cures type weakness).
        if pat.semantic_boundary:
            line += f" [b: {pat.semantic_boundary[:100]}]"
        lines.append(line)
    # topology edges among kept patterns (IS-A + dep/con/comp)
    topo = []
    for e in meta.meta_edges:
        if e.src not in keep or e.dst not in keep:
            continue
        if e.relation == "subclass_of":
            topo.append(f"{e.src} subclass_of {e.dst}")
        elif e.relation in ("depends_on", "constrains", "composes"):
            topo.append(f"{e.src} {e.relation} {e.dst}")
    if topo:
        lines.append("Topology: " + "; ".join(topo))
    return "\n".join(lines)


def _parse_hg_response(raw: str | None, node_id: str) -> tuple[list[HGNode], list[Hyperedge], str]:
    """Parse LLM output into HGNode/Hyperedge. Rewrites nids with a node_id
    prefix so nodes from different DAG nodes never collide."""
    if not raw:
        return [], [], ""
    parsed = parse_json_response(raw)
    if not isinstance(parsed, dict):
        return [], [], ""
    raw_nodes = parsed.get("nodes", []) or []
    raw_hes = parsed.get("hyperedges", []) or []
    summary = parsed.get("summary", "") or ""

    # nid remap: LLM-local nid -> global nid
    remap: dict[str, str] = {}
    nodes: list[HGNode] = []
    for n in raw_nodes:
        if not isinstance(n, dict):
            continue
        local = str(n.get("nid", ""))
        if not local:
            continue
        gid = f"{node_id}_{local}"
        remap[local] = gid
        labels = n.get("labels", []) or []
        if isinstance(labels, str):
            labels = [labels]
        nodes.append(HGNode(
            nid=gid, labels=[str(l) for l in labels if l],
            surface=str(n.get("surface", "")),
            properties=n.get("properties", {}) if isinstance(n.get("properties"), dict) else {},
            evidence_span=str(n.get("evidence_span", "")),
        ))
    hes: list[Hyperedge] = []
    for i, h in enumerate(raw_hes):
        if not isinstance(h, dict):
            continue
        local_ids = h.get("node_ids", []) or []
        gids = [remap.get(str(x), str(x)) for x in local_ids]  # remap if known
        roles = [str(r) for r in (h.get("node_roles", []) or [])]
        quals = h.get("qualifiers", {}) if isinstance(h.get("qualifiers"), dict) else {}
        quals = {str(k): str(v) for k, v in quals.items()}
        hes.append(Hyperedge(
            eid=f"{node_id}_e{i}", pattern_type=str(h.get("pattern_type", "")),
            node_ids=gids, node_roles=roles, qualifiers=quals,
            evidence_span=str(h.get("evidence_span", "")),
        ))
    return nodes, hes, summary


CHUNK_THRESH = 4000  # over-long sections (>4k chars) drown one LLM call —
                     # Discussion/Results can be 23k chars -> 0 edges. Chunk on
                     # sentence boundaries into ~6k pieces, multiple calls,
                     # merge. (the cross-chunk surface dedup handles node reuse.)


def _chunk_text(text: str, thresh: int = CHUNK_THRESH) -> list[str]:
    """Split over-long section text into ~thresh-char chunks on sentence
    boundaries. Returns [text] if short enough."""
    if len(text) <= thresh:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + thresh, len(text))
        if end < len(text):
            # extend to next sentence end to avoid mid-sentence cuts
            for sep in (". ", "? ", "! "):
                pos = text.rfind(sep, start + thresh // 2, end + thresh)
                if pos > 0:
                    end = pos + 1
                    break
            else:
                end = min(end + 200, len(text))
        chunks.append(text[start:end])
        start = end
    return [c for c in chunks if c.strip()]


# ============================================================================
# MULTI-STEP EXTRACTION (vs old single-pass 21K-char prompt)
# Problem: 21K-char EXTRACT_HG_PROMPT -> deepseek-chat doesn't fully follow
# fine-grained rules (PIV excluded, composed_of boundary, label correctness).
# Fix: split into 3 short focused prompts — each does ONE thing, LLM follows
# better (short-context instruction adherence).
# ============================================================================

_JOINT_PROMPT = """You are extracting a SCIENTIFIC KNOWLEDGE HYPERGRAPH from ONE chunk of a {domain} paper section ({section_name}).

Schema patterns (the CURRENT schema — pick pattern_type from here; [boundary: ...] tells you WHEN each applies — it is the decision criterion, not decoration):
{schema_prompt}

Section chunk text:
{section_text}

Task: in ONE pass, extract the entities AND the n-ary hyperedges connecting them, reading each sentence and binding participants AS you read it.

Output JSON:
{{"nodes":[{{"nid":"n1","surface":"...","type":"METHOD","evidence_span":"verbatim phrase where the entity appears"}}],
 "hyperedges":[{{"eid":"e1","pattern_type":"...","node_ids":["n1","n2"],"node_roles":["...","..."],"evidence_span":"...","qualifiers":{{}}}}]}}

SLOT-BINDING DISCIPLINE (what may and may not occupy a node slot — post-checks reject violations):
- A node must be the THING the sentence puts in that role. Time/position/repetition adverbials
  ("after 15–20 revolutions", "at short times", "near the wall") are NOT entities — never node slots.
- PERSON/RESEARCHER names are never METHOD/INSTRUMENT nodes (a person may author a method; the
  method is the method).
- An ATTRIBUTE of a thing ("the accuracy of X") is not the thing itself: a measures edge's object
  is the QUANTITY the method measures, not a property of the method.
- A context the definition lives in (the MDP, the model, the experiment) is not the DEFINED entity;
  bind the entity actually being defined ("we define X as Y": subject=X).
- When the sentence names a vague aggregate ("all previous algorithms"), bind that aggregate
  faithfully — do not substitute or invent a specific name.

PATTERN-SELECTION CRITERIA (the highest-error types — apply these tests before committing to a pattern_type):
- influences ONLY means X functionally DEPENDS on Y / X varies with Y in a law or
  mechanism ("the value of Q depends on the discount factor"). A sentence that
  STATES A PROPERTY ("the task is partially observed"), a TEMPORAL CONDITION
  ("feedback is received after N steps"), an OBSERVATION LIMIT ("impossible to
  understand X from Y"), a DEFINITION ("X is Y"), or an ASSUMPTION ("sequences
  are assumed to terminate") is NOT an influences edge — properties/conditions/
  limits are not functional dependencies. When unsure, do NOT emit the edge.
- NEGATION = INDEPENDENCE: "X does not depend on Y", "X is independent of Y",
  "would not influence", "not affected by", "no impact on", "X is fairly
  constant as Y varies" DENY a dependence — NEVER emit an influences edge from
  such a sentence (if the schema has an absence/independence pattern and the
  independence is itself the finding, use that; otherwise emit nothing). The
  negation counts wherever it sits — including inside a relative or causal
  clause ("the law, which does not depend on Y", "remains well-posed since the
  condition is not affected by Y"): Y is NOT an input/parameter/cause. A
  sentence may carry a negated clause AND a positive one ("X is not affected
  by A but depends linearly on B") — extract the positive dependence (B), and
  the independence from A only via an absence/independence pattern.
- DEPENDENCY DIRECTION: in "A depends on B" / "A is a function of B" / "A
  varies with B", B is the source/cause and A is the target/effect — the
  DEPENDING quantity is the target, never the source.
- compares requires an explicit side-by-side EVALUATION of two entities.
  "X is a (finite-difference) approximation of Y" / "X reduces to Y" is
  formulation equivalence (equivalent_formulation if the schema has it);
  "observation matches the theoretical prediction" is agreement (agrees_with),
  not a comparison.
- PORTING: "we employ (a prior) relation X in a new geometry/configuration" is
  adapts (reuse in a new setting), not extends (extends = direct technical
  generalization of a method).
- BACKGROUND direction: from = the CURRENT work, to = the prior context it
  cites as motivation. A system used to TEST a theory is validates_against
  territory, not background — and validates_against binds object = the thing
  VALIDATED (usually the theory/claim), reference = the experiment/benchmark
  it is checked against ("this system has been used to test the validity of
  the theory": object=theory, reference=system).
- constitutive_law slots hold PHYSICAL QUANTITIES (the law's inputs/outputs/
  parameters), never the law/equation itself: "using the Bellman equation to
  estimate Q" does not put the Bellman equation in the input slot — the law
  IS the pattern; its quantities are the slots.
- composed_of ONLY means structural parts of a whole ("DQN consists of a replay
  buffer and a target network"). "X combines/combines the ideas of paradigm A
  and B" is lineage/derivation, not structural composition; a preprocessing
  pipeline step's substeps are composition, a taxonomy merge is not.
- defines ONLY means definitional identity ("X is defined as Y", "we call X the
  Y"). "selected by", "chosen by", "obtained from" are not definitions.
- ablation/disable-and-measure statements go to an ablation-like pattern if the
  schema has one, never to composed_of.

Entity types (pick the best fit per entity):
- METHOD: a NAMED scientific modeling approach/theory/law/model (μ(I) rheology, Transformer, DQN, Arrhenius kinetics). NOT measurement tools/procedures, NOT setups, NOT benchmarks, NOT a whole research field (reinforcement learning is a FIELD, not a method instance).
- PARAMETER: a named quantity/symbol in equations (μ, learning rate, E_a). NOT generic quantities (stress, accuracy) → PROPERTY.
- PHENOMENON: an effect/behavior (overfitting, nonlocal creep).
- REGIME: an operational condition (dense/quasi-static, training/inference).
- MATERIAL: a substance/sample/cell line.
- NUMERIC: a specific value (0.38, 75%).
- PROPERTY: a generic quantity/metric/benchmark/task that fits none of the above (Atari 2600, game score, accuracy).

HARD RULES (a post-check rejects violations — a rule-violating edge is a wasted edge):
1. An edge's evidence_span = the MINIMAL CONTINUOUS sentence(s) from the text that STATE this relation. Copy VERBATIM.
2. EVERY node of an edge must have its surface LITERALLY PRESENT inside that edge's evidence_span. Bind participants from the SAME sentence that states the relation — NEVER substitute an entity from a different sentence, and NEVER generalize ("these 6 games" must not become "49 games").
3. The evidence sentence must STATE the relation (not imply it, not describe a setup). A sentence that only describes what a baseline IS does not support a comparison edge; a sentence that only says what was done does not support a result edge.
4. POLARITY: bind roles to match the sentence's direction exactly. 'A outperforms B' → winner=A, loser=B. 'A is comparable to B' or 'A achieves 75% of B' is NOT outperforming. 'A fails where B works' reverses the roles.
5. n-ary: parallel participants in one sentence go into ONE edge — a comparison with a task and a metric is ONE edge with 4 nodes (method A, method B, task, metric), not separate edges.
6. node_roles MUST be exactly the roles DECLARED for that pattern_type in the schema above (the schema's role slots override anything else you remember).
7. qualifiers: only keys the pattern declares. Enum values only for enum keys (method: experiment|simulation|theory|review; evidence_strength: measured|derived|hypothesized|assumed; cited_from: this_work|prior_art|definition).
8. Prefer a schema pattern over inventing a new pattern_type. New pattern names (only if nothing fits): clean snake_case, never slash-joined compounds.
9. SETUP/IMPLEMENTATION/PLOTTING sentences ("a constant flux rate is set by a
  syringe pump attached to ...", "store transition ... in D", "execute action
  in emulator", "we plot X as a function of Y", "solid/open triangles") say
  what was DONE or how data is displayed — they are NOT scientific claims: no
  influences/measures/composed_of/defines edges from them. A measurement
  instrument MEASURES a quantity; a plot axis, a storage target, or an
  apparatus location is not an instrument, component, or cause. This targets
  apparatus/protocol/display text ONLY — a stated modeling or design choice
  ("we use the full sequence as the state representation") IS a claim and may
  support defines/composed_of.

If nothing extractable, output {{"nodes":[],"hyperedges":[]}}.
"""


def _run_hg_node_joint(node: dict, sections: list, blocks: list,
                       schema_prompt: str, bb, llm: str, domain: str,
                       meta=None, feedback_hint: str = "") -> tuple[list[HGNode], list[Hyperedge], str]:
    """JOINT extraction (B+ rewrite, 2026-08-25): ONE LLM call per chunk
    extracts entities + types + hyperedges together, so binding happens in
    sentence context (the three-step decomposition bound entities to roles
    from a decontextualized list — the probe's slot-binding failure mode).

    Deterministic post-checks (binding-locality etc.) live in
    ExtractionAgent's rule gate, not here — this function is the LLM call +
    parse. Returns (nodes, edges, summary) like the old runners."""
    sec_text = section_text_for_node(node, sections, blocks)
    if not sec_text:
        print(f"  [joint] {node.get('id','?')}: no sec_text", flush=True)
        return [], [], ""
    predecessor = bb.predecessor_summary(node.get("deps", []))
    chunks = _chunk_text(sec_text)
    all_nodes: list[HGNode] = []
    all_edges: list[Hyperedge] = []
    summary = ""
    print(f"  [joint] {node.get('id','?')}: {len(chunks)} chunks, sec_text={len(sec_text)} chars", flush=True)

    def _process_chunk(ci_chunk):
        ci, chunk = ci_chunk
        nid_prefix = f"{node['id']}c{ci}" if len(chunks) > 1 else node["id"]
        p = _JOINT_PROMPT.format(domain=domain,
                                 section_name=node.get("section", ""),
                                 schema_prompt=schema_prompt,
                                 section_text=chunk)
        if predecessor:
            p = p + "\n\nPredecessor context:\n" + predecessor
        if feedback_hint:
            p = p + "\n\nFEEDBACK (prioritize):\n" + feedback_hint
        raw = _call(p, llm, max_tokens=12288)
        parsed = parse_json_response(raw) or {}
        if not parsed and raw:
            p_retry = p + "\n\nIMPORTANT: output ONLY valid JSON, no prose."
            raw2 = _call(p_retry, llm, max_tokens=8192)
            parsed = parse_json_response(raw2) or {}
            print(f"  [joint] c{ci} RETRY parsed nodes={len(parsed.get('nodes',[]) or [])} "
                  f"hes={len(parsed.get('hyperedges',[]) or [])}", flush=True)

        # --- nodes ---
        chunk_nodes: list[HGNode] = []
        local2gid: dict[str, str] = {}
        for n in (parsed.get("nodes", []) or []):
            if not isinstance(n, dict):
                continue
            local = str(n.get("nid", ""))
            if not local or not n.get("surface"):
                continue
            labels = n.get("type") or n.get("labels") or []
            if isinstance(labels, str):
                labels = [labels]
            gid = f"{nid_prefix}_{local}"
            local2gid[local] = gid
            chunk_nodes.append(HGNode(
                nid=gid,
                labels=[str(l).upper() for l in labels if l],
                surface=str(n.get("surface", "")),
                evidence_span=str(n.get("evidence_span", "")),
            ))

        # --- hyperedges (node_ids remapped local -> prefixed gid) ---
        chunk_edges: list[Hyperedge] = []
        for i, h in enumerate((parsed.get("hyperedges", []) or [])):
            if not isinstance(h, dict):
                continue
            gids = [local2gid.get(str(x), str(x)) for x in (h.get("node_ids", []) or [])]
            roles = [str(r) for r in (h.get("node_roles", []) or [])]
            quals = h.get("qualifiers", {}) if isinstance(h.get("qualifiers"), dict) else {}
            quals = {str(k): str(v) for k, v in quals.items()}
            chunk_edges.append(Hyperedge(
                eid=f"{nid_prefix}_e{i}",
                pattern_type=str(h.get("pattern_type", "")),
                node_ids=gids, node_roles=roles, qualifiers=quals,
                evidence_span=str(h.get("evidence_span", "")),
            ))
        csum = str(parsed.get("summary", ""))[:300]
        print(f"  [joint] c{ci} parsed nodes={len(chunk_nodes)} hes={len(chunk_edges)}", flush=True)
        return chunk_nodes, chunk_edges, csum

    # chunk-level concurrency (independent text segments). 8 workers — joint
    # is 1 call/chunk (vs 3 in the old multistep), so the same API budget
    # supports deeper parallelism.
    if len(chunks) <= 1:
        for ci_chunk in enumerate(chunks):
            cn, ce, cs = _process_chunk(ci_chunk)
            all_nodes.extend(cn); all_edges.extend(ce); summary = cs or summary
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        max_workers = min(8, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_chunk, (ci, chunk)): ci
                       for ci, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                try:
                    cn, ce, cs = future.result()
                    all_nodes.extend(cn); all_edges.extend(ce)
                    if cs:
                        summary = (summary + " " + cs)[:600] if summary else cs
                except Exception as e:
                    print(f"  [joint] chunk {futures[future]} failed: {e!r}", flush=True)

    return all_nodes, all_edges, summary

def _discourse_for_node(node: dict, sections: list) -> str:
    sec = next((s for s in sections if s.get("name") == node.get("section")), None)
    return sec.get("discourse_role", "unknown") if sec else "unknown"


_ORG_HINTS = ("university", "department", "institute", "faculty", "laboratory",
              "college", "school of", "center for", "division of")


def _extract_metadata(blocks: list, paper_id: str) -> dict:
    """Heuristically pull paper-level metadata from the leading mineru blocks:
    the first text block is usually the title; subsequent text blocks (until
    an org affiliation / abstract / section heading) are author names.
    doi/year/venue are left empty here — they need upstream lookup and are
    filled by the caller when available. This gives every extracted graph a
    minimal provenance header so downstream can map hyperedge -> paper without
    a separate join."""
    md = {"paper_id": paper_id, "title": "", "authors": [],
          "doi": "", "year": "", "venue": ""}
    texts = [b.get("text", "").strip() for b in blocks
             if b.get("type", "") in ("", "text") and b.get("text", "").strip()]
    if not texts:
        return md
    md["title"] = texts[0][:300]
    # scan the leading ~8 blocks for doi/year/venue — many mineru extractions
    # carry a citation line like "Citation: J. Rheol. 23, 243 (1979); doi: 10.xxx"
    import re as _re
    lead = " ".join(texts[:8])
    doi_m = _re.search(r"10\.\d{4,9}/[^\s\"';]+", lead)
    if doi_m:
        md["doi"] = doi_m.group(0).rstrip(".,);")
    year_m = _re.search(r"\b(1[89]\d{2}|20[0-3]\d)\b", lead)
    if year_m:
        md["year"] = year_m.group(0)
    # venue: text before the year in a "Citation:" line, e.g. "J. Rheol. 23, 243"
    cit_m = _re.search(r"(?:Citation|Published in)[:\s]+([^.]*?)\d{1,3}\s*,?\s*\d+", lead)
    if cit_m:
        md["venue"] = cit_m.group(1).strip(" ,;")[:120]
    authors = []
    for t in texts[1:6]:
        tl = t.lower()
        if any(h in tl for h in _ORG_HINTS) or len(t) > 120 or "abstract" in tl:
            break
        if "doi" in tl or "citation" in tl or "view online" in tl or "http" in tl:
            break  # citation/metadata line, not an author
        if "." in t and len(t.split()) > 8:
            break
        authors.append(t[:120])
    md["authors"] = authors[:8]
    # if doi still missing, look it up by title via Crossref (no key needed).
    # Cached per paper_id so re-extraction doesn't re-hit the network. Falls
    # back silently to the heuristic values on any failure. We cross-check
    # the year: if Crossref's top hit year != the heuristic year, the title
    # query likely matched a same-named different paper -> drop the crossref
    # doi (keep heuristic year which came from the paper's own "Dated:" line).
    if not md["doi"] and md["title"]:
        cr = _crossref_lookup(md["title"], paper_id)
        if cr:
            cr_year = str(cr.get("year", "") or "")
            if md["year"] and cr_year and md["year"] != cr_year:
                pass  # year mismatch -> likely wrong match, skip crossref doi
            else:
                md["doi"] = md["doi"] or cr.get("doi", "")
                md["venue"] = md["venue"] or cr.get("venue", "")
                md["lookup_source"] = "crossref"
    return md


_CROSSREF_CACHE: dict[str, dict] = {}


def _crossref_lookup(title: str, paper_id: str) -> dict | None:
    """Query Crossref by title to fill doi/year/venue. Cached per paper_id.
    No API key needed (polite pool). Returns None on any failure."""
    if paper_id in _CROSSREF_CACHE:
        return _CROSSREF_CACHE[paper_id] or None
    import urllib.request, urllib.parse
    try:
        q = urllib.parse.quote(title[:200])
        url = (f"https://api.crossref.org/works?query.title={q}&rows=1"
               f"&select=DOI,title,issued,container-title")
        req = urllib.request.Request(url, headers={"User-Agent": "LogicKG/0.1 (mailto:noreply)"})
        raw = urllib.request.urlopen(req, timeout=12).read()
        items = json.loads(raw).get("message", {}).get("items", [])
        if not items:
            _CROSSREF_CACHE[paper_id] = {}; return None
        it = items[0]
        yr = it.get("issued", {}).get("date-parts", [[None]])
        yr = yr[0][0] if yr and yr[0] else None
        res = {"doi": it.get("DOI", ""),
               "title_match": (it.get("title") or [""])[0],
               "year": yr, "venue": (it.get("container-title") or [""])[0][:120]}
        _CROSSREF_CACHE[paper_id] = res
        return res
    except Exception:
        _CROSSREF_CACHE[paper_id] = {}
        return None


def extract_hypergraph(structure_map: dict, blocks: list, meta: MetaHypergraph,
                       llm: str = "deepseek", paper_id: str = "",
                       domain: str = "granular flow physics",
                       trigger: EvolutionTrigger | None = None,
                       include_topology: bool = True,
                       evolve: bool = True,
                       propagate_intra_dag: bool = True,
                       feedback_hint: str = "") -> dict:
    """Phase 1: run all DAG nodes in topo order, producing an InstanceHypergraph
    and evolving the meta-hypergraph in place (deep self-evolution closed loop).

    `meta` is mutated in place (caller holds the shared schema for cross-paper
    evolution). `trigger` may be passed in to persist cross-paper recurrence
    accounting; a fresh one is created if None.

    Ablation switches (default behavior unchanged):
      - evolve=False: skip run_evolution_loop entirely (frozen schema arm).
      - propagate_intra_dag=False: compute the schema prompt ONCE before the
        loop and do not re-fetch per node (disables intra-DAG propagation;
        evolution still mutates meta for cross-paper + downstream maintenance).

    Returns {instance, evolutions, n_calls, n_nodes, n_hyperedges, validation_failures}.
    """
    sections = structure_map.get("sections", [])
    dag = structure_map.get("dag", {"nodes": []})
    nodes = topo_order(dag)

    instance = InstanceHypergraph(paper_id=paper_id)
    instance.metadata = _extract_metadata(blocks, paper_id)
    trigger = trigger or EvolutionTrigger()
    bb = HGBlackboard()
    n_calls = 0
    all_evolutions: list[dict] = []
    total_failures = 0
    failed_edges: list[dict] = []  # debug: rejected edges + reasons
    surface2nid: dict[str, str] = {}  # cross-section surface dedup

    # schema prompt: re-fetched per node when propagate_intra_dag (P4 forward
    # propagation); computed once and frozen across nodes when not (ablation).
    schema_prompt = meta.to_prompt(include_topology=include_topology)
    for node in nodes:
        # P4 forward propagation: re-fetch the (possibly evolved) schema prompt
        if propagate_intra_dag:
            schema_prompt = meta.to_prompt(include_topology=include_topology)
        # joint extraction (B+ rewrite): 1 LLM call per chunk — entities +
        # types + hyperedges together, binding in sentence context
        _sec = section_text_for_node(node, sections, blocks) or ""
        hg_nodes, hg_edges, summary = _run_hg_node_joint(
            node, sections, blocks, schema_prompt, bb, llm, domain, meta,
            feedback_hint=feedback_hint)
        n_calls += max(1, len(_chunk_text(_sec)))

        # add nodes to the instance graph (dedup by SURFACE, cross-section):
        # the LLM re-extracts "granular materials"/"fabric" in each section
        # with fresh nids. Merge by normalized surface: reuse the existing
        # nid, union labels, keep first evidence. Remap this batch's edges.
        local_remap: dict[str, str] = {}
        for n in hg_nodes:
            key = _norm_surface(n.surface)
            if key and key in surface2nid:
                existing = instance.nodes.get(surface2nid[key])
                if existing:
                    for l in n.labels:
                        if l not in existing.labels:
                            existing.labels.append(l)
                    local_remap[n.nid] = existing.nid
                    continue
            if n.nid not in instance.nodes:
                instance.add_node(n)
                if key:
                    surface2nid[key] = n.nid
            local_remap[n.nid] = n.nid
        # remap edge node_ids to dedup'd nids
        for he in hg_edges:
            he.node_ids = [local_remap.get(nid, nid) for nid in he.node_ids]

        # validate each hyperedge; collect structural failures
        failures: list[tuple[Hyperedge, str]] = []
        for he in hg_edges:
            # only validate edges whose nodes all exist in the instance
            if not all(nid in instance.nodes for nid in he.node_ids):
                continue
            ok, reason = meta.validate(he, instance)
            if not ok:
                failures.append((he, reason))
                failed_edges.append({"node": node["id"], "pattern_type": he.pattern_type,
                    "arity": len(he.node_ids), "roles": he.node_roles,
                    "qualifier_keys": list(he.qualifiers.keys()), "reason": reason,
                    "evidence": he.evidence_span[:80]})
            else:
                instance.add_hyperedge(he)
        total_failures += len(failures)

        # closed loop: failures -> probe -> validate -> apply (mutates meta in place)
        if failures and evolve:
            evols, nc = run_evolution_loop(meta, failures, trigger, node["id"], paper_id,
                                           domain=domain, llm=llm, instance=instance)
            n_calls += nc
            all_evolutions.extend(evols)
            # After evolution, re-validate the failed edges that were held out:
            # an accepted new pattern/type may now accommodate them.
            for he, _ in failures:
                ok2, _ = meta.validate(he, instance)
                if ok2:
                    instance.add_hyperedge(he)

        bb.add(node["id"], summary)

    return {
        "instance": instance,
        "evolutions": all_evolutions,
        "n_calls": n_calls,
        "n_nodes": len(instance.nodes),
        "n_hyperedges": len(instance.hyperedges),
        "validation_failures": total_failures,
        "failed_edges": failed_edges,
    }
