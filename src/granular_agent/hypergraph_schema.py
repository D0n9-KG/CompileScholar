"""Knowledge Hypergraph + Meta-Hypergraph (self-evolving schema).

Two layers:
1. Instance hypergraph: hyperedges extracted from papers (connect >2 nodes,
   carry qualifiers — n-ary facts, equations, regime-scoped claims).
2. Meta-hypergraph (the SCHEMA): a hypergraph itself — meta-nodes = node types,
   meta-hyperedges = edge patterns + type relations + subclass hierarchy.
   This meta-hypergraph EVOLVES during extraction.

Design (deep self-evolution):
- Instance hyperedge validated against meta-hypergraph patterns.
- Structural mismatch (no matching meta-pattern, or unstable repeated match)
  triggers evolution probe → propose meta-structure change (add meta-node,
  add meta-hyperedge, split/merge meta-nodes, add subclass edge).
- Probe is evidence-anchored (verbatim spans) + LLM-judged.
- Accepted change propagates forward (downstream DAG nodes re-extract with
  the evolved meta-hypergraph).

Node: a typed entity with properties (multi-label: a node can be both
MATERIAL and DIMENSIONLESS_NUMBER, resolving the old single-inheritance issue).
Hyperedge: connects N nodes, has a pattern_type, qualifiers (key→value, e.g.
applies_in="dense regime"), evidence_span.
"""
from __future__ import annotations
import json, copy
from dataclasses import dataclass, field, asdict
from typing import Any
import re, unicodedata


def _norm_surface(s: str) -> str:
    """Normalize a node surface for cross-paper alignment: NFKC, lowercase,
    collapse whitespace/punctuation. "Granular Materials" == "granular
    materials". Surface-only alignment (no embedding entity-resolution) — the
    honest same limitation as intra-paper dedup."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s.lower())
    s = re.sub(r"[\s\-_]+", " ", s).strip(" .,;:()")
    return s


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for embedding entity-resolution (local, no llm_client
    dep). 0 if either is empty/zero."""
    if not a or not b or len(a) != len(b):
        return 0.0
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class HGNode:
    """A node in the instance hypergraph. Multi-label (can have >1 type)."""
    nid: str
    labels: list[str] = field(default_factory=list)   # multi-label (MATERIAL, DIMENSIONLESS_NUMBER, ...)
    surface: str = ""                                   # the mention text
    properties: dict[str, Any] = field(default_factory=dict)  # extra attrs (value, unit, ...)
    evidence_span: str = ""
    source_paper: str = ""    # paper_id this node was extracted from (for
                              # cross-paper merged graphs — node attribution/
                              # provenance for downstream lookup). Defaults to
                              # the instance's paper_id when single-paper.

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Hyperedge:
    """A hyperedge connecting N nodes. n-ary fact / equation / scoped claim."""
    eid: str
    pattern_type: str                                   # e.g. "constitutive_law", "measures", "claim_relation"
    node_ids: list[str] = field(default_factory=list)   # the N nodes it connects (ordered)
    node_roles: list[str] = field(default_factory=list) # role of each node (input/output/subject/object/...)
    qualifiers: dict[str, str] = field(default_factory=dict)  # applies_in_regime, condition, time, ...
    evidence_span: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InstanceHypergraph:
    """The extracted hypergraph for one paper."""
    paper_id: str = ""
    nodes: dict[str, HGNode] = field(default_factory=dict)      # nid -> HGNode
    hyperedges: dict[str, Hyperedge] = field(default_factory=dict)  # eid -> Hyperedge
    metadata: dict[str, Any] = field(default_factory=dict)  # paper-level
        # provenance for downstream lookup + node attribution: title, authors,
        # doi, year, venue. Populated by the extractor from the source blocks
        # (title/authors) + upstream (doi/year when available). Lets a query
        # go hyperedge -> evidence -> paper title/doi without a separate lookup.

    def add_node(self, n: HGNode):
        if not n.source_paper:
            n.source_paper = self.paper_id
        self.nodes[n.nid] = n

    def add_hyperedge(self, he: Hyperedge):
        self.hyperedges[he.eid] = he

    def to_dict(self) -> dict:
        return {"paper_id": self.paper_id,
                "metadata": self.metadata,
                "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
                "hyperedges": {k: v.to_dict() for k, v in self.hyperedges.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "InstanceHypergraph":
        """Inverse of to_dict. Reconstructs nodes/hyperedges from their asdict form.
        Used by corpus_driver.load_instance to restore a persisted instance."""
        inst = cls(paper_id=d.get("paper_id", ""), metadata=d.get("metadata", {}) or {})
        for nid, nd in (d.get("nodes") or {}).items():
            n = HGNode(nid=nd.get("nid", nid),
                       labels=list(nd.get("labels", [])),
                       surface=nd.get("surface", ""),
                       properties=dict(nd.get("properties", {}) or {}),
                       evidence_span=nd.get("evidence_span", ""),
                       source_paper=nd.get("source_paper", ""))
            inst.nodes[n.nid] = n
        for eid, ed in (d.get("hyperedges") or {}).items():
            he = Hyperedge(eid=ed.get("eid", eid),
                           pattern_type=ed.get("pattern_type", ""),
                           node_ids=list(ed.get("node_ids", [])),
                           node_roles=list(ed.get("node_roles", [])),
                           qualifiers=dict(ed.get("qualifiers", {}) or {}),
                           evidence_span=ed.get("evidence_span", ""))
            inst.hyperedges[he.eid] = he
        return inst


# ---------------------------------------------------------------------------

@dataclass
class MetaNode:
    """A node-type in the meta-hypergraph (schema-level type)."""
    type_id: str                                        # e.g. "MATERIAL", "DIMENSIONLESS_NUMBER"
    description: str = ""
    # subclass edges live as meta-edges (subclass_of) in MetaHypergraph.meta_edges
    is_abstract: bool = False                           # abstract types can't be instantiated directly


@dataclass
class MetaHyperedgePattern:
    """A pattern for instance hyperedges (schema-level edge pattern).
    Defines which node types connect, in what roles, with what qualifiers."""
    pattern_id: str                                     # e.g. "constitutive_law", "claim_relation"
    description: str = ""
    # ordered (role -> type_id) slots; a hyperedge of this pattern must have nodes
    # matching these roles (type can be abstract, matched by subclass)
    role_slots: list[dict] = field(default_factory=list)  # [{"role":"output","type":"DIMENSIONLESS_NUMBER"},...]
    allowed_qualifiers: list[str] = field(default_factory=list)  # ["applies_in_regime","condition",...]
    # repair provenance: a deprecated pattern is kept (not deleted) so historical
    # hyperedges keep a valid schema reference; new extraction must not emit it.
    deprecated: bool = False
    split_from: str = ""        # parent pattern_id if this pattern was born from a split
    # Taxonomy flag (Path C, "real ontology" upgrade): a pattern split into
    # sub-patterns becomes ABSTRACT — kept as a generalization, NOT deprecated.
    # An abstract parent participates in a pattern-level IS-A taxonomy
    # (subclass_of meta-edges: child IS-A parent), is RENDERED as a tree node
    # in to_prompt (so the extractor sees "constitutive_law (abstract) > {
    # power_law, density_dependent, ...}"), and is a VALIDATION FALLBACK — a
    # new edge that no concrete child matches falls back to the abstract
    # parent. This is the difference between a lineage log (split_from alone)
    # and a queryable ontology (IS-A hierarchy). Contrast with `deprecated`
    # (merge/retire — truly retired, never matches, never shown).
    is_abstract: bool = False
    # Seed-skeleton bounded op (Path C): every pattern belongs to ONE fixed
    # top-level CONTENT FAMILY. add_pattern REJECTS a pattern whose family is
    # not in TOP_LEVEL_FAMILIES — the agent may specialize WITHIN a family
    # (add_pattern with an existing family, or split), but may NOT invent a
    # free-form new top-level dimension (the divergence disease A6). Splits
    # inherit the parent's family. Empty family is allowed ONLY for the seed
    # (so old seeds without the field still load); the bounded-op gate treats
    # an unset family as "needs assignment" and rejects free-form adds.
    family: str = ""
    # semantic_boundary (v2 design supplement section 4 / 断点: schema-in-context
    # for type-judgment weakness). A short prose description of WHEN this pattern
    # applies vs nearby patterns it's easy to confuse with — e.g. influences =
    # "X functionally depends on Y, X varies with Y; NOT a co-listing / input /
    # test setup" (vs composed_of / measures). Rendered in to_prompt so the
    # extractor sees the boundary, not just the pattern name + role slots.
    # Hand-written for the seed patterns; the evolver generates one when it
    # adds a new pattern (from evidence + rationale). Cures the re-type weakness
    # (54% mis-typed edges) at the schema-in-context level, not via brittle rules.
    semantic_boundary: str = ""


# ---------------------------------------------------------------------------
# Path C: seed skeleton + bounded operations (the novelty pieces).
#
# TOP_LEVEL_FAMILIES fixes the 6 content dimensions the schema grows WITHIN
# (borrowed from AdaKGC's fixed top-level major classes). The 4 CONTEXT
# dimensions live as controlled QUALIFIER keys (QUALIFIER_REGISTRY) on the
# instance hyper-relational layer — they are an EXTENSION POINT (can only
# attach to an existing hyperedge), NOT free-form new patterns. This split
# (fixed top-level content families + controlled qualifier extension) is what
# cures the "hypergraph too free -> divergent non-converging schema" disease
# while preserving the n-ary + split novelty (HRKG's controllability borrowed
# without surrendering the hypergraph novelty anchor — see PATH-C-DESIGN.md).
# ---------------------------------------------------------------------------

TOP_LEVEL_FAMILIES = (
    "constitutive_law",   # output quantity lawfully related to >=1 input + params
    "dependency",         # one quantity depends on / influences / scales with another
    "definition",         # one entity defined-as / identified-with another (definitional)
    "composition",        # one entity composed-of / part-of others
    "measure",            # a quantity is measured by / characterizes a measure
    "claim",              # discourse relation between claims/approaches (supports/contrasts)
)

# Controlled qualifier registry: the instance-layer hyper-relational extension
# point. A qualifier key NOT in this registry is rejected by validate() —
#治发散: the LLM cannot spawn ad-hoc qualifier keys. Values are either a
# controlled enum (discrete -> enables deterministic split, the headline) or
# normalized free-text. This RECONCILES the goal's 4 context dimensions
# (condition / method / evidence_strength / cited_from) with the qualifier
# keys the working code already depends on (applies_in_regime for regime-
# scoped retrieval 0.735 + split-by-regime; dependency_type the controlled
# enum that makes split FIRE on real free-text data — see HYPERGRAPH-
# EVOLUTION.md 'SPLIT FIRES'). The goal's 4-key set was a pre-compaction
# simplification; dropping applies_in_regime/dependency_type would break the
# headline mechanism. method/evidence_strength are NEW (address P-E2
# attribution + P-E4 discourse filtering).
QUALIFIER_REGISTRY: dict[str, tuple[str, tuple[str, ...] | None]] = {
    # --- the 4 context dimensions (goal's承重设计) ---
    "condition":        ("free_text", None),   # 干湿/温度/压力 or regime condition
    "method":           ("enum", ("experiment", "simulation", "theory", "review")),
    "evidence_strength":("enum", ("measured", "derived", "hypothesized", "assumed")),
    "cited_from":       ("enum", ("this_work", "prior_art", "definition")),
    # --- load-bearing qualifiers the working code depends on (kept, not dropped) ---
    "applies_in_regime":("free_text", None),  # domain-dependent: dense/quasi-static
        # for physics, LSTF/train/test for ML, etc. Enum-restricting to physics
        # regimes rejects every general-paper edge whose regime tag is a valid ML
        # task tag (see DECISION-validation-rejects-all-general-edges). Free-text
        # here; split-by-regime still works via discrete-value detection on the
        # actual values present (qualifier_is_discrete), not a fixed enum.
    "dependency_type":  ("enum", ("monotonic", "derivation", "analogy", "composition")),
    "relation_type":   ("free_text", None),    # claim discourse verbs (supports/contrasts/...)
    "function_form":    ("free_text", None),    # constitutive_law equation text
    "parameters":       ("free_text", None),    # named params list (also NUMERIC nodes)
}


def qualifier_key_allowed(key: str) -> bool:
    return key in QUALIFIER_REGISTRY


def qualifier_value_ok(key: str, value: str) -> bool:
    """A qualifier value is valid if its key is registered AND, for an enum-
    typed qualifier, the value is in the enum (case-insensitive). Free-text-
    typed qualifiers accept any non-empty value. This is the deterministic
    gate that keeps the new context dimensions (method/evidence_strength) AND
    the load-bearing split enum (dependency_type) controlled — without it the
    LLM fills free-text values ('seminar discussion', 'qualitative') and the
    dimensions become un-clusterable, breaking split (the headline)."""
    spec = QUALIFIER_REGISTRY.get(key)
    if spec is None:
        return False
    kind, enum = spec
    if kind == "enum":
        return str(value).strip().lower() in {e.lower() for e in (enum or ())}
    return bool(str(value).strip())


@dataclass
class MetaEdge:
    """An edge in the meta-hypergraph (between meta-nodes / patterns).
    Encodes: subclass_of, type_relations, pattern_dependencies."""
    src: str        # meta-node type_id or pattern_id
    dst: str
    relation: str   # "subclass_of" | "type_relation" | "pattern_dependency"
    props: dict[str, Any] = field(default_factory=dict)


def _role_sig(role_slots: list[dict]) -> tuple:
    """Structural signature of a pattern's role-slots: (role-seq, type-seq).
    Two patterns with the same signature have the same structure — split is
    semantic-only, so sub-patterns must share the parent's signature."""
    rs = role_slots or []
    return (tuple(s.get("role", "") for s in rs),
            tuple(s.get("type", "") for s in rs))


def _match_variadic_set(slots: list[dict], roles: list[str]) -> tuple[bool, list[int]]:
    """Order-independent SET+COUNT match for a variadic pattern.

    Every edge role must equal the role of SOME slot in `slots`. Each
    repeatable slot (repeatable=True) may be matched by >=1 nodes; each
    non-repeatable slot must be matched by exactly 1 node. Roles not in the
    pattern reject the edge. This lets a constitutive_law pattern
    [output*, input*] match edges like [output, output, output, input, input,
    input] (3 dependents + 3 independents) — strict slot-position matching
    would reject these and silently collapse n-ary extraction to binary.

    Returns (ok, node_idx -> slot_idx mapping) for type-checking.
    """
    # role name -> slot index (patterns must not declare duplicate role names)
    role_to_slot: dict[str, int] = {}
    for i, s in enumerate(slots):
        role_to_slot[s.get("role", "")] = i
    # count constraints: non-repeatable slots need exactly 1; repeatable slots
    # are OPTIONAL (min=0) — a pattern declares the roles it ACCEPTS, not the
    # roles every edge must fill. Requiring >=1 of every declared repeatable role
    # makes adding optional auxiliary roles (output/parameter on an influences
    # edge) STRICTER not looser, rejecting exactly the rich n-ary edges we want.
    # The arity>=2 invariant is enforced separately (a hyperedge has >=2 nodes).
    needed = {i: (1, 1) for i in range(len(slots))}  # (min, max)
    for i, s in enumerate(slots):
        if s.get("repeatable"):
            needed[i] = (0, 10**9)
    counts = [0] * len(slots)
    node_to_slot: list[int] = []
    for r in roles:
        si = role_to_slot.get(r)
        if si is None:
            return False, []  # role not declared in pattern
        counts[si] += 1
        node_to_slot.append(si)
    for i in range(len(slots)):
        lo, hi = needed[i]
        if not (lo <= counts[i] <= hi):
            return False, []
    return True, node_to_slot


class MetaHypergraph:
    """The schema: a hypergraph of types + patterns + meta-edges.
    This is what evolves during extraction (deep structural self-evolution)."""

    def __init__(self):
        self.meta_nodes: dict[str, MetaNode] = {}           # type_id -> MetaNode
        self.patterns: dict[str, MetaHyperedgePattern] = {} # pattern_id -> MetaHyperedgePattern
        self.meta_edges: list[MetaEdge] = []                # subclass_of, type_relation, ...
        self.version: str = "0.1"
        # family -> root pattern_id (the abstract generalization for that
        # top-level family). Seed sets these; add_pattern attaches a new
        # same-family pattern IS-A its family root so EVERY pattern lives in
        # the IS-A tree (not an orphan with only a family tag). This is what
        # makes the schema a real ontology: a pattern added by the evolution
        # probe (not via split) still gets a subclass_of edge to its family
        # root, so the taxonomy is complete (no orphan leaves).
        self.family_roots: dict[str, str] = {}

    # ---- introspection ----
    def types(self) -> list[str]:
        return list(self.meta_nodes.keys())

    def patterns_ids(self) -> list[str]:
        return list(self.patterns.keys())

    def subclasses_of(self, type_id: str) -> list[str]:
        """Direct + transitive subclasses."""
        direct = [e.src for e in self.meta_edges
                  if e.relation == "subclass_of" and e.dst == type_id]
        seen = set(direct)
        stack = list(direct)
        while stack:
            t = stack.pop()
            for e in self.meta_edges:
                if e.relation == "subclass_of" and e.dst == t and e.src not in seen:
                    seen.add(e.src); stack.append(e.src)
        return list(seen)

    def is_subtype(self, sub: str, sup: str) -> bool:
        if sub == sup:
            return True
        return sub in self.subclasses_of(sup)

    def match_pattern(self, he: Hyperedge) -> str | None:
        """Return the pattern_id matching this instance hyperedge, or None.
        Lightweight match (not NP-hard): role-slot type compatibility by subtype."""
        for pid, pat in self.patterns.items():
            if len(he.node_roles) != len(pat.role_slots):
                continue
            ok = True
            for role_slot, nid, role in zip(pat.role_slots, he.node_ids, he.node_roles):
                if role_slot.get("role") != role:
                    ok = False; break
                node = None
                # node lookup happens in InstanceHypergraph; here we only check type compat
                # via the node's labels (passed in he? no). This is a schema-side check;
                # full validation uses the instance graph. We expose a type-check helper below.
            if ok:
                return pid
        return None

    def validate(self, he: Hyperedge, instance: InstanceHypergraph) -> tuple[bool, str]:
        """Validate an instance hyperedge against the meta-hypergraph.
        Returns (ok, reason). ok=False means structural mismatch → evolution trigger.

        Supports VARIADIC roles: a role_slot with repeatable=True may match
        multiple nodes of that role (enables n-ary hyperedges). For a variadic
        pattern we use SET+COUNT matching (order-independent): every edge role
        must be one of the pattern's declared roles, and each repeatable role
        needs >=1 node; non-repeatable roles need exactly 1. This handles edges
        like [output, output, output, input, input, input] (3 dependents, 3
        independents) which strict slot-position matching would reject.

        TAXONOMY (Path C "real ontology"): a CONCRETE pattern (not abstract,
        not deprecated) is tried FIRST; an ABSTRACT pattern (a parent that was
        split into sub-patterns) is a FALLBACK — an edge that no concrete child
        matches falls back to the abstract parent. So a generic constitutive_law
        edge with no specific sub-pattern match still validates against the
        abstract constitutive_law parent (kept as a generalization, not retired).
        Deprecated (merge/retire) patterns never match."""
        # two passes: concrete first (most specific), abstract fallback.
        concrete = [(pid, p) for pid, p in self.patterns.items()
                    if not p.deprecated and not p.is_abstract]
        abstract = [(pid, p) for pid, p in self.patterns.items()
                    if not p.deprecated and p.is_abstract]
        pid = None  # init: if no patterns (empty-seed schema), loop body never runs
        for pat_id, pat in concrete + abstract:
            pid = self._try_match(pat, he, instance)
            if pid:
                break
        if pid is None:
            return False, "no-matching-meta-pattern"
        return True, pid

    def _try_match(self, pat: MetaHyperedgePattern, he: Hyperedge,
                   instance: InstanceHypergraph) -> str | None:
        """Try matching one hyperedge against one pattern. Returns pat.pattern_id
        if it matches (role + type + qualifier), else None. Factored out of
        validate() so the concrete-then-abstract two-pass loop can reuse it."""
        slots = pat.role_slots
        has_repeat = any(s.get("repeatable") for s in slots)
        if has_repeat:
            ok, role_map = _match_variadic_set(slots, he.node_roles)
            if not ok:
                return None
        else:
            if len(he.node_roles) != len(slots):
                return None
            if not all(rs.get("role") == r for rs, r in zip(slots, he.node_roles)):
                return None
            role_map = list(range(len(slots)))  # identity
        # type compatibility (subtype-aware)
        if len(role_map) != len(he.node_ids):
            return None
        for node_idx, slot_idx in enumerate(role_map):
            if slot_idx >= len(slots) or node_idx >= len(he.node_ids):
                return None
            rs = slots[slot_idx]
            node = instance.nodes.get(he.node_ids[node_idx])
            if not node:
                return None
            req_type = rs.get("type")
            if req_type and not any(self.is_subtype(l, req_type) for l in node.labels):
                return None
        # qualifier check (REDESIGN v2): a qualifier key must be in the
        # pattern's allowed_qualifiers set (pattern-level control — the
        # pattern declares what qualifiers it carries). The GLOBAL registry
        # is no longer a hard gate — new keys can be dynamically registered
        # by the evolution probe (add_qualifier op), so the schema isn't
        # locked to a fixed key set. For KNOWN enum keys, the value must
        # still be in the enum (治发散: LLM can't fill free-text where an
        # enum exists); for new/free keys, any non-empty value is accepted.
        if pat.allowed_qualifiers:
            if not all(q in pat.allowed_qualifiers for q in he.qualifiers.keys()):
                return None
            # enum-value check only for keys registered with an enum
            if not all(qualifier_value_ok(q, v) for q, v in he.qualifiers.items()
                       if q in QUALIFIER_REGISTRY and QUALIFIER_REGISTRY[q][0] == "enum"):
                return None
        # if pattern has no allowed_qualifiers set, accept any key (legacy/
        # seed patterns that didn't declare) — controlled at probe level.
        return pat.pattern_id

    def add_meta_node(self, type_id: str, description: str, evidence: str = "",
                      paper_id: str = "") -> str:
        if type_id in self.meta_nodes:
            return self.version
        self.meta_nodes[type_id] = MetaNode(type_id=type_id, description=description)
        self.version = self._bump()
        return self.version

    def add_pattern(self, pattern: MetaHyperedgePattern, evidence: str = "",
                    paper_id: str = "") -> str:
        # Case-normalize the pattern_id to the seed's lowercase snake_case form.
        # deepseek emits BOTH lower and UPPER names run-to-run, creating
        # case-duplicate patterns (PROPERTY_DEPENDENCY_ON_PACKING_FRACTION vs
        # property_dependency_on_packing_fraction) the merge gate catches
        # unreliably (id-name embedding cosine is noisy). Folding at the source
        # prevents the dup rather than detecting it after.
        pattern.pattern_id = pattern.pattern_id.lower()
        if pattern.pattern_id in self.patterns:
            return self.version
        # REDESIGN v2: families are now GROWABLE, not locked to the 6 seed
        # families. A pattern may declare ANY family; if it's new, this
        # pattern becomes the family's root (family_roots auto-register).
        # Divergence is NOT prevented by locking families — it's prevented by
        # the conservative gate (cross_node>=2, only recurring gaps add) +
        # merge/retire (dedup + cleanup). Locking families was over-restrictive:
        # it forced relations not in the 6 physical families (causal/temporal/
        # uncertainty/etc) into wrong slots or dropped them — real info loss.
        # The 6 seed families are a STARTING skeleton, not a ceiling.
        self.patterns[pattern.pattern_id] = pattern
        # TAXONOMY COMPLETENESS: attach the new pattern IS-A its family root
        # so every pattern lives in the IS-A tree (no orphan leaves). A new
        # family's first pattern auto-becomes the root.
        root = self.family_roots.get(pattern.family) if pattern.family else None
        if root and root != pattern.pattern_id and root in self.patterns:
            self.meta_edges.append(MetaEdge(
                src=pattern.pattern_id, dst=root, relation="subclass_of",
                props={"paper_id": paper_id, "evidence": evidence, "via": "family_root"}))
        elif pattern.family and not root:
            # first pattern of a family (seed OR newly-grown) becomes its root
            self.family_roots[pattern.family] = pattern.pattern_id
        self.version = self._bump()
        return self.version

    def add_subclass(self, sub: str, sup: str, evidence: str = "", paper_id: str = "") -> str:
        # avoid cycles
        if self.is_subtype(sup, sub):
            return self.version
        self.meta_edges.append(MetaEdge(src=sub, dst=sup, relation="subclass_of"))
        self.version = self._bump()
        return self.version

    def split_meta_node(self, type_id: str, new_sub: str, evidence: str = "",
                        paper_id: str = "") -> str:
        """Split: new_sub becomes a subclass of type_id (instances matching new_sub
        will be re-typed)."""
        if new_sub in self.meta_nodes:
            return self.version
        self.meta_nodes[new_sub] = MetaNode(type_id=new_sub, description=f"split from {type_id}")
        self.meta_edges.append(MetaEdge(src=new_sub, dst=type_id, relation="subclass_of"))
        self.version = self._bump()
        return self.version

    def merge_meta_nodes(self, a: str, b: str, into: str = None, evidence: str = "",
                          paper_id: str = "") -> str:
        """Merge two meta-nodes: make b subclass of a (or both into a new `into`),
        re-point meta-edges."""
        keep = into or a
        if keep not in self.meta_nodes:
            self.meta_nodes[keep] = MetaNode(type_id=keep, description=f"merge of {a},{b}")
        other = b if keep == a else a
        self.meta_edges.append(MetaEdge(src=other, dst=keep, relation="subclass_of"))
        self.version = self._bump()
        return self.version

    # ---- pattern-level repair (split / merge / retire) ----
    # These are the headline operations. split_pattern is the ONLY one not done
    # by DIAL-KG (DIAL-KG does merge/retire on triplets; pattern-level split on
    # a hypergraph schema is the novel contribution — see ISSUES-AND-PRIORITY.md).
    # merge/retire are reimplemented on the hypergraph setting, cited as prior.

    def split_pattern(self, pattern_id: str, sub_patterns: list[MetaHyperedgePattern],
                      evidence: str = "", paper_id: str = "") -> str | None:
        """Split an over-wide pattern into >=2 sub-patterns.

        The sub-patterns INHERIT the parent's role_slots (structure unchanged —
        what splits is the SEMANTIC boundary, detected by instance clustering),
        each gets a distinguishing description + allowed_qualifiers. The parent
        becomes ABSTRACT (is_abstract=True) — it is KEPT as a generalization in
        a pattern-level IS-A taxonomy, NOT deprecated. A subclass_of meta-edge
        records each child IS-A parent (a queryable ontology, not just a lineage
        log). Abstract parents are still VALIDATION FALLBACK (an edge no concrete
        child matches falls back to the parent) and RENDERED as tree roots in
        to_prompt. Historical hyperedges of the parent (re-attributed to children
        by run_split) keep resolving via the children.

        Returns new meta version, or None if the split is invalid (parent
        missing, <2 sub-patterns, or a sub reuses an existing pattern_id).
        """
        if pattern_id not in self.patterns:
            return None
        if len(sub_patterns) < 2:
            return None
        parent = self.patterns[pattern_id]
        # RECURSIVE-SPLIT GUARD: an already-abstract parent has no direct
        # instances (they live in its children) — splitting it again is a
        # no-op that would orphan its existing children's subclass_of edges.
        # To refine further, split a CHILD (a concrete sub-pattern that grew
        # over-wide), not the abstract parent. Reject here so the caller can
        # re-target the split at a child.
        if parent.is_abstract:
            return None
        # each sub must inherit the parent's role-structure (split is semantic,
        # not structural — a structural split would be add_pattern, not split)
        parent_sig = _role_sig(parent.role_slots)
        for sp in sub_patterns:
            if sp.pattern_id in self.patterns:
                return None  # would clobber an existing pattern
            if _role_sig(sp.role_slots) != parent_sig:
                return None  # sub-pattern must keep the parent's structure
            sp.split_from = pattern_id
            # inherit the parent's top-level family so the bounded-op gate does
            # not reject sub-patterns (they are specializations WITHIN a family,
            # the legitimate growth path the gate must NOT block).
            if not sp.family:
                sp.family = parent.family
            self.patterns[sp.pattern_id] = sp
        # TAXONOMY (Path C "real ontology"): the parent becomes an abstract
        # generalization, NOT deprecated. It stays queryable (subclass_of edges)
        # + rendered + a validation fallback. Children are IS-A subclasses.
        parent.is_abstract = True
        # refresh the now-abstract parent's description so the taxonomy root
        # reflects its specialization (not the stale pre-split description).
        kid_names = ", ".join(sp.pattern_id for sp in sub_patterns)
        parent.description = (f"{parent.description} [abstract generalization, "
                              f"specialized into: {kid_names}]")
        # REDESIGN v2 step 4: children INHERIT the parent's pattern_dependency
        # edges (topology not lost on split). If parent depends_on X, children
        # also depend_on X (they're specializations of the same relation).
        parent_deps = [e for e in self.meta_edges
                      if e.relation in ("depends_on", "constrains", "composes")
                      and (e.src == pattern_id or e.dst == pattern_id)]
        for sp in sub_patterns:
            for e in parent_deps:
                if e.src == pattern_id:
                    self.meta_edges.append(MetaEdge(
                        src=sp.pattern_id, dst=e.dst, relation=e.relation,
                        props={"paper_id": paper_id, "evidence": evidence, "inherited_from": pattern_id}))
                elif e.dst == pattern_id:
                    self.meta_edges.append(MetaEdge(
                        src=e.src, dst=sp.pattern_id, relation=e.relation,
                        props={"paper_id": paper_id, "evidence": evidence, "inherited_to": pattern_id}))
        # subclass_of edge: each child IS-A parent (the ontology relation)
        for sp in sub_patterns:
            self.meta_edges.append(MetaEdge(
                src=sp.pattern_id, dst=pattern_id, relation="subclass_of",
                props={"paper_id": paper_id, "evidence": evidence}))
        self.version = self._bump()
        return self.version

    # ---- REDESIGN v2 step 2: pattern-level dependency/constraint topology ----
    # These give the schema layer REAL topology beyond the IS-A tree (what
    # DIAL-KG's flat schema lacks). The value is schema constraint-violation
    # detection (a DIAL-KG-can't-do capability, quantifiable).

    def add_pattern_dependency(self, dependent: str, depended_on: str,
                               rel: str = "depends_on", evidence: str = "",
                               paper_id: str = "") -> str | None:
        """Record a pattern-level dependency/constraint edge (REDESIGN v2).
        rel in {depends_on, constrains, composes}. E.g. a constitutive_law
        edge that references a parameter defined by a `defines` edge =>
        constitutive_law depends_on defines. This is inferred from instances
        (graph reachability: pattern A's node appears in pattern B's edge
        where B is a definition-family pattern) — deterministic, not LLM."""
        if dependent not in self.patterns or depended_on not in self.patterns:
            return None
        if dependent == depended_on:
            return None
        # avoid duplicate
        for e in self.meta_edges:
            if (e.src == dependent and e.dst == depended_on
                    and e.relation == rel):
                return self.version
        self.meta_edges.append(MetaEdge(
            src=dependent, dst=depended_on, relation=rel,
            props={"paper_id": paper_id, "evidence": evidence}))
        self.version = self._bump()
        return self.version

    def pattern_dependents(self, pattern_id: str, rel: str = "depends_on") -> list[str]:
        """Patterns that depend_on / constrains / composes the given pattern."""
        return [e.src for e in self.meta_edges
                if e.relation == rel and e.dst == pattern_id]

    def pattern_dependencies(self, pattern_id: str, rel: str = "depends_on") -> list[str]:
        """Patterns the given pattern depends on / constrains / composes."""
        return [e.dst for e in self.meta_edges
                if e.relation == rel and e.src == pattern_id]

    def detect_constraint_violations(self, instance: "InstanceHypergraph",
                                     domain: str = "") -> list[dict]:
        """Schema constraint-violation detection (REDESIGN v2 — the富拓扑's
        real value, DIAL-KG can't do this). Deterministic (graph reachability,
        no LLM). Independent of depends_on edges (detects the gap directly).

        NARROWED (after real-paper validation): only NUMERIC nodes (parameters/
        constants/values) referenced by non-definition edges must be defined —
        generic PROPERTY terms (stress/shear_rate) are universal physics
        vocabulary that don't need per-paper definition. The first run flagged
        71 violations almost all false-positives (stress/shear_rate flagged as
        undefined = wrong). Only NUMERIC nodes (a paper-specific parameter like
        a friction coefficient μ_s) referenced-but-undefined is a real schema
        gap (the constitutive law uses a constant the schema never defined).

        REFINED (2026-08-14): exclude single-letter universal physics symbols
        (d grain diameter / ρ density / g gravity / I inertial number /
        σ stress / τ shear stress / P pressure / v velocity / h depth /
        φ friction angle / θ angle) — these are field-wide vocabulary, not
        paper-specific constants. Real violations are paper-specific constants
        (e.g. μ_s base friction, b slope, I_0 scaling) the schema must define.

        DOMAIN-GATED (2026-08-15, anti-overfit): the universal-symbol whitelist
        is granular-flow vocabulary (μ/d/I/σ/τ...). In non-physics domains 'n'
        (sample size), 'k' (class count), 'T' (target) are NOT universal — they
        are paper-specific params the schema SHOULD require defined. Suppressing
        those would hide real violations. The whitelist runs ONLY when domain
        starts with 'granular'; other domains get NO whitelist (every NUMERIC
        referenced-but-undefined is a violation). Domain passed by caller."""
        _GRANULAR_UNIVERSAL = {
            "d", "ρ", "ρ_s", "g", "i", "σ", "τ", "p", "v", "h", "φ", "θ",
            "γ", "γ̇", "μ", "n", "e", "k", "λ", "l", "t", "x", "y", "z",
            # also full-word forms (extractor may emit surface as word)
            "diameter", "density", "gravity", "stress", "shear stress",
            "pressure", "velocity", "depth", "angle", "time",
        }
        _is_granular = (domain or "").lower().startswith("granular")
        def _is_universal(surface: str) -> bool:
            if not _is_granular:
                return False  # no whitelist in non-physics domains
            s = surface.strip().lower()
            if s in _GRANULAR_UNIVERSAL:
                return True
            # multi-char with subscript (μ_s, I_0, b_1) -> paper-specific, NOT universal
            return False

        violations = []
        # build: set of node surfaces that ARE defined (in definition-family edges)
        defined_surfaces: set[str] = set()
        for he in instance.hyperedges.values():
            pat = self.patterns.get(he.pattern_type)
            if pat and pat.family == "definition":
                for nid in he.node_ids:
                    n = instance.nodes.get(nid)
                    if n:
                        defined_surfaces.add(n.surface)
        # check non-definition edges: their NUMERIC nodes should be defined
        for he in instance.hyperedges.values():
            pat = self.patterns.get(he.pattern_type)
            if not pat or pat.family == "definition":
                continue
            for nid in he.node_ids:
                n = instance.nodes.get(nid)
                if not n:
                    continue
                # only NUMERIC nodes (parameters/constants) need definition;
                # generic PROPERTY terms are universal vocab, not violations
                if "NUMERIC" not in n.labels:
                    continue
                if _is_universal(n.surface):
                    continue  # d/ρ/g etc are field vocabulary, not a schema gap
                if n.surface not in defined_surfaces:
                    violations.append({
                        "violation": "referenced_undefined_numeric",
                        "pattern": he.pattern_type,
                        "node_surface": n.surface,
                        "edge_eid": he.eid,
                        "evidence": he.evidence_span[:80]})
        return violations

    def rename_pattern(self, old_id: str, new_id: str,
                       evidence: str = "", paper_id: str = "") -> str | None:
        """Rename a pattern (the 5th bounded op, ANNEAL/SCION edit type).
        Re-keys the pattern + rewrites ALL references: subclass_of /
        merged_into meta-edges, the renamed pattern's split_from, and the
        caller's instance-hyperedge pattern_type fields (re-attribution done
        by the caller via the returned old->new mapping).

        Returns new version, or None if old_id missing / new_id taken /
        new_id invalid. Case-folds new_id to the seed's lowercase form
        (consistency with add_pattern / split / merge naming)."""
        if old_id not in self.patterns:
            return None
        new_id = str(new_id).strip().lower()
        if not new_id or new_id.replace("_", "").isalnum() is False:
            return None
        if new_id == old_id:
            return self.version
        if new_id in self.patterns:
            return None  # would clobber
        pat = self.patterns.pop(old_id)
        pat.pattern_id = new_id
        self.patterns[new_id] = pat
        # rewrite meta-edge references (src/dst both)
        for e in self.meta_edges:
            if e.src == old_id:
                e.src = new_id
            if e.dst == old_id:
                e.dst = new_id
        # rewrite split_from lineage pointing at the renamed pattern
        for p in self.patterns.values():
            if p.split_from == old_id:
                p.split_from = new_id
        # record the rename for provenance
        self.meta_edges.append(MetaEdge(
            src=old_id, dst=new_id, relation="renamed_to",
            props={"paper_id": paper_id, "evidence": evidence}))
        self.version = self._bump()
        return self.version

    def pattern_subclasses(self, pattern_id: str) -> list[str]:
        """Direct + transitive sub-patterns of an abstract parent (the
        pattern-level IS-A taxonomy). Returns the descendant pattern_ids."""
        direct = [e.src for e in self.meta_edges
                  if e.relation == "subclass_of" and e.dst == pattern_id]
        seen = set(direct)
        stack = list(direct)
        while stack:
            t = stack.pop()
            for e in self.meta_edges:
                if e.relation == "subclass_of" and e.dst == t and e.src not in seen:
                    seen.add(e.src); stack.append(e.src)
        return list(seen)

    def merge_patterns(self, pattern_ids: list[str], into: str,
                       evidence: str = "", paper_id: str = "") -> str | None:
        """Merge >=2 near-duplicate patterns into one (DIAL-KG cross-batch
        canonicalization, reimplemented on the hypergraph schema). The `into`
        pattern is kept (or created if it's a new id); the others are
        deprecated. Historical hyperedges are re-attributed by the caller
        (retrace) — here we only mark + record lineage.

        TAXONOMY-AWARE (Path C): if a merged-away pattern is an ABSTRACT parent
        (has IS-A children), its children's subclass_of edges are RE-PARENTED
        to the survivor — the taxonomy is preserved, not orphaned. An abstract
        parent merged into a concrete survivor makes the survivor abstract
        (it now generalizes the reparented children). Mixing an abstract parent
        with a concrete leaf in a merge is rejected (a generalization and a
        leaf are not duplicates — that's a taxonomy relation, not a merge)."""
        if len(pattern_ids) < 2:
            return None
        # taxonomy guard: refuse to merge an abstract parent with a concrete
        # leaf (different ontological rank -> not a duplicate, a misfire).
        abstractions = [pid for pid in pattern_ids if pid in self.patterns
                        and self.patterns[pid].is_abstract]
        concretes = [pid for pid in pattern_ids if pid in self.patterns
                     and not self.patterns[pid].is_abstract]
        if abstractions and concretes:
            return None  # don't merge a generalization with a leaf
        # all but `into` get deprecated; `into` must be one of them or new
        keepers = pattern_ids if into in pattern_ids else pattern_ids + [into]
        if into not in self.patterns:
            # merge into a fresh pattern id: clone structure from the first
            donor = self.patterns.get(pattern_ids[0])
            if not donor:
                return None
            self.patterns[into] = MetaHyperedgePattern(
                pattern_id=into, description=f"merge of {','.join(pattern_ids)}",
                role_slots=[dict(s) for s in donor.role_slots],
                allowed_qualifiers=list(donor.allowed_qualifiers),
                split_from=f"merge:{','.join(pattern_ids)}",
                family=donor.family)
        # if the survivor absorbs an abstract parent, it becomes abstract too
        # (it now generalizes the reparented children).
        if any(self.patterns.get(k) and self.patterns[k].is_abstract
               for k in keepers if k != into):
            self.patterns[into].is_abstract = True
        for pid in keepers:
            if pid == into:
                continue
            if pid in self.patterns:
                # RE-PARENT children of a merged-away abstract parent to the
                # survivor (preserve the taxonomy, don't orphan the children).
                if self.patterns[pid].is_abstract:
                    for e in list(self.meta_edges):
                        if e.relation == "subclass_of" and e.dst == pid:
                            e.dst = into
                self.patterns[pid].deprecated = True
            self.meta_edges.append(MetaEdge(
                src=pid, dst=into, relation="merged_into",
                props={"paper_id": paper_id, "evidence": evidence}))
        self.version = self._bump()
        return self.version

    def retire_pattern(self, pattern_id: str, evidence: str = "",
                       paper_id: str = "") -> str | None:
        """Soft-deprecate a pattern (DIAL-KG retire). Kept for provenance;
        to_prompt() stops advertising it to the extractor so it won't be
        re-emitted."""
        if pattern_id not in self.patterns:
            return None
        self.patterns[pattern_id].deprecated = True
        self.meta_edges.append(MetaEdge(
            src=pattern_id, dst=pattern_id, relation="retired",
            props={"paper_id": paper_id, "evidence": evidence}))
        self.version = self._bump()
        return self.version

    def active_patterns(self) -> dict[str, MetaHyperedgePattern]:
        """Patterns not deprecated — what to_prompt advertises + what
        match_pattern considers (deprecated patterns only match historical
        edges for re-attribution, not new extraction)."""
        return {pid: p for pid, p in self.patterns.items() if not p.deprecated}

    # ---- persistence (full-field save/load, Path C "self-evolution must be
    # recoverable across sessions — the evolved schema is the asset) ----
    def to_dict(self) -> dict:
        """Full-field serialization of the meta-hypergraph (not the summary
        dump that lost is_abstract/family/deprecated/family_roots). Round-
        trippable via from_dict. Use this for save_meta so an evolved schema
        can be LOADED back and incrementally evolved further — closing the
        'schema gone after the run' gap (the user's earliest complaint)."""
        return {
            "version": self.version,
            "meta_nodes": {tid: {"type_id": n.type_id, "description": n.description,
                                 "is_abstract": n.is_abstract}
                           for tid, n in self.meta_nodes.items()},
            "patterns": {pid: {"pattern_id": p.pattern_id, "description": p.description,
                               "role_slots": [dict(s) for s in p.role_slots],
                               "allowed_qualifiers": list(p.allowed_qualifiers),
                               "deprecated": p.deprecated, "is_abstract": p.is_abstract,
                               "split_from": p.split_from, "family": p.family,
                               "semantic_boundary": p.semantic_boundary}
                         for pid, p in self.patterns.items()},
            "meta_edges": [{"src": e.src, "dst": e.dst, "relation": e.relation,
                            "props": dict(e.props)} for e in self.meta_edges],
            "family_roots": dict(self.family_roots),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MetaHypergraph":
        """Rebuild a meta-hypergraph from to_dict output. Restores ALL fields
        (is_abstract, family, deprecated, split_from, family_roots, meta_edges)
        so the loaded schema behaves identically to the in-memory one —
        validate/to_prompt/evolution all work on a loaded schema."""
        m = cls()
        m.version = d.get("version", "0.1")
        for tid, n in d.get("meta_nodes", {}).items():
            m.meta_nodes[tid] = MetaNode(type_id=tid, description=n.get("description", ""),
                                         is_abstract=n.get("is_abstract", False))
        for pid, p in d.get("patterns", {}).items():
            m.patterns[pid] = MetaHyperedgePattern(
                pattern_id=p.get("pattern_id", pid), description=p.get("description", ""),
                role_slots=p.get("role_slots", []), allowed_qualifiers=p.get("allowed_qualifiers", []),
                deprecated=p.get("deprecated", False), is_abstract=p.get("is_abstract", False),
                split_from=p.get("split_from", ""), family=p.get("family", ""),
                semantic_boundary=p.get("semantic_boundary", ""))
        for e in d.get("meta_edges", []):
            m.meta_edges.append(MetaEdge(src=e.get("src",""), dst=e.get("dst",""),
                                         relation=e.get("relation",""), props=e.get("props", {})))
        m.family_roots = dict(d.get("family_roots", {}))
        return m

    def _bump(self) -> str:
        parts = self.version.split(".")
        try:
            return f"{parts[0]}.{int(parts[1]) + 1}" if len(parts) == 2 else "0.2"
        except ValueError:
            return "0.2"

    def to_prompt(self, compact: bool = False, include_topology: bool = True) -> str:
        """Render the meta-hypergraph as a prompt fragment for extraction.
        (Forward propagation: downstream nodes get the evolved schema via this.)

        Patterns are rendered as a TAXONOMY TREE: an abstract parent (split
        into sub-patterns) is shown as a root marked (abstract), its concrete
        children indented beneath. The extractor sees the IS-A hierarchy, not
        a flat list — so it knows constitutive_law is a generalization with
        concrete sub-kinds power_law/density_dependent/etc., and can pick the
        most specific one or fall back to the abstract parent.

        compact=True: omit per-pattern role_slots + qualifiers (the bulky
        part). MEASURED COST: compact dropped grounding ~0.89 -> ~0.79 across
        8 papers (the LLM loses qualifier/role detail, edges slightly weaken).
        So compact is NOT a free win — default False (full prompt). Only set
        True when schema > ~150 patterns and the 32k context is the binding
        constraint (accepting the grounding tradeoff). validate() is unaffected
        either way (it uses the in-memory pattern, not this string)."""
        types = ", ".join(self.meta_nodes.keys()) or "(none)"
        # type-level subclass edges (meta-node hierarchy)
        type_sub = [f"{e.src} subclass_of {e.dst}" for e in self.meta_edges
                    if e.relation == "subclass_of" and e.src in self.meta_nodes]
        type_sub_str = "; ".join(type_sub) if type_sub else "(none)"
        # build pattern taxonomy: parent -> [children]
        children: dict[str, list[str]] = {}
        for e in self.meta_edges:
            if e.relation == "subclass_of" and e.src in self.patterns and e.dst in self.patterns:
                children.setdefault(e.dst, []).append(e.src)
        is_child = {c for cs in children.values() for c in cs}
        # roots = non-deprecated patterns that are NOT a child of anyone
        roots = [pid for pid, p in self.patterns.items()
                 if not p.deprecated and pid not in is_child]

        def render_pat(pid: str, indent: int) -> list[str]:
            pat = self.patterns[pid]
            fam = f"<{pat.family}>" if pat.family else ""
            abs_tag = " (abstract)" if pat.is_abstract else ""
            pad = "  " * indent
            if compact:
                # omit role_slots + qualifiers (the bulky part); LLM picks by
                # pattern_id + description, knows conventions from the prompt.
                line = f"{pad}{pid}{fam}{abs_tag} — {pat.description}"
            else:
                slots = ", ".join(f"{s.get('role')}:{s.get('type','?')}" for s in pat.role_slots)
                quals = ",".join(pat.allowed_qualifiers) if pat.allowed_qualifiers else ""
                line = f"{pad}{pid}{fam}{abs_tag}({slots})[qualifiers:{quals}] — {pat.description}"
                if pat.semantic_boundary:
                    # render the boundary so the extractor sees WHEN to use this
                    # pattern vs its confusables (schema-in-context, 断点 fix).
                    line += f" [boundary: {pat.semantic_boundary}]"
            out = [line]
            for ch in sorted(children.get(pid, [])):
                if not self.patterns[ch].deprecated:
                    out.extend(render_pat(ch, indent + 1))
            return out

        pats_str = "\n".join(line for r in sorted(roots) for line in render_pat(r, 1)) if roots else "  (none)"
        # show ALL current families (seed 6 + any grown), so the probe sees
        # what already exists before proposing a new one.
        families_str = ", ".join(sorted(set(p.family for p in self.patterns.values() if p.family))) or "(none)"
        # REDESIGN v2 step 4: render the pattern-level TOPOLOGY edges
        # (depends_on / constrains / composes) so the schema's富拓扑 is visible
        # to the extractor, not just the IS-A tree. These are the edges that
        # make the schema layer a directed constrained hypergraph; without
        # rendering them the extractor never sees that e.g. a measure edge
        # is CONSTRAINED BY a law, so it can't use the constraint to guide
        # extraction. Only edges between NON-deprecated patterns are shown.
        topo = []
        for e in self.meta_edges:
            if e.relation not in ("depends_on", "constrains", "composes"):
                continue
            src_p = self.patterns.get(e.src)
            dst_p = self.patterns.get(e.dst)
            if not src_p or not dst_p or src_p.deprecated or dst_p.deprecated:
                continue
            topo.append(f"  {e.src} {e.relation} {e.dst}")
        topo_str = "\n".join(topo) if topo else "  (none)"
        # include_topology (REDESIGN v2): the pattern-level topology edges can
        # be omitted from the prompt for an ablation — does rendering them
        # improve extraction quality? Default True (show富拓扑). validate() is
        # unaffected either way (uses the in-memory pattern, not this string).
        topo_section = ""
        if include_topology:
            topo_section = (f"Pattern-level topology (directed constrained hypergraph — A depends_on/constrains/composes B):\n{topo_str}\n")
        return (f"Meta-Hypergraph (schema v{self.version}):\n"
                f"Node types: {types}\n"
                f"Node subclass hierarchy: {type_sub_str}\n"
                f"Seed families (growable — specialize within, or propose a new one if none fits): {families_str}\n"
                f"Hyperedge patterns (taxonomy — indented = IS-A specialization of parent):\n{pats_str}\n"
                f"{topo_section}")


def seed_meta_hypergraph() -> MetaHypergraph:
    """Seed the meta-hypergraph with the Path C skeleton: one pattern per
    top-level CONTENT FAMILY (TOP_LEVEL_FAMILIES), so every family has a seed
    representative the agent specializes WITHIN (bounded op). Minimal per-
    family so evolution has room, but NO family is missing — a missing family
    would force the LLM to propose a new top-level dimension (which the
    bounded-op gate now rejects)."""
    m = MetaHypergraph()
    # common root so cross-type n-ary edges pass validation (METHOD whole +
    # PARAMETER components, etc.) — mirrors seed_meta_hypergraph_general's THING root.
    m.meta_nodes["THING"] = MetaNode(type_id="THING", description="root: any node type is-a THING")
    # node types — fine-grained so the rich topology (method/parameter/phenomenon
    # relations) can be read directly from instance hyperedges without co-occurrence
    # guessing. METHOD/PHENOMENON/PARAMETER added so extract labels them distinctly
    # (was all PROPERTY, losing the method↔parameter↔phenomenon structure).
    for t, d in [("MATERIAL", "a granular material or substance"),
                 ("METHOD", "a named modeling approach/theory/law/model (e.g. μ(I) rheology, kinetic theory, NGF, DEM, RFT)"),
                 ("PARAMETER", "a named physical quantity/constant/coefficient in equations (e.g. μ, I, d, P, τ, g, ρ_s, T, A, ξ, μ_s, b)"),
                 ("PHENOMENON", "a physical effect/behavior methods capture (e.g. nonlocal creep, thin-body strengthening, secondary rheology, segregation, clogging)"),
                 ("PROPERTY", "a generic physical property/quantity not fitting the above (kept for residual cases)"),
                 ("NUMERIC", "a numeric value / measured number"),
                 ("REGIME", "a flow regime (dense/quasi-static/inertial)")]:
        m.meta_nodes[t] = MetaNode(type_id=t, description=d)
        m.meta_edges.append(MetaEdge(src=t, dst="THING", relation="subclass_of"))
    T = "THING"  # permissive slot type (any labeled node is-a THING)
    # one seed pattern per top-level family. method/evidence_strength added
    # as allowed qualifiers on the physics families (constitutive_law /
    # dependency / measure) — the new context dimensions (P-E2 attribution +
    # P-E4 discourse filtering). applies_in_regime + dependency_type kept
    # (load-bearing: regime retrieval + the split trigger enum).
    m.patterns["constitutive_law"] = MetaHyperedgePattern(
        pattern_id="constitutive_law", family="constitutive_law",
        description="a constitutive law relating output quantity/quantities to >=1 input quantity (n-ary)",
        role_slots=[{"role": "output", "type": T, "repeatable": True}, {"role": "input", "type": T, "repeatable": True}, {"role": "parameter", "type": T, "repeatable": True}, {"role": "coefficient", "type": T, "repeatable": True}, {"role": "exponent", "type": T, "repeatable": True}],
        allowed_qualifiers=["applies_in_regime", "function_form", "parameters", "method", "evidence_strength", "cited_from"],
        semantic_boundary="a FORMAL/QUANTITATIVE law output=f(inputs+params), stated as a formula or explicit functional relation. NOT a prose 'X depends on Y' (that is influences), NOT a definition (defines), NOT a measurement setup (measures).")
    m.patterns["influences"] = MetaHyperedgePattern(
        pattern_id="influences", family="dependency",
        description="one quantity influences / depends on >=1 target quantity (n-ary, general dependence)",
        role_slots=[{"role": "source", "type": T, "repeatable": True}, {"role": "target", "type": T, "repeatable": True}, {"role": "cause", "type": T, "repeatable": True}, {"role": "effect", "type": T, "repeatable": True}],
        allowed_qualifiers=["dependency_type", "applies_in_regime", "method", "evidence_strength", "cited_from"],
        semantic_boundary="X functionally depends on / varies with Y (a genuine causal or scaling dependence). NOT a co-listing of parallel items (composed_of), NOT a classification (defines), NOT 'we tested X' (measures), NOT 'X is part of Y' (composed_of).")
    m.patterns["defines"] = MetaHyperedgePattern(
        pattern_id="defines", family="definition",
        description="one entity is defined-as / identified-with / named-by another (definitional identity, n-ary)",
        role_slots=[{"role": "subject", "type": T, "repeatable": True}, {"role": "definition", "type": T, "repeatable": True}, {"role": "object", "type": T, "repeatable": True}],
        allowed_qualifiers=["relation_type", "method", "evidence_strength", "cited_from"],
        semantic_boundary="a DEFINITIONAL identity: X is defined as / is / denotes / is named / is classified into Y. NOT a structural part-of (composed_of), NOT a functional dependence (influences). 'X is analogous to Y' is a CLAIM not a definition.")
    m.patterns["composed_of"] = MetaHyperedgePattern(
        pattern_id="composed_of", family="composition",
        description="one whole is composed-of / part-of >=1 component (n-ary composition)",
        role_slots=[{"role": "whole", "type": T, "repeatable": True}, {"role": "component", "type": T, "repeatable": True}],
        allowed_qualifiers=["relation_type", "method", "evidence_strength", "cited_from"],
        semantic_boundary="STRUCTURAL composition: parts that make up a whole (model A = B + C). NOT classification (defines), NOT 'X has control params' (influences), NOT experimental setup description, NOT 'we tested X by doing Y'.")
    m.patterns["measures"] = MetaHyperedgePattern(
        pattern_id="measures", family="measure",
        description="a quantity is measured by / characterizes >=1 measure",
        role_slots=[{"role": "object", "type": T}, {"role": "instrument", "type": T, "repeatable": True}],
        allowed_qualifiers=["condition", "applies_in_regime", "method", "evidence_strength", "cited_from"],
        semantic_boundary="a measurement/evaluation relation: X is measured/evaluated by instrument/method Y. NOT a functional law (constitutive_law), NOT a dependence (influences).")
    m.patterns["claim_relation"] = MetaHyperedgePattern(
        pattern_id="claim_relation", family="claim",
        description="a discourse relation between >=2 claims/approaches (supports/contrasts/extends/...)",
        role_slots=[{"role": "from", "type": T, "repeatable": True}, {"role": "to", "type": T, "repeatable": True}],
        allowed_qualifiers=["relation_type", "applies_in_regime", "method", "evidence_strength", "cited_from"],
        semantic_boundary="a DISCOURSE relation between claims/findings (supports/contrasts/outperforms). NOT a method-to-method development (extends/improves/compares — those are evolution), NOT a measurement (measures).")
    # evolution family: cross-METHOD (or PHENOMENON) relations STATED in the
    # text — extends/improves/compares/replaces/adapts/background. These are
    # the relations the text explicitly states (NOT inferred — inference is a
    # downstream agent's job). Each connects 2+ METHODS/PHENOMENA; the relation
    # verb IS the pattern_id. Extracted when text says "X extends Y" etc.
    _evo_family = "evolution"
    _evo_slots = [{"role": "from", "type": T, "repeatable": True},
                  {"role": "to", "type": T, "repeatable": True}]
    _evo_qual = ["relation_type", "cited_from", "evidence_strength", "method"]
    # semantic_boundary for each evolution verb (M3 fix — these 6 were missing and
    # P0 showed improves-vs-compares is the worst type-confusion source). Each
    # boundary says WHEN this verb applies vs its confusables, so the extractor
    # + verifier pick the right evolution verb, not a discourse claim_relation.
    _evo_boundary = {
        "extends": "X GENERALIZES Y (X extends Y's scope/regime/params; X is a direct "
                   "technical extension of Y). NOT improves (X need not be more accurate, "
                   "just broader), NOT background (X is a direct technical inheritance, "
                   "not a motivation citation), NOT claim_relation (this is method-to-method).",
        "improves": "X RESOLVES a limitation of Y (X is more accurate/applicable BECAUSE "
                    "it fixes something Y fails at; first-principles vs phenomenological = "
                    "improves). NOT compares (X is NOT a parallel different-mechanism "
                    "alternative; X builds on Y and is better), NOT extends (X is better "
                    "not just broader).",
        "compares": "A and B are PARALLEL different-mechanism alternatives with overlapping "
                    "scope (both model the same phenomenon, different assumptions, each "
                    "pros/cons). NOT improves (neither fixes the other's limitation), NOT "
                    "extends (neither generalizes the other).",
        "replaces": "X SUBSTITUTES Y outright (X is used INSTEAD of Y; Y is retired). "
                    "NOT improves (replaces = Y is gone, improves = Y still used + X better).",
        "adapts": "X PORTS Y to a NEW scenario/regime/domain Y wasn't designed for "
                  "(adaptation of Y's mechanism). NOT extends (adapts = new scenario, "
                  "extends = broader scope within same setting).",
        "background": "X cites Y as MOTIVATION/prior context only (Y inspired X but X is "
                      "NOT a direct technical extension/inheritance of Y). NOT extends "
                      "(no direct technical lineage), NOT improves (no limitation fix).",
    }
    for _eid, _desc in [
        ("extends", "X generalizes/extends Y (X METHOD/PHENOMENON -> Y)"),
        ("improves", "X improves Y's accuracy/applicability, resolving Y's limitation"),
        ("compares", "X is compared with Y"),
        ("replaces", "X replaces Y"),
        ("adapts", "X adapts Y to a new scenario"),
        ("background", "X uses Y as background/motivation"),
    ]:
        m.patterns[_eid] = MetaHyperedgePattern(
            pattern_id=_eid, family=_evo_family, description=_desc,
            role_slots=_evo_slots, allowed_qualifiers=_evo_qual,
            semantic_boundary=_evo_boundary[_eid])
    # each seed pattern is the root of its top-level family — add_pattern
    # attaches new same-family patterns IS-A this root.
    m.family_roots = {p.family: p.pattern_id for p in m.patterns.values()}
    return m


def seed_meta_hypergraph_general() -> MetaHypergraph:
    """General-domain seed (non-physics): same 6-family skeleton as the physics
    seed but with domain-agnostic node types (ENTITY/METHOD/RESULT/NUMERIC) and
    general-paper relation descriptions. Use for non-physics corpora (e.g.
    ResearchQA, SciERC) so the schema isn't biased toward physical quantities.

    Same family structure as seed_meta_hypergraph() — so the topology-inference
    logic (depends_on needs definition family; constrains needs constitutive_law;
    composes needs composition family) works identically. Only the type semantics
    change (general entities vs MATERIAL/PROPERTY/REGIME).

    ROLE VOCAB (DECISION-validation-rejects-all-general-edges): each pattern now
    declares the FULL role vocabulary the EXTRACT_HG_PROMPT encourages for its
    family (output/parameter/coefficient on laws; output/cause/effect on
    influences; object on defines; parameter on claims). Previously only 2 roles
    were declared, so every n-ary edge the LLM emitted with the encouraged extra
    roles was rejected by _match_variadic_set (role not in pattern) -> 0 edges.

    TYPE ROOT (same DECISION): general papers mix METHOD+ENTITY+RESULT in one
    relation. The 4 general types are independent with no shared supertype, so a
    METHOD whole with ENTITY components failed is_subtype -> rejected. THING is a
    common root and all 4 types subclass_of THING; slot types are THING so
    cross-type n-ary edges pass while node types stay informative."""
    m = MetaHypergraph()
    # common root so cross-type n-ary edges pass (METHOD whole + ENTITY components)
    m.meta_nodes["THING"] = MetaNode(type_id="THING", description="root: any node type is-a THING")
    # general academic node types (not physics-specific), all under THING
    for t, d in [("ENTITY", "a research entity/concept (model, method, dataset, task, etc.)"),
                 ("METHOD", "a method/technique/approach/model"),
                 ("RESULT", "a finding/metric/result"),
                 ("NUMERIC", "a numeric value (parameter, score, count)")]:
        m.meta_nodes[t] = MetaNode(type_id=t, description=d)
        m.meta_edges.append(MetaEdge(src=t, dst="THING", relation="subclass_of"))
    T = "THING"  # permissive slot type (any labeled node is-a THING)
    # same 6 families, general-paper descriptions, expanded role vocab, type THING
    m.patterns["constitutive_law"] = MetaHyperedgePattern(
        pattern_id="constitutive_law", family="constitutive_law",
        description="a quantitative/formal relation (output computed from inputs + parameters): loss, score formula, scaling law",
        role_slots=[{"role": "output", "type": T, "repeatable": True}, {"role": "input", "type": T, "repeatable": True},
                    {"role": "parameter", "type": T, "repeatable": True}, {"role": "coefficient", "type": T, "repeatable": True},
                    {"role": "exponent", "type": T, "repeatable": True}],
        allowed_qualifiers=["applies_in_regime", "function_form", "parameters", "method", "evidence_strength", "cited_from"])
    m.patterns["influences"] = MetaHyperedgePattern(
        pattern_id="influences", family="dependency",
        description="one concept/quantity influences / depends on / improves another (n-ary)",
        role_slots=[{"role": "source", "type": T, "repeatable": True}, {"role": "target", "type": T, "repeatable": True},
                    {"role": "output", "type": T, "repeatable": True}, {"role": "cause", "type": T, "repeatable": True},
                    {"role": "effect", "type": T, "repeatable": True}, {"role": "condition", "type": T, "repeatable": True}],
        allowed_qualifiers=["dependency_type", "applies_in_regime", "method", "evidence_strength", "cited_from"])
    m.patterns["defines"] = MetaHyperedgePattern(
        pattern_id="defines", family="definition",
        description="one entity is defined-as / identified-with / named-by another (definitional identity, n-ary)",
        role_slots=[{"role": "subject", "type": T, "repeatable": True}, {"role": "definition", "type": T, "repeatable": True},
                    {"role": "object", "type": T, "repeatable": True}],
        allowed_qualifiers=["relation_type", "method", "evidence_strength", "cited_from"])
    m.patterns["composed_of"] = MetaHyperedgePattern(
        pattern_id="composed_of", family="composition",
        description="one whole (model/pipeline/system) is composed-of / part-of >=1 component (n-ary)",
        role_slots=[{"role": "whole", "type": T, "repeatable": True}, {"role": "component", "type": T, "repeatable": True}],
        allowed_qualifiers=["relation_type", "method", "evidence_strength", "cited_from"])
    m.patterns["measures"] = MetaHyperedgePattern(
        pattern_id="measures", family="measure",
        description="a method/approach is used to evaluate/measure >=1 target (n-ary)",
        role_slots=[{"role": "object", "type": T, "repeatable": True}, {"role": "instrument", "type": T, "repeatable": True}],
        allowed_qualifiers=["condition", "applies_in_regime", "method", "evidence_strength", "cited_from"])
    m.patterns["claim_relation"] = MetaHyperedgePattern(
        pattern_id="claim_relation", family="claim",
        description="a discourse relation between >=2 claims/findings/approaches (supports/contrasts/outperforms/extends)",
        role_slots=[{"role": "from", "type": T, "repeatable": True}, {"role": "to", "type": T, "repeatable": True},
                    {"role": "parameter", "type": T, "repeatable": True}, {"role": "condition", "type": T, "repeatable": True}],
        allowed_qualifiers=["relation_type", "applies_in_regime", "method", "evidence_strength", "cited_from"])
    m.family_roots = {p.family: p.pattern_id for p in m.patterns.values()}
    return m
