"""A-box: cross-paper Concept HYPERGRAPH (the instance-layer knowledge graph).

Per DESIGN_full.md section 3 (A-box). The ConceptGraph is an n-ary hypergraph
(multi-endpoint edges, NOT pairwise-binary) — this is the innovation: laws
(I=γ̇d/√(P/ρ)) stay ONE n-ary hyperedge [law, I, γ̇, d, P, ρ], not 10
pairwise edges. Binary would lose the n-ary structure that is our selling point.

- Concept: concrete entity (method/phenomenon/parameter/regime/material/numeric)
  with surface_variants provenance (per-paper surface+evidence+year+section).
- ConceptHyperedge: n-ary edge (node_ids list + roles + kind) with provenance.
  kind = rich-topology kind (method_parameter/captures/composition/nary/
  law_parameter/method_regime) from infer_rich_topology_direct, NOT raw
  pattern_type. Emergence (P2) adds evolution kinds here.
- Cross-paper alignment: LLM batched for METHOD/PHENOMENON (noisy naming);
  PARAMETER/NUMERIC use SYMBOL matching (μ/I/d not surface — 'inertial number I'
  and 'I' merge by symbol). Symbol ambiguity (μ friction vs μ(I) law) via LLM.

T-box (schema/MetaHypergraph) defines types + patterns; A-box references them.
Emergence (P2) reads structure signals (temporal/citation/content) from this
hypergraph to surface evolution relation candidates.
"""
from __future__ import annotations
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class SurfaceVariant:
    """One paper's surface mention of a concept (provenance)."""
    surface: str
    paper_id: str
    evidence: str = ""
    year: str = ""
    section: str = ""


@dataclass
class Concept:
    """A concrete concept entity in the cross-paper hypergraph (A-box)."""
    concept_id: str
    type: str
    surface_variants: list[SurfaceVariant] = field(default_factory=list)
    source_papers: list[str] = field(default_factory=list)
    year_range: tuple[str, str] = ("", "")
    deprecated: bool = False
    canonical_name: str = ""
    symbol: str = ""  # for PARAMETER/NUMERIC: extracted symbol (μ/I/d/...) for alignment

    def add_variant(self, surface, paper_id, evidence="", year="", section=""):
        self.surface_variants.append(SurfaceVariant(surface, paper_id, evidence, year, section))
        if paper_id and paper_id not in self.source_papers:
            self.source_papers.append(paper_id)
        if year:
            years = [v.year for v in self.surface_variants if v.year]
            if years:
                self.year_range = (min(years), max(years))

    def surfaces(self):
        seen = []
        for v in self.surface_variants:
            if v.surface and v.surface not in seen:
                seen.append(v.surface)
        return seen

    def to_dict(self):
        d = asdict(self)
        d["year_range"] = list(self.year_range)
        return d


@dataclass
class ConceptHyperedge:
    """An n-ary edge in the concept hypergraph (A-box). Multi-endpoint.

    kind = rich-topology kind (from infer_rich_topology_direct) or evolution
    kind (added by P2 emergence). NOT raw pattern_type.
    """
    he_id: str
    node_ids: list[str] = field(default_factory=list)  # concept_ids (n-ary, order matters for roles)
    node_roles: list[str] = field(default_factory=list)
    kind: str = ""
    provenance: list[dict] = field(default_factory=list)  # [{paper_id, evidence, year, section}]
    emergence_signals: dict[str, Any] = field(default_factory=dict)  # P2 fills
    confidence: float = 0.0

    def to_dict(self):
        return asdict(self)


# greek + latin single-symbol tokens used in granular-flow equations
_SYMBOL_RE = re.compile(
    r'\b(μ_s|μ_d|μ_eff|μ|I|d|P|τ|g|ρ_s|ρ|ξ|ν|η|κ|T|A|B|b|ℓ|θ|ϕ|φ|γ|Γ|e|p|s|t|v|n|k|c|f)\b'
)
# γ̇ (gamma-dot, shear rate) has a combining mark that breaks \b word boundaries;
# granular flow uses γ̇ heavily — handle explicitly before the regex.
_GAMMA_DOT = "γ̇"

# normalization map for common surface-level symbol variants
_GREEK = {'mu': 'μ', 'rho': 'ρ', 'tau': 'τ', 'xi': 'ξ', 'nu': 'ν', 'eta': 'η',
          'kappa': 'κ', 'theta': 'θ', 'phi': 'φ', 'gamma': 'γ', 'Gamma': 'Γ', 'ell': 'ℓ'}

# law-name patterns: 'μ(I)'/'mu(I)'/'μ(I) rheology' is a LAW (METHOD), NOT the
# parameter μ. Symbol extraction must NOT return μ for these (would merge the
# law with the friction-coefficient parameter). Per DESIGN C3 disambiguation.
_LAW_NAME_RE = re.compile(r'\b(mu|μ)\s*\(\s*I\s*\)', re.I)


def _extract_symbol(surface: str) -> str:
    """Extract a parameter/numeric symbol token from a surface mention.
    e.g. 'inertial number I' -> 'I', 'friction coefficient μ' -> 'μ',
    'μ_s' -> 'μ_s', 'cooperativity length ξ' -> 'ξ', 'γ̇' -> 'γ̇'.
    Returns '' if none, OR if the surface is a law-name like 'μ(I) rheology'
    (law = METHOD, not the parameter μ — DESIGN C3 disambiguation, so the
    law does not merge with friction coefficient μ via symbol)."""
    s = surface.strip()
    # γ̇ first (combining mark defeats \b)
    if _GAMMA_DOT in s:
        return _GAMMA_DOT
    # law-name (μ(I)) — NOT a parameter symbol; caller should treat as METHOD
    # (returns '' so get_or_create falls back to surface, no symbol merge)
    if _LAW_NAME_RE.search(s):
        return ""
    # exact symbol match (handles greek letters and latin singles)
    m = _SYMBOL_RE.search(s)
    if m:
        return m.group(1)
    # spelled-out greek ('mu', 'rho'...) -> greek letter
    low = s.lower()
    for word, letter in _GREEK.items():
        if re.search(r'\b' + word + r'\b', low):
            return letter
    return ""


def _norm_surface(s: str) -> str:
    """normalize surface for first-pass exact-surface lookup."""
    return s.strip().lower()


class ConceptGraph:
    """Cross-paper n-ary concept hypergraph (A-box). Accumulates concepts +
    n-ary hyperedges across papers with semantic/symbol alignment."""

    def __init__(self):
        self.concepts: dict[str, Concept] = {}
        self.hyperedges: list[ConceptHyperedge] = []
        self._next_id = 0
        # first-pass lookup indices
        self._surface2concept: dict[str, str] = {}   # norm surface -> concept_id
        self._symbol2concept: dict[str, str] = {}    # symbol -> concept_id (PARAMETER/NUMERIC)

    def _new_id(self, type_: str) -> str:
        self._next_id += 1
        prefix = {"METHOD": "M", "PHENOMENON": "F", "PARAMETER": "P",
                  "REGIME": "R", "MATERIAL": "MAT", "NUMERIC": "N"}.get(type_, "C")
        return f"C{prefix}{self._next_id:04d}"

    def get_or_create(self, type_: str, surface: str, paper_id: str,
                      evidence: str = "", year: str = "",
                      section: str = "") -> Concept:
        """First-pass alignment:
        - PARAMETER/NUMERIC with extractable symbol → align by SYMBOL
          ('inertial number I' ~ 'I' merge by symbol 'I')
        - PARAMETER/NUMERIC with NO symbol (function-like 'V_surf(y)',
          'h_max', '|γ̇|') → fall back to surface-string (not 'always new')
        - other types → exact-surface lookup
        LLM semantic alignment (merge near-synonyms) in align_concepts, not here.
        """
        if type_ in ("PARAMETER", "NUMERIC"):
            sym = _extract_symbol(surface)
            if sym:  # symbol-based alignment
                if sym in self._symbol2concept:
                    cid = self._symbol2concept[sym]
                    self.concepts[cid].add_variant(surface, paper_id, evidence, year, section)
                    return self.concepts[cid]
                cid = self._new_id(type_)
                c = Concept(concept_id=cid, type=type_, symbol=sym)
                c.add_variant(surface, paper_id, evidence, year, section)
                self.concepts[cid] = c
                self._symbol2concept[sym] = cid
                return c
            # no symbol extractable → fall back to surface-string (avoids
            # creating a new concept every ingest for 'V_surf(y)'-like surfaces)
            key = _norm_surface(surface)
            cid = self._surface2concept.get(key)
            if cid and cid in self.concepts:
                self.concepts[cid].add_variant(surface, paper_id, evidence, year, section)
                return self.concepts[cid]
            cid = self._new_id(type_)
            c = Concept(concept_id=cid, type=type_, symbol="")
            c.add_variant(surface, paper_id, evidence, year, section)
            self.concepts[cid] = c
            self._surface2concept[key] = cid
            return c
        # non-symbol types: exact-surface lookup
        key = _norm_surface(surface)
        cid = self._surface2concept.get(key)
        if cid and cid in self.concepts:
            self.concepts[cid].add_variant(surface, paper_id, evidence, year, section)
            return self.concepts[cid]
        cid = self._new_id(type_)
        c = Concept(concept_id=cid, type=type_)
        c.add_variant(surface, paper_id, evidence, year, section)
        self.concepts[cid] = c
        self._surface2concept[key] = cid
        return c

    def add_hyperedge(self, node_ids: list[str], kind: str, roles: list[str] = None,
                      paper_id: str = "", evidence: str = "", year: str = "",
                      section: str = "") -> ConceptHyperedge:
        """Add an n-ary hyperedge. DEDUP by (frozenset(node_ids), kind):
        same nodes+kind → accumulate provenance (not duplicate). Skips empty/
        single-node 'edges' (not a relation). Self-loops (all same id) skipped."""
        node_ids = [n for n in node_ids if n]
        if len(node_ids) < 2:
            return None
        if len(set(node_ids)) == 1:
            return None  # self-loop
        key = (frozenset(node_ids), kind)
        for he in self.hyperedges:
            if (frozenset(he.node_ids), he.kind) == key:
                if paper_id:
                    he.provenance.append({"paper_id": paper_id, "evidence": evidence,
                                          "year": year, "section": section})
                return he
        he = ConceptHyperedge(he_id=f"H{self._next_he()}", node_ids=node_ids,
                               node_roles=roles or [], kind=kind,
                               provenance=[{"paper_id": paper_id, "evidence": evidence,
                                            "year": year, "section": section}]
                               if paper_id else [])
        self.hyperedges.append(he)
        return he

    _he_counter = 0

    def _next_he(self) -> int:
        ConceptGraph._he_counter += 1
        return ConceptGraph._he_counter

    def merge_concepts(self, into_id: str, merge_id: str) -> None:
        """Merge two Concepts (semantic near-synonyms, called by align_concepts).
        Redirects hyperedge node_ids, dedups resulting duplicate/self-loop edges."""
        if into_id == merge_id or merge_id not in self.concepts or into_id not in self.concepts:
            return
        keep = self.concepts[into_id]
        drop = self.concepts[merge_id]
        for v in drop.surface_variants:
            keep.add_variant(v.surface, v.paper_id, v.evidence, v.year, v.section)
        if drop.symbol and not keep.symbol:
            keep.symbol = drop.symbol
        # redirect hyperedge endpoints
        for he in self.hyperedges:
            he.node_ids = [into_id if n == merge_id else n for n in he.node_ids]
        # dedup: after redirect, edges with same (frozenset, kind) collapse; self-loops drop
        seen = {}
        new_edges = []
        for he in self.hyperedges:
            if len(set(he.node_ids)) < 2:
                continue  # self-loop after merge
            key = (frozenset(he.node_ids), he.kind)
            if key in seen:
                # merge provenance into existing
                seen[key].provenance.extend(he.provenance)
                continue
            seen[key] = he
            new_edges.append(he)
        self.hyperedges = new_edges
        # update indices
        for s in drop.surfaces():
            self._surface2concept[_norm_surface(s)] = into_id
        if drop.symbol:
            self._symbol2concept[drop.symbol] = into_id
        del self.concepts[merge_id]

    def concepts_by_type(self, type_: str):
        return [c for c in self.concepts.values() if c.type == type_ and not c.deprecated]

    def ingest_instance(self, inst, year: str = "") -> None:
        """Accumulate one paper's instance into the concept hypergraph.

        Does NOT use raw pattern_type. Runs consolidate_instance +
        infer_rich_topology_direct to get rich-topology edges (correct kind:
        method_parameter/captures/composition/nary/law_parameter/method_regime),
        then ingests those as n-ary ConceptHyperedges (no pairwise split).

        inst: InstanceHypergraph (or dict). year: for P2 time-order signals.
        """
        from granular_agent.hypergraph_evolution import (
            consolidate_instance, infer_rich_topology_direct)
        # accept dict or InstanceHypergraph
        is_obj = hasattr(inst, "nodes")
        nodes = inst.nodes if is_obj else inst["nodes"]
        paper_id = (getattr(inst, "paper_id", "") if is_obj else inst.get("paper_id", "")) or ""

        def _section(nid: str) -> str:
            return nid.split("_")[0] if "_" in nid else nid

        # 1. consolidate + infer rich-topology (correct kinds)
        consolidate_instance(inst)
        rich_edges = infer_rich_topology_direct(inst, paper_id=paper_id)
        # fallback: if all edges classify as "other" (param↔param only), the
        # paper would contribute ZERO concepts. Re-run including "other" so
        # the paper still ingests its labeled nodes (no silent skip).
        if not rich_edges:
            rich_edges = infer_rich_topology_direct(inst, paper_id=paper_id,
                                                     include_other=True)

        # 2. align each node mentioned in rich edges to a Concept (with
        # provenance: surface + paper_id + evidence + year + section — section
        # now carried per rich edge from infer_rich_topology_direct).
        nid2concept = {}
        for re_ in rich_edges:
            sec = re_.get("section", "")
            for nd in re_["nodes"]:
                surf = nd["surface"]
                labels = nd.get("labels", [])
                t = next((l for l in labels if l != "THING"), (labels[0] if labels else "PROPERTY"))
                c = self.get_or_create(t, surf, paper_id, re_.get("evidence", ""),
                                       year, sec)
                nid2concept[(t, surf)] = c.concept_id

        # 3. ingest each rich-topology edge as an n-ary ConceptHyperedge.
        # For evolution edges, preserve the SPECIFIC subtype (extends/improves/
        # compares/replaces/adapts/background) as kind — not generic 'evolution'
        # — so eval can match gold evolution_edges.type. The subtype is in the
        # rich edge's pattern_type (infer_rich sets kind='evolution' but
        # pattern_type carries the verb).
        for re_ in rich_edges:
            kind = re_["kind"]
            if kind == "evolution":
                pt = re_.get("pattern_type", "")
                # pattern_type is the verb (extends/improves/...) per schema seed
                if pt in ("extends", "improves", "compares", "replaces",
                          "adapts", "background"):
                    kind = pt  # specific subtype, eval-alignable
            ev = re_.get("evidence", "")
            sec = re_.get("section", "")
            cids = []
            for nd in re_["nodes"]:
                surf = nd["surface"]
                labels = nd.get("labels", [])
                t = next((l for l in labels if l != "THING"), (labels[0] if labels else "PROPERTY"))
                cid = nid2concept.get((t, surf))
                if cid:
                    cids.append(cid)
            if len(cids) >= 2:
                self.add_hyperedge(cids, kind, paper_id=paper_id,
                                   evidence=ev, year=year, section=sec)

    # ---- cross-paper semantic alignment (C2) ----
    def align_concepts(self, llm_fn, type_filter=("METHOD", "PHENOMENON"),
                       batch_size: int = 20) -> int:
        """LLM semantic alignment: batch concept surfaces per type, ask LLM
        which are the SAME concept (near-synonyms). Merge each group into one
        (keep = the concept with most surface_variants; tie-break by id).

        Only METHOD/PHENOMENON (noisy naming). PARAMETER/NUMERIC use symbol
        matching (in get_or_create)."""
        n_merged = 0
        for t in type_filter:
            concepts = [c for cid, c in self.concepts.items()
                        if c.type == t and not c.deprecated]
            items = [(c.concept_id, c.surfaces()[0] if c.surfaces() else "")
                     for c in concepts if c.surfaces()]
            if len(items) < 2:
                continue
            for i in range(0, len(items), batch_size):
                batch = items[i:i+batch_size]
                groups = _llm_align_batch(batch, t, llm_fn)
                for group in groups:
                    if len(group) < 2:
                        continue
                    # keep = concept with most variants (stable, not arbitrary)
                    cands = [self.concepts[g] for g in group if g in self.concepts]
                    if len(cands) < 2:
                        continue
                    keep = max(cands, key=lambda c: (len(c.surface_variants), -ord(c.concept_id[-1])))
                    for dup in cands:
                        if dup.concept_id != keep.concept_id:
                            self.merge_concepts(keep.concept_id, dup.concept_id)
                            n_merged += 1
        return n_merged

    def to_dict(self):
        return {
            "n_concepts": len(self.concepts),
            "n_hyperedges": len(self.hyperedges),
            "concepts": {cid: c.to_dict() for cid, c in self.concepts.items()},
            "hyperedges": [he.to_dict() for he in self.hyperedges],
        }

    @classmethod
    def from_dict(cls, d):
        cg = cls()
        for cid, cd in d.get("concepts", {}).items():
            c = Concept(concept_id=cid, type=cd["type"], deprecated=cd.get("deprecated", False),
                        canonical_name=cd.get("canonical_name", ""), symbol=cd.get("symbol", ""))
            for v in cd.get("surface_variants", []):
                c.add_variant(v["surface"], v.get("paper_id", ""),
                              v.get("evidence", ""), v.get("year", ""), v.get("section", ""))
            cg.concepts[cid] = c
            for s in c.surfaces():
                cg._surface2concept[_norm_surface(s)] = cid
            if c.symbol:
                cg._symbol2concept[c.symbol] = cid
        for he in d.get("hyperedges", []):
            cg.hyperedges.append(ConceptHyperedge(
                he_id=he["he_id"], node_ids=he.get("node_ids", []),
                node_roles=he.get("node_roles", []), kind=he.get("kind", ""),
                provenance=he.get("provenance", []),
                emergence_signals=he.get("emergence_signals", {}),
                confidence=he.get("confidence", 0.0)))
        cg._next_id = len(cg.concepts)
        return cg


_ALIGN_PROMPT = """下面是抽取出的多个{type_label}实体(每个有一个代表性surface, 来自不同论文).
判断哪些是【同一个概念】(近义/同义/只是表述不同, 如 "non-local rheology" 和 "I-gradient model" 是同一个方法).
只把确属同一概念的归一组; 不同概念不要合并; 不确定的不合并.

{type_label}列表(id: surface):
{items}

输出 JSON: {{"groups": [[id1, id2], [id3], ...]}}  每组是同一概念的id列表, 单个的也列出.
"""


def _llm_align_batch(items, type_label, llm_fn):
    """items: list[(concept_id, surface)]. Returns list of groups
    (each a list of concept_ids). Uses INDEX-based mapping so the LLM
    returning index numbers (not raw ids) still maps correctly."""
    type_map = {"METHOD": "建模方法", "PHENOMENON": "物理现象"}
    tl = type_map.get(type_label, type_label)
    # present items with a stable numeric index the LLM can refer to
    idx_items = "\n".join(f"{i}: {surf}" for i, (_, surf) in enumerate(items))
    prompt = _ALIGN_PROMPT.format(type_label=tl, items=idx_items).replace(
        "id1, id2", "index1, index2").replace("id3", "index3")
    try:
        from granular_agent.llm_client import parse_json_response
        resp = llm_fn(prompt)
        obj = parse_json_response(resp) or {}
        groups = obj.get("groups", [])
        out = []
        for g in groups:
            # g may be list of int indices OR strings; map to concept_ids
            cids = []
            for x in g:
                try:
                    idx = int(x)
                    if 0 <= idx < len(items):
                        cids.append(items[idx][0])
                except (ValueError, TypeError):
                    # maybe returned the id directly
                    if x in [cid for cid, _ in items]:
                        cids.append(x)
            out.append(cids)
        return out
    except Exception:
        return []
