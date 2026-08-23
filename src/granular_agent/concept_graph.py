"""A-box: cross-paper Concept Graph (the knowledge graph itself).

Per DESIGN_full.md section 3 (A-box). The ConceptGraph accumulates concrete
concept entities (methods/phenomena/parameters/...) across papers, with
cross-paper semantic alignment (LLM, batched) so the same method mentioned in
different papers merges into one Concept — the alignment anchor that the
InstanceCorpus surface-merge was too weak to provide.

This is the A-box (instance layer). The T-box (schema/MetaHypergraph) defines
types + relation patterns; ConceptGraph holds concrete entities + relations
referencing those types/patterns.

Provenance: every Concept records surface_variants (per-paper surface +
evidence + year + section), every ConceptRelation records source papers +
evidence — so any concept/edge traces back to source-paper original text.

Emergence (P2) reads structure signals from this graph to surface evolution
relation candidates — NOT done here (P1 = data structure + accumulation +
alignment only).
"""
from __future__ import annotations
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
    """A concrete concept entity in the cross-paper graph (A-box)."""
    concept_id: str
    type: str                                   # METHOD/PHENOMENON/PARAMETER/REGIME/MATERIAL/NUMERIC (T-box type)
    surface_variants: list[SurfaceVariant] = field(default_factory=list)
    source_papers: list[str] = field(default_factory=list)
    year_range: tuple[str, str] = ("", "")      # (min, max) year
    deprecated: bool = False
    canonical_name: str = ""                    # optional, set when stable

    def add_variant(self, surface: str, paper_id: str, evidence: str = "",
                    year: str = "", section: str = "") -> None:
        self.surface_variants.append(SurfaceVariant(surface, paper_id, evidence, year, section))
        if paper_id and paper_id not in self.source_papers:
            self.source_papers.append(paper_id)
        if year:
            years = [v.year for v in self.surface_variants if v.year]
            if years:
                self.year_range = (min(years), max(years))

    def surfaces(self) -> list[str]:
        """All distinct surfaces (for alignment/lookup)."""
        seen = []
        for v in self.surface_variants:
            if v.surface and v.surface not in seen:
                seen.append(v.surface)
        return seen

    def to_dict(self) -> dict:
        d = asdict(self)
        d["year_range"] = list(self.year_range)
        return d


@dataclass
class ConceptRelation:
    """A concrete relation between two Concepts (A-box)."""
    src_concept: str
    tgt_concept: str
    kind: str                                    # T-box pattern (constitutive_law/method_parameter/extends/...)
    provenance: list[dict] = field(default_factory=list)  # [{paper_id, evidence, year}]
    emergence_signals: dict[str, Any] = field(default_factory=dict)  # P2 fills
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class ConceptGraph:
    """Cross-paper concept graph (A-box). Accumulates concepts + relations
    across papers with semantic alignment. Replaces InstanceCorpus surface-merge."""

    def __init__(self):
        self.concepts: dict[str, Concept] = {}
        self.relations: list[ConceptRelation] = []
        self._next_id = 0
        # surface -> concept_id index for fast first-pass lookup (exact surface)
        self._surface2concept: dict[str, str] = {}

    def _new_id(self, type_: str) -> str:
        self._next_id += 1
        prefix = {"METHOD": "M", "PHENOMENON": "F", "PARAMETER": "P",
                   "REGIME": "R", "MATERIAL": "MAT", "NUMERIC": "N"}.get(type_, "C")
        return f"C{prefix}{self._next_id:04d}"

    def get_or_create(self, type_: str, surface: str, paper_id: str,
                      evidence: str = "", year: str = "",
                      section: str = "") -> Concept:
        """First-pass: exact-surface lookup, else create. LLM semantic
        alignment (merge near-synonyms) happens in align_concepts, not here."""
        key = surface.strip().lower()
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

    def add_relation(self, src: str, tgt: str, kind: str,
                     paper_id: str = "", evidence: str = "",
                     year: str = "") -> ConceptRelation:
        rel = ConceptRelation(src_concept=src, tgt_concept=tgt, kind=kind,
                              provenance=[{"paper_id": paper_id, "evidence": evidence, "year": year}]
                              if paper_id else [])
        self.relations.append(rel)
        return rel

    def merge_concepts(self, into_id: str, merge_id: str) -> None:
        """Merge two Concepts (semantic near-synonyms, called by align_concepts).
        Redirects relations, merges variants + provenance."""
        if into_id == merge_id or merge_id not in self.concepts or into_id not in self.concepts:
            return
        keep = self.concepts[into_id]
        drop = self.concepts[merge_id]
        for v in drop.surface_variants:
            keep.add_variant(v.surface, v.paper_id, v.evidence, v.year, v.section)
        # redirect relations
        for r in self.relations:
            if r.src_concept == merge_id:
                r.src_concept = into_id
            if r.tgt_concept == merge_id:
                r.tgt_concept = into_id
        # update surface index
        for s in drop.surfaces():
            self._surface2concept[s.strip().lower()] = into_id
        del self.concepts[merge_id]

    def concepts_by_type(self, type_: str) -> list[Concept]:
        return [c for c in self.concepts.values() if c.type == type_ and not c.deprecated]

    def to_dict(self) -> dict:
        return {
            "n_concepts": len(self.concepts),
            "n_relations": len(self.relations),
            "concepts": {cid: c.to_dict() for cid, c in self.concepts.items()},
            "relations": [r.to_dict() for r in self.relations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptGraph":
        cg = cls()
        for cid, cd in d.get("concepts", {}).items():
            c = Concept(concept_id=cid, type=cd["type"], deprecated=cd.get("deprecated", False),
                        canonical_name=cd.get("canonical_name", ""))
            for v in cd.get("surface_variants", []):
                c.add_variant(v["surface"], v.get("paper_id", ""),
                              v.get("evidence", ""), v.get("year", ""), v.get("section", ""))
            cg.concepts[cid] = c
            for s in c.surfaces():
                cg._surface2concept[s.strip().lower()] = cid
        for r in d.get("relations", []):
            cg.relations.append(ConceptRelation(
                src_concept=r["src_concept"], tgt_concept=r["tgt_concept"],
                kind=r["kind"], provenance=r.get("provenance", []),
                emergence_signals=r.get("emergence_signals", {}),
                confidence=r.get("confidence", 0.0)))
        cg._next_id = len(cg.concepts)
        return cg

    # ---- cross-paper semantic alignment (C2) ----
    def align_concepts(self, llm_fn, type_filter=("METHOD", "PHENOMENON"),
                       batch_size: int = 20) -> int:
        """LLM semantic alignment: for each type, batch its concept surfaces
        and ask LLM which are the SAME concept (near-synonyms surface-string
        miss). Merge each LLM-identified group into one.

        Replaces InstanceCorpus surface-only merge (which missed
        'non-local rheology' ~ 'I-gradient' synonyms).

        llm_fn(prompt) -> response text. Returns number of merges done.
        Only aligns METHOD/PHENOMENON (the noisy-naming types); PARAMETER/
        NUMERIC use symbol matching elsewhere."""
        n_merged = 0
        for t in type_filter:
            concepts = [c for cid, c in self.concepts.items()
                        if c.type == t and not c.deprecated]
            # collect (concept_id, representative surface) — first surface per concept
            items = [(c.concept_id, c.surfaces()[0] if c.surfaces() else "")
                     for c in concepts if c.surfaces()]
            if len(items) < 2:
                continue
            for i in range(0, len(items), batch_size):
                batch = items[i:i+batch_size]
                merges = _llm_align_batch(batch, t, llm_fn)
                # merges: list of groups (each a list of concept_ids that are the same)
                for group in merges:
                    if len(group) < 2:
                        continue
                    keep = group[0]
                    for dup in group[1:]:
                        if dup in self.concepts and keep in self.concepts:
                            self.merge_concepts(keep, dup)
                            n_merged += 1
        return n_merged


_ALIGN_PROMPT = """下面是抽取出的多个{type_label}实体(每个有一个代表性surface, 来自不同论文).
判断哪些是【同一个概念】(近义/同义/只是表述不同, 如 "non-local rheology" 和 "I-gradient model" 是同一个方法).
只把确属同一概念的归一组; 不同概念不要合并; 不确定的不合并.

{type_label}列表(id: surface):
{items}

输出 JSON: {{"groups": [[id1, id2], [id3], ...]}}  每组是同一概念的id列表, 单个的也列出.
"""


def _llm_align_batch(items, type_label, llm_fn):
    type_map = {"METHOD": "建模方法", "PHENOMENON": "物理现象"}
    tl = type_map.get(type_label, type_label)
    item_str = "\n".join(f"{cid}: {surf}" for cid, surf in items)
    try:
        from granular_agent.llm_client import parse_json_response
        resp = llm_fn(_ALIGN_PROMPT.format(type_label=tl, items=item_str))
        obj = parse_json_response(resp) or {}
        groups = obj.get("groups", [])
        # map back to concept_ids (items[i][0] is the concept_id)
        id2idx = {cid: i for i, (cid, _) in enumerate(items)}
        return [[g for g in group if g in id2idx] for group in groups]
    except Exception:
        return []
