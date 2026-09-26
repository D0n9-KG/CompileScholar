"""Lineage-driven walk primitive (2026-09-25, user-directed design).

Second knowledge-model-driven retrieval primitive. Generic citation
walkers BFS uniformly over the citation graph; our genealogy view carries
971 TYPED method-evolution edges (extends/improves/replaces/motivated_by/
compares_with/uses...) inside the corpus. lineage_walk:

  1. IN-CORPUS walk: typed edges from the entity — direction and relation
     selectable (e.g. successors = extends|improves|replaces|generalizes).
     The typed relation is the difference: 'who improved X' is a different
     question than 'who cited X', and our edges already answer it.
  2. BOUNDARY DETECTION: the walk reaches nodes with no further in-corpus
     edges — those frontier methods are exactly where EXTERNAL retrieval
     should continue (their successors live outside the corpus).
  3. EXTERNAL continuation: frontier entities -> semantic/keyword search
     through the shared tiered engine (the successor-generation query is
     synthesized from the lineage context: entity + its relation type),
     plus optional citation-graph lookup when a DOI is known.
  4. Successor filter: external candidates are annotated as likely
     successors when they mention the entity (title/abstract) — cheap
     lexical check, no claims of certainty.

The STRATEGY is ours (typed in-corpus walk -> frontier -> directed
external continuation); engines are shared with blind search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

# relations that propagate "improvement/succession" (vs comparison/use)
SUCCESSOR_RELATIONS = {"extends", "improves", "replaces", "generalizes"}
PREDECESSOR_RELATIONS = {"motivated_by", "component_of"}


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


@dataclass(frozen=True)
class LineageNode:
    entity_id: str
    canonical: str
    via_relation: str | None = None
    depth: int = 0
    in_corpus: bool = True


@dataclass(frozen=True)
class LineageWalkResult:
    entity: str
    direction: str
    in_corpus: list[LineageNode] = field(default_factory=list)
    frontier: list[LineageNode] = field(default_factory=list)
    attached_external: list[dict] = field(default_factory=list)
    external_candidates: list[Any] = field(default_factory=list)
    successor_annotations: list[dict] = field(default_factory=list)
    latency: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def walk_in_corpus(
    entity: str,
    genealogy: dict,
    *,
    direction: str = "successors",
    relations: set[str] | None = None,
    max_depth: int = 2,
    max_nodes: int = 25,
) -> tuple[list[LineageNode], list[LineageNode]]:
    """Typed BFS over in-corpus genealogy edges. Returns (reached, frontier):
    frontier = nodes at the walk boundary with no further edges in the
    chosen direction — the handoff points for external continuation."""
    nodes = genealogy.get("nodes", {}) or {}
    edges = genealogy.get("edges", []) or []
    if relations is None:
        relations = (SUCCESSOR_RELATIONS if direction == "successors"
                     else PREDECESSOR_RELATIONS if direction == "predecessors"
                     else SUCCESSOR_RELATIONS | PREDECESSOR_RELATIONS)

    # entity -> id resolution (name match against node canonicals)
    start_ids = [nid for nid, n in nodes.items()
                 if _norm(str(n.get("canonical") or "")) == _norm(entity)]
    if not start_ids:
        # substring tolerance
        start_ids = [nid for nid, n in nodes.items()
                     if _norm(entity) and _norm(entity) in
                     _norm(str(n.get("canonical") or ""))][:1]
    if not start_ids:
        return [], []

    # adjacency in the chosen direction
    def _outgoing(nid: str):
        for e in edges:
            if str(e.get("from")) == nid and e.get("relation") in relations:
                yield str(e.get("to")), e.get("relation")
    def _incoming(nid: str):
        for e in edges:
            if str(e.get("to")) == nid and e.get("relation") in relations:
                yield str(e.get("from")), e.get("relation")

    step = _outgoing if direction != "predecessors" else _incoming
    reached: list[LineageNode] = []
    seen = set(start_ids)
    frontier: list[LineageNode] = []
    layer = [(sid, None, 0) for sid in start_ids]
    while layer and len(reached) < max_nodes:
        nxt = []
        for nid, via, depth in layer:
            n = nodes.get(nid, {})
            reached.append(LineageNode(
                entity_id=nid, canonical=str(n.get("canonical") or nid),
                via_relation=via, depth=depth,
                in_corpus=bool(n.get("in_corpus"))))
            children = list(step(nid))
            if not children:
                # terminal: no further edges — the lineage continues outside
                frontier.append(reached[-1])
            elif depth >= max_depth - 1 and depth + 1 > max_depth - 1:
                # depth-truncated with known children: the walk stops here
                # but the lineage demonstrably continues (InstructBLIP
                # -extends-> BLIP-2 case) — frontier with a note, external
                # continuation is still the right move
                frontier.append(reached[-1])
            for child, rel in children:
                if child not in seen:
                    seen.add(child)
                    if depth + 1 <= max_depth:
                        nxt.append((child, rel, depth + 1))
        layer = nxt
    # frontier dedup (a node can be terminal only once)
    _fseen = set()
    frontier = [f for f in frontier
                if not (f.entity_id in _fseen or _fseen.add(f.entity_id))]
    return reached, frontier


def collect_attached_external(
    reached: list[LineageNode], genealogy: dict,
) -> list[dict]:
    """Backflow payoff (batch-2): external_mention edges from walked nodes
    point at papers already coarse-extracted into the library — known
    external continuations, surfaced at zero retrieval cost, BEFORE any
    blind search. Each entry carries the paper's coarse records so the
    caller gets the claims without another tool call."""
    nodes = genealogy.get("nodes", {}) or {}
    reached_ids = {n.entity_id for n in reached}
    out: list[dict] = []
    seen_nodes = set()
    for e in genealogy.get("edges", []) or []:
        if e.get("relation") != "external_mention":
            continue
        if str(e.get("from")) not in reached_ids:
            continue
        to_id = str(e.get("to"))
        node = nodes.get(to_id)
        if node is None or to_id in seen_nodes:
            continue
        seen_nodes.add(to_id)
        out.append({
            "title": str(node.get("canonical") or to_id),
            "year": node.get("year"), "doi": node.get("doi"),
            "attached_via": str(e.get("from_name") or ""),
            "records": (node.get("records") or [])[:6],
        })
    return out[:6]


def _successor_queries(node: LineageNode, relation_ctx: str) -> list[str]:
    """Frontier entity -> external successor/predecessor queries. The
    lineage context (what relation chain led here) sharpens the query."""
    name = node.canonical
    qs = [f"{name} improvement successor method"]
    if relation_ctx == "successors":
        qs.append(f"{name} extends improved method recent")
    else:
        qs.append(f"{name} original method prior work foundation")
    return qs[:2]


def lineage_walk(
    entity: str,
    views: dict,
    *,
    direction: str = "successors",
    search_fn: Callable[[str, int], Any] | None = None,
    max_depth: int = 2,
    external: bool = True,
    limit: int = 8,
) -> LineageWalkResult:
    """In-corpus typed walk -> frontier -> directed external continuation."""
    import time
    lat: dict[str, float] = {}
    genealogy = (views or {}).get("genealogy", {}) or {}

    t0 = time.time()
    reached, frontier = walk_in_corpus(
        entity, genealogy, direction=direction, max_depth=max_depth)
    lat["in_corpus_walk_ms"] = round((time.time() - t0) * 1000, 1)
    notes = []
    if not reached:
        notes.append(f"entity '{entity}' not found in genealogy — "
                     f"caller should fall back to blind search")
        return LineageWalkResult(entity=entity, direction=direction,
                                 latency=lat, notes=notes)

    # backflow first: papers already attached to the library via mentions
    # (batch-2) — known continuations cost nothing and outrank blind hits
    attached = collect_attached_external(reached, genealogy)
    if attached:
        notes.append(f"{len(attached)} attached external paper(s) already "
                     f"in library (coarse) — library-growth payoff")

    ext_cands: list[Any] = []
    succ_ann: list[dict] = []
    if external and frontier:
        if search_fn is None:
            from retrieval.search_service import SearchService
            svc = SearchService(limit=limit)
            search_fn = lambda q, k: svc.search(q, mode="full", limit=k)  # noqa: E731
        t0 = time.time()
        for f in frontier[:3]:  # top frontier nodes only (latency budget)
            for q in _successor_queries(f, direction):
                res = search_fn(q, limit)
                for c in (getattr(res, "candidates", None) or []):
                    ext_cands.append(c)
                    text = _norm(str(getattr(c, "title", "") or ""))
                    raw = getattr(c, "raw", None)
                    if isinstance(raw, dict):
                        text += " " + _norm(
                            str(raw.get("abstract") or "")[:600])
                    if _norm(f.canonical) and _norm(f.canonical) in text:
                        succ_ann.append({
                            "candidate_title":
                                str(getattr(c, "title", ""))[:100],
                            "frontier_entity": f.canonical,
                            "relation": "likely successor (mentions entity)",
                        })
        lat["external_ms"] = round((time.time() - t0) * 1000, 1)

    # dedupe external candidates by title
    seen, uniq = set(), []
    for c in ext_cands:
        key = _norm(str(getattr(c, "title", "") or ""))[:80]
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        uniq.append(c)
    return LineageWalkResult(entity=entity, direction=direction,
                             in_corpus=reached, frontier=frontier,
                             attached_external=attached,
                             external_candidates=uniq[:limit],
                             successor_annotations=succ_ann[:15],
                             latency=lat, notes=notes)
