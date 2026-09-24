# -*- coding: utf-8 -*-
"""Mentions backflow: coarse-extracted external papers attach to the
in-corpus genealogy (library-growth minimal loop, batch-2 2026-09-25).

The insight (user directive): the coarse extractor's `mentions` field is
citation intent from the EXTERNAL paper's perspective — "this outside paper
references method M" is the same signal a citation-context extractor would
produce, obtained for free from the abstract. Backflow converts it into
graph edges:

  coarse-extract(P) -> P.mentions=[M...] -> match M against registry
  surface_index -> edge(M_entity --external_mention--> P_node)

P becomes an `external_paper` node carrying its coarse records; the edge is
deliberately WEAK (external_mention, provenance="coarse") — the abstract
says P mentions M, not HOW P relates to M. Relation upgrade rides on Tier-2
promotion (deep extraction produces real genealogy edges that replace the
weak one).

Payoff wiring: lineage_walk treats external_mention edges from reached
nodes as KNOWN external continuations — surfaced before blind search, at
zero retrieval cost. Each coarse-extracted paper makes the next question's
walk cheaper and better targeted.

Idempotency: node id = "ext:"+hash(doi or title); re-extracting the same
paper never duplicates nodes or edges. Persistence is JSONL (one line per
paper) replayed at load; batch-3 moves it to the sci-evo SQLite registry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", "src"))


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _tokens(t: str) -> list[str]:
    return _norm(t).split()


def _paper_node_id(paper_key: str) -> str:
    return "ext:" + hashlib.md5((paper_key or "").encode("utf-8")).hexdigest()[:12]


def _build_norm_surface(surface_index: dict) -> dict[str, list[str]]:
    """surface_index {surface: entity_id} -> {normalized surface: [entity_ids]}."""
    out: dict[str, list[str]] = {}
    for surface, eid in (surface_index or {}).items():
        n = _norm(str(surface))
        if n:
            out.setdefault(n, []).append(str(eid))
    return out


def match_mentions(mentions: list[str], surface_index: dict,
                   blocklist: list | None = None) -> list[dict]:
    """Deterministic mention -> registry entity matching.

    Tier 1: exact normalized surface hit.
    Tier 2: contiguous token containment (either direction) for mentions
    with >=2 tokens — "classifier guidance" hits surface "classifier
    guidance (CFG)".
    Safety: blocklist names are rejected outright (cross-domain homonyms,
    e.g. GAP=band gap vs GAN variants); tier-2 hits resolving to more than
    one distinct entity are dropped as ambiguous (honest miss > wrong edge).
    """
    block_norm = {_norm(str(b.get("name") if isinstance(b, dict) else b))
                  for b in (blocklist or [])}
    block_norm.discard("")
    norm_surface = _build_norm_surface(surface_index)

    # tier-2 index: multi-token surfaces keyed by their token tuples is
    # expensive to scan per mention; build once per call over surfaces with
    # >=2 tokens (bounded by registry size, ~12k — fine at call frequency)
    multi_token = [(n.split(), eids) for n, eids in norm_surface.items()
                   if len(n.split()) >= 2]

    out = []
    for m in (mentions or []):
        nm = _norm(str(m))
        if not nm or nm in block_norm:
            continue
        mt = nm.split()
        hit_eids: list[str] = []
        match_type = None
        if nm in norm_surface:
            hit_eids = norm_surface[nm]
            match_type = "exact"
        elif len(mt) >= 2:
            for toks, eids in multi_token:
                if (len(mt) <= len(toks) and _subseq(mt, toks)) or \
                   (len(toks) < len(mt) and _subseq(toks, mt)):
                    hit_eids.extend(eids)
                    match_type = "containment"
        uniq = sorted(set(hit_eids))
        if len(uniq) == 1:
            out.append({"mention": str(m), "entity_id": uniq[0],
                        "match": match_type})
        # 0 hits = honest miss; >1 = ambiguous, dropped
    return out


def _subseq(needle: list[str], hay: list[str]) -> bool:
    """ contiguous token subsequence """
    n, h = len(needle), len(hay)
    return any(hay[i:i + n] == needle for i in range(h - n + 1))


def build_backflow(coarse_payload: dict, paper_meta: dict,
                   registry: dict, blocklist: list | None = None) -> dict:
    """Coarse payload + paper metadata -> backflow attachment
    {node, edges, matches}. Pure — no views mutation here.
    `registry` is the full registry dict (entities + surface_index) so
    matched entity ids resolve to canonical names for edge from_name."""
    title = (paper_meta.get("title") or "").strip()
    doi = (paper_meta.get("doi") or "").strip() or None
    paper_key = doi or title
    if not paper_key:
        return {"node": None, "edges": [], "matches": []}
    records = (coarse_payload or {}).get("records") or []
    eid2name = {str(e.get("entity_id")): str(e.get("canonical"))
                for e in (registry or {}).get("entities", [])
                if e.get("entity_id")}
    surface_index = (registry or {}).get("surface_index") or {}

    # collect mentions per record so each edge carries its evidence quote
    per_record: list[tuple[dict, list[dict]]] = []
    all_matches: dict[str, dict] = {}
    for r in records:
        ms = match_mentions(r.get("mentions") or [], surface_index, blocklist)
        if ms:
            per_record.append((r, ms))
            for m in ms:
                all_matches.setdefault(m["entity_id"], m)

    node = {
        "id": _paper_node_id(paper_key),
        "canonical": title or paper_key,
        "entity_type": "external_paper",
        "year": paper_meta.get("year"),
        "in_corpus": False,
        "doi": doi,
        "paper_key": paper_key,
        # trimmed coarse records — everything a walk needs to surface the
        # paper's claims without another lookup
        "records": [{"id": r["id"], "kind": r["kind"], "subject": r["subject"],
                     "claim": r["claim"], "quote": (r.get("quote") or "")[:250]}
                    for r in records[:10]],
    }
    edges = []
    seen_pairs = set()
    for r, ms in per_record:
        for m in ms:
            key = (m["entity_id"], node["id"])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            edges.append({
                "from": m["entity_id"], "to": node["id"],
                "from_name": eid2name.get(m["entity_id"], m["entity_id"]),
                "to_name": node["canonical"],
                "relation": "external_mention",
                "scope": {}, "evidence_basis": "explicit_mention",
                "year": paper_meta.get("year"),
                "paper_id": paper_key,
                "record_id": r["id"],
                "quote": (r.get("quote") or "")[:250],
                "provenance": "coarse",
            })
    return {"node": node, "edges": edges, "matches": list(all_matches.values())}


def apply_backflow(views: dict, bf: dict, persist_path: str | None = None) -> dict:
    """Attach a backflow result to views["genealogy"] in place (idempotent).
    Returns {attached, already_present, matched_entities}."""
    node = bf.get("node")
    if not node:
        return {"attached": 0, "already_present": True, "matched_entities": []}
    g = views.setdefault("genealogy", {"nodes": {}, "edges": []})
    nodes = g.setdefault("nodes", {})
    edges = g.setdefault("edges", [])
    existed = node["id"] in nodes
    if not existed:
        nodes[node["id"]] = {k: v for k, v in node.items() if k != "id"}
    existing_pairs = {(str(e.get("from")), str(e.get("to")))
                      for e in edges if e.get("relation") == "external_mention"}
    new_edges = [e for e in bf.get("edges", [])
                 if (e["from"], e["to"]) not in existing_pairs]
    edges.extend(new_edges)

    if persist_path and (new_edges or not existed):
        rec = {"ts": datetime.now(timezone.utc).isoformat(),
               "node": node, "edges": new_edges, "replayed": existed}
        os.makedirs(os.path.dirname(persist_path) or ".", exist_ok=True)
        with open(persist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    names = sorted({e["from_name"] for e in bf.get("edges", [])})
    return {"attached": len(new_edges), "already_present": existed,
            "node_id": node["id"], "matched_entities": names}


def load_backflow(views: dict, persist_path: str) -> int:
    """Replay persisted backflow lines into views (cross-session growth).
    Returns the number of papers attached."""
    if not persist_path or not os.path.exists(persist_path):
        return 0
    n = 0
    for line in open(persist_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("node"):
            apply_backflow(views, rec, persist_path=None)
            n += 1
    return n
