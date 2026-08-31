# -*- coding: utf-8 -*-
"""ScholarGraph demo backend (2026-08-29) — the contest video/presentation
surface for the knowledge-hypergraph system.

  GET  /                       → the single-page demo UI (demo/index.html)
  POST /api/search             → full pipeline run (recall → grade → rank)
  GET  /api/graphs             → list stored graphs (name + stats) for a selector
  GET  /api/graph              → a stored graph as vis-network nodes/edges.
                                 Hyperedges render in the BIPARTITE (hub-node)
                                 form the design review picked: every non-cites
                                 n-ary hyperedge becomes a ◆ hub connected to its
                                 members; cites stays a directed intent-colored
                                 edge. Optional center+hops returns an ego
                                 subgraph (lazy expansion).
  GET  /api/graph/papers       → ego-subgraph union for a set of paper titles
                                 (the search view's embedded subgraph mode).

UI serves from demo/index.html. Run:
  PYTHONPATH=src python demo/app.py   →  http://127.0.0.1:8899
"""
import json, os, re, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
os.chdir(_REPO)
sys.path.insert(0, os.path.join(_REPO, "src"))

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ScholarGraph Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


# ---------------------------------------------------------------------------
# Color system — single source of truth; the UI builds its legend from these.
# Tuned for a dark graph canvas.
# ---------------------------------------------------------------------------
INTENT_COLORS = {          # cites edge intent
    "extends": "#3b82f6", "improves": "#10b981", "compares": "#f59e0b",
    "replaces": "#ef4444", "adapts": "#8b5cf6", "background": "#94a3b8",
}
TYPE_COLORS = {            # concept node type
    "METHOD": "#6366f1", "PARAMETER": "#f59e0b", "TASK": "#10b981",
    "PHENOMENON": "#ec4899", "LAW": "#ef4444", "DATASET": "#06b6d4",
    "MATERIAL": "#84cc16", "METRIC": "#14b8a6",
}
KIND_COLORS = {            # non-cites hyperedge (hub) rich-topology kind
    "method_parameter": "#a78bfa", "law_parameter": "#c084fc",
    "method_phenomenon": "#f472b6", "composition": "#38bdf8",
    "nary": "#fbbf24", "method_method": "#818cf8",
}
_DEFAULT_CONCEPT = "#64748b"
_DEFAULT_HUB = "#a78bfa"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _concept_label(c) -> str:
    return c.canonical_name or (c.surface_variants[0].surface
                                if c.surface_variants else c.concept_id)


# ---------------------------------------------------------------------------
# vis-node builders
# ---------------------------------------------------------------------------
def _concept_node(c) -> dict:
    label = _concept_label(c)
    if c.type == "PAPER":
        return {"id": c.concept_id, "label": label[:46], "shape": "box",
                "color": {"background": "#1e3a8a", "border": "#3b82f6"},
                "font": {"color": "#dbeafe", "size": 12},
                "title": label, "group": "paper",
                "meta": {"kind": "paper", "label": label,
                         "papers": list(c.source_papers),
                         "definition": c.definition}}
    return {"id": c.concept_id, "label": label[:26], "shape": "dot",
            "size": 14 if c.central else 9,
            "color": {"background": TYPE_COLORS.get(c.type, _DEFAULT_CONCEPT),
                      "border": "#0f172a"},
            "font": {"color": "#cbd5e1", "size": 11},
            "title": f"{label} · {c.type}", "group": "concept",
            "meta": {"kind": "concept", "label": label, "type": c.type,
                     "symbol": c.symbol, "definition": c.definition,
                     "central": c.central, "papers": list(c.source_papers),
                     "years": list(c.year_range)}}


def _emit_hyperedge(h, present: set):
    """Return (nodes, edges) for one hyperedge restricted to `present` ids.
    cites → directed intent edge; everything else → ◆ hub + member spokes."""
    nodes, edges = [], []
    # order-preserving dedupe: a concept may fill several roles of one n-ary
    # hyperedge; duplicate member spokes collide on the same vis edge id
    # ("he_H28-CC0012 already exists", measured on the 50-paper graph).
    members = list(dict.fromkeys(n for n in h.node_ids if n in present))
    ev = h.provenance[0].get("evidence", "") if h.provenance else ""
    src_papers = sorted({p.get("paper_id", "") for p in h.provenance
                         if p.get("paper_id")})
    if h.kind == "cites":
        a, b = h.node_ids[0], h.node_ids[-1]
        if a not in present or b not in present:
            return nodes, edges
        intent = h.pattern_type or "cites"
        edges.append({"id": h.he_id, "from": a, "to": b, "arrows": "to",
                      "color": {"color": INTENT_COLORS.get(intent, "#94a3b8"),
                                "opacity": 0.9},
                      "width": 1.8, "label": intent,
                      "font": {"size": 8, "color": "#94a3b8", "strokeWidth": 0},
                      "title": f"{intent}\n{ev[:220]}",
                      "meta": {"kind": "cites", "intent": intent, "evidence": ev,
                               "papers": src_papers, "label": intent}})
        return nodes, edges
    if len(members) < 2:
        return nodes, edges
    hub_id = "he_" + h.he_id
    kind = h.kind or "relates"
    hcol = KIND_COLORS.get(kind, _DEFAULT_HUB)
    label = h.pattern_type or kind
    nodes.append({"id": hub_id, "label": "", "shape": "diamond", "size": 7,
                  "color": {"background": hcol, "border": hcol},
                  "title": f"{label} ({kind})", "group": "hub",
                  "meta": {"kind": "hyperedge", "label": label, "type": kind,
                           "pattern": h.pattern_type, "roles": list(h.node_roles),
                           "members": members, "evidence": ev,
                           "papers": src_papers}})
    for m in members:
        edges.append({"id": f"{hub_id}-{m}", "from": hub_id, "to": m,
                      "color": {"color": hcol, "opacity": 0.45},
                      "width": 1.0})
    return nodes, edges


def _build_vis(kb, present: set, he_ids=None):
    """Serialize concepts in `present` (+ hyperedges, optionally restricted to
    `he_ids`) into vis-network nodes/edges."""
    nodes = [_concept_node(kb.abox.concepts[cid]) for cid in present
             if cid in kb.abox.concepts]
    edges, extra_nodes = [], []
    for h in kb.abox.hyperedges:
        if he_ids is not None and h.he_id not in he_ids:
            continue
        n, e = _emit_hyperedge(h, present)
        extra_nodes.extend(n)
        edges.extend(e)
    return nodes + extra_nodes, edges


# ---------------------------------------------------------------------------
# graph stats / selection helpers
# ---------------------------------------------------------------------------
def _graph_stats(kb) -> dict:
    concepts = kb.abox.concepts
    papers = sum(1 for c in concepts.values() if c.type == "PAPER")
    cites = sum(1 for h in kb.abox.hyperedges if h.kind == "cites")
    return {"n_concepts": len(concepts), "n_papers": papers,
            "n_hyperedges": len(kb.abox.hyperedges), "n_cites": cites}


def _degrees(kb) -> dict:
    deg = {}
    for h in kb.abox.hyperedges:
        for n in h.node_ids:
            deg[n] = deg.get(n, 0) + 1
    return deg


def _ego(kb, center: str, hops: int):
    """BFS over the bipartite concept↔hyperedge structure. Returns
    (concept_ids, hyperedge_ids) within `hops` concept-hops of `center`."""
    he_by_node = {}
    for h in kb.abox.hyperedges:
        for n in h.node_ids:
            he_by_node.setdefault(n, []).append(h)
    concepts, chosen, frontier = {center}, set(), {center}
    for _ in range(max(1, hops)):
        nxt = set()
        for n in frontier:
            for h in he_by_node.get(n, []):
                chosen.add(h.he_id)
                for m in h.node_ids:
                    if m not in concepts:
                        concepts.add(m)
                        nxt.add(m)
        frontier = nxt
        if not nxt:
            break
    return concepts, chosen


# ---------------------------------------------------------------------------
# search endpoint (unchanged behavior — wraps contest.pipeline)
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str
    max_out: int = 20
    fast: bool = True      # interactive demo config; false = full SPAR pipeline


@app.get("/")
def index():
    return FileResponse(os.path.join(_HERE, "index.html"))


@app.post("/api/search")
def search(req: SearchRequest):
    from contest.pipeline import recall, grade_and_rank
    from contest.cost_ledger import ledger, CostLedger

    fresh = CostLedger()
    ledger.__dict__.update(fresh.__dict__)
    t0 = time.time()
    try:
        cands = recall(req.query)
        t_recall = time.time() - t0
        # fast=True trims latency for the interactive demo (citation trim + one
        # global embedding pass + a single grading chunk); fast=False runs the
        # exact SPAR pipeline configuration.
        if req.fast:
            kept = grade_and_rank(req.query, cands, rank="authority",
                                  semantic_trim=False, grade_window=60, grade_chunk=60)
        else:
            kept = grade_and_rank(req.query, cands, rank="authority")
        t_total = time.time() - t0
        papers = [{
            "title": c.get("title", ""),
            "year": c.get("year"),
            "citations": c.get("citation_count") or c.get("citationCount") or 0,
            "abstract": (c.get("abstract") or "")[:300],
            "sem": c.get("_sem"),
        } for c in kept]
        n_high = max(1, len(kept) // 3)
        for i, p in enumerate(papers):
            p["rank"] = i + 1
            p["band"] = "high" if i < n_high else "partial"
        return {"query": req.query, "n_recall": len(cands), "papers": papers,
                "timing": {"recall_s": round(t_recall, 1),
                           "total_s": round(t_total, 1)},
                "cost": ledger.report()}
    except Exception as e:
        return JSONResponse({"error": repr(e)[:300]}, status_code=500)


# ---------------------------------------------------------------------------
# graph endpoints
# ---------------------------------------------------------------------------
@app.get("/api/graphs")
def graphs():
    """List stored graphs + manifest stats (drives the graph selector and the
    'building…' empty state)."""
    from granular_agent.graph_store import STORE_ROOT
    out = []
    if os.path.isdir(STORE_ROOT):
        for name in sorted(os.listdir(STORE_ROOT)):
            mp = os.path.join(STORE_ROOT, name, "manifest.json")
            if not os.path.exists(mp):
                continue
            try:
                m = json.load(open(mp, encoding="utf-8"))
            except Exception:
                m = {}
            out.append({"name": name,
                        "n_concepts": m.get("n_concepts", 0),
                        "n_hyperedges": m.get("n_hyperedges", 0),
                        "n_papers": len(m.get("papers", [])),
                        "version": m.get("version", ""),
                        "updated_at": m.get("updated_at", "")})
    return {"graphs": out}


@app.get("/api/graph")
def graph(name: str = "llm_domain", max_nodes: int = 160,
          center: str | None = None, hops: int = 1):
    """vis-network payload for a stored graph. Without `center` → a bounded
    overview (all PAPER nodes + the highest-degree concepts). With `center` →
    the ego subgraph within `hops` concept-hops (lazy expansion)."""
    from granular_agent.graph_store import load_graph
    kb = load_graph(name)
    if kb is None:
        return JSONResponse({"error": f"graph {name!r} not found",
                             "empty": True, "stats": None,
                             "hint": "The knowledge graph is still building."},
                            status_code=404)
    if center:
        if center not in kb.abox.concepts:
            return JSONResponse({"error": f"node {center!r} not found"},
                                status_code=404)
        present, he_ids = _ego(kb, center, hops)
        nodes, edges = _build_vis(kb, present, he_ids)
    else:
        papers = {cid for cid, c in kb.abox.concepts.items() if c.type == "PAPER"}
        deg = _degrees(kb)
        others = sorted((cid for cid in kb.abox.concepts if cid not in papers),
                        key=lambda c: -deg.get(c, 0))
        present = papers | set(others[:max(0, max_nodes - len(papers))])
        nodes, edges = _build_vis(kb, present)
    return {"nodes": nodes, "edges": edges, "stats": _graph_stats(kb),
            "legend": {"intents": INTENT_COLORS, "types": TYPE_COLORS,
                       "kinds": KIND_COLORS}}


@app.get("/api/graph/papers")
def graph_for_papers(name: str = "llm_domain", titles: str = "", hops: int = 1):
    """Ego-subgraph union for a list of paper titles (the search view's graph
    mode). Titles not yet extracted into the graph come back in `unmatched`."""
    from granular_agent.graph_store import load_graph
    kb = load_graph(name)
    if kb is None:
        return JSONResponse({"error": f"graph {name!r} not found", "empty": True,
                             "nodes": [], "edges": [], "matched": [],
                             "unmatched": [t for t in titles.split("|") if t]},
                            status_code=404)
    title_list = [t for t in titles.split("|") if t.strip()]
    norm_to_pid = {}
    for cid, c in kb.abox.concepts.items():
        if c.type != "PAPER":
            continue
        norm_to_pid[_norm(_concept_label(c))] = cid
    present, he_ids = set(), set()
    matched, unmatched = [], []
    for t in title_list:
        pid = norm_to_pid.get(_norm(t))
        if not pid:
            unmatched.append(t)
            continue
        matched.append({"title": t, "id": pid})
        c, hs = _ego(kb, pid, hops)
        present |= c
        he_ids |= hs
    nodes, edges = _build_vis(kb, present, he_ids) if present else ([], [])
    return {"nodes": nodes, "edges": edges, "matched": matched,
            "unmatched": unmatched, "stats": _graph_stats(kb),
            "legend": {"intents": INTENT_COLORS, "types": TYPE_COLORS,
                       "kinds": KIND_COLORS}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8899)
