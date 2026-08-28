# -*- coding: utf-8 -*-
"""Graph store — persistent, ACCUMULATING knowledge-base storage (2026-08-28).

The gap this closes: every run built a fresh KB from scratch and left a
disposable per-run bundle in .research_tmp/runs/. Incremental graph building
(query-loop rounds, corpus ingest) needs a named, durable graph that survives
process exits: build 3 papers today, 5 more tomorrow, query against all 8.

Design:
  - store root: data/graphs/<name>/kb.json (full KB snapshot: tbox+abox+
    skills+ledger+paper_meta) + manifest.json (provenance: which papers are
    ingested, when, with what schema version).
  - load_graph(name) → KnowledgeBase | None; save_graph(kb, name) atomic-write
    (tmp file + rename — a crash never corrupts the store).
  - ingest_papers(kb, paper_ids, ...) → the incremental entry point: the
    standard kernel pipeline (process_paper_via_kernel) per paper + save.
  - The kernel's own from_dict/to_dict roundtrip carries EVERYTHING (PAPER
    nodes, cites edges, surface map, ledger with mutation ids) — verified
    roundtrip, no bespoke serialization here.

Store layout is a REPORT ASSET too: 'the graph is a first-class persistent
artifact, not a per-run byproduct' — the structured-10% evidence object.
"""
import json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

STORE_ROOT = os.path.join("data", "graphs")


def _dir(name: str) -> str:
    d = os.path.join(STORE_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def save_graph(kb, name: str, manifest_extra: dict | None = None) -> str:
    """Atomic full-snapshot save. Returns the path written."""
    d = _dir(name)
    payload = kb.to_dict()
    payload["_paper_meta"] = getattr(kb, "_paper_meta", {})
    tmp = os.path.join(d, "kb.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(d, "kb.json"))     # atomic on same volume
    # manifest: merge with previous (papers accumulate)
    mp = os.path.join(d, "manifest.json")
    manifest = {}
    if os.path.exists(mp):
        try:
            manifest = json.load(open(mp, encoding="utf-8"))
        except Exception:
            manifest = {}
    papers = manifest.get("papers", [])
    for pid, meta in (manifest_extra or {}).get("papers", {}).items():
        papers.append({"paper_id": pid, **meta, "ingested_at": time.strftime("%Y-%m-%d %H:%M")})
    manifest["papers"] = papers
    manifest["n_concepts"] = len(kb.abox.concepts)
    manifest["n_hyperedges"] = len(kb.abox.hyperedges)
    manifest["version"] = kb.version
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M")
    tmp = os.path.join(d, "manifest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    os.replace(tmp, mp)
    return os.path.join(d, "kb.json")


def load_graph(name: str):
    """Load a stored KB, or None if the store has no snapshot yet."""
    p = os.path.join(STORE_ROOT, name, "kb.json")
    if not os.path.exists(p):
        return None
    from granular_agent.knowledge_base import KnowledgeBase
    d = json.load(open(p, encoding="utf-8"))
    paper_meta = d.pop("_paper_meta", {})
    kb = KnowledgeBase.from_dict(d)
    kb._paper_meta = paper_meta
    return kb


def ingest_papers(name: str, papers: list[dict], arm: str = "add_only",
                  domain: str = "ml") -> dict:
    """The incremental entry point: load-or-create the named graph, run the
    kernel pipeline on each new paper, save after each (crash-safe).
    papers: [{pid, title, surnames, years}] — surnames/years feed the corpus
    registry for intra-corpus citation edges.
    Returns a summary dict."""
    from granular_agent.agent import GranularFlowAgent
    from granular_agent.citation_stage import (register_corpus_papers,
                                               CORPUS_REGISTRY)

    kb = load_graph(name)
    agent = GranularFlowAgent(llms=["DeepSeek-V4-Flash"], domain=domain)
    if kb is not None:
        agent._kb = kb                      # continue on the ACCUMULATED graph
        # restore the citation-stage corpus registry so new papers can cite
        # already-ingested ones (registry is process-global; rebuild from
        # stored PAPER nodes)
        for cid, c in kb.abox.concepts.items():
            if c.type == "PAPER":
                title = c.canonical_name or (c.surface_variants[0].surface if c.surface_variants else "")
                pid = (c.source_papers or [""])[0]
                if title and pid:
                    CORPUS_REGISTRY[pid] = {"title": title, "surnames": [],
                                            "years": []}
    else:
        kb = agent._get_kernel()

    # register the incoming batch (citation join targets)
    register_corpus_papers(kb, papers)

    summary = {"name": name, "papers": []}
    for p in papers:
        pid = p["pid"]
        print(f"[graph:{name}] ingesting {pid} ({p.get('title','')[:45]})", flush=True)
        try:
            rep = agent.process_paper_via_kernel(pid, arm=arm)
            ci = rep.get("citation_intents") or {}
            summary["papers"].append({
                "pid": pid, "concepts": rep["n_concepts"],
                "edges": rep["n_hyperedges"],
                "cites_committed": ci.get("n_committed", 0)})
        except Exception as e:
            summary["papers"].append({"pid": pid, "error": repr(e)[:120]})
        # save after EVERY paper — a crash loses at most one paper's work
        save_graph(agent._get_kernel(), name,
                   {"papers": {pid: {"title": p.get("title", "")}}})
    summary["n_concepts"] = len(agent._get_kernel().abox.concepts)
    summary["n_hyperedges"] = len(agent._get_kernel().abox.hyperedges)
    return summary


if __name__ == "__main__":
    # CLI: python -m granular_agent.graph_store ingest <name> <papers.json>
    #       python -m granular_agent.graph_store info <name>
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ingest", "info"])
    ap.add_argument("name")
    ap.add_argument("papers_json", nargs="?")
    a = ap.parse_args()
    if a.cmd == "info":
        mp = os.path.join(STORE_ROOT, a.name, "manifest.json")
        print(json.dumps(json.load(open(mp, encoding="utf-8")),
                         ensure_ascii=False, indent=1) if os.path.exists(mp)
              else f"no graph named {a.name!r}")
    elif a.cmd == "ingest":
        papers = json.load(open(a.papers_json, encoding="utf-8"))
        print(json.dumps(ingest_papers(a.name, papers), ensure_ascii=False, indent=1))
