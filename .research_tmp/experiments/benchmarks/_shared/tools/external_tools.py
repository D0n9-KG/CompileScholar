# -*- coding: utf-8 -*-
"""External retrieval tools for the answering loop (broker wiring, 2026-09-25).

Injects four tools onto the KBTools instance used by the Multi answering
harness:

  search_papers(query, k?)      blind external search (shared tiered engine)
  gap_search(query)             gap-driven: absence coordinates -> directed queries
  lineage_walk(entity, direction?)  typed genealogy walk -> frontier -> external continuation
  citation_graph(doi|title, direction?)  external citation neighbors + corpus annotation

Design decisions:
  - injected in multi_ours_run.build_tools_multi (fork-side); the frozen home
    modules (evidence_b2_tools / harness core) stay untouched — the harness
    already dispatches any kb attribute present in TOOL_WHITELIST, which we
    extend at runtime
  - every observation is a self-contained dict with the same provenance
    discipline as typed tools (title/doi/year/abstract per paper), so note
    lines can cite [paper title] or [doi] backrefs — SQA2's citation format
    (LLM attribution over snippets) accepts free-form ids
  - external candidates render TITLE + ABSTRACT (the official asta tool uses
    pd["text"] = pd["abstract"]; we match that shape) — enough for the model
    to judge relevance AND to cite as evidence at SQA2
  - Tier-1 coarse extraction trigger is EXPOSED as a flag on the observation
    (deep_dive: true) — the answering prompt decides when to promote; the
    promotion itself runs coarse_extract (one light call) and is wired here
    as extract_paper(abstract) so the loop can call it as a tool too
"""

from __future__ import annotations

import json
import os
import sys

_SRC = r"C:\Users\D0n9\Desktop\CompileScholar\src"
_SCIEVO = r"C:\Users\D0n9\Desktop\sci-evo-extract\src"
for _p in (_SRC, _SCIEVO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _ensure_env():
    if not os.environ.get("SCIVERSE_API_TOKEN"):
        try:
            for line in open(os.path.join(
                    r"C:\Users\D0n9\Desktop\CompileScholar", ".env"),
                    encoding="utf-8"):
                if line.strip().startswith("SCIVERSE_API_TOKEN"):
                    _, _, v = line.strip().partition("=")
                    os.environ["SCIVERSE_API_TOKEN"] = v.strip()
        except Exception:
            pass
    os.environ.setdefault("LOCAL_MODEL", os.environ.get("OURS_MODEL", "Qwen3.8-27B").split(":")[-1])
    os.environ.setdefault("LOCAL_EMBED_MODEL", "qwen3-embedding-8b-local")
    os.environ.setdefault("EMBEDDING_MODEL", "qwen3-embedding-8b-local")


def _cand_row(c) -> dict:
    """One external candidate -> harness observation row (title+abstract
    evidence, official asta shape)."""
    raw = getattr(c, "raw", None) or {}
    abstract = ""
    for k in ("abstract", "summary"):
        v = raw.get(k) if isinstance(raw, dict) else None
        if isinstance(v, str) and v.strip():
            abstract = v
            break
    # OpenAlex inverted-index reconstruction
    inv = raw.get("abstract_inverted_index") if isinstance(raw, dict) else None
    if not abstract and isinstance(inv, dict) and inv:
        slots = {int(p): str(w) for w, ps in inv.items()
                 for p in (ps if isinstance(ps, list) else [])}
        abstract = " ".join(slots[k] for k in sorted(slots))
    return {
        "title": getattr(c, "title", None),
        "year": getattr(c, "year", None),
        "doi": getattr(c, "normalized_doi", None),
        "source": getattr(c, "source_name", None),
        "abstract": (abstract[:900] + "…") if len(abstract) > 900 else abstract,
    }


def _truncate_obs(rows: list[dict], cap: int = 6500) -> list[dict]:
    """Fit observation to the harness OBS_CAP envelope: keep rows until the
    cap, always keep title/doi/source, shrink abstracts first."""
    out = []
    used = 200
    for r in rows:
        row_json = json.dumps(r, ensure_ascii=False)
        if used + len(row_json) <= cap:
            out.append(r)
            used += len(row_json)
            continue
        # shrink abstract to fit
        budget = cap - used - 250
        if budget > 200 and r.get("abstract"):
            r = dict(r, abstract=r["abstract"][:budget] + "…")
            out.append(r)
            used += len(json.dumps(r, ensure_ascii=False))
        break
    return out


class ExternalTools:
    """Lazy holder — the engine stack is built once per answering process."""

    def __init__(self, views: dict, kb_manifest: dict, model: str = "local:Qwen3.8-27B",
                 registry: dict = None, blocklist: list = None,
                 backflow_path: str = None, tier_db: str = None):
        _ensure_env()
        self._model = model
        self._views = views
        self._kb_manifest = kb_manifest
        self._registry = registry
        self._blocklist = blocklist
        self._backflow_path = backflow_path
        self._tier_db = tier_db
        self._tiers = None
        self._svc = None
        self._gap_docs = None
        self._embed = None
        self._citgraph = None

    def _tier_store(self):
        """Lazy sci-evo SQLite tier store (batch-3): the system of record
        for coarse/deep extraction state — re-extraction is refused across
        sessions, promotion candidates queue for Tier 2."""
        if self._tiers is None:
            if not self._tier_db:
                self._tiers = False
                return None
            try:
                from sci_evo_extract.library.coarse_store import TierStore
                root = os.path.dirname(self._tier_db)
                self._tiers = TierStore(self._tier_db,
                                        os.path.join(root, "growth_library"))
            except Exception:
                self._tiers = False
        return self._tiers or None

    # ---- lazy singletons -------------------------------------------------

    def _service(self):
        if self._svc is None:
            from sci_evo_extract.library.search_service import SearchService
            self._svc = SearchService(limit=10)
        return self._svc

    def _gap_index(self):
        if self._gap_docs is None:
            from sci_evo_extract.library.gap_search import build_gap_docs
            self._gap_docs = build_gap_docs(self._views)
        return self._gap_docs

    def _embed_fn(self):
        if self._embed is None:
            try:
                from kb_infra.embedding import embed_local
                self._embed = embed_local
            except Exception:
                self._embed = False  # token-overlap fallback in gap_search
        return self._embed or None

    def _citation(self):
        if self._citgraph is None:
            from sci_evo_extract.library.citation_graph import CitationGraphService
            manifest_rows = [
                {"paper_id": pid, "doi": (m or {}).get("doi"),
                 "title": (m or {}).get("title")}
                for pid, m in (self._kb_manifest or {}).items()
            ]
            self._citgraph = CitationGraphService(corpus_manifest=manifest_rows)
        return self._citgraph

    # ---- tool surfaces (harness contract: plain dicts) --------------------

    def search_papers(self, query: str, k: int = 8) -> dict:
        """Blind external search through the shared tiered engine
        (Sciverse semantic leads, keyword fallbacks). Returns title+abstract
        rows usable as evidence."""
        try:
            # agent mode (P14): the loop's query is already refined keyword
            # vocabulary — skip the A3 decomposition, semantic-first single
            # pass with keyword fallback
            r = self._service().search(query, mode="agent", limit=k)
            rows = [_cand_row(c) for c in r.candidates[:k]]
            return {"tool": "search_papers", "n": len(rows), "papers": _truncate_obs(rows),
                    "latency_ms": {k2: v for k2, v in r.latency.items()},
                    "note": "external papers; cite by [title] or [doi] in notes"}
        except Exception as e:
            return {"tool": "search_papers", "n": 0,
                    "error": f"external search failed: {str(e)[:120]}"}

    def gap_search(self, query: str) -> dict:
        """Gap-driven search: matches the query against the knowledge base's
        recorded ABSENCES (what methods cannot do / what has no results),
        then searches externally for papers that FILL the matched gap —
        using vocabulary the question itself does not contain."""
        try:
            from sci_evo_extract.library.gap_search import gap_search as gs
            r = gs(query, gap_docs=self._gap_index(),
                   search_fn=lambda q, k: self._service().search(q, mode="agent", limit=k),
                   embed=self._embed_fn(), limit=8)
            gaps = [{"gap": h.gap_doc["doc"][:140], "score": h.score,
                     "kind": h.gap_doc.get("kind")}
                    for h in r.gaps]
            rows = [_cand_row(c) for c in r.candidates]
            return {"tool": "gap_search", "n": len(rows), "gaps": gaps,
                    "papers": _truncate_obs(rows),
                    "fill_annotations": r.fill_annotations[:8],
                    "note": "candidates annotated as gap fillers where matched"}
        except Exception as e:
            return {"tool": "gap_search", "n": 0,
                    "error": f"gap search failed: {str(e)[:120]}"}

    def lineage_walk(self, entity: str, direction: str = "successors") -> dict:
        """Typed method-genealogy walk: in-corpus extends/improves/replaces
        edges to the frontier, then external continuation for successors.
        Use for 'latest advances in X / what replaced X' questions."""
        try:
            from sci_evo_extract.library.lineage_walk import lineage_walk as lw
            r = lw(entity, self._views, direction=direction, external=True, limit=6)
            chain = [{"entity": n.canonical, "via": n.via_relation,
                      "depth": n.depth} for n in r.in_corpus]
            frontier = [n.canonical for n in r.frontier]
            rows = [_cand_row(c) for c in r.external_candidates]
            return {"tool": "lineage_walk", "entity": entity,
                    "corpus_chain": chain, "frontier": frontier,
                    "attached_external": r.attached_external,
                    "n_external": len(rows), "papers": _truncate_obs(rows),
                    "successor_annotations": r.successor_annotations[:8],
                    "note": "frontier = corpus lineage ends here; papers are external continuations; "
                            "attached_external = papers already coarse-extracted into the library "
                            "(their records are included — no re-search needed)"}
        except Exception as e:
            return {"tool": "lineage_walk", "n_external": 0,
                    "error": f"lineage walk failed: {str(e)[:120]}"}

    def citation_graph(self, doi: str = None, title: str = None,
                       direction: str = "both") -> dict:
        """External citation neighbors of a paper (S2; OpenAlex fallback for
        outgoing). Endpoints matched against our corpus are marked."""
        try:
            g = self._citation().graph(doi=doi, title=title,
                                       direction=direction, max_total=20)
            edges = [{"dir": e.direction, "title": e.title, "year": e.year,
                      "doi": e.doi,
                      "in_corpus": e.in_corpus_paper_id}
                     for e in g.edges[:20]]
            return {"tool": "citation_graph", "anchor": g.anchor,
                    "n_edges": len(edges), "edges": _truncate_obs(edges, 5500),
                    "n_in_corpus": g.n_in_corpus,
                    "note": "in_corpus=paper_id means the neighbor is already in our KB"}
        except Exception as e:
            return {"tool": "citation_graph", "n_edges": 0,
                    "error": f"citation graph failed: {str(e)[:120]}"}

    def extract_paper(self, title: str, abstract: str) -> dict:
        """Tier-1 coarse extraction: digest ONE external paper's abstract into
        structured records (method/finding/limitation + lineage mentions).
        Run this when an external paper looks core to the question; the
        records are abstract-level (provenance=coarse), not full-text."""
        try:
            from kb_compiler.records.coarse_extract import extract_coarse
            r = extract_coarse(title, abstract, None, self._model)
            recs = [{"kind": x["kind"], "subject": x["subject"],
                     "claim": x["claim"], "quote": x["quote"][:200],
                     "mentions": x["mentions"],
                     # citation handle per record so note lines can backref
                     # the paper (behavior probe: the model invented
                     # [extract_paper] tool-name citations when records
                     # carried no citable id — 4 consecutive reject steps)
                     "title": title}
                    for x in r["records"]]
            obs = {"tool": "extract_paper", "n": len(recs), "records": recs,
                   "title": title,
                   "note": "coarse records from abstract only; cite this paper in notes as [title]"}
            # mentions backflow (batch-2): the records' mentions match
            # registry entities -> attach this paper to the corpus
            # genealogy as an external continuation node. The library
            # grows; later lineage_walks reach this paper at zero cost.
            try:
                if self._registry is not None and r["records"]:
                    from kb_compiler.records.backflow import build_backflow, apply_backflow
                    bf = build_backflow(
                        r, {"title": title, "year": None},
                        self._registry, self._blocklist)
                    res = apply_backflow(self._views, bf,
                                         persist_path=self._backflow_path)
                    if res.get("attached"):
                        obs["attached_to_lineage"] = res["matched_entities"]
                        obs["library_growth"] = (
                            "paper attached as external continuation of: "
                            + ", ".join(res["matched_entities"]))
            except Exception as e:
                obs["backflow_error"] = str(e)[:120]
            # batch-3: tier state in the sci-evo SQLite registry — the
            # paper is recorded as coarse-extracted (re-extraction refused
            # across sessions); the deterministic promotion rule (limitation
            # signal / multi-lineage anchor) is surfaced to the loop
            try:
                ts = self._tier_store()
                if ts is not None:
                    matched = res["matched_entities"] if res.get("attached") or res.get("matched_entities") else []
                    st = ts.register_coarse(
                        title=title, year=None, abstract=abstract[:6000],
                        records=r["records"], matched_entities=matched)
                    if st.get("reused"):
                        obs["tier_note"] = ("paper already registered "
                                            f"(tier={st['tier']}) — no re-extraction")
                    if st.get("promotion_candidate"):
                        obs["promotion_candidate"] = True
                        obs["promotion_note"] = (
                            "limitation/multi-lineage signal — strong Tier-2 "
                            "candidate (full-text extraction at library-growth time)")
            except Exception as e:
                obs["tier_error"] = str(e)[:120]
            return obs
        except Exception as e:
            return {"tool": "extract_paper", "n": 0,
                    "error": f"coarse extraction failed: {str(e)[:120]}"}


def attach_external_tools(kb, views: dict, kb_manifest: dict,
                          model: str = "local:Qwen3.8-27B",
                          registry: dict = None, blocklist: list = None,
                          backflow_path: str = None,
                          tier_db: str = None) -> ExternalTools:
    """Inject the five external tools onto a KBTools instance and extend the
    harness tool whitelist/catalog at runtime. `registry` enables mentions
    backflow (extract_paper -> genealogy attachment); `backflow_path`
    persists + replays library growth across sessions; `tier_db` puts
    coarse/deep extraction state into the sci-evo SQLite registry."""
    ext = ExternalTools(views, kb_manifest, model, registry=registry,
                        blocklist=blocklist, backflow_path=backflow_path,
                        tier_db=tier_db)
    if backflow_path:
        try:
            from kb_compiler.records.backflow import load_backflow
            n = load_backflow(views, backflow_path)
            if n:
                print(f"[backflow] replayed {n} external papers into "
                      f"genealogy", flush=True)
        except Exception:
            pass
    kb.search_papers = ext.search_papers
    kb.gap_search = ext.gap_search
    kb.lineage_walk_ext = ext.lineage_walk
    kb.citation_graph = ext.citation_graph
    kb.extract_paper = ext.extract_paper
    # whitelist/catalog extension (runtime, fork-side; frozen home untouched)
    try:
        from evidence_b2_tools import TOOL_WHITELIST, TOOL_SIGS
        TOOL_WHITELIST.update({"search_papers", "gap_search",
                               "lineage_walk_ext", "citation_graph",
                               "extract_paper"})
        TOOL_SIGS.update({
            "search_papers": "(query,k?)",
            "gap_search": "(query)",
            "lineage_walk_ext": "(entity,direction?)",
            "citation_graph": "(doi?,title?,direction?)",
            "extract_paper": "(title,abstract)",
        })
    except Exception:
        pass
    return ext
