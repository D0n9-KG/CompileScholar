# -*- coding: utf-8 -*-
"""Memory recall — cross-query accumulation (GOAL 2026-08-30 下午).

The structural ceiling of ②'s one-hop exploration: once K seeds are picked,
recall is locked (measured: 62% of holdout misses were never-reached; boost1
q21-23: in-pool golds still died, and 5/10 of those misses sat in titles
mined by EARLIER queries). The compensation is not more hops — it is reuse:
every reference title ever mined (cached citreports) forms a growing pool
the current query can search. Zero new LLM cost; one embedding pass.

This is "检索行为沉淀为系统能力" as a mechanism, not a slogan: each query's
exploration permanently widens recall for all future queries.

BLIND by construction: matches the query against mined titles only; gold
never enters anything.
"""
import glob, json, os

_CACHE_DIR = os.path.join(".research_tmp", "contest_survey", "_explore_cache")
_MEM_CACHE = os.path.join(_CACHE_DIR, "_memtitles_cache.json")


def _all_mined_titles_with_intent() -> list[tuple]:
    """Union of every reference title ever mined, carrying the STRONGEST
    non-background intent label seen for that title (extends/improves/... >
    background). Cached; refreshes when a new citreport file appears.
    Returns [(norm_title, display_title, intent), ...]."""
    mtime_key = 0
    for f in glob.glob(os.path.join(_CACHE_DIR, "citreport_*.json")):
        mtime_key = max(mtime_key, int(os.path.getmtime(f)))
    if os.path.exists(_MEM_CACHE):
        try:
            c = json.load(open(_MEM_CACHE, encoding="utf-8"))
            if c.get("key") == mtime_key:
                return [tuple(t) for t in c["titles"]]
        except Exception:
            pass
    import re
    _rank = {"extends": 4, "improves": 4, "replaces": 4,
             "compares": 3, "adapts": 3, "background": 1}
    def nt(s): return re.sub(r"[^a-z]", "", (s or "").lower())
    titles: dict[str, tuple[str, str, int]] = {}   # norm -> (display, intent, rank)
    for f in glob.glob(os.path.join(_CACHE_DIR, "citreport_*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for e in d.get("edges", []):
            t = (e.get("ref_title") or "").strip()
            it = e.get("intent") or "background"
            if not t:
                continue
            k = nt(t)
            r = _rank.get(it, 1)
            if k not in titles or r > titles[k][2]:
                titles[k] = (t[:160], it, r)
    out = [(k, v[0], v[1]) for k, v in titles.items()]
    try:
        json.dump({"key": mtime_key, "titles": out},
                  open(_MEM_CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        pass
    return out


def hub_recall(query: str, top_k: int = 10, min_seeds: int = 3) -> list[dict]:
    """HUB recall — the minimal second hop (GOAL 2026-08-30): papers cited
    by >= min_seeds DISTINCT ingested papers are field hubs (measured head:
    ResNet 11 seeds, Attention 10, BERT 9 — 278 hubs at >=2). Cross-seed
    co-citation is a representative-paper detector — exactly the gold
    culture of AutoScholar (Related-Work lists) and RSQ (expert-picked
    classics). Semantic match against hub titles only; zero new ingestion
    (pure citreport cache query — the cheap form of walking the citation
    graph one extra level)."""
    import re
    import math
    srcs_per_title: dict[str, tuple[set, str]] = {}
    for f in glob.glob(os.path.join(_CACHE_DIR, "citreport_*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        src = d.get("paper_id", "")
        for e in d.get("edges", []):
            t = (e.get("ref_title") or "").strip()
            if t:
                k = re.sub(r"[^a-z]", "", t.lower())
                srcs_per_title.setdefault(k, (set(), t[:160]))
                srcs_per_title[k][0].add(src)
    hubs = [(v[1], len(v[0])) for v in srcs_per_title.values()
            if len(v[0]) >= min_seeds]
    if not hubs:
        return []
    try:
        from granular_agent.hypergraph_evolution import _embed_texts_robust
        texts = [query] + [h[0] for h in hubs]
        embs = _embed_texts_robust(texts)
        if not embs or len(embs) != len(texts):
            return []
        qe = embs[0]
        def _cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na and nb else 0.0
        scored = sorted(((h, _cos(qe, e)) for h, e in zip(hubs, embs[1:])),
                        key=lambda t: -t[1])[:top_k]
    except Exception:
        return []
    from contest.google_recall import _s2_match_title
    out = []
    for (disp, n_seeds), _sc in scored:
        m = _s2_match_title(disp) or {}
        out.append({"title": m.get("title") or disp,
                    "year": m.get("year", ""),
                    "citationCount": m.get("citationCount", 0),
                    "abstract": m.get("abstract", ""),
                    "_q": f"hub:{n_seeds}seeds", "_via": disp[:60],
                    "externalIds": m.get("externalIds", {})})
    return out


def memory_recall(query: str, top_k: int = 20) -> list[dict]:
    """Semantic match of the query against ALL accumulated mined titles ->
    candidates. INTENT-ROUTED (GOAL 2026-08-30, user direction): titles the
    classifier labeled extends/improves/compares/replaces/adapts get a
    score boost BEFORE the semantic cut — the intent edges are the
    exploration directions the mechanism design says to walk; background
    titles (survey enumerations) stay searchable but ranked behind intent
    edges at equal similarity. Graph-growth recall: earlier queries'
    exploration serves the current one.

    EMBEDDING CACHE (GOAL 2026-08-30 01:30 fix): the title pool is
    append-only — re-embedding all 4400+ titles per query measured 5.4h at
    night provider rates (the RSQ q1 killer). Title embeddings now live in
    a disk cache keyed by content hash; only NEW titles embed. The query
    itself always embeds fresh (1 batch)."""
    import re
    import math
    import hashlib
    pairs = _all_mined_titles_with_intent()
    if not pairs:
        return []
    cache_path = os.path.join(_CACHE_DIR, "title_embs.json")
    try:
        emb_cache = json.load(open(cache_path, encoding="utf-8"))
    except Exception:
        emb_cache = {}
    # figure out which titles are new (cache stores {norm_title: vec})
    new_pairs = [p for p in pairs if p[0] not in emb_cache]
    if new_pairs:
        try:
            from granular_agent.hypergraph_evolution import _embed_texts_robust
            new_embs = _embed_texts_robust([p[1] for p in new_pairs])
            if new_embs and len(new_embs) == len(new_pairs):
                for p, e in zip(new_pairs, new_embs):
                    emb_cache[p[0]] = e
                try:
                    os.makedirs(_CACHE_DIR, exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(emb_cache, f)
                except Exception:
                    pass
        except Exception:
            pass
    try:
        from granular_agent.hypergraph_evolution import _embed_texts_robust
        q_embs = _embed_texts_robust([query])
        if not q_embs:
            return []
        qe = q_embs[0]
    except Exception:
        return []
    embs = [emb_cache.get(p[0]) for p in pairs]
    # fall back: any title still missing an embedding is skipped (not scored)
    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0
    _INTENT_BOOST = {"extends": 0.15, "improves": 0.15, "compares": 0.10,
                     "replaces": 0.15, "adapts": 0.10}
    scored = []
    for p, e in zip(pairs, embs):
        if not e:
            continue
        scored.append((p, _cos(qe, e) + _INTENT_BOOST.get(p[2], 0.0)))
    scored.sort(key=lambda t: -t[1])
    scored = scored[:top_k]
    if not scored:
        return []
        qe = embs[0]
        def _cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na and nb else 0.0
    # resolve top-k display titles to real rows via S2 batch (title words)
    from contest.google_recall import _s2_batch_resolve
    from contest.pipeline import _s2_bulk_one
    out = []
    seen = set()
    for (norm, disp), _sc in scored:
        if norm in seen:
            continue
        seen.add(norm)
        toks = re.sub(r"[^a-z0-9 ]", " ", disp.lower()).split()
        tq = " ".join(toks[:7])
        try:
            for row in _s2_bulk_one(tq)[:3]:
                out.append({"title": row.get("title") or disp,
                            "year": row.get("year", ""),
                            "citationCount": row.get("citationCount", 0),
                            "abstract": row.get("abstract", ""),
                            "_q": "memory:recall", "_via": disp[:60],
                            "externalIds": row.get("externalIds", {})})
                break                     # top-1 resolution per title
        except Exception:
            continue
    return out
