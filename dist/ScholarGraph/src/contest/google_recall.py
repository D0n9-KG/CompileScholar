# -*- coding: utf-8 -*-
"""Google source (SerpAPI) — the GA-baseline lever (GOAL 2026-08-29 晚).

Published evidence (SPAR paper, arXiv:2507.15245, SPARBench F1): a bare
"Google restricted to site:arxiv.org" baseline scores 0.2451 — second only
to SPAR itself (0.3015) — while every single-API+LLM system sits at
0.004-0.024. Google's ranking is the strongest public proxy for the
"expert-picked representative papers" gold culture. This module brings that
signal into the pool as one more recall source; the grading/ranking funnel
stays unchanged.

BUDGET HARD CONSTRAINT: SerpAPI free plan, ~249 searches left this month
(2026-08-29). Design: original query + top schema-expansion queries per
benchmark query (<=4 searches), num=20 each, disk-cached forever after.
Every search must go through _serp_search() which enforces the cache —
never call SerpAPI raw.

Metadata: candidates resolve through S2's anonymous batch endpoint
(measured working: POST graph/v1/paper/batch -> title/year/citationCount/
abstract), so Google candidates carry REAL citation counts into grading and
hybrid ordering (no cited-0 handicap). Unresolved ids keep the SerpAPI
title, cited 0 (honest degrade).

BLIND by construction: search queries derive from the user query + frozen
schema expansions only. Gold never enters query generation.
"""
import json, os, re, time, urllib.parse, urllib.request

from contest.cost_ledger import ledger

_CACHE_DIR = os.path.join(".research_tmp", "contest_survey", "_google_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

_ARXIV_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})"
    r"(?:v\d+)?", re.I)


def _api_key() -> str:
    for line in open(".env", encoding="utf-8"):
        if line.startswith("SERPER_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


def _get(url: str, data=None, timeout=30) -> bytes:
    """(walled 2026-08-31) open+read inside a daemon thread with a hard wall.
    urllib's timeout does not stop slow-drip servers — same disease as the
    Paratera walls in granular_agent.llm_client (an s2match hang froze the
    demo server mid-recording). Also ignores the Windows system proxy (it
    hijacks requests to some hosts; SerpAPI/S2 are directly reachable)."""
    import threading
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=data)
    box = {}

    def _run():
        try:
            with direct.open(req, timeout=timeout) as r:
                box["b"] = r.read()
        except Exception as e:
            box["e"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout + 30)
    if "b" in box:
        return box["b"]
    if "e" in box:
        raise box["e"]
    raise TimeoutError(f"http wall {timeout + 30}s (thread-abandon): {url[:80]}")


def _cache_path(key: str) -> str:
    import hashlib
    return os.path.join(_CACHE_DIR,
                        hashlib.md5(key.encode()).hexdigest()[:16] + ".json")


def _serp_search(q: str, num: int = 20) -> list[dict]:
    """SerpAPI search, disk-cached (BUDGET: never search the same thing
    twice). Returns organic_results entries with arxiv id attached."""
    ck = _cache_path(f"serp|{q}|{num}")
    if os.path.exists(ck):
        try:
            ledger._record("http", source="google-cache", dt=0.0, ok=True)
            return json.loads(open(ck, encoding="utf-8").read())
        except Exception:
            pass
    url = ("https://serpapi.com/search?" + urllib.parse.urlencode(
        {"q": q, "num": num, "api_key": _api_key()}))
    t0 = time.time()
    d = json.loads(_get(url, timeout=60))
    out = []
    for r in (d.get("organic_results") or [])[:num]:
        link = r.get("link", "")
        m = _ARXIV_RE.search(link)
        out.append({"title": (r.get("title") or "").strip(),
                    "link": link, "position": r.get("position", 0),
                    "snippet": (r.get("snippet") or "")[:300],
                    "arxiv_id": m.group(1) if m else None})
    ledger._record("http", source="google", dt=round(time.time() - t0, 2),
                   ok=bool(out))
    with open(ck, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    time.sleep(0.5)                       # be gentle; budget is count, not rate
    return out


def _s2_batch_resolve(arxiv_ids: list[str]) -> dict[str, dict]:
    """Anonymous S2 batch lookup (measured working): {arxiv_id: {title, year,
    citationCount, abstract}}. ids not in S2 are absent from the result."""
    out: dict[str, dict] = {}
    for i in range(0, len(arxiv_ids), 100):
        batch = arxiv_ids[i:i + 100]
        body = json.dumps({"ids": [f"ARXIV:{a}" for a in batch]}).encode()
        url = ("https://api.semanticscholar.org/graph/v1/paper/batch"
               "?fields=title,year,citationCount,abstract")
        d = None
        for attempt in range(3):           # transient 429s measured on bursts
            t0 = time.time()
            try:
                d = json.loads(_get(url, data=body, timeout=60))
                ledger._record("http", source="s2batch",
                               dt=round(time.time() - t0, 2), ok=True)
                break
            except Exception:
                ledger._record("http", source="s2batch",
                               dt=round(time.time() - t0, 2), ok=False)
                time.sleep(3 + 3 * attempt)
        if d is None:
            continue
        for aid, p in zip(batch, d):
            if p and p.get("title"):
                out[aid] = {"title": p["title"],
                            "year": p.get("year") or "",
                            "citationCount": p.get("citationCount") or 0,
                            "abstract": p.get("abstract") or ""}
        time.sleep(1.5)
    return out


def google_recall(query: str, expansion_queries: list[str],
                  max_searches: int = 4, num: int = 20,
                  plain: bool = False) -> list[dict]:
    """Pool candidates from Google. Search set = the original query + top
    expansions. site:arxiv.org mode (GA evidence) or PLAIN mode (G evidence:
    recall 0.20 on AutoScholar — small-gold benchmarks with recent golds,
    where site-restriction measured HURTS: GA=0.04 vs G=0.20 there). Each
    search's results form their own _q group so they survive per-query trim
    as a group; Google positions kept on the candidate (_g).

    PLAIN mode metadata: non-arxiv results are resolved by S2's
    /paper/search/match endpoint (title match, anonymous); unresolved keep
    the SerpAPI title, cited 0 (honest degrade)."""
    searches = [query] + [q for q in expansion_queries
                          if q and q.strip() != query][:max_searches - 1]
    cands: list[dict] = []
    seen_ax = set()
    for sq in searches:
        qg = sq if plain else f"site:arxiv.org {sq}"
        for r in _serp_search(qg, num=num):
            if not r.get("title"):
                continue
            ax = r.get("arxiv_id")
            if plain and not ax:
                # non-arxiv hit: keep with title only; resolved below
                cands.append({"title": r["title"], "year": "",
                              "citationCount": 0, "abstract": "",
                              "_q": f"google:{sq[:40]}",
                              "_g": r["position"], "externalIds": {}})
                continue
            if not ax or ax in seen_ax:
                continue
            seen_ax.add(ax)
            cands.append({"title": r["title"], "year": "",
                          "citationCount": 0, "abstract": "",
                          "_q": f"google:{sq[:40]}", "_g": r["position"],
                          "externalIds": {"ArXiv": ax}})
    if not cands:
        return []
    # resolve metadata: arxiv ids via batch; plain-mode title-only rows via
    # S2 match endpoint (best effort, small N)
    ax_ids = [c["externalIds"]["ArXiv"] for c in cands if c["externalIds"]]
    meta = _s2_batch_resolve(ax_ids) if ax_ids else {}
    if plain:
        title_only = [c for c in cands if not c["externalIds"]]
        for c in title_only[:15]:            # cap: match endpoint is 1 call each
            m = _s2_match_title(c["title"])
            if m:
                c.update(m)
                c["citationCount"] = m.get("citationCount", 0)
    for c in cands:
        m = meta.get((c.get("externalIds") or {}).get("ArXiv"))
        if m:
            c.update(m)
            c["citationCount"] = m["citationCount"]
    return cands


def _s2_match_title(title: str) -> dict | None:
    """S2 /paper/search/match: single-paper title matching (anonymous).
    Returns {title, year, citationCount, abstract, arxiv_id} or None."""
    import urllib.parse
    url = ("https://api.semanticscholar.org/graph/v1/paper/search/match?"
           + urllib.parse.urlencode(
               {"query": title,
                "fields": "title,year,citationCount,abstract,externalIds"}))
    for attempt in range(2):
        t0 = time.time()
        try:
            d = json.loads(_get(url, timeout=30))
            ledger._record("http", source="s2match",
                           dt=round(time.time() - t0, 2), ok=True)
            data = (d.get("data") or [None])[0]
            if not data:
                return None
            return {"title": data.get("title") or title,
                    "year": data.get("year") or "",
                    "citationCount": data.get("citationCount") or 0,
                    "abstract": data.get("abstract") or "",
                    "externalIds": data.get("externalIds") or {}}
        except Exception:
            ledger._record("http", source="s2match",
                           dt=round(time.time() - t0, 2), ok=False)
            time.sleep(2 + 2 * attempt)
    return None
