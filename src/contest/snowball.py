# -*- coding: utf-8 -*-
"""Citation snowball via OpenAlex (2026-08-28) — the recall-coverage leg.

From graded seed papers (H/S from the grader), expand the pool one hop:
  backward = seed's referenced_works (who it cites)
  forward  = papers citing the seed (cited_by_api_url)
Only ONE hop (SPAR's RefChain discipline: precision over recall, cost cap).

Budget discipline (the $0.1/day = 1000-request lesson):
  - disk cache keyed by openalex work id (a work's references never change
    within a day; re-runs are free)
  - batch reference resolution via the filter=openalex:W1|W2|... endpoint
    (50 ids/call = 20x cheaper than per-work calls)
  - dedup against the existing pool BEFORE fetching metadata
"""
import json, os, re, time
import urllib.parse

from .cost_ledger import ledger

_CACHE_DIR = ".research_tmp/contest_survey/_oa_snowball_cache"
_MAILTO = "2447197731@qq.com"


def _norm_title(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def _cache_path(key: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, key + ".json")


def _get(url: str, timeout: float = 40) -> dict | None:
    if "mailto=" not in url:
        url += ("&" if "?" in url else "?") + "mailto=" + _MAILTO
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"LogicKG-research (mailto:{_MAILTO})"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        ledger._record("http", source="openalex", dt=time.time() - t0, ok=True)
        return d
    except Exception as e:
        ledger._record("http", source="openalex", dt=time.time() - t0, ok=False,
                       err=repr(e)[:80])
        return None


def resolve_seeds(titles: list[str]) -> dict[str, str]:
    """Title -> openalex W-id via /works?search=title (1 request per title,
    cached). Returns {norm_title: W-id}."""
    out = {}
    for t in titles:
        key = "seed_" + _norm_title(t)[:40]
        cp = _cache_path(key)
        if os.path.exists(cp):
            d = json.loads(open(cp, encoding="utf-8").read())
        else:
            d = _get("https://api.openalex.org/works?" + urllib.parse.urlencode(
                {"search": t, "per-page": "1"})) or {}
            json.dump(d, open(cp, "w", encoding="utf-8"), ensure_ascii=False)
        results = d.get("results") or []
        if results:
            wid = results[0].get("id", "").split("/")[-1]
            if wid:
                out[_norm_title(t)] = wid
        time.sleep(0.15)   # ~6 rps polite ceiling under 1 rps key-less budget
    return out


def references_batch(wids: list[str]) -> dict[str, list[str]]:
    """W-id -> its referenced_works W-ids, via batched filter calls (50/call).
    Cached per W-id set."""
    out: dict[str, list[str]] = {}
    for i in range(0, len(wids), 50):
        chunk = wids[i:i + 50]
        ckey = "refs_" + _norm_title("|".join(chunk))[:60]
        cp = _cache_path(ckey)
        if os.path.exists(cp):
            per_work = json.loads(open(cp, encoding="utf-8").read())
        else:
            flt = "|".join(chunk)
            per_work = {}
            cursor = "*"
            while cursor:
                d = _get("https://api.openalex.org/works?" + urllib.parse.urlencode(
                    {"filter": f"openalex:{flt}", "select": "id,referenced_works",
                     "per-page": "50", "cursor": cursor})) or {}
                for w in d.get("results", []):
                    wid = w.get("id", "").split("/")[-1]
                    per_work[wid] = [r.split("/")[-1] for r in w.get("referenced_works", [])]
                cursor = ((d.get("meta") or {}).get("next_cursor"))
            json.dump(per_work, open(cp, "w", encoding="utf-8"))
        out.update(per_work)
        time.sleep(0.15)
    return out


def works_metadata(wids: list[str]) -> list[dict]:
    """Batch resolve W-ids to candidate rows (title/year/cited_by_count), 50/call."""
    rows = []
    for i in range(0, len(wids), 50):
        chunk = wids[i:i + 50]
        ckey = "meta_" + _norm_title("|".join(chunk))[:60]
        cp = _cache_path(ckey)
        if os.path.exists(cp):
            batch = json.loads(open(cp, encoding="utf-8").read())
        else:
            flt = "|".join(chunk)
            d = _get("https://api.openalex.org/works?" + urllib.parse.urlencode(
                {"filter": f"openalex:{flt}",
                 "select": "id,title,publication_year,cited_by_count",
                 "per-page": "50"})) or {}
            batch = [{"source_name": "openalex", "title": w.get("title"),
                      "year": w.get("publication_year"),
                      "citationCount": w.get("cited_by_count"),
                      "openalex_id": w.get("id", "").split("/")[-1]}
                     for w in d.get("results", []) if w.get("title")]
            json.dump(batch, open(cp, "w", encoding="utf-8"), ensure_ascii=False)
        rows.extend(batch)
        time.sleep(0.15)
    return rows


def snowball(seed_titles: list[str], pool_titles: set[str],
             max_seeds: int = 8) -> list[dict]:
    """One-hop expansion: seeds' references ∪ seed-resolution works, minus the
    existing pool. Returns new candidate rows."""
    seeds = resolve_seeds(seed_titles[:max_seeds])
    if not seeds:
        return []
    refs = references_batch(list(seeds.values()))
    # candidate wids = union of references (backward only — forward citations
    # need the cited_by endpoint per work, 8x the budget; second iteration)
    cand_wids = list({w for lst in refs.values() for w in lst})
    if not cand_wids:
        return []
    metas = works_metadata(cand_wids)
    new = [m for m in metas
           if _norm_title(m.get("title")) not in pool_titles]
    return new
