# -*- coding: utf-8 -*-
"""Multi-source reference graph for citation expansion (moved from benchmarks/_shared/tools/refgraph.py; routing,
pacing, caching and budget guards unchanged).

Routes per seed paper, stopping at the first source with >= 5 references (each step cached):
  1. OpenAlex title search (10 credits; batched across a question's seeds by prefetch(): 6 titles = 10 credits)
  2. S2 (only when not cooling down after a 429; anonymous pool ~1 req/s per IP)
  3. Crossref
  4. arXiv HTML reference list (off by default; robots Crawl-delay 15)
Co-cited candidate abstracts: OpenAlex DOI batch (1 credit / 50) then exact-title arXiv lookup.

Changes from the original: the HTTP cache lives in CS_REFGRAPH_CACHE (default: the original
benchmarks/_shared/refgraph_cache, so existing cached responses keep being used); the OpenAlex key comes from
compilescholar.core.secrets; per-source pacing state stays in ~/.pace_<src>.json (shared with running jobs).
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from ..core import paths, secrets

_CACHE = os.environ.get("CS_REFGRAPH_CACHE") or str(paths.cache() / "refgraph")
os.makedirs(_CACHE, exist_ok=True)
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_UA = "compilescholar-eval/0.1 (mailto:compilescholar-eval@example.org)"
# Access etiquette (hard constraints): export.arxiv.org <= 1 req / 3 s; arxiv.org/html robots Crawl-delay 15;
# S2 without a key ~1 req/s; OpenAlex with a key is billed per credit (0.15 s gap kept); Crossref polite ~1 req/s.
_GAP = {"arxiv": 3.1, "arxiv_html": 15.5, "s2": 1.1, "openalex": 0.15, "crossref": 1.0}
_S2_COOLDOWN_S = 1200
_OA_MIN_REMAINING = 500
STATS: dict[str, int] = {}


def _bump(k: str, n: int = 1):
    STATS[k] = STATS.get(k, 0) + n


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").lower())).strip()


def clean_title(s: str) -> str:
    """去掉题录残留的引号与尾部标点（Crossref unstructured 常见 '“C3: …,”'）。"""
    return re.sub(r"^[\s\"'“”‘’]+|[\s\"'“”‘’,.;]+$", "", s or "")


def _state_path(name: str) -> str:
    return os.path.join(os.path.expanduser("~"), f".pace_{name}.json")


def _pace(source: str):
    from filelock import FileLock
    p = _state_path(source)
    with FileLock(p + ".lock"):
        try:
            st = json.load(open(p, encoding="utf-8"))
        except Exception:
            st = {}
        gap = _GAP[source] - (time.time() - st.get("last", 0.0))
        if gap > 0:
            time.sleep(gap)
        st["last"] = time.time()
        json.dump(st, open(p, "w", encoding="utf-8"))


def _s2_cooling() -> bool:
    try:
        return time.time() < json.load(open(_state_path("s2_cooldown"), encoding="utf-8"))["until"]
    except Exception:
        return False


def _s2_cool():
    json.dump({"until": time.time() + _S2_COOLDOWN_S}, open(_state_path("s2_cooldown"), "w", encoding="utf-8"))
    _bump("s2_429")


def _oa_key() -> str | None:
    return secrets.get("OPENALEX_API_KEY")


_OA_REMAINING = [None]

# W1-9 (2026-10-04): set when any HTTP call in the current thread failed TRANSIENTLY (S2 cooling / 429, OpenAlex
# budget skip, network error, retries exhausted). references() never caches an empty route while it is set, so a
# temporary failure is retried next time instead of being remembered as "this paper has no references" forever.
_TL = __import__("threading").local()


def _mark_transient():
    _TL.transient = True
    _bump("transient_fail")


def _write_atomic(fn: str, text: str):
    tmp = fn + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, fn)


def _http(source: str, url: str, *, headers=None, data=None, timeout=40, cache=True) -> str | None:
    """Cached, paced GET/POST. Three outcomes: text (cached); a definitive negative — 404 — returns None and is
    cached as empty; a transient failure returns None, is NOT cached, and marks the thread (see _mark_transient)."""
    key = url + ("|" + data.decode() if data else "")
    fn = os.path.join(_CACHE, hashlib.sha1(key.encode()).hexdigest() + ".txt")
    if cache and os.path.exists(fn):
        t = open(fn, encoding="utf-8").read()
        _bump(f"{source}_cache")
        return t or None
    if source == "s2" and _s2_cooling():
        _mark_transient()
        return None
    for attempt in range(3):
        _pace(source)
        try:
            h = {"User-Agent": _UA, **(headers or {})}
            r = _OPENER.open(urllib.request.Request(url, data=data, headers=h), timeout=timeout)
            t = r.read().decode("utf-8", "replace")
            if source == "openalex":
                rem = r.headers.get("X-RateLimit-Remaining")
                _OA_REMAINING[0] = int(rem) if rem and rem.isdigit() else _OA_REMAINING[0]
            _bump(f"{source}_net")
            if cache:
                _write_atomic(fn, t)
            return t
        except urllib.error.HTTPError as e:
            if e.code == 404:
                if cache:
                    _write_atomic(fn, "")
                return None
            if e.code == 429:
                if source == "s2":
                    _s2_cool()
                    _mark_transient()
                    return None
                if source == "openalex":
                    _OA_REMAINING[0] = 0
                    _mark_transient()
                    return None
            time.sleep(4 * (attempt + 1))
        except Exception:
            time.sleep(4 * (attempt + 1))
    _mark_transient()
    return None


def _json(t: str | None):
    try:
        return json.loads(t) if t else None
    except Exception:
        return None


def _oa_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


# ---------------------------------------------------------------- S2
def s2_references(title: str = "", arxiv_id: str | None = None) -> list[dict]:
    pid = f"arXiv:{arxiv_id}" if arxiv_id else None
    if not pid:
        m = _json(_http("s2", "https://api.semanticscholar.org/graph/v1/paper/search/match?" +
                        urllib.parse.urlencode({"query": title[:300], "fields": "paperId,title"})))
        row = ((m or {}).get("data") or [{}])[0]
        if not row.get("paperId") or norm(row.get("title"))[:40] != norm(title)[:40]:
            return []
        pid = row["paperId"]
    d = _json(_http("s2", f"https://api.semanticscholar.org/graph/v1/paper/{urllib.parse.quote(pid)}/references?" +
                    urllib.parse.urlencode({"fields": "title,year,abstract,publicationDate,externalIds", "limit": 200})))
    out = []
    for x in (d or {}).get("data") or []:
        c = x.get("citedPaper") or {}
        if c.get("title"):
            ext = c.get("externalIds") or {}
            out.append({"title": c["title"], "year": c.get("year"), "date": c.get("publicationDate"),
                        "abstract": (c.get("abstract") or "").strip(),
                        "ids": {"arxiv": ext.get("ArXiv"), "doi": ext.get("DOI")}})
    return out


# ---------------------------------------------------------------- OpenAlex
def _oa_get(params: dict, cost: int) -> dict | None:
    key = _oa_key()
    if not key:
        return None
    if cost >= 10 and _OA_REMAINING[0] is not None and _OA_REMAINING[0] < _OA_MIN_REMAINING:
        _bump("openalex_budget_skip")
        _mark_transient()
        return None
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    return _json(_http("openalex", url, headers={"Authorization": f"Bearer {key}"}))


def _oa_refs_from_work(w: dict) -> list[dict]:
    ids = [x.split("/")[-1] for x in (w.get("referenced_works") or [])]
    out = []
    for i in range(0, len(ids), 50):
        d = _oa_get({"filter": "openalex:" + "|".join(ids[i:i + 50]), "per-page": 50,
                     "select": "id,display_name,publication_year,publication_date,abstract_inverted_index,doi,ids"}, 1)
        for x in (d or {}).get("results") or []:
            if x.get("display_name"):
                doi = (x.get("doi") or "").replace("https://doi.org/", "")
                arx = doi.split("arxiv.")[-1] if "10.48550/arxiv." in doi.lower() else None
                out.append({"title": x["display_name"], "year": x.get("publication_year"),
                            "date": x.get("publication_date"), "abstract": _oa_abstract(x.get("abstract_inverted_index")),
                            "ids": {"openalex": x["id"].split("/")[-1], "doi": doi or None, "arxiv": arx}})
    return out


def openalex_references(title: str, arxiv_id: str | None = None, allow_search: bool = False) -> list[dict]:
    w = None
    if arxiv_id:
        d = _oa_get({"filter": f"doi:https://doi.org/10.48550/arxiv.{arxiv_id}", "per-page": 1,
                     "select": "id,display_name,referenced_works"}, 1)
        w = ((d or {}).get("results") or [None])[0]
    if (not w or not w.get("referenced_works")) and allow_search:
        q = re.sub(r"[,:|()\"?!]", " ", title)[:250]
        d = _oa_get({"filter": f"title.search:{q}", "per-page": 5, "select": "id,display_name,referenced_works"}, 10)
        cands = [x for x in (d or {}).get("results") or [] if norm(x.get("display_name"))[:40] == norm(title)[:40]]
        if cands:
            w = max(cands, key=lambda x: len(x.get("referenced_works") or []))
    return _oa_refs_from_work(w) if w and w.get("referenced_works") else []


def openalex_abstracts_by_ids(rows: list[dict]) -> None:
    """就地补摘要：按 arXiv DOI / DOI 批量查（1 credit/50 篇）。"""
    need = [r for r in rows if not r.get("abstract") and ((r.get("ids") or {}).get("arxiv") or (r.get("ids") or {}).get("doi"))]
    dois = {}
    for r in need:
        ids = r["ids"]
        d = (f"10.48550/arxiv.{ids['arxiv']}" if ids.get("arxiv") else ids["doi"]).lower()
        dois.setdefault(d, []).append(r)
    keys = list(dois)
    for i in range(0, len(keys), 50):
        d = _oa_get({"filter": "doi:" + "|".join("https://doi.org/" + k for k in keys[i:i + 50]), "per-page": 50,
                     "select": "doi,abstract_inverted_index,publication_year,publication_date"}, 1)
        for x in (d or {}).get("results") or []:
            k = (x.get("doi") or "").replace("https://doi.org/", "").lower()
            ab = _oa_abstract(x.get("abstract_inverted_index"))
            for r in dois.get(k, []):
                if ab:
                    r["abstract"] = ab
                r["date"] = r.get("date") or x.get("publication_date")


# ---------------------------------------------------------------- arXiv
def arxiv_lookup(title: str) -> dict | None:
    q = re.sub(r'["\\]', " ", title)[:200]
    t = _http("arxiv", "http://export.arxiv.org/api/query?" +
              urllib.parse.urlencode({"search_query": f'ti:"{q}"', "max_results": 3}))
    if not t:
        return None
    for e in re.findall(r"<entry>(.*?)</entry>", t, re.S):
        tt = re.sub(r"\s+", " ", (re.search(r"<title>(.*?)</title>", e, re.S) or [None, ""])[1]).strip()
        if norm(tt)[:40] != norm(title)[:40]:
            continue
        aid = re.search(r"<id>https?://arxiv.org/abs/([^<]+?)(v\d+)?</id>", e)
        pub = re.search(r"<published>(\d{4})-(\d{2})", e)
        summ = re.sub(r"\s+", " ", (re.search(r"<summary>(.*?)</summary>", e, re.S) or [None, ""])[1]).strip()
        return {"arxiv": aid.group(1) if aid else None, "title": tt,
                "year": int(pub.group(1)) if pub else None, "month": int(pub.group(2)) if pub else None,
                "abstract": summ}
    return None


_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_ARXIV_IN_REF = re.compile(r"\b(?:arXiv:|arxiv\.org/abs/)(\d{4}\.\d{4,5})", re.I)


def _bib_title(entry: str) -> str:
    """ltx_bibitem 纯文本 → 题名（启发式；'Authors. Year. Title. Venue.' 或 '[n] Authors, Title, Venue, Year.'）。"""
    s = re.sub(r"^\s*(\[\d+\]|\(\d+\)|[A-Z][^()]{0,60}\(\d{4}[a-z]?\))\s*", "", entry)
    parts = [p.strip() for p in re.split(r"(?<=[a-z0-9\)])\.\s+", s) if p.strip()]

    def clean(p):
        p = re.split(r"\s+\.\s+|\s+arXiv preprint|\s+In\s+(?:Proceedings|Advances)|,\s*(?:19|20)\d\d\b", p)[0]
        return p.strip(" .,")

    def authorish(p):
        segs = [x.strip() for x in re.split(r",|\band\b", p) if x.strip()]
        return len(segs) >= 2 and all(re.fullmatch(r"(?:[A-Z][\w'\-]*\.?\s*){1,4}", x) for x in segs)

    venue = re.compile(r"^(In |Proceedings|arXiv|CoRR|URL|https?:|Advances in|Journal|Transactions)")
    for p in parts[1:]:
        p = clean(p)
        if p and not _YEAR.fullmatch(p) and not authorish(p) and not venue.match(p) and len(p.split()) >= 2:
            return p
    for p in parts:
        p = clean(p)
        if len(p.split()) >= 3 and not authorish(p) and not venue.match(p):
            return p
    return clean(parts[0]) if parts else s


def arxiv_html_references(arxiv_id: str) -> list[dict]:
    t = _http("arxiv_html", f"https://arxiv.org/html/{arxiv_id}")
    if not t:
        return []
    out = []
    for x in re.findall(r'<li[^>]*class="ltx_bibitem"[^>]*>(.*?)</li>', t, re.S):
        raw = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(x))).strip()
        ys = [int(y) for y in _YEAR.findall(raw)]
        m = _ARXIV_IN_REF.search(raw)
        out.append({"title": _bib_title(raw), "year": max(ys) if ys else None, "date": None, "abstract": "",
                    "ids": {"arxiv": m.group(1) if m else None}})
    return out


# ---------------------------------------------------------------- Crossref
def crossref_references(title: str) -> list[dict]:
    d = _json(_http("crossref", "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": title[:300], "rows": 3, "select": "DOI,title,reference",
         "mailto": "compilescholar-eval@example.org"}), timeout=30))
    for it in ((d or {}).get("message") or {}).get("items") or []:
        if norm((it.get("title") or [""])[0])[:40] != norm(title)[:40]:
            continue
        out = []
        for r in it.get("reference") or []:
            tt = r.get("article-title") or r.get("volume-title") or (_bib_title(r["unstructured"]) if r.get("unstructured") else "")
            if tt:
                y = r.get("year")
                out.append({"title": tt, "year": int(y) if str(y or "").isdigit() else None, "date": None,
                            "abstract": "", "ids": {"doi": r.get("DOI")}})
        return out
    return []


# ---------------------------------------------------------------- batched prefetch（一题所有种子一次性解析）
_PREFETCHED: set[str] = set()  # 本进程内已由批量搜索尝试过的种子（不再单篇花 10 credits 重搜）
def _route_fn(title: str) -> str:
    return os.path.join(_CACHE, "route_" + hashlib.sha1(norm(title).encode()).hexdigest() + ".json")


def prefetch(titles: list[str], deadline: float | None = None, chunk: int = 6) -> None:
    """把一题全部种子的 OpenAlex 记录按 OR 批量标题搜索一次解析（实测 6 个标题/次 = 10 credits，逐篇则 60），
    再把所有种子的 referenced_works 合并去重后按 id 批量取元数据（1 credit/50 篇）。命中且 ≥5 条参考文献的种子
    直接写路由缓存（source=openalex_search）；其余留给 references() 逐篇走 S2/Crossref。"""
    todo = [t for t in dict.fromkeys(titles) if t and not os.path.exists(_route_fn(t))]
    _PREFETCHED.update(norm(t) for t in todo)
    works: dict[str, dict] = {}
    for i in range(0, len(todo), chunk):
        if deadline is not None and time.time() > deadline:
            break
        part = todo[i:i + chunk]
        q = "|".join(re.sub(r"[,:|()\"?!]", " ", t)[:200] for t in part)
        d = _oa_get({"filter": f"title.search:{q}", "per-page": 50, "select": "id,display_name,referenced_works"}, 10)
        res = (d or {}).get("results") or []
        for t in part:
            cands = [x for x in res if norm(x.get("display_name"))[:40] == norm(t)[:40]]
            if cands:
                works[t] = max(cands, key=lambda x: len(x.get("referenced_works") or []))
    ids = sorted({x.split("/")[-1] for w in works.values() for x in (w.get("referenced_works") or [])})
    meta: dict[str, dict] = {}
    for i in range(0, len(ids), 50):
        if deadline is not None and time.time() > deadline:
            break
        d = _oa_get({"filter": "openalex:" + "|".join(ids[i:i + 50]), "per-page": 50,
                     "select": "id,display_name,publication_year,publication_date,abstract_inverted_index,doi"}, 1)
        for x in (d or {}).get("results") or []:
            doi = (x.get("doi") or "").replace("https://doi.org/", "")
            meta[x["id"].split("/")[-1]] = {
                "title": clean_title(x.get("display_name") or ""), "year": x.get("publication_year"),
                "date": x.get("publication_date"), "abstract": _oa_abstract(x.get("abstract_inverted_index")),
                "ids": {"openalex": x["id"].split("/")[-1], "doi": doi or None,
                        "arxiv": doi.split("arxiv.")[-1] if "10.48550/arxiv." in doi.lower() else None}}
    for t, w in works.items():
        refs = [meta[x.split("/")[-1]] for x in (w.get("referenced_works") or []) if x.split("/")[-1] in meta]
        refs = [r for r in refs if r["title"]]
        if len(refs) >= 5:
            json.dump({"refs": refs, "source": "openalex_search"}, open(_route_fn(t), "w", encoding="utf-8"),
                      ensure_ascii=False)
            _bump("prefetch_hit")
        else:
            _bump("prefetch_short")


# ---------------------------------------------------------------- router
def references(title: str, deadline: float | None = None, use_arxiv_html: bool = False) -> tuple[list[dict], str]:
    """种子论文的参考文献：按路由顺序取第一个 ≥5 条的来源；都不足则取条数最多的。返回 (refs, source)。

    路由（10-03 dev v9 实测后修订）：S2 无 key 在多进程下一放开就 429、冷却期一结束再 429，实际不可用；
    arXiv HTML 15.5s/页 → 种子解析仅 0.48 篇/分钟（dev 需 ~180 篇 ≈ 6h）。改为：
      1. OpenAlex 标题搜索（10 credits，无间隔；同题多条记录取 referenced_works 最多者，预印本与正式版互补）
      2. S2（仅在不冷却时，单次尝试）
      3. Crossref
      4. arXiv HTML（默认关；use_arxiv_html=True 时启用，离线补缓存用）
    deadline：墙钟上限；超时后跳过慢来源。"""
    fn = _route_fn(title)
    if os.path.exists(fn):
        d = json.load(open(fn, encoding="utf-8"))
        _bump(f"route_cache_{d['source']}")
        return d["refs"], d["source"]
    _TL.transient = False
    best: tuple[list[dict], str] = ([], "none")

    def take(refs, src):
        nonlocal best
        if len(refs) > len(best[0]):
            best = (refs, src)
        return len(refs) >= 5

    late = lambda: deadline is not None and time.time() > deadline  # noqa: E731
    # prefetch() 已对本题全部种子做过批量 OpenAlex 解析；这里只对未命中的种子补单篇 OpenAlex 搜索（10 credits）
    done = take(openalex_references(title, allow_search=norm(title) not in _PREFETCHED), "openalex_search")
    if not done and not late() and not _s2_cooling():
        done = take(s2_references(title), "s2")
    if not done and not late():
        done = take(crossref_references(title), "crossref")
    if not done and use_arxiv_html and not late():
        a = arxiv_lookup(title)
        if a and a.get("arxiv"):
            take(arxiv_html_references(a["arxiv"]), "arxiv_html")
    refs, src = best
    for r in refs:
        r["title"] = clean_title(r["title"])
    _bump(f"route_{src}")
    # Cache the route when there are references, or when it is a definitive empty: every source answered and none
    # failed transiently, and the deadline did not cut the route short (W1-9: the old condition only checked the S2
    # cooldown, so timeouts / budget skips / network errors were cached as "no references" forever).
    definitive_empty = not getattr(_TL, "transient", False) and not late() and not _s2_cooling()
    if refs or definitive_empty:
        _write_atomic(fn, json.dumps({"refs": refs, "source": src}, ensure_ascii=False))
    else:
        _bump("route_not_cached_transient")
    return refs, src


def resolve_abstracts(rows: list[dict], deadline: float | None = None) -> None:
    """就地为共引候选补摘要：OpenAlex 按 DOI 批量（便宜）→ arXiv 精确标题（每条 3s，受 deadline 约束）。"""
    openalex_abstracts_by_ids(rows)
    for r in rows:
        if deadline is not None and time.time() > deadline:
            break
        if r.get("abstract") and len(r["abstract"]) > 80:
            continue
        a = arxiv_lookup(r["title"])
        if a and a.get("abstract") and len(a["abstract"]) > 80:
            r["abstract"], r["title"] = a["abstract"], a["title"]
            r["year"] = r.get("year") or a.get("year")
            r["date"] = r.get("date") or (f"{a['year']}-{a['month']:02d}" if a.get("year") and a.get("month") else None)
            r.setdefault("ids", {})["arxiv"] = a.get("arxiv")
