# -*- coding: utf-8 -*-
"""多来源参考文献图（引文扩展用），不依赖 S2 key。

10-03 实测（本机出口 IP）：S2 无 key 被 IP 级限流（20/20 次 429）；OpenAlex 无 key 当日免费额度耗尽
（"$0 remaining, resets at midnight UTC"）；Crossref 可用但 CS 论文参考文献覆盖差（8 篇 arXiv/会议论文仅 1 篇有列表）；
arXiv HTML（arxiv.org/html/<id>）10/10 可解析出 ltx_bibitem 参考文献条目、单篇 ~2s。
→ 来源顺序：arXiv HTML（主）→ Crossref（有 DOI 的期刊/会议兜底）。
   S2/OpenAlex 不再走（额度不稳定、按 IP 共享，并行即被封）。

接口：references(title) -> [{"title": str, "year": int|None, "raw": str}]（种子论文的参考文献，题录级，无摘要）
      resolve_abstract(title) -> {"title","year","abstract","arxiv"} | None（只对共引排序后的前 k 条调用）
所有网络请求跨进程节流（文件锁，每个来源独立间隔）并缓存到 _shared/refgraph_cache/。
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.join(_HERE, "..", "refgraph_cache")
os.makedirs(_CACHE, exist_ok=True)
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_UA = {"User-Agent": "compilescholar-eval/0.1 (mailto:compilescholar-eval@example.org)"}
# 访问礼仪（硬约束）：export.arxiv.org API 使用条款 = 每 3 秒不超过 1 次；arxiv.org（/html 页）robots.txt
# Crawl-delay: 15（10-03 实查；此前误用 3.1s，已改）。Crossref polite pool ~1 req/s。
_GAP = {"arxiv": 3.1, "arxiv_html": 15.5, "crossref": 1.0}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").lower())).strip()


def _pace(source: str):
    from filelock import FileLock
    p = os.path.join(os.path.expanduser("~"), f".pace_{source}.json")
    with FileLock(p + ".lock"):
        try:
            last = json.load(open(p, encoding="utf-8"))["last"]
        except Exception:
            last = 0.0
        gap = _GAP[source] - (time.time() - last)
        if gap > 0:
            time.sleep(gap)
        json.dump({"last": time.time()}, open(p, "w", encoding="utf-8"))


def _get(source: str, url: str, timeout: int = 40) -> str | None:
    fn = os.path.join(_CACHE, hashlib.sha1(url.encode()).hexdigest() + ".txt")
    if os.path.exists(fn):
        t = open(fn, encoding="utf-8").read()
        return t or None
    for attempt in range(3):
        _pace(source)
        try:
            t = _OPENER.open(urllib.request.Request(url, headers=_UA), timeout=timeout).read().decode("utf-8", "replace")
            open(fn, "w", encoding="utf-8").write(t)
            return t
        except urllib.error.HTTPError as e:
            if e.code == 404:
                open(fn, "w", encoding="utf-8").write("")  # 负缓存：没有 HTML 版
                return None
            time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


# ---------------------------------------------------------------- arXiv
def arxiv_lookup(title: str) -> dict | None:
    """标题 → arXiv 条目（id、标题、年份、摘要）。只接受标题规范化后前 40 字符一致的命中。"""
    q = re.sub(r'["\\]', " ", title)[:200]
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f'ti:"{q}"', "max_results": 3})
    t = _get("arxiv", url)
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


def _bib_title(entry: str) -> str:
    """从 ltx_bibitem 纯文本里取题名：作者串之后、第一个句点结束的那一段（启发式；arXiv HTML 多为
    'Authors. Year. Title. Venue.' 或 '[n] Authors, Title, Venue, Year.'）。"""
    s = re.sub(r"^\s*(\[\d+\]|\(\d+\)|[A-Z][^()]{0,60}\(\d{4}[a-z]?\))\s*", "", entry)
    parts = [p.strip() for p in re.split(r"(?<=[a-z0-9\)])\.\s+", s) if p.strip()]
    # 去掉作者段与纯年份段，取第一个像标题的段
    def clean(p):
        # 去掉尾部的会议/预印本信息（" . ICLR"、" arXiv preprint arXiv:…"、", 2019"）
        p = re.split(r"\s+\.\s+|\s+arXiv preprint|\s+In\s+(?:Proceedings|Advances)|,\s*(?:19|20)\d\d\b", p)[0]
        return p.strip(" .,")

    def authorish(p):
        # "Jason Weston, Sumit Chopra, and Antoine Bordes"：逗号分隔、每段都是首字母大写的人名
        segs = [x.strip() for x in re.split(r",|\band\b", p) if x.strip()]
        return len(segs) >= 2 and all(re.fullmatch(r"(?:[A-Z][\w'\-]*\.?\s*){1,4}", x) for x in segs)

    venue = re.compile(r"^(In |Proceedings|arXiv|CoRR|URL|https?:|Advances in|Journal|Transactions)")
    # 优先：作者段之后的第一个非年份、非出处段（标题可以很短，如 "Memory networks"）
    for p in parts[1:]:
        p = clean(p)
        if p and not _YEAR.fullmatch(p) and not authorish(p) and not venue.match(p) and len(p.split()) >= 2:
            return p
    for p in parts:
        p = clean(p)
        if len(p.split()) >= 3 and not authorish(p) and not venue.match(p):
            return p
    return clean(parts[0]) if parts else s


def arxiv_references(arxiv_id: str) -> list[dict]:
    t = _get("arxiv_html", f"https://arxiv.org/html/{arxiv_id}")
    if not t:
        return []
    out = []
    for x in re.findall(r'<li[^>]*class="ltx_bibitem"[^>]*>(.*?)</li>', t, re.S):
        raw = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(x))).strip()
        ys = [int(y) for y in _YEAR.findall(raw)]
        out.append({"title": _bib_title(raw), "year": max(ys) if ys else None, "raw": raw[:600]})
    return out


# ---------------------------------------------------------------- Crossref（兜底：有 DOI 的论文）
def crossref_references(title: str) -> list[dict]:
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": title[:300], "rows": 3, "select": "DOI,title,reference",
         "mailto": "compilescholar-eval@example.org"})
    t = _get("crossref", url, timeout=30)
    if not t:
        return []
    try:
        items = json.loads(t)["message"]["items"]
    except Exception:
        return []
    for it in items:
        if norm((it.get("title") or [""])[0])[:40] != norm(title)[:40]:
            continue
        out = []
        for r in it.get("reference") or []:
            tt = r.get("article-title") or r.get("volume-title") or ""
            if not tt and r.get("unstructured"):
                tt = _bib_title(r["unstructured"])
            if tt:
                y = r.get("year")
                out.append({"title": tt, "year": int(y) if str(y or "").isdigit() else None,
                            "raw": (r.get("unstructured") or tt)[:600]})
        return out
    return []


# ---------------------------------------------------------------- entry
def references(title: str) -> tuple[list[dict], str]:
    """种子论文的参考文献（题录级）。返回 (refs, source)。"""
    a = arxiv_lookup(title)
    if a and a.get("arxiv"):
        refs = arxiv_references(a["arxiv"])
        if len(refs) >= 5:
            return refs, "arxiv_html"
    refs = crossref_references(title)
    if refs:
        return refs, "crossref"
    return [], "none"


def resolve_abstract(title: str) -> dict | None:
    """共引候选（题录）→ 带摘要的论文（arXiv 精确标题匹配）。"""
    a = arxiv_lookup(title)
    if a and a.get("abstract") and len(a["abstract"]) > 80:
        return a
    return None
