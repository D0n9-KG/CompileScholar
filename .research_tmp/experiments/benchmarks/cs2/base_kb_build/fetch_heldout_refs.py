# -*- coding: utf-8 -*-
"""领域层直接检验（叙事 v8 §3.3）的输入：20 篇留出综述各自引用的论文（标题+摘要+年份）。

来源：S2 /paper/arXiv:{id}/references（一篇综述 1-2 次调用）；S2 缺摘要的、有 arXiv id 的，用 arXiv API 批量补。
只取发表早于综述的条目（综述引用天然满足，仍按年份兜底）。综述本身不进输入集。
S2 无 key ~1 req/s：全程串行、间隔 3s——**不要与 CS2 引文扩展同时跑**（共享匿名配额）。
输出 base_kb_v2/heldout_refs/<survey_pid>.json：[{key,title,year,abstract,arxiv,doi,src}] + 统计。
用法：python fetch_heldout_refs.py
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "base_kb_v2")
OUT = os.path.join(V2, "heldout_refs")
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",api.semanticscholar.org,export.arxiv.org"
UA = {"User-Agent": "compilescholar-eval/0.1"}


def _get(url, tries=5, spacing=3.0):
    for a in range(tries):
        time.sleep(spacing)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(10 * 2 ** a, 120))
                continue
            print("  HTTP", e.code, url[:90])
            return None
        except Exception as e:
            print("  ERR", str(e)[:80])
            time.sleep(10)
    return None


def s2_refs(arxiv_id):
    rows, off = [], 0
    while True:
        q = urllib.parse.urlencode({"fields": "title,abstract,year,externalIds,publicationDate", "limit": 1000,
                                    "offset": off})
        b = _get(f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}/references?{q}")
        if b is None:
            break
        d = json.loads(b)
        for x in d.get("data") or []:
            p = x.get("citedPaper") or {}
            if p.get("title"):
                rows.append(p)
        if d.get("next") is None:
            break
        off = d["next"]
    return rows


def arxiv_abstracts(ids):
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        b = _get("http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)}), spacing=3.5)
        if b is None:
            continue
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for e in ET.fromstring(b).findall("a:entry", ns):
            aid = re.sub(r"v\d+$", "", (e.findtext("a:id", "", ns) or "").rsplit("/abs/", 1)[-1])
            summ = re.sub(r"\s+", " ", e.findtext("a:summary", "", ns) or "").strip()
            if aid and summ:
                out[aid] = summ
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    gold = json.load(open(os.path.join(V2, "survey_gold.json"), encoding="utf-8"))
    stats = {}
    for pid, g in gold.items():
        dest = os.path.join(OUT, f"{pid}.json")
        if os.path.exists(dest):
            rows = json.load(open(dest, encoding="utf-8"))
            stats[pid] = {"n": len(rows), "with_abstract": sum(1 for r in rows if r["abstract"]), "cached": True}
            continue
        raw = s2_refs(g["arxiv_id"])
        rows, seen = [], set()
        for p in raw:
            t = re.sub(r"\s+", " ", p["title"]).strip()
            k = re.sub(r"\W+", " ", t.lower()).strip()
            if not k or k in seen:
                continue
            seen.add(k)
            if p.get("year") and g.get("year") and int(p["year"]) > int(g["year"]):
                continue
            ext = p.get("externalIds") or {}
            rows.append({"key": k, "title": t, "year": p.get("year"), "abstract": (p.get("abstract") or "").strip(),
                         "arxiv": ext.get("ArXiv"), "doi": ext.get("DOI"), "src": "s2"})
        need = [r["arxiv"] for r in rows if not r["abstract"] and r["arxiv"]]
        if need:
            ab = arxiv_abstracts(need)
            for r in rows:
                if not r["abstract"] and r["arxiv"] in ab:
                    r["abstract"], r["src"] = ab[r["arxiv"]], "s2+arxiv"
        json.dump(rows, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        stats[pid] = {"n": len(rows), "with_abstract": sum(1 for r in rows if r["abstract"])}
        print(f"{pid} refs={len(rows)} with_abstract={stats[pid]['with_abstract']} | {g['title'][:60]}", flush=True)
    json.dump(stats, open(os.path.join(OUT, "_stats.json"), "w"), indent=1)
    print("total", sum(s["n"] for s in stats.values()), "with abstract", sum(s["with_abstract"] for s in stats.values()))


if __name__ == "__main__":
    main()
