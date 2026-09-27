# -*- coding: utf-8 -*-
"""MDAQA 300 题分层选样（预注册冻结，PREREG-MDAQA-subset.md）。

分层维度（设计档裁定）：
  1. 比较型优先（regex: compare/difference/versus/differ/contrast——
     实测 3,390/6,804=50%，预注册口径"34%"是旧调研；以本脚本冻结的
     regex 为准并披露）
  2. 社区大小分层：2 篇 / 3 篇 / 4 篇（实测 6391/405/8）
  3. 领域分层：support 论文主类目（arXiv 类别取第一篇的第一个类目，
     简化为领域桶——CS 子类合并为 cs，其余单列）

规模：300（用户裁定 2026-09-28）。随机种子=20260928，选样清单落盘
（qid+question+support 全量存档，可复现）。
"""
import json
import random
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

SEED = 20260928
N = 300
DATA = (r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp"
        r"\benchmark-audit-0926\mdaqa-data.json")
OUT = (r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp"
       r"\experiments\benchmarks\mdaqa\subset_300.json")


def domain_of(support, cache):
    """support 第一篇 arXiv id -> 主类目（简化领域桶）。"""
    aid = support[0]
    if aid in cache:
        return cache[aid]
    cat = "unknown"
    try:
        url = (f"http://export.arxiv.org/api/query?id_list={aid}"
               "&max_results=1")
        req = urllib.request.Request(url, headers={
            "User-Agent": "CompileScholar/0.1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            root = ET.fromstring(r.read())
        ns = {"a": "http://www.w3.org/2005/Atom"}
        cats = [c.get("term") for c in root.findall(
            ".//a:category", ns)]
        if cats:
            c = cats[0]
            cat = "cs" if c.startswith("cs.") else c.split(".")[0]
    except Exception:
        pass
    cache[aid] = cat
    import time
    time.sleep(3.05)   # arXiv 限流纪律（每 3 秒 1 请求；实测 71 分钟
    #  走完 1370 个唯一 id——超采阶段接受，正式冻结只跑一次）
    return cat


def main():
    d = json.load(open(DATA, encoding="utf-8"))
    comp_re = re.compile(
        r"compare|difference|versus|differ|contrast", re.I)
    # strata: (is_comparison, community_size, domain)
    cache = {}
    strata = defaultdict(list)
    print("building strata (arXiv category lookups, rate-limited)...",
          flush=True)
    for x in d:
        size = len(x["support"])
        comp = bool(comp_re.search(x["question"]))
        # 领域只对抽样候选查（全量 6804×3s=5.7h 太慢）——两步法：
        # 先按 (comp, size) 分层随机抽 5×超采，再对超采集查领域做配额
        strata[(comp, size)].append(x)
    rng = random.Random(SEED)
    # 超采：每 (comp,size) 层按比例抽 5 倍
    oversample = []
    quota = {}
    for key, rows in sorted(strata.items(), key=str):
        k_share = len(rows) / len(d)
        n_take = max(1, int(N * k_share * 5))
        rng.shuffle(rows)
        oversample.append((key, rows[:n_take]))
        quota[key] = max(1, round(N * k_share))
    flat = [(key, x) for key, rows in oversample for x in rows]
    print(f"oversampled {len(flat)}; resolving domains for them...",
          flush=True)
    dom = defaultdict(list)
    for key, x in flat:
        d_ = domain_of(x["support"], cache)
        dom[(key[0], key[1], d_)].append(x)
    # 领域桶内配额：同 (comp,size) 下按领域比例分
    final = []
    by_cs = defaultdict(list)
    for (comp, size, d_), rows in dom.items():
        by_cs[(comp, size)].extend(rows)
    for key, rows in by_cs.items():
        q = quota[key]
        # 领域多样性：每领域轮转取（round-robin），不足从别的领域补
        by_dom = defaultdict(list)
        for x in rows:
            dom_of_x = next(
                dk for (c, s, dk), rs in dom.items()
                if (c, s) == key and x in rs)
            by_dom[dom_of_x].append(x)
        picked = []
        while len(picked) < q and any(by_dom.values()):
            for dk in sorted(by_dom):
                if by_dom[dk]:
                    picked.append(by_dom[dk].pop(0))
                    if len(picked) >= q:
                        break
        final.extend(picked[:q])
    print(f"selected {len(final)} questions", flush=True)
    # 存档：全量字段（可复现）
    out = {
        "seed": SEED, "n": len(final), "source": DATA,
        "strata_key": ["is_comparison(regex)", "community_size",
                       "domain(support[0] primary category)"],
        "comparison_regex": comp_re.pattern,
        "questions": [
            {"qid": x["id"], "question": x["question"],
             "answer": x["answer"], "support": x["support"]}
            for x in sorted(final, key=lambda z: z["id"])],
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    sizes = Counter(len(x["support"]) for x in final)
    ncomp = sum(1 for x in final if comp_re.search(x["question"]))
    print(f"saved -> {OUT}")
    print(f"community sizes: {dict(sorted(sizes.items()))} | "
          f"comparison: {ncomp}/{len(final)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
