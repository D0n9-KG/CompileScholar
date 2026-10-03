# -*- coding: utf-8 -*-
"""领域层直接检验的金标（叙事 v8 §3.3）：从留出综述自身的抽取记录构造"综述作者的领域判断"。

选综述：taxonomy/comparison 记录 ≥ 20 条的综述（有足够领域结构），按 primary_category 分层、固定种子抽 N 篇。
每篇金标（全部来自综述原文的逐字记录，不由我们生成）：
  families    —— taxonomy_node 快照的节点名（综述作者划分的方法族/子主题）
  properties  —— 节点下的性质主张（做法/优劣/适用条件）
  limitations —— challenges 类综述主张 + survey_claimed 缺口（族级局限与开放问题）
评测时（另一脚本）：这些综述从 KB 中整体移除，只给系统它们引用的论文，系统构建状态后与金标做 nugget 式对齐。
确定性、零 LLM、零网络。输出 base_kb_v2/survey_gold.json + 统计。
"""
import json
import os
import random
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "base_kb")
V2 = os.path.join(HERE, "..", "base_kb_v2")
N = int(os.environ.get("N_GOLD_SURVEYS", "20"))


def main():
    rs = json.load(open(os.path.join(BASE, "records_survey.json"), encoding="utf-8"))
    sm = {r["paper_id"]: r for r in json.load(open(os.path.join(BASE, "survey_manifest.json"), encoding="utf-8"))}
    merged = json.load(open(os.path.join(V2, "records.json"), encoding="utf-8"))
    cand = []
    for pid, p in rs.items():
        recs = p["records"]
        n_struct = sum(1 for r in recs if r.get("semantic_label") in ("taxonomy", "comparison"))
        if n_struct >= 20 and pid in sm:
            cand.append(pid)
    by_cat = defaultdict(list)
    for pid in sorted(cand):
        by_cat[sm[pid].get("primary_category") or "?"].append(pid)
    rng = random.Random(20261003)
    for v in by_cat.values():
        rng.shuffle(v)
    picked, cats = [], sorted(by_cat)
    while len(picked) < min(N, len(cand)):
        for c in cats:
            if by_cat[c] and len(picked) < N:
                picked.append(by_cat[c].pop())
    gold = {}
    for pid in picked:
        recs = (merged.get(pid) or {}).get("records") or []
        fams, props, lims = [], [], []
        for r in recs:
            if r.get("snapshot_type") == "taxonomy_node" and r.get("subject"):
                fams.append(r["subject"].strip())
                for c in r.get("claims") or []:
                    if c.get("claim"):
                        props.append({"family": r["subject"].strip(), "text": c["claim"], "quote": (r.get("quote") or "")[:400]})
            elif r.get("kind") == "survey_claim" and r.get("semantic_label") == "challenges" and r.get("claim"):
                lims.append({"about": r.get("claims_about"), "text": r["claim"], "quote": (r.get("quote") or "")[:400],
                             "src": "challenges"})
            elif r.get("kind") == "absence" and r.get("absence_type") == "survey_claimed":
                t = next((r[k] for k in ("missing_translated", "missing") if isinstance(r.get(k), str) and r[k].strip()), "")
                if t:
                    lims.append({"about": r.get("subject"), "text": t, "quote": (r.get("quote") or "")[:400], "src": "gap"})
        gold[pid] = {"title": sm[pid].get("title"), "year": sm[pid].get("year"), "arxiv_id": sm[pid].get("arxiv_id"),
                     "category": sm[pid].get("primary_category"),
                     "families": sorted(set(fams)), "properties": props, "limitations": lims}
    json.dump(gold, open(os.path.join(V2, "survey_gold.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"candidates {len(cand)} | picked {len(gold)} | categories {dict((c, sum(1 for g in gold.values() if g['category']==c)) for c in cats)}")
    for pid, g in gold.items():
        print(f"  {g['arxiv_id']:11s} fam={len(g['families']):3d} prop={len(g['properties']):3d} lim={len(g['limitations']):3d} | {g['title'][:60]}")


if __name__ == "__main__":
    main()
