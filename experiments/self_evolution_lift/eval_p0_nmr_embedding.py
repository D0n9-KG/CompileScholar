# -*- coding: utf-8 -*-
"""P0-2 NMR (Node Matching Rate) via embedding + Hungarian.

替代 LLM 对齐做节点匹配: 更稳定可复现, 解 M1/M17 歧义靠 aliases 丰富 gold 表征.
gold 方法 = name + 所有 aliases 都 embed 取均值作 gold 表征; lift 方法 = name embed.
Hungarian (scipy linear_sum_assignment) 最优匹配, cosine >= 阈值算命中.

NMR 口径:
  NMR_recall   = 命中数 / 53            (lift 节点对 gold 的覆盖)
  NMR_prec     = 命中数 / len(lift方法)  (lift 节点里多少真对上 gold)
  fair_NMR     = 命中数 / 输入论文涉及的gold方法数 (公平分母, 需 --papers)
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from scipy.optimize import linear_sum_assignment

from granular_agent.llm_client import embed_batch, cosine_sim

GOLD = os.path.join(os.path.dirname(__file__), "gold", "ARFM2024_gold.json")


def gold_repr(m):
    """gold 方法表征文本: name + aliases (英文/符号, embedding 友好)."""
    parts = [m["name"]] + m.get("aliases", [])
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lift_json")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threshold", type=float, default=0.65, help="cosine 命中阈值")
    ap.add_argument("--papers", default=None, help="逗号分隔输入论文前缀, 算 fair_NMR")
    ap.add_argument("--gold", default=None, help="gold json 路径(默认ARFM2024)")
    args = ap.parse_args()

    lift = json.load(open(args.lift_json, encoding='utf-8'))
    gold_path = args.gold or GOLD
    gold = json.load(open(gold_path, encoding='utf-8'))
    lift_names = [m["name"] for m in lift["methods"]]
    gold_methods = gold["methods"]
    gold_texts = [gold_repr(m) for m in gold_methods]

    print(f"lift: {len(lift_names)} methods | gold: {len(gold_methods)} methods")
    print(f"embedding (lift={len(lift_names)}, gold={len(gold_texts)})...", flush=True)
    le = embed_batch(lift_names) if lift_names else []
    ge = embed_batch(gold_texts)
    if not le or not ge:
        print("[ERR] embedding empty"); return

    # cosine 代价矩阵
    nl, ng = len(le), len(ge)
    sim = np.zeros((nl, ng))
    for i in range(nl):
        for j in range(ng):
            sim[i, j] = cosine_sim(le[i], ge[j])
    # Hungarian on -sim (最大化 sim)
    rows, cols = linear_sum_assignment(-sim)
    matches = []
    for i, j in zip(rows, cols):
        s = sim[i, j]
        if s >= args.threshold:
            matches.append({"lift": lift_names[i], "gold_id": gold_methods[j]["id"],
                            "gold_name": gold_methods[j]["name"], "cosine": round(s, 3)})

    print(f"\n=== Hungarian 匹配 (cosine>={args.threshold}) ===")
    for m in matches:
        print(f"  [{m['cosine']:.3f}] {m['lift'][:35]} -> {m['gold_id']} {m['gold_name'][:25]}")
    hit = len(matches)
    nmr_recall = hit / len(gold_methods)
    nmr_prec = hit / len(lift_names) if lift_names else 0.0
    print(f"\nNMR_recall = {hit}/53 = {nmr_recall:.3f}")
    print(f"NMR_prec   = {hit}/{len(lift_names)} = {nmr_prec:.3f}")

    # 检查 M1/M17 歧义: gold M1 (μ(I) rheology) vs M17 (I-gradient) 是否被正确区分
    m1 = next((m for m in matches if m["gold_id"] == "M1"), None)
    m17 = next((m for m in matches if m["gold_id"] == "M17"), None)
    print(f"\nM1/M17 歧义检查: M1命中={m1['lift'][:30] if m17 else None}, M17命中={m17['lift'][:30] if m17 else None}")

    fair = None
    if args.papers:
        import re
        def _pr(r):
            r = r.replace('é','e').replace('ü','u').replace('ç','c')
            m = re.search(r'([A-Za-z]+)', r); y = re.search(r'(19|20)\d{2}', r)
            return (m.group(1).lower()[:5], int(y.group(0))) if m and y else None
        inp = set(_pr(p.replace('_',' ')) for p in args.papers.split(","))
        inp_cov = sum(1 for m in gold_methods
                      if any(_pr(r) and _pr(r) in inp for r in m.get("refs",[])))
        hit_in_inp = sum(1 for m in matches
                         if any(_pr(r) and _pr(r) in inp for r in gold_methods[[g["id"] for g in gold_methods].index(m["gold_id"])].get("refs",[])))
        fair = {"input_methods": inp_cov, "hit_in_input": hit_in_inp,
                "fair_NMR": hit_in_inp / inp_cov if inp_cov else 0.0}
        print(f"fair_NMR = {hit_in_inp}/{inp_cov} = {fair['fair_NMR']:.3f}")

    tag = args.tag or os.path.splitext(os.path.basename(args.lift_json))[0]
    out = {"tag": tag, "threshold": args.threshold, "lift_methods": len(lift_names),
           "matches": matches, "n_hits": hit, "nmr_recall": nmr_recall,
           "nmr_prec": nmr_prec, "fair_nmr": fair}
    outp = os.path.join(os.path.dirname(args.lift_json), f"{tag}_nmr.json") if not args.gold or args.gold == GOLD else \
           os.path.join(os.path.dirname(args.lift_json), f"{tag}_nmr_{os.path.splitext(os.path.basename(gold_path))[0]}.json")
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
