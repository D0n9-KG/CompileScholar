# -*- coding: utf-8 -*-
"""领域状态 v2 的实体同一性（族合并 + 局限/比较挂载），取代 build_state_v2 的"规范化名精确匹配"（挂载率 ~11%）。

方法（候选生成 + 判定，确定性可复现）：
  1. 节点名、未挂局限/比较的 about 名 → 同一嵌入模型（local qwen3-embedding-8b）向量；
  2. 族合并：节点名两两余弦 ≥ T_MERGE 且共享 ≥1 个非泛化内容词 → 并查集合并；
  3. 挂载：about 名与族代表名最近邻余弦 ≥ T_ATTACH 且共享内容词 → 挂到该族；
  4. 泛化名（DF 过高的单词名，如 "methods"/"models"/"deep learning" 这类跨多领域的大族）不参与合并的传递，防止雪球。
阈值从数据里标定并打印分布供人工抽检（输出 merge_audit.json：每档阈值附 20 个随机样本）。
输出 base_kb_v2/state_merged.json。
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "src")))
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",192.168.199.73"
from kb_infra.embedding import embed_local  # noqa: E402

V2 = os.path.join(HERE, "..", "base_kb_v2")
T_MERGE = float(os.environ.get("T_MERGE", "0.86"))
T_ATTACH = float(os.environ.get("T_ATTACH", "0.80"))
STOP = set("method methods approach approaches technique techniques model models based the of and for in on with using "
           "learning deep neural network networks system systems framework frameworks algorithm algorithms data".split())


def content(s):
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP and len(w) > 2}


def emb(texts):
    out = []
    for i in range(0, len(texts), 64):
        out.extend(embed_local(texts[i:i + 64], batch_size=64))
    M = np.asarray(out, dtype="float32")
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def main():
    st = json.load(open(os.path.join(V2, "state.json"), encoding="utf-8"))
    fams = st["families"]
    names = [f["names"][0] for f in fams]
    E = emb(names)
    # 泛化名：内容词集合为空或节点出现在 ≥ 3% 的综述里（名太泛，不作合并传递桥）
    generic = {i for i, n in enumerate(names) if not content(n)}
    parent = list(range(len(fams)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    S = E @ E.T
    np.fill_diagonal(S, 0)
    pairs = np.argwhere(S >= T_MERGE)
    merged_pairs = []
    for i, j in pairs:
        if i >= j or i in generic or j in generic:
            continue
        if not (content(names[i]) & content(names[j])):
            continue
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a
            merged_pairs.append((names[i], names[j], float(S[i, j])))
    groups = defaultdict(list)
    for i in range(len(fams)):
        groups[find(i)].append(i)
    new = []
    for root, idx in groups.items():
        g = {"names": sorted({n for i in idx for n in fams[i]["names"]}), "props": [], "limits": [], "compares": [],
             "n_sources": 0, "members": [fams[i]["key"] for i in idx]}
        srcs = set()
        for i in idx:
            for k in ("props", "limits", "compares"):
                g[k].extend(fams[i][k])
            srcs |= {p["paper_id"] for p in fams[i]["props"]}
        g["n_sources"] = len(srcs)
        g["rep"] = names[max(idx, key=lambda i: len(fams[i]["props"]))]
        new.append(g)
    # 挂载未归族的局限/比较
    reps = [g["rep"] for g in new]
    R = emb(reps)
    attached = Counter()
    audit = {"merge_samples": random.Random(0).sample(merged_pairs, min(25, len(merged_pairs))), "attach_samples": []}
    for slot in ("limits", "compares"):
        items = st["unattached"][slot]
        abouts = [str(it.get("about") or it["text"][:80]) for it in items]
        if not abouts:
            continue
        A = emb(abouts)
        sims = A @ R.T
        best = sims.argmax(1)
        still = []
        for k, it in enumerate(items):
            j, s = int(best[k]), float(sims[k, best[k]])
            if s >= T_ATTACH and (content(abouts[k]) & content(reps[j])):
                new[j][slot].append(it)
                attached[slot] += 1
                if len(audit["attach_samples"]) < 40 and random.Random(k).random() < 0.05:
                    audit["attach_samples"].append({"slot": slot, "about": abouts[k], "family": reps[j], "sim": round(s, 3)})
            else:
                still.append(it)
        st["unattached"][slot] = still
    new.sort(key=lambda g: (-g["n_sources"], -(len(g["props"]) + len(g["limits"]) + len(g["compares"]))))
    out = {"families": new, "unattached": st["unattached"], "params": {"T_MERGE": T_MERGE, "T_ATTACH": T_ATTACH}}
    json.dump(out, open(os.path.join(V2, "state_merged.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(audit, open(os.path.join(V2, "merge_audit.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    stats = {"families_before": len(fams), "families_after": len(new), "merges": len(merged_pairs),
             "multi_source_after": sum(1 for g in new if g["n_sources"] >= 2),
             "limits_newly_attached": attached["limits"], "compares_newly_attached": attached["compares"],
             "limits_attached_total": sum(len(g["limits"]) for g in new),
             "compares_attached_total": sum(len(g["compares"]) for g in new),
             "limits_unattached": len(st["unattached"]["limits"]), "compares_unattached": len(st["unattached"]["compares"])}
    json.dump(stats, open(os.path.join(V2, "state_merged_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))
    for m in audit["merge_samples"][:10]:
        print("  MERGE", m)
    for a in audit["attach_samples"][:10]:
        print("  ATTACH", a)


if __name__ == "__main__":
    main()
