"""KB v2 的 dev/test 偏向复测（与 kb_dev_test_bias.py 同法）：题面→KB 论文 TF-IDF top-1 中位余弦。"""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

B = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/"
P = json.load(open(B + "cs2/base_kb_v2/papers.json", encoding="utf-8"))
R = json.load(open(B + "cs2/base_kb_v2/records.json", encoding="utf-8"))
docs = []
for pid, m in P.items():
    t = (m.get("title") or "") + ". " + (m.get("abstract") or "")
    if not m.get("abstract"):  # 综述/无摘要：用前 30 条记录的 claim 代表内容
        t += " " + " ".join((r.get("claim") or "") for r in R.get(pid, {}).get("records", [])[:30])
    docs.append(t)
dev = [q["question"] for q in json.load(open(B + "scholarqa_multi/sqa2_rubrics_v1_recomputed.json", encoding="utf-8"))]
tst = [q["question"] for q in json.load(open(B + "scholarqa_multi/sqa2_rubrics_v2_recomputed.json", encoding="utf-8"))]
v = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit(docs + dev + tst)
D = v.transform(docs)
for name, qs in (("dev", dev), ("test", tst)):
    S = (v.transform(qs) @ D.T).toarray()
    top = np.sort(S, axis=1)
    print(f"{name}: KB v2 papers={len(docs)} | top1 median {np.median(top[:, -1]):.3f} | top10 mean {top[:, -10:].mean():.3f}")
