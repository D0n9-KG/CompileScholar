"""KB 对 dev(v1) vs test(v2) 题面的词法覆盖偏向（TF-IDF top-k 余弦），分层 demand_core / 其余。"""
import json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
B="C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/"
man=json.load(open(B+"cs2/base_kb/manifest_all.json",encoding="utf-8"))
kb=set(json.load(open(B+"cs2/base_kb/records_merged.json",encoding="utf-8")))
rows=[r for r in man if r.get("paper_id") in kb]
docs=[(r.get("title") or "")+". "+(r.get("abstract") or "") for r in rows]
isdem=np.array([r.get("primary_category")=="demand_core" for r in rows])
dev=[q["question"] for q in json.load(open(B+"scholarqa_multi/sqa2_rubrics_v1_recomputed.json",encoding="utf-8"))]
tst=[q["question"] for q in json.load(open(B+"scholarqa_multi/sqa2_rubrics_v2_recomputed.json",encoding="utf-8"))]
v=TfidfVectorizer(stop_words="english",sublinear_tf=True,min_df=2).fit(docs+dev+tst)
D=v.transform(docs)
def stats(qs,mask=None,k=10):
    S=(v.transform(qs)@D.T).toarray()
    if mask is not None: S=S[:,mask]
    top=np.sort(S,axis=1)[:,-k:]
    return top.mean(), np.median(top[:,-1])
print("KB papers with manifest rows:",len(rows),"demand:",isdem.sum())
for name,qs in [("dev",dev),("test",tst)]:
    a=stats(qs); d=stats(qs,isdem); o=stats(qs,~isdem)
    print(f"{name}: all top10 mean={a[0]:.3f} top1 med={a[1]:.3f} | demand-only top10={d[0]:.3f} | non-demand top10={o[0]:.3f}")
