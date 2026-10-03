"""金标 nugget 与 (a) 被引文献摘要 (b) 目标论文摘要 的词法最大覆盖——粗估 nugget 是"单篇可抄"还是"综合/定位"。"""
import csv, json, re, os, statistics as st
csv.field_size_limit(10**9)
D="dataset/"
STOP=set("the a an of and or to in for on with by is are be as that this from at its their it into via using based can than more most such these those which".split())
tok=lambda s:{w for w in re.findall(r"[a-z0-9]+",s.lower()) if w not in STOP and len(w)>2}
pc={r["arxiv_id"]:r for r in csv.DictReader(open(D+"paper_content.csv",encoding="utf-8"))}
ic=list(csv.DictReader(open(D+"important_citations.csv",encoding="utf-8")))
rows=[]
for i in sorted([x for x in os.listdir(D+"gt_nuggets_outputs") if x.isdigit()],key=int):
    p=D+f"gt_nuggets_outputs/{i}/res.json"
    if not os.path.exists(p): continue
    d=json.load(open(p,encoding="utf-8")); q=d["qid"]
    cabs=[tok(c["cited_paper_title"]+" "+c["cited_paper_abstract"]) for c in ic if c["parent_paper_arxiv_id"]==q]
    tabs=tok(pc[q]["abstract"]+" "+pc[q]["paper_title"]) if q in pc else set()
    for n in d["nuggets"]:
        t=tok(n["text"])
        if not t: continue
        best=max([len(t&c)/len(t) for c in cabs] or [0])
        union=len(t&set().union(*cabs))/len(t) if cabs else 0
        rows.append((n["importance"],best,union,len(t&tabs)/len(t)))
print("nuggets:",len(rows))
for imp in ["vital","okay"]:
    R=[r for r in rows if r[0]==imp]
    b=[r[1] for r in R]; u=[r[2] for r in R]; t=[r[3] for r in R]
    print(imp,len(R),"| best single cited-abs cover med %.2f, >=0.6: %.0f%%"%(st.median(b),100*sum(x>=0.6 for x in b)/len(b)),
          "| union cited-abs med %.2f"%st.median(u), "| target-abs cover med %.2f"%st.median(t))
