"""Oracle 测试床输入：48 道唯一 gt 题（DSB gt_nuggets_outputs 66 目录含 18 对重复 qid，取每 qid 最小目录号）。
每题=目标论文摘要+官方 DeepScholar-base 查询模板（600 词/编号引用）+金标参考文献（important_citations）。"""
import csv, json, os, re
import pandas as pd
csv.field_size_limit(10**9)
DSB = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\deepscholar\dsb"
GT = os.path.join(DSB, "dataset", "gt_nuggets_outputs")
QUERY_TEMPLATE = ("Your task is to write a Related Works section for an academic paper given the paper's abstract. "
  "Your response should provide the Related Works section and references. Only include references from arXiv that are "
  "published before {cutoff_date}. Mention them in a separate, numbered reference list at the end and use the reference "
  "numbers to provide in-line citations in the Related Works section for all claims referring to a source (e.g., description "
  "of source [3]. Further details [6][7][8][9][10].) Each in-line citation must consist of a single reference number within "
  "a pair of brackets. Do not use any other citation format. Do not exceed 600 words for the related works section. "
  "Here is the paper abstract: {abstract}")
pc = pd.read_csv(os.path.join(DSB, "dataset", "paper_content.csv"))
ds = pd.read_csv(os.path.join(DSB, "dataset", "papers_with_related_works.csv"))
ic = pd.read_csv(os.path.join(DSB, "dataset", "important_citations.csv"))
by_q = {}
for d in sorted([d for d in os.listdir(GT) if d.isdigit()], key=int):
    j = json.load(open(os.path.join(GT, d, "res.json"), encoding="utf-8"))
    by_q.setdefault(j["qid"], d)
rows = []
n_ref = n_noabs = 0
for qid, d in sorted(by_q.items(), key=lambda x: int(x[1])):
    p = pc[pc.arxiv_id == qid].iloc[0]
    meta = ds[ds.arxiv_id == qid].iloc[0]
    refs = []
    for _, r in ic[ic.parent_paper_arxiv_id == qid].iterrows():
        abs_ = r.cited_paper_abstract if isinstance(r.cited_paper_abstract, str) else ""
        refs.append({"title": str(r.cited_paper_title).strip(), "abstract": abs_.strip(),
                     "url": r.cited_paper_arxiv_link if isinstance(r.cited_paper_arxiv_link, str) else "",
                     "authors": r.cited_paper_authors if isinstance(r.cited_paper_authors, str) else "",
                     "year": str(r.bib_paper_year) if not pd.isna(r.bib_paper_year) else ""})
        n_ref += 1; n_noabs += (abs_.strip() == "")
    rows.append({"gt_dir": d, "qid": qid, "title": p.paper_title, "abstract": p.abstract,
                 "published_date": str(meta.published_date),
                 "query": QUERY_TEMPLATE.format(cutoff_date=meta.published_date, abstract=p.abstract),
                 "refs": refs})
json.dump(rows, open("oracle_inputs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"questions={len(rows)} refs={n_ref} mean_refs={n_ref/len(rows):.1f} missing_abstract={n_noabs} ({n_noabs/n_ref:.1%})")
