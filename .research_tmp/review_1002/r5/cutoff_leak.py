# -*- coding: utf-8 -*-
"""R5：各臂引用的知识截止（2025-05-01）违规统计——只读，零 LLM。
年份来源：harness/ours=citation.metadata.year；GPTR=url 中 arXiv DOI(10.48550/arXiv.YYMM)
或 doi 前缀推断不了则记 unknown。ours 另按 KB 内/外部候选（ext:）分层。"""
import json, os, re, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CS2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2"
os.chdir(CS2)

def arxiv_ym(s):
    m = re.search(r"(?:arxiv[./:]|abs/)(\d{2})(\d{2})\.\d{4,5}", s or "", re.I)
    return (2000 + int(m.group(1)), int(m.group(2))) if m else None

def year_of(c):
    md = c.get("metadata") or {}
    y = md.get("year")
    ym = arxiv_ym(str(md.get("arxiv") or "")) or arxiv_ym(str(md.get("url") or "")) \
        or arxiv_ym(str(md.get("doi") or ""))
    if ym: return ym[0], ym[1], "arxiv"
    try:
        y = int(str(y)[:4]) if y else None
    except Exception:
        y = None
    return (y, None, "year") if y else (None, None, "unknown")

def stat(fi, label, split_ext=False):
    rows = json.load(open(fi, encoding="utf-8"))
    n = unk = post = y2025 = 0; qpost = set(); qany = set()
    by_tier = collections.Counter()
    for r in rows:
        for s in r["sections"]:
            for c in s.get("citations") or []:
                n += 1
                y, mth, src = year_of(c)
                if y is None: unk += 1; continue
                # 违规：>2025，或 arxiv 2025-05 及之后
                viol = y > 2025 or (y == 2025 and mth is not None and mth >= 5)
                if y == 2025 and mth is None: y2025 += 1
                if viol:
                    post += 1; qpost.add(r["qid"])
                    if split_ext:
                        ev = (c.get("metadata") or {}).get("evidence_loc")
                        by_tier["kb_rec" if ev and any((e or {}).get("chunk_id") for e in ev) else "abstract/ext"] += 1
    print(f"{label:<14} cites={n:5d} unknown_year={unk/n if n else 0:5.2f} "
          f"post-cutoff={post/n if n else 0:5.3f} ({post}) yr2025-ambig={y2025} "
          f"q_with_post={len(qpost)}/{len(rows)} {dict(by_tier) if by_tier else ''}")
    return qpost

stat("judge_input_harness_100.json", "harness100")
stat("arm_gptr/judge_input_gptr_cs2.json", "gptr")
stat("arm_storm/judge_input_storm_cs2.json", "storm")
for b in ["32b", "33a", "33b", "34a", "34b", "34c", "34d", "34e"]:
    stat(f"judge_input_ours_batch{b}.json", f"ours{b}", split_ext=True)

# GPTR url 形态分布（年份不可直接得时说明原因）
rows = json.load(open("arm_gptr/judge_input_gptr_cs2.json", encoding="utf-8"))
kinds = collections.Counter()
for r in rows:
    for s in r["sections"]:
        for c in s["citations"]:
            u = (c.get("metadata") or {}).get("url") or ""
            kinds["doi-arxiv" if "10.48550" in u else "doi" if "doi.org" in u
                  else "scholar-q" if "scholar.google" in u else "other"] += 1
print("gptr url kinds:", dict(kinds))

# 外部候选落盘里的年份（ours 运行时 search_papers 候选）
for p in ["base_kb/ext_candidates.jsonl", "arm_ours/kb_snapshot_writes/manifest_all.json"]:
    if not os.path.exists(p): continue
    if p.endswith(".jsonl"):
        cands = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    else:
        cands = [r for r in json.load(open(p, encoding="utf-8"))
                 if str(r.get("paper_id", "")).startswith("ext:")]
    ys = collections.Counter()
    for c in cands:
        y = c.get("year")
        try: y = int(str(y)[:4]) if y else None
        except Exception: y = None
        ys["none" if y is None else ">2025" if y > 2025 else "2025" if y == 2025 else "<=2024"] += 1
    print(f"{p}: n={len(cands)} {dict(ys)}")
