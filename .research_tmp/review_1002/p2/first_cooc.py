"""时间剖分客观 gold 可行性探针：本地 arXiv OAI 快照(2024-04)流式扫一遍，
记录若干 (方法, 任务) 概念对在 CS 摘要中的首次共现日期与论文数。"""
import json, re, time
SNAP = "//192.168.199.138/Share400T/pub/LLM_Data/data/JournalPapers/arXiv_Dataset/arxiv-metadata-oai-snapshot.json"
PAIRS = [("contrastive learning","recommend"),("diffusion model","time series"),
         ("mixture of experts","speech recognition"),("graph neural network","weather forecast"),
         ("transformer","tabular data"),("reinforcement learning from human feedback","code generation"),
         ("knowledge distillation","object detection"),("large language model","protein"),
         ("dropout","bayesian"),("mamba","point cloud"),("retrieval-augmented","medical"),
         ("low-rank adaptation","diffusion")]
pats=[(a,b,re.compile(r"\b"+re.escape(a),re.I),re.compile(r"\b"+re.escape(b),re.I)) for a,b in PAIRS]
first={p:None for p in PAIRS}; cnt={p:0 for p in PAIRS}; n=0; t0=time.time()
with open(SNAP,encoding="utf-8",errors="replace") as f:
    for line in f:
        n+=1
        if '"cs.' not in line: continue
        d=json.loads(line); txt=(d.get("title","")+" "+d.get("abstract",""))
        v=d.get("versions") or [{}]; date=(v[0].get("created") or "")
        dt=d.get("update_date") if not date else None
        aid=d.get("id")
        for a,b,pa,pb in pats:
            if pa.search(txt) and pb.search(txt):
                cnt[(a,b)]+=1
                # arXiv id 编码投稿年月（新式 YYMM.xxxxx），比 created 字符串更易比较
                m=re.match(r"(\d{2})(\d{2})\.",aid or "")
                ym=("20"+m.group(1)+"-"+m.group(2)) if m else (dt or "")
                if first[(a,b)] is None or ym<first[(a,b)][0]: first[(a,b)]=(ym,aid,d.get("title","")[:80])
print(f"scanned {n} rows in {time.time()-t0:.0f}s")
for p in PAIRS: print(p, cnt[p], first[p])
