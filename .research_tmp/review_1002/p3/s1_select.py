"""Step1: 本地 OAI 快照筛序列推荐候选（2019-2024），并统计 GNN 节点分类作选型对照。"""
import json, re, random, collections, sys
sys.path.insert(0, "."); from common import SNAP, P3
SR = re.compile(r"\bsequential recommend|\bsession-based recommend|\bnext[- ]item recommend", re.I)
SR_B = re.compile(r"SASRec|BERT4Rec|GRU4Rec|Caser\b", re.I)
GNN = re.compile(r"\bnode classification\b", re.I)
GNN_B = re.compile(r"\bCora\b|Citeseer|PubMed|ogbn-arxiv", re.I)
cand = {"sr": [], "gnn": []}
with open(SNAP, encoding="utf-8", errors="replace") as f:
    for line in f:
        if '"cs.' not in line: continue
        d = json.loads(line); aid = d["id"]
        m = re.match(r"(\d{2})(\d{2})\.\d+$", aid)
        if not m: continue
        yr = 2000 + int(m.group(1))
        if not 2019 <= yr <= 2024: continue
        t = d["title"] + " " + d["abstract"]
        if SR.search(t): cand["sr"].append({"id": aid, "year": yr, "title": " ".join(d["title"].split()), "base": bool(SR_B.search(t))})
        if GNN.search(t) and GNN_B.search(t): cand["gnn"].append({"id": aid, "year": yr})
for k, v in cand.items():
    print(k, len(v), dict(sorted(collections.Counter(x["year"] for x in v).items())))
sr = cand["sr"]
print("SR with baseline names in abstract:", sum(x["base"] for x in sr))
random.seed(7)
by = collections.defaultdict(list)
for x in sr: by[x["year"]].append(x)
pick = []
for y in sorted(by):
    pool = sorted(by[y], key=lambda x: (not x["base"], random.random()))
    pick += pool[:22]
json.dump(pick, open(P3 + "/s1_candidates.json", "w", encoding="utf-8"), indent=1)
print("picked", len(pick), dict(collections.Counter(x["year"] for x in pick)))
