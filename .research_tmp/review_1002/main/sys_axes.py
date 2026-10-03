"""对照：系统输出（STORM / harness）的小节标题与目标论文标题+摘要的词重叠，vs 人写 0.84。"""
import csv
import glob
import json
import os
import random
import re

csv.field_size_limit(10**9)
B = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/"
pc = {r["arxiv_id"]: r for r in csv.DictReader(open(B + "deepscholar/dsb/dataset/paper_content.csv", encoding="utf-8"))}
ids = list(pc)
STOP = set("the a an of and or to in for on with by is are be as that this from at its their it into via "
           "using based can than more most such these those which we our also have has been while however "
           "related work works recent approaches methods method models model learning overview introduction "
           "conclusion summary background references".split())


def toks(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 3}


def md_heads(text):
    return [h.strip("# ").strip() for h in re.findall(r"^#{2,4}\s+.+$", text, flags=re.M)]


def score(pairs):
    random.seed(0)
    own, oth = [], []
    for aid, heads in pairs:
        if aid not in pc:
            continue
        t = toks(pc[aid]["paper_title"] + " " + pc[aid]["abstract"])
        o = pc[random.choice([x for x in ids if x != aid])]
        to = toks(o["paper_title"] + " " + o["abstract"])
        for h in heads:
            ht = toks(h)
            if ht:
                own.append(len(ht & t) / len(ht))
                oth.append(len(ht & to) / len(ht))
    m = lambda x: sum(x) / len(x) if x else float("nan")
    return len(own), m(own), m(oth), (sum(x >= 0.5 for x in own) / len(own) if own else float("nan"))


storm = []
for d in glob.glob(B + "cs2/arm_storm_dsb/*v*"):
    aid = os.path.basename(d)
    for f in glob.glob(d + "/*/storm_gen_article.md") + glob.glob(d + "/storm_gen_article.md"):
        storm.append((aid, md_heads(open(f, encoding="utf-8", errors="replace").read())))
harness = []
for r in json.load(open(B + "cs2/arm_harness_dsb/answers_harness_dsb.json", encoding="utf-8")):
    aid = next((a for a in ids if a.startswith(str(r.get("qid") or "")) or pc[a]["paper_title"][:40] == (r.get("title") or "")[:40]), None)
    if aid:
        harness.append((aid, [h for h in md_heads(r.get("result") or "") if h.lower() != "related work"]))
for name, pairs in [("STORM", storm), ("harness", harness)]:
    n, a, b, f = score(pairs)
    print(f"{name}: papers={len(pairs)} heads={n} | overlap OWN target {a:.2f} | RANDOM target {b:.2f} | >=50% own {f:.0%}")
print("human (rw_axes.py): heads=45 | OWN 0.84 | RANDOM 0.05 | >=50% own 89%")
