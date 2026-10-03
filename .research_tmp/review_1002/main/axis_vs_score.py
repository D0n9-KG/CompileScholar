"""系统内逐题：小节标题与目标论文的对齐度 vs nugget coverage（harness/STORM 各自题内相关，控制系统差异）。"""
import csv
import glob
import json
import os
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


def align(aid, text):
    t = toks(pc[aid]["paper_title"] + " " + pc[aid]["abstract"])
    hs = [h.strip("# ").strip() for h in re.findall(r"^#{2,4}\s+.+$", text, flags=re.M)]
    v = [len(toks(h) & t) / len(toks(h)) for h in hs if toks(h) and h.lower() != "related work"]
    return sum(v) / len(v) if v else None


def spearman(x, y):
    def rank(a):
        o = sorted(range(len(a)), key=lambda i: a[i])
        r = [0] * len(a)
        for k, i in enumerate(o):
            r[i] = k
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** .5
    vy = sum((b - my) ** 2 for b in ry) ** .5
    return cov / (vx * vy)


def load_scores(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        k = os.path.basename(r["folder_path"].replace("\\", "/"))
        out[k] = float(r["nugget_coverage"])
    return out


# harness: indexed/<i> 对应 answers 顺序
h_ans = json.load(open(B + "cs2/arm_harness_dsb/answers_harness_dsb.json", encoding="utf-8"))
h_sc = load_scores(B + "deepscholar/dsb/results_harness_dsb/nugget_coverage/storm.csv")
xs, ys = [], []
for i, r in enumerate(h_ans):
    aid = next((a for a in ids if pc[a]["paper_title"][:40] == (r.get("title") or "")[:40]), None)
    if aid and str(i) in h_sc:
        a = align(aid, r.get("result") or "")
        if a is not None:
            xs.append(a)
            ys.append(h_sc[str(i)])
print(f"harness: n={len(xs)} spearman(align, nugget)={spearman(xs, ys):.3f}" if len(xs) > 5 else f"harness n={len(xs)}")

# STORM: folder 名=arxiv id
s_paths = glob.glob(B + "deepscholar/dsb/results_storm_rest/nugget_coverage/*.csv")
s_sc = {}
for p in s_paths:
    if "aggregated" not in p:
        s_sc.update(load_scores(p))
xs, ys = [], []
for d in glob.glob(B + "cs2/arm_storm_dsb/*v*"):
    aid = os.path.basename(d)
    fs = glob.glob(d + "/*/storm_gen_article.md")
    if aid in pc and fs and aid in s_sc:
        a = align(aid, open(fs[0], encoding="utf-8", errors="replace").read())
        if a is not None:
            xs.append(a)
            ys.append(s_sc[aid])
print(f"STORM: n={len(xs)} spearman(align, nugget)={spearman(xs, ys):.3f}" if len(xs) > 5 else f"STORM n={len(xs)} (score keys sample: {list(s_sc)[:3]})")
