"""量化：人写 related work 的段落/小节组织轴是否 = 目标论文标题/摘要中的设计成分。
对每个有标题的 related work，计算小节标题与目标论文标题+摘要的词重叠；
无标题的按段落首句做同样计算。对照：随机配对另一篇目标论文。"""
import csv
import random
import re

csv.field_size_limit(10**9)
D = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/deepscholar/dsb/dataset/"
pc = list(csv.DictReader(open(D + "paper_content.csv", encoding="utf-8")))
BS = chr(92)
head_pat = re.compile(re.escape(BS) + r"(?:sub)*section\*?\{([^}]*)\}|"
                      + re.escape(BS) + r"paragraph\*?\{([^}]*)\}|"
                      + re.escape(BS) + r"textbf\{([^}]*)\}")
cite_pat = re.compile(re.escape(BS) + r"cite[pt]?\*?\{[^}]*\}")
STOP = set("the a an of and or to in for on with by is are be as that this from at its their it into via "
           "using based can than more most such these those which we our also have has been while however "
           "related work works recent approaches methods method models model learning".split())


def toks(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 3}


def units(rw):
    hs = [a or b or c for a, b, c in head_pat.findall(rw)]
    if hs:
        return hs, "heads"
    paras = [p for p in re.split(r"\n\s*\n", rw) if len(p.strip()) > 80]
    firsts = [re.split(r"(?<=[.!?])\s", cite_pat.sub("", p).strip())[0] for p in paras]
    return firsts, "para-first-sentence"


def cover(unit, target):
    t = toks(unit)
    return len(t & target) / len(t) if t else None


random.seed(0)
res = {"heads": [[], []], "para-first-sentence": [[], []]}
for i, r in enumerate(pc):
    tgt = toks(r["paper_title"] + " " + r["abstract"])
    other = pc[(i + random.randint(1, len(pc) - 1)) % len(pc)]
    tgt_o = toks(other["paper_title"] + " " + other["abstract"])
    us, kind = units(r["related_works_section"])
    for u in us:
        a, b = cover(u, tgt), cover(u, tgt_o)
        if a is not None:
            res[kind][0].append(a)
            res[kind][1].append(b)
for k, (own, oth) in res.items():
    if own:
        m = lambda x: sum(x) / len(x)
        print(f"{k}: n={len(own)} | overlap with OWN target title+abstract {m(own):.2f} "
              f"| with RANDOM other target {m(oth):.2f} | units with >=50% own-overlap "
              f"{sum(x >= 0.5 for x in own) / len(own):.0%}")
