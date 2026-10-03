"""Step3 精度核对：随机 30 张结果表 → 自动接地检查（值/方法/数据集字符串是否出现在表/标题）
+ 打印其中 10 张表的样本元组供人工逐条语义核对（行列对应、角色、数据集归属）。"""
import json, glob, random, re, sys
sys.path.insert(0, ".")
from common import P3

files = []
for f in sorted(glob.glob(P3 + "/tuples/*.json")):
    d = json.load(open(f, encoding="utf-8"))
    if d["tuples"]:
        files.append(d)
random.seed(11)
samp = random.sample(files, min(30, len(files)))


def fmt_nums(v):
    out = {f"{v}", f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", f"{v:.4f}"}
    if v < 1:
        out |= {f"{v*100:.1f}", f"{v*100:.2f}", f".{str(v).split('.')[-1]}"}
    return {x.rstrip("0").rstrip(".") if "." in x else x for x in out} | out


agg = {"n": 0, "value": 0, "method": 0, "dataset": 0}
per = []
for d in samp:
    src = (d["caption"] + " " + " ".join(d["rows"])).lower()
    src_n = re.sub(r"[\s,]", "", src)
    tup = random.sample(d["tuples"], min(10, len(d["tuples"])))
    ok = {"value": 0, "method": 0, "dataset": 0}
    for t in tup:
        ok["value"] += any(x.lower() in src_n for x in fmt_nums(t["value"]))
        ok["method"] += re.sub(r"\s", "", str(t["method"]).lower())[:12] in src_n
        ok["dataset"] += bool(t["dataset"]) and re.sub(r"\s", "", str(t["dataset"]).lower())[:6] in src_n
    for k in ok:
        agg[k] += ok[k]
    agg["n"] += len(tup)
    per.append({"paper": d["paper"], "table": d["table_id"], "n": len(tup), **ok})
print("auto-grounding over", len(samp), "tables /", agg["n"], "tuples:",
      {k: round(agg[k] / agg["n"], 3) for k in ("value", "method", "dataset")})


# 行对齐：值必须出现在含方法名的那一行（常规表），或该方法名所在列（转置表：表头行含方法名）
def aligned(d, t):
    m = re.sub(r"\s", "", str(t["method"]).lower())[:10]
    nums = {x.lower() for x in fmt_nums(t["value"])}
    rows = [re.sub(r"[\s,]", "", r.lower()) for r in d["rows"]]
    for r in rows:
        if m and m in r and any(x in r for x in nums):
            return True
    # 转置：找表头中方法名所在列号，再在各行同列取值
    for hi, h in enumerate(d["rows"][:3]):
        cells = [re.sub(r"\s", "", c.lower()) for c in h.split("|")]
        if any(m and m in c for c in cells):
            ci = [i for i, c in enumerate(cells) if m and m in c]
            for r in d["rows"][hi + 1:]:
                rc = [re.sub(r"[\s,]", "", c.lower()) for c in r.split("|")]
                for off in (0, -1, 1, -2, 2):
                    for i in ci:
                        j = i + off
                        if 0 <= j < len(rc) and any(x == rc[j] or x in rc[j].split() for x in nums):
                            return True
    return False


al = n_al = 0
for d in samp:
    random.seed(d["paper"])
    for t in random.sample(d["tuples"], min(10, len(d["tuples"]))):
        n_al += 1
        al += aligned(d, t)
print("row/column alignment (method<->value same row or same column):", al, "/", n_al, round(al / n_al, 3))
json.dump({"agg": agg, "per": per}, open(P3 + "/s6_auto.json", "w", encoding="utf-8"), indent=1)
# 人工核对材料：前 10 张样本表，每张 4 条元组 + 表头前 6 行
with open(P3 + "/s6_manual_sheet.txt", "w", encoding="utf-8") as f:
    for d in samp[:10]:
        f.write(f"=== {d['paper']} {d['table_id']} | {d['caption'][:160]}\n")
        for r in d["rows"][:7]:
            f.write("  " + r[:220] + "\n")
        for t in random.sample(d["tuples"], min(4, len(d["tuples"]))):
            f.write(f"  -> {t['method']} | {t['dataset']} | {t['metric']} | {t['value']} | {t['role']} | set={t['setting'][:40]}\n")
print("manual sheet written")
