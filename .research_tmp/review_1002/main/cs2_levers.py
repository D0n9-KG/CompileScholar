"""CS2 改造可行性快测（只读现有产物）：
1. 答案长度 vs IR：ours（34 系）与 harness 的词数分布、各自臂内 长度–IR 相关。
2. harness 截止泄漏：引用了 >2025 年论文的题 vs 没引用的题，分数差（混杂，仅作量级参考）。
3. 同题配对（34 系 offset 75-89 与 harness 交集）各 facet 差。"""
import glob
import json
import re
import statistics as st

B = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/cs2/"


def facets(v):
    ir = v["ingredient_recall"]["ingredient_recall"]
    ap = v["answer_precision"]["answer_precision"]
    c = v["citation"]
    ct = c.get("f1", (c["citation_recall"] + c["citation_precision"]) / 2)
    return {"IR": ir, "AP": ap, "CT": ct, "G": (ir + ap + c["citation_recall"] + c["citation_precision"]) / 4}


def words_of_sections(secs):
    return sum(len((s.get("text") or "").split()) for s in secs)


# harness
hs = json.load(open(B + "direct_scores_harness_100_ds.json", encoding="utf-8"))
ha = {r["qid"]: r for r in json.load(open(B + "arm_harness/answers_harness_dev20.json", encoding="utf-8"))}
H = {}
for q, v in hs.items():
    try:
        f = facets(v)
    except Exception:
        continue
    res = (ha.get(q) or ha.get(q[:24]) or {}).get("result") or ""
    m = re.search(r"\{.*\}", res, flags=re.S)
    w = None
    try:
        w = words_of_sections(json.loads(m.group(0))["sections"]) if m else None
    except Exception:
        w = len(res.split())
    leak = bool(re.search(r'"year"\s*:\s*"?(2026|2025)', res)) and bool(re.search(r'"year"\s*:\s*"?2026', res))
    H[q[:24]] = {**f, "words": w or len(res.split()), "leak2026": leak}

# ours 34 系
O = {}
for b in "abcde":
    sc = json.load(open(B + f"direct_scores_ours_batch34{b}_ds.json", encoding="utf-8"))
    ji = json.load(open(B + f"judge_input_ours_batch34{b}.json", encoding="utf-8"))
    rows = ji if isinstance(ji, list) else list(ji.values())
    wd = {}
    for r in rows:
        qid = (r.get("qid") or r.get("case_id") or "")[:24]
        secs = r.get("sections") or (r.get("response") or {}).get("sections") or []
        wd[qid] = words_of_sections(secs)
    for q, v in sc.items():
        try:
            f = facets(v)
        except Exception:
            continue
        O.setdefault(q[:24], []).append({**f, "words": wd.get(q[:24])})


def corr(xs, ys):
    p = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    return st.correlation([a for a, _ in p], [b for _, b in p]) if len(p) > 3 else float("nan")


hw = [h["words"] for h in H.values()]
print(f"harness n={len(H)} words median {st.median(hw):.0f} | corr(words, IR) {corr(hw, [h['IR'] for h in H.values()]):+.2f}")
ow = [r["words"] for rs in O.values() for r in rs if r["words"]]
oir = [r["IR"] for rs in O.values() for r in rs if r["words"]]
print(f"ours(34系) runs={len(ow)} words median {st.median(ow):.0f} | corr(words, IR) {corr(ow, oir):+.2f}")
lk = [h for h in H.values() if h["leak2026"]]
nl = [h for h in H.values() if not h["leak2026"]]
for k in ("G", "IR", "AP", "CT"):
    print(f"harness cites-2026 (n={len(lk)}) {k} {st.mean(h[k] for h in lk):.3f} vs no-2026 (n={len(nl)}) {st.mean(h[k] for h in nl):.3f}")
com = [q for q in O if q in H]
print(f"paired ours(34系均值)-harness on n={len(com)} common questions:")
for k in ("G", "IR", "AP", "CT"):
    d = [st.mean(r[k] for r in O[q]) - H[q][k] for q in com]
    print(f"   {k} {st.mean(d):+.3f}")
dw = [st.mean(r["words"] for r in O[q] if r["words"]) - H[q]["words"] for q in com if any(r["words"] for r in O[q])]
print(f"   words ours-harness median {st.median(dw):+.0f}")
