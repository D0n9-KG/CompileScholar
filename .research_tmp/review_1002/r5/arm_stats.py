# -*- coding: utf-8 -*-
"""R5 审查：各臂判分输入形态 + 分数口径 + 失败/缺题统计（只读，零 LLM 调用）。"""
import json, re, os, sys, statistics as st
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CS2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2"
os.chdir(CS2)

ARMS = {
    "harness100": ("judge_input_harness_100.json", "direct_scores_harness_100_ds.json"),
    "gptr": ("arm_gptr/judge_input_gptr_cs2.json", "arm_gptr/direct_scores_gptr_cs2_ds.json"),
    "perplexity": ("judge_input_perplexity_dev.json", "direct_scores_perplexity_ds.json"),
    "storm": ("arm_storm/judge_input_storm_cs2.json", "direct_scores_storm_cs2.json"),
}
for b in ["32b", "33a", "33b", "34a", "34b", "34c", "34d", "34e"]:
    ARMS[f"ours{b}"] = (f"judge_input_ours_batch{b}.json", f"direct_scores_ours_batch{b}_ds.json")

def alpha(s): return re.sub(r"[^a-zA-Z]", "", s or "").lower()

def facets(s):
    if not isinstance(s, dict): return None
    try:
        ir = s["ingredient_recall"]["ingredient_recall"]
        ap = s["answer_precision"]["answer_precision"]
        c = s["citation"]
        return ir, ap, c["citation_recall"], c["citation_precision"], c["f1"]
    except Exception:
        return None

def ans_rows(b):
    p = f"arm_ours/answers_pilot_cs2batch{b}.json"
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

print(f"{'arm':<12}{'ans':>5}{'in':>5}{'sc':>5}{'err':>5} | {'IR':>6}{'AP':>6}{'CR':>6}{'CP':>6}{'F1':>6} | {'g4(off)':>8}{'g3(F1)':>8}")
for arm, (fi, fs) in ARMS.items():
    if not (os.path.exists(fi) and os.path.exists(fs)):
        print(arm, "missing", fi if not os.path.exists(fi) else fs); continue
    inp = json.load(open(fi, encoding="utf-8"))
    sc = json.load(open(fs, encoding="utf-8"))
    qin = {r["qid"] for r in inp}
    ok = {q: facets(v) for q, v in sc.items() if facets(v)}
    err = len(sc) - len(ok)
    na = "-"
    if arm.startswith("ours"):
        a = ans_rows(arm[4:]); na = len(a) if a else "-"
    F = list(ok.values())
    m = [st.mean(x[i] for x in F) for i in range(5)]
    g4 = st.mean((x[0] + x[1] + x[2] + x[3]) / 4 for x in F)
    g3 = st.mean((x[0] + x[1] + x[4]) / 3 for x in F)
    print(f"{arm:<12}{str(na):>5}{len(inp):>5}{len(sc):>5}{err:>5} | " + "".join(f"{v:6.3f}" for v in m) + f" | {g4:8.3f}{g3:8.3f}")
    # 缺题
    miss_sc = qin - set(sc)
    if miss_sc: print(f"   in-but-not-scored: {len(miss_sc)}")

print("\n=== 引用形态（judge 输入层）===")
print(f"{'arm':<12}{'cites':>6}{'empty':>7}{'noTitle':>8}{'strSnip':>8}{'snipInText':>11}{'idInText':>9}{'avgSnipCh':>10}{'cit/q':>6}{'words/q':>8}")
for arm, (fi, fs) in ARMS.items():
    if not os.path.exists(fi): continue
    inp = json.load(open(fi, encoding="utf-8"))
    n = empty = notitle = strsnip = inl = idin = 0; L = []; words = []
    for r in inp:
        w = 0
        for s in r["sections"]:
            txt = s.get("text") or ""
            w += len(txt.split())
            ta = alpha(txt)
            for c in s.get("citations") or []:
                n += 1
                sn = c.get("snippets")
                if isinstance(sn, str): strsnip += 1; sn_l = [sn]
                else: sn_l = sn or []
                if not sn or not any((x or "").strip() for x in sn_l): empty += 1
                if not c.get("title"): notitle += 1
                if sn_l and any(alpha(x) and alpha(x) in ta for x in sn_l): inl += 1
                if c.get("id") and c["id"] in txt: idin += 1
                L.append(sum(len(x or "") for x in sn_l))
        words.append(w)
    print(f"{arm:<12}{n:>6}{empty/n if n else 0:>7.2f}{notitle:>8}{strsnip:>8}{inl/n if n else 0:>11.3f}{idin/n if n else 0:>9.2f}{(st.mean(L) if L else 0):>10.0f}{n/len(inp):>6.1f}{st.mean(words):>8.0f}")
