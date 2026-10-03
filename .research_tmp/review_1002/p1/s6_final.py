# -*- coding: utf-8 -*-
"""Manual adjudication (P1 reviewer read every refuted/unclear title+abstract) -> final tables."""
import json
import os
from collections import defaultdict

from s4_review import wilson

P1 = os.path.dirname(os.path.abspath(__file__))
# V=valid refutation (independent or source-contradicted, published before claim date)
# U=unclear (plausible but abstract/date insufficient)  I=judge refutation rejected (-> not refuted)
# reasons recorded for audit
MANUAL = {
    0: ("I", "weak 'need for more real-world testing'; single unrelated paper"),
    1: ("I", "weak vague"), 2: ("I", "weak vague"),
    3: ("V", "open: safety guarantees for learning-based control; Koller18/Fisac17 predate (medium conf)"),
    5: ("I", "weak 'underexplored'; 1-2 papers don't refute quantity"),
    6: ("I", "weak"), 7: ("I", "weak; refuter irrelevant"), 8: ("I", "weak attributed survey"),
    9: ("U", "weak"),
    11: ("V", "DP-SCAFFOLD 2021 = DP + heterogeneity FL; stale Kairouz-era gap stated as current"),
    15: ("U", "attributed: ViT benign overfitting (2024) vs source date same-year ambiguous"),
    17: ("I", "refuter is the cited source; weak"), 18: ("I", "refuter irrelevant"),
    19: ("I", "refuter is the cited source [4]"),
    21: ("U", "'quantifying reality gap is open' vs 2019 quantification paper; 'open' ambiguous"),
    23: ("I", "historical 'had remained open' - correct as stated"),
    25: ("I", "first-person novelty copied from SqlCompose; refuter = source itself"),
    27: ("I", "claim truncated, uncheckable"),
    28: ("V", "tree-covering practical implementation exists (2024, before cutoff); report states none"),
    30: ("U", "cross-lingual watermark: refuter 2025 same-year; X-SIR-type prior claims"),
    31: ("U", "judge refuter (SHAP on XGBoost geotech) does not evaluate explanation methods; claim likely false but not shown"),
    32: ("U", "LLM bimanual planning: refuter RoboPARA 2025 may post-date cutoff"),
    33: ("U", "first-person novelty (Borisov survey) copied; earlier survey not shown pre-dating"),
    34: ("I", "first-person novelty copied from TAPEX; refuter = source"),
    40: ("U", "duplicate of 33 in another report"),
    41: ("I", "weak-ish 'most models black boxes' attributed 2022; one counterexample insufficient"),
    42: ("V", "HaluQuestQA (2024) manually annotated long-form hallucination dataset; stale 2023 survey gap"),
    43: ("I", "first-person novelty copied (dropout runtime); refuter = source"),
    46: ("I", "garbled first-person novelty (TAPEX/TAPAS)"),
    50: ("U", "weak"), 51: ("I", "weak"), 52: ("I", "weak"), 53: ("I", "weak"),
    54: ("V", "astronomy QA benchmark AstroMLab-1 (2024) exists"),
    55: ("I", "weak"), 56: ("I", "weak"), 57: ("I", "weak"), 58: ("I", "refuter = cited source"),
    63: ("I", "garbled sentence"),
    66: ("V", "source-contradicted: cited 2021 paper itself proposes the framework the sentence says nobody proposed"),
    68: ("I", "correctly scoped 'prior to this development'"),
    69: ("V", "'prior to 2020 no automatic review generation' vs Bartoli et al. 2016"),
    71: ("I", "weak vague attributed"), 72: ("I", "irrelevant"), 73: ("U", "weak"), 74: ("U", "weak"),
    75: ("V", "weak 'lack of multi-discipline multimodal sci datasets' vs MMSci 2024 (medium)"),
    76: ("I", "weak vague"), 77: ("U", "weak"),
    79: ("V", "weak 'limited to simulations' vs 2022/2024 experimental ISAC demos (medium)"),
    84: ("I", "'constructing NEW gap examples' is an agenda, not refuted by existing gaps"),
    85: ("I", "same as 84"),
}


def final(i, o):
    if i in MANUAL:
        return MANUAL[i][0]
    v = o.get("verdict")
    return {"not_refuted": "N", "error": "E"}.get(v, "N")


def main():
    out = json.load(open(os.path.join(P1, "s3_verified.json"), encoding="utf-8"))
    agg = defaultdict(lambda: defaultdict(int))
    for i, o in enumerate(out):
        f = final(i, o)
        key = (o["sysg"], o["label"])
        agg[key]["n"] += f != "E"
        agg[key][f] += 1
        for g in (("ALL", o["label"]), ("ALL", "strong+open") if o["label"] in ("strong", "open_problem") else None):
            if g:
                agg[g]["n"] += f != "E"
                agg[g][f] += 1
    rows = []
    for k in sorted(agg):
        a = agg[k]
        p, lo, hi = wilson(a["V"], a["n"])
        pu, lou, hiu = wilson(a["V"] + a["U"], a["n"])
        rows.append({"group": k, "n": a["n"], "valid": a["V"], "unclear": a["U"],
                     "rate": round(p, 3), "ci": [round(lo, 3), round(hi, 3)],
                     "rate_incl_unclear": round(pu, 3), "ci_incl": [round(lou, 3), round(hiu, 3)]})
        print(k, rows[-1])
    # findability of valid refutations
    finds = [(i, r["q_idx"], r["rank"]) for i, o in enumerate(out) if final(i, o) == "V"
             for r in o["refuting"][:1]]
    print("valid refutation (first refuter) query idx / rank:", finds)
    # judge precision on its own 'refuted'
    jr = [i for i, o in enumerate(out) if o.get("verdict") == "refuted"]
    print("judge refuted", len(jr), "-> manual V", sum(final(i, out[i]) == "V" for i in jr),
          "U", sum(final(i, out[i]) == "U" for i in jr))
    # survey part
    sg = json.load(open(os.path.join(P1, "s5_survey_gaps.json"), encoding="utf-8"))
    SMAN = {12: "V"}  # CoTTA (CVPR'22) vs survey 2301.00265 future-work on continually changing test domains
    chk = [i for i, o in enumerate(sg) if o.get("verdict") != "not_checkable"]
    v = sum(1 for i in chk if SMAN.get(i) == "V")
    jd = sum(1 for o in sg if o.get("verdict") == "already_done")
    p, lo, hi = wilson(v, len(chk))
    print("survey gaps: sampled", len(sg), "checkable", len(chk), "judge already_done", jd,
          "manual valid", v, "rate", round(p, 3), [round(lo, 3), round(hi, 3)])
    json.dump({"rows": [{**r, "group": list(r["group"])} for r in rows], "finds": finds,
               "manual": MANUAL, "survey": {"n": len(sg), "checkable": len(chk), "judge_done": jd,
                                            "valid": v}},
              open(os.path.join(P1, "final_tables.json"), "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)


if __name__ == "__main__":
    main()
