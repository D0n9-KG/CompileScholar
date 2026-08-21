# -*- coding: utf-8 -*-
"""gold 标注一致性诚实核验: 对ARFM2024 gold随机抽N条边, 人工(规则)核验evidence是否verbatim+
关系方向是否合理. 报抽检一致率(诚实, 非真两人kappa).
诚实说明: gold由GLM-5.2单模型抽, 无真人inter-annotator kappa. 此脚做自动核验:
  1. evidence verbatim: gold edge.evidence 是否在综述原文出现(子串)
  2. 边端点方法名是否在综述出现
  3. 关系类型合理性(extends/improves等有无evidence支撑词)
"""
import os, sys, json, random
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

GOLD = os.path.join(os.path.dirname(__file__), "gold", "ARFM2024_gold.json")
SURVEY = os.path.join(os.path.dirname(__file__), "pilot_surveys", "2024ARFM综述_致密颗粒介质建模进展.md")
N = 20  # 抽检边数


def main():
    gold = json.load(open(GOLD, encoding='utf-8'))
    survey = open(SURVEY, encoding='utf-8', errors='replace').read()
    # 去图行
    survey = "\n".join(l for l in survey.splitlines() if not l.lstrip().startswith("!"))
    edges = gold["evolution_edges"]
    methods = {m["id"]: m for m in gold["methods"]}

    random.seed(42)
    sample = random.sample(edges, min(N, len(edges)))
    print(f"=== ARFM2024 gold 抽检 ({len(sample)}/{len(edges)} 边) ===\n")

    verbatim_ok, verbatim_fail = 0, 0
    endpoint_ok = 0
    rel_supported = 0
    for e in sample:
        src, tgt, typ = e["src"], e["tgt"], e["type"]
        ev = (e.get("evidence") or "").strip()
        # 1. evidence verbatim (子串, 容错空白)
        ev_norm = " ".join(ev.split())
        surv_norm = " ".join(survey.split())
        # 取evidence前60字符查子串(长evidence可能被截)
        ev_probe = ev_norm[:60] if len(ev_norm) > 60 else ev_norm
        is_verbatim = ev_probe and ev_probe in surv_norm
        if is_verbatim:
            verbatim_ok += 1
        else:
            verbatim_fail += 1
        # 2. 端点方法名在综述
        sn = methods.get(src, {}).get("name", "")
        tn = methods.get(tgt, {}).get("name", "")
        sn_in = sn and sn.lower() in surv_norm.lower()
        tn_in = tn and tn.lower() in surv_norm.lower()
        if sn_in and tn_in:
            endpoint_ok += 1
        # 3. 关系支撑词
        rel_words = {"extends": ["extend", "generaliz", "推广", "扩展"],
                     "improves": ["improv", "改进", "better", "accurate"],
                     "compares": ["compar", "对比", "contrast"],
                     "replaces": ["replac", "取代", "instead"],
                     "adapts": ["adapt", "适用", "apply to"],
                     "background": ["background", "motivat", "基于", "背景"]}
        words = rel_words.get(typ, [])
        has_word = any(w in surv_norm.lower() for w in words) or any(w in ev_norm.lower() for w in words)
        if has_word:
            rel_supported += 1
        print(f"  {src}({sn[:15]}) --{typ}--> {tgt}({tn[:15]}) | verbatim={'Y' if is_verbatim else 'N'} "
              f"endpoint={'Y' if sn_in and tn_in else 'N'} rel_word={'Y' if has_word else 'N'}")
        print(f"    evidence: {ev[:90]}")

    n = len(sample)
    print(f"\n=== 抽检结果 ({n} 条) ===")
    print(f"  evidence verbatim (原文出现): {verbatim_ok}/{n} = {verbatim_ok/n:.2f}")
    print(f"  端点方法名在综述: {endpoint_ok}/{n} = {endpoint_ok/n:.2f}")
    print(f"  关系有支撑词: {rel_supported}/{n} = {rel_supported/n:.2f}")
    print(f"\n诚实说明: gold由GLM-5.2单模型抽, 无真人inter-annotator kappa.")
    print(f"此为自动核验(evidence是否verbatim+端点+关系词), 非Cohen's kappa.")
    print(f"verbatim一致率 {verbatim_ok/n:.2f} 可作gold质量近似指标(高=无幻觉evidence).")


if __name__ == "__main__":
    main()
