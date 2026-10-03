# -*- coding: utf-8 -*-
"""P4 stage 6: my manual verdicts (after reading source passages) + summary tables.
Version drift verified via arXiv v1 abstract pages (2511.14718v1, 2509.26173v1)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import P4, calls, wilson  # noqa: E402

# label in: in_scope | overreach_report (report-introduced scope broadening) | misrepresented |
#           misattributed | version_drift (faithful to superseded v1)
M = {
 "openai_dr": {"2302.04732": ("misattributed", "Zeno paper has no 'small/darkly lit objects' finding (grep 0 hits)"),
               "2407.12514": ("in_scope", ""), "2504.13629": ("in_scope", ""), "1703.0056": ("in_scope", "")},
 "harness": {"2509.26173": ("version_drift", "quotes v1; 2nd clause absent from current version"),
             "2512.03868": ("in_scope", "background fact"),
             "2511.14718": ("version_drift", "v1 abstract: accuracy 50->75% p=0.002; current version reverses: 46% vs 64% favoring Snowflake"),
             "2410.01774": ("in_scope", ""),
             "1906.09936": ("in_scope", "author-level generality inherited; critical hidden limitation: single centre, 52 recordings"),
             "2112.08787": ("in_scope", ""),
             "2412.16926": ("in_scope", "author-level generality inherited; critical hidden limitation: Gemini Pro/Flash only"),
             "2411.13868": ("in_scope", ""), "2105.06442": ("in_scope", ""),
             "2510.21970": ("misattributed", "quote is the paper's related-work sentence about LoRA Land [15], not its own result"),
             "2209.11302": ("in_scope", ""), "2509.15356": ("in_scope", ""), "2511.13722": ("in_scope", ""),
             "1812.05473": ("overreach_report", "node-classification gain on Cora/Pubmed that paper calls 'not that significant' -> 'beat node-only baselines on network classification tasks'; critical hidden limitation")},
 "gptr": {"2207.14711": ("in_scope", ""), "1707.06209": ("in_scope", ""), "1212.1186": ("in_scope", ""),
          "2007.04439": ("in_scope", ""), "2104.13457": ("in_scope", ""), "2211.16971": ("in_scope", ""),
          "2211.08473": ("in_scope", ""),
          "2211.11798": ("overreach_report", "tested only social-media toxicity/bias classification with Megatron PLMs -> 'efficacy of active learning in few-shot settings'; critical hidden limitation"),
          "2212.04214": ("in_scope", "author-level generality inherited (SSN only)"),
          "1012.5696": ("in_scope", ""),
          "2008.12228": ("overreach_report", "borderline: 'We develop a framework' -> 'General RL frameworks can learn...'"),
          "2010.07586": ("in_scope", ""), "2207.03034": ("in_scope", "")},
 "elicit": {"2311.03099": ("misrepresented", "2.20->66.34 GSM8K is Task-Arithmetic merging of LM&Math (table), attributed to DARE"),
            "2302.02104": ("in_scope", ""), "2110.07143": ("in_scope", ""), "2312.10136": ("in_scope", ""),
            "2403.02884": ("in_scope", ""), "2210.08823": ("in_scope", "judge FP: 'examined methods' hedged + 'specific contexts'"),
            "2202.03734": ("in_scope", "critical hidden limitation: 4 tabular datasets, logistic regression only"),
            "2201.04687": ("in_scope", ""), "2306.08891": ("in_scope", ""),
            "2303.06247": ("overreach_report", "tableware rearrangement, 3-5 objects, GPT-3 -> 'LLMs can effectively contribute to both simple and complex robot planning scenarios'; critical hidden limitation"),
            "2305.14992": ("in_scope", ""),
            "2305.10276": ("misattributed", "Q-learning maze / self-planning codegen results not in this paper"),
            "2309.11987": ("in_scope", ""), "2304.01904": ("in_scope", "")},
}
CRIT_HIDDEN = {"1906.09936", "2412.16926", "1812.05473", "2211.11798", "2202.03734", "2303.06247"}

a = json.load(open(os.path.join(P4, "s5_judged.json"), encoding="utf-8"))
b = json.load(open(os.path.join(P4, "s5b_judged_elicit1.json"), encoding="utf-8"))
valid = [o for o in a if o["system"] != "elicit" and o["judge"].get("label") != "no_text"] + b
rows, agree = [], {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
for o in valid:
    mine, note = M[o["system"]][o["arxiv_id"]]
    jl = o["judge"].get("label")
    rows.append({"system": o["system"], "arxiv_id": o["arxiv_id"], "sent": o["sent"], "judge": jl,
                 "manual": mine, "note": note, "critical_hidden_limitation": o["arxiv_id"] in CRIT_HIDDEN})
    m_prob, j_prob = mine != "in_scope", jl != "in_scope"
    agree[("tp" if j_prob else "fn") if m_prob else ("fp" if j_prob else "tn")] += 1
summ = {}
for s in ["openai_dr", "harness", "gptr", "elicit", "ALL"]:
    rs = [r for r in rows if s == "ALL" or r["system"] == s]
    n = len(rs)
    cnt = {k: sum(1 for r in rs if r["manual"] == k) for k in
           ["in_scope", "overreach_report", "misrepresented", "misattributed", "version_drift"]}
    summ[s] = {"n": n, **cnt,
               "overreach_rate": wilson(cnt["overreach_report"], n),
               "any_report_error_rate": wilson(cnt["overreach_report"] + cnt["misrepresented"] + cnt["misattributed"], n),
               "critical_hidden_limitation": wilson(sum(r["critical_hidden_limitation"] for r in rs), n)}
summ["overreach_strict_excl_borderline_and_strength"] = wilson(2, len(rows))
summ["judge_vs_manual_problem_flag"] = {**agree, "precision": round(agree["tp"] / (agree["tp"] + agree["fp"]), 3),
                                        "recall": round(agree["tp"] / (agree["tp"] + agree["fn"]), 3)}
summ["judge_overreach_flags"] = {"flagged": 2, "confirmed": 2, "manual_overreach": 4}
summ["calls"] = calls()
json.dump({"rows": rows, "summary": summ}, open(os.path.join(P4, "s6_final.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(summ, indent=1))
