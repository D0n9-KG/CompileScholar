# -*- coding: utf-8 -*-
"""GOLDCOV step 5: aggregate kb_cov + ans_cov into the 2x2 decomposition,
per-type / per-paper breakdowns, and the batch-5 hit list.

Usage:
  py -3.13 report.py            # writes GOLDCOV-REPORT.md + report.json
"""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    points = json.load(open(os.path.join(HERE, "points.json"), encoding="utf-8"))
    kb, ans = {}, {}
    for l in open(os.path.join(HERE, "kb_cov.jsonl"), encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            kb[r["key"]] = r
    if os.path.exists(os.path.join(HERE, "ans_cov.jsonl")):
        for l in open(os.path.join(HERE, "ans_cov.jsonl"), encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                ans[r["key"]] = r
    return points, kb, ans


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    points, kb, ans = load()
    qmeta = {qid: {"ptype": q["prompt_type"], "question": q["question"][:200]}
             for qid, q in points.items()}

    rows = []
    for qid, q in points.items():
        for p in q["points"]:
            key = f"{qid}#{p['point_id']}"
            k = kb.get(key)
            a = ans.get(key)
            if not k:
                continue
            kb_has = k["verdict"] in ("covered", "partial")
            rows.append({
                "key": key, "qid": qid, "prompt_type": q["prompt_type"],
                "point_id": p["point_id"], "ptype": p["ptype"],
                "point": p["text"], "kb_verdict": k["verdict"],
                "kb_tier": k.get("tier"), "kb_reason": k.get("reason", ""),
                "in_answer": a.get("in_answer") if a else None,
                "ans_reason": a.get("reason", "") if a else "",
                "kb_has": kb_has,
            })

    n = len(rows)
    # ---- 2x2 ----
    both = [r for r in rows if r["kb_has"] and r["in_answer"]]
    kb_only = [r for r in rows if r["kb_has"] and r["in_answer"] is False]
    ans_only = [r for r in rows if not r["kb_has"] and r["in_answer"]]
    neither = [r for r in rows if not r["kb_has"] and r["in_answer"] is False]
    unknown = [r for r in rows if r["in_answer"] is None and r in rows]

    cov = Counter(r["kb_verdict"] for r in rows)
    cov_score = (cov["covered"] + 0.5 * cov["partial"]) / max(1, n)

    by_type = defaultdict(lambda: Counter())
    for r in rows:
        by_type[r["prompt_type"]][r["kb_verdict"]] += 1

    by_paper = defaultdict(lambda: Counter())
    # evidence paper attribution comes from kb_cov evidence ids -> rec_map
    rmap = json.load(open(os.path.join(HERE, "rec_map.json"), encoding="utf-8"))
    kb_raw = {}
    for l in open(os.path.join(HERE, "kb_cov.jsonl"), encoding="utf-8"):
        if l.strip():
            rr = json.loads(l)
            kb_raw[rr["key"]] = rr
    for r in rows:
        if r["kb_verdict"] in ("covered", "partial"):
            for k in (kb_raw[r["key"]].get("evidence") or []):
                m = rmap.get(k)
                if m:
                    by_paper[m["paper_id"]]["evidence"] += 1

    # per-question missing points (batch-5 hit list)
    per_q_missing = defaultdict(list)
    for r in rows:
        if r["kb_verdict"] == "missing":
            per_q_missing[r["qid"]].append(r)

    out = {
        "n_points": n,
        "kb_coverage": {"covered": cov["covered"], "partial": cov["partial"],
                        "missing": cov["missing"], "score": round(cov_score, 4)},
        "matrix": {
            "kb_and_answer": len(both), "kb_only_answer_missing": len(kb_only),
            "answer_only_kb_missing": len(ans_only), "neither": len(neither),
            "answer_unknown": len(unknown),
        },
        "by_type": {k: dict(v) for k, v in by_type.items()},
        "per_q_missing": {q: len(v) for q, v in per_q_missing.items()},
    }
    json.dump(out | {"rows": rows}, open(os.path.join(HERE, "report.json"), "w",
                                         encoding="utf-8"), ensure_ascii=False, indent=1)

    md = ["# GOLDCOV 报告（gold 要点覆盖率）", ""]
    md.append(f"总要点：**{n}**（30 题）")
    md.append(f"KB 覆盖率：**{cov_score:.1%}** "
              f"(covered {cov['covered']} / partial {cov['partial']} / missing {cov['missing']})")
    md.append("")
    md.append("## 2×2 分解（KB × 答案）")
    md.append("")
    md.append("| | answer 有 | answer 无 |")
    md.append("|---|---|---|")
    md.append(f"| **KB 有** | {len(both)} | {len(kb_only)} |")
    md.append(f"| **KB 无** | {len(ans_only)} (A7 兜底?) | {len(neither)} (抽取缺口) |")
    if unknown:
        md.append(f"| answer 判定缺失: {len(unknown)}")
    md.append("")
    md.append("## 分题型")
    md.append("")
    md.append("| 题型 | covered | partial | missing | 覆盖率 |")
    md.append("|---|---|---|---|---|")
    for t in sorted(by_type):
        c = by_type[t]
        s = (c["covered"] + 0.5 * c["partial"]) / max(1, sum(c.values()))
        md.append(f"| {t} | {c['covered']} | {c['partial']} | {c['missing']} | {s:.1%} |")
    md.append("")
    md.append("## 批 5 动刀清单（missing 要点，按题）")
    md.append("")
    for qid in sorted(per_q_missing, key=lambda q: -len(per_q_missing[q])):
        q = per_q_missing[qid]
        md.append(f"### {qid}（{qmeta[qid]['ptype']}，{len(q)} missing）")
        for r in q:
            md.append(f"- [{r['ptype']}] {r['point'][:220]}")
        md.append("")
    open(os.path.join(HERE, "GOLDCOV-REPORT.md"), "w", encoding="utf-8").write("\n".join(md))
    print(f"n={n} cov={cov_score:.1%} matrix={out['matrix']}")
    print(f"-> {HERE}/GOLDCOV-REPORT.md")


if __name__ == "__main__":
    main()
