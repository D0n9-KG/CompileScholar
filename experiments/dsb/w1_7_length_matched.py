# -*- coding: utf-8 -*-
"""W1-7 (DSB): length-matched comparison. Our DSB answers (vnext_v3_nocite, median 2,130 words) are cut, per
question, to the harness_v2 answer's word count for the same question (questions the harness failed on: cut to the
harness median, 668). Cutting keeps whole paragraphs from the top while they fit, so the text stays readable;
citation markers inside kept paragraphs are untouched. Judged with the same nugget judge; quiet output.

Writes runs/w1-7-dsb-length-matched-20261004/{gen/<gt>.md, judged.json, summary.json}.
"""
import json
import statistics as st
import sys
from pathlib import Path

from compilescholar.core import paths
from compilescholar.eval import dsb

P6 = paths.benchmarks("dsb") / "arms"
OUT = paths.runs() / "w1-7-dsb-length-matched-20261004"


def cut_to_words(text: str, budget: int) -> str:
    paras = [p for p in text.split("\n\n")]
    out, n = [], 0
    for p in paras:
        w = len(p.split())
        if out and n + w > budget:
            break
        out.append(p)
        n += w
    return "\n\n".join(out)


def main():
    qs = dsb.question_set()
    h = {r["gt_dir"]: r for r in json.load(open(P6 / "judged" / "harness_v2.json", encoding="utf-8"))}
    h_med = int(st.median(r["words"] for r in h.values()))
    gen = OUT / "gen"
    gen.mkdir(parents=True, exist_ok=True)
    budgets = {}
    for g in qs:
        src = P6 / "gen" / "vnext_v3_nocite" / f"{g}.md"
        if not src.exists():
            continue
        b = h[g]["words"] if g in h and "error" not in h[g] else h_med
        budgets[g] = b
        (gen / f"{g}.md").write_text(cut_to_words(src.read_text(encoding="utf-8"), b), encoding="utf-8")
    for attempt in range(3):
        res = dsb.judge_dir(str(gen), str(OUT / "judged.json"), ids=list(budgets), workers=6)
        print(f"[w1-7 dsb] attempt {attempt + 1}: {res}", flush=True)
        if res["error_rows"] == 0:
            break
    words_after = [len((gen / f"{g}.md").read_text(encoding="utf-8").split()) for g in budgets]
    s = dsb.summarize(str(OUT / "judged.json"), qs)
    p = dsb.paired(str(OUT / "judged.json"), str(P6 / "judged" / "harness_v2.json"), qs)
    summary = {"n": len(qs), "ours_length_matched": s["nugget_coverage"], "ours_original": dsb.summarize(
        str(P6 / "judged" / "vnext_v3_nocite.json"), qs)["nugget_coverage"],
        "harness_v2": dsb.summarize(str(P6 / "judged" / "harness_v2.json"), qs)["nugget_coverage"],
        "paired_ours_matched_minus_harness": p, "words_median_after_cut": st.median(words_after),
        "harness_words_median": h_med}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
