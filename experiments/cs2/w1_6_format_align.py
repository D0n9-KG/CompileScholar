# -*- coding: utf-8 -*-
"""W1-6 format-aligned re-judgment of the frozen v9b CS2 test answers (pre-registered reading rule: PREREG §4).

Builds variants from results/cs2-test100-v9b-20261004 and judges them with the SAME judge as the main table
(JudgeAdapter.legacy(), DeepSeek-V4.1-Flash), quietly (no per-question scores printed). Outputs go to
runs/w1-6-format-align-20261004/<variant>/{judge_input.json, scores.json}.

  ours_r{1,2}_trim     our snippets trimmed to the harness median length (240 chars, most-relevant sentences)
  ours_r{1,2}_title    our citations with titles only
  harness_title        harness citations with titles only
"""
import asyncio
import gzip
import json
import sys
from pathlib import Path

from compilescholar.core import paths
from compilescholar.eval.cs2 import format_align as F
from compilescholar.eval.cs2 import judge as J

RES = paths.REPO / "results" / "cs2-test100-v9b-20261004"
OUT = paths.runs() / "w1-6-format-align-20261004"
JOBS = [("ours_r1_trim", "ours_r1", "trim"), ("ours_r2_trim", "ours_r2", "trim"),
        ("ours_r1_title", "ours_r1", "title"), ("ours_r2_title", "ours_r2", "title"),
        ("harness_title", "harness", "title")]


def main(only: list[str]):
    for name, arm, kind in JOBS:
        if only and name not in only:
            continue
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        rows = json.load(gzip.open(RES / arm / "judge_input.json.gz"))
        v = F.variant(rows, kind)
        json.dump(v, open(d / "judge_input.json", "w", encoding="utf-8"), ensure_ascii=False)
        json.dump({"source": arm, "variant": kind, "stats_before": F.snippet_stats(rows), "stats_after": F.snippet_stats(v)},
                  open(d / "variant.json", "w"), indent=1)
        for attempt in range(3):  # judge-error rows are retried on each call until none remain
            res = asyncio.run(J.judge_file(str(d / "judge_input.json"), str(d / "scores.json"), "test",
                                           J.JudgeAdapter.legacy(), parallel=8, quiet_scores=True))
            print(f"[w1-6] {name} attempt {attempt + 1}: {res}", flush=True)
            if res["judge_error_rows"] == 0:
                break


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1:])
