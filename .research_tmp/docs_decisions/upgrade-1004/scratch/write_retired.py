# -*- coding: utf-8 -*-
"""Write .research_tmp/RETIRED.md from the NAS verification reports."""
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
NAS = r"\\192.168.199.138\Share400T\personal\jhd0n9\CompileScholar-cold-20261004"
DIRS = ["experiments/archive", "experiments/baselines", "experiments/e2_need_gap", "experiments/stageB",
        "archive_oneoff", "benchmark-audit-0926"]
rows = []
for d in DIRS:
    v = json.load(open(os.path.join(NAS, "_logs", "verify_" + d.replace("/", "_") + ".json")))
    now = NAS + "\\" + d.replace("/", "\\")
    rows.append(f"| `.research_tmp/{d}/` | `{now}` | {v['files']:,} | {v['bytes'] / 1e9:.1f} GB |")
txt = ("# Retired to cold storage (2026-10-04)\n\n"
       "These directories held retired experiments (PaperScope / AirQA archive, baseline model weights, the E2 need-gap\n"
       "study, Stage B pilots, one-off scratch, the 09-26 benchmark audit). Each was copied to the NAS, every file was\n"
       "verified by size and sha256 against the original (per-directory reports in `_logs/verify_*.json` there), and only\n"
       "then deleted here. Two tracked scripts under stageB were among them; they remain in git history.\n\n"
       "| Was | Now | Files | Size |\n|---|---|---|---|\n" + "\n".join(rows) +
       "\n\nRestore one with `robocopy <Now> <Was> /E`. No reported number depends on them (see `legacy/INDEX.md`).\n")
open(os.path.join(REPO, ".research_tmp", "RETIRED.md"), "w", encoding="utf-8").write(txt)
print(txt)
