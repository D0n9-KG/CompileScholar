# -*- coding: utf-8 -*-
"""W6 S7: move code that the new package replaces into legacy/ with `git mv` (history kept), and write legacy/INDEX.md.

Only CODE moves (.py / .sh / .ps1 under the listed directories). Data, logs and outputs stay where they are (that is
S8, data move). The frozen v9b test is reproducible from tag freeze-cs2-v9b; nothing here needs to keep running in
place. Dry run by default; pass --apply to execute.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
B = ".research_tmp/experiments/benchmarks"
MOVES = [  # (source dir, legacy dir, patterns)
    (f"{B}/_shared/tools", "legacy/benchmarks/_shared/tools", ("*.py",)),
    (f"{B}/_shared/mcp", "legacy/benchmarks/_shared/mcp", ("*.py",)),
    (f"{B}/cs2", "legacy/benchmarks/cs2", ("*.py", "*.sh", "*.ps1")),
    (f"{B}/cs2/base_kb_build", "legacy/benchmarks/cs2/base_kb_build", ("*.py",)),
    (f"{B}/cs2/judge_runs_ours", "legacy/benchmarks/cs2/judge_runs_ours", ("*.py",)),
    (f"{B}/scholarqa_multi", "legacy/benchmarks/scholarqa_multi", ("*.py",)),
    (f"{B}/review_1002_p6_link", None, ()),  # placeholder, nothing
]


def tracked(pattern: str) -> list[str]:
    r = subprocess.run(["git", "-C", str(REPO), "ls-files", "--", pattern], capture_output=True, text=True)
    return [x for x in r.stdout.splitlines() if x]


def plan() -> list[tuple[str, str]]:
    out = []
    for src, dst, pats in MOVES:
        if dst is None:
            continue
        for p in pats:
            for f in tracked(f"{src}/{p}"):
                if Path(f).parent.as_posix() != src:  # top level of that dir only
                    continue
                out.append((f, f"{dst}/{Path(f).name}"))
    return out


if __name__ == "__main__":
    moves = plan()
    print(f"{len(moves)} files")
    for a, b in moves[:8]:
        print(f"  {a} -> {b}")
    if "--apply" in sys.argv:
        for a, b in moves:
            (REPO / b).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "-C", str(REPO), "mv", a, b], check=True)
        print("moved")
