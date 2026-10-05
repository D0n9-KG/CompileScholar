# -*- coding: utf-8 -*-
"""P2 analysis: per survey and pooled, mean over runs per arm, and paired arm differences (reception/both vs self/memory)
with a bootstrap over (survey, run-pair) units. Runs whose state is empty (failed generation) are excluded and listed.
Writes runs/pilot-p2-reception-vs-self-20261005/analysis.json."""
import itertools
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "pilot-p2-reception-vs-self-20261005"
PARTS = ("families", "properties", "limitations")
ARMS = ("self", "reception", "both", "memory")


def load():
    rows, failed = {}, []
    for sp in OUT.glob("*.scores.json"):
        name = sp.name[: -len(".scores.json")]
        sid = ".".join(name.split(".")[:2])
        arm, rk = name.split(".")[2], name.split(".")[3]
        st = json.load(open(OUT / f"{name}.json", encoding="utf-8"))
        if not st.get("families"):
            failed.append(name)
            continue
        s = json.load(open(sp, encoding="utf-8"))
        rows.setdefault(sid, {}).setdefault(arm, {})[rk] = {p: s[p]["recall"] or 0.0 for p in PARTS}
    return rows, failed


def boot(xs, B=10000, seed=0):
    r = random.Random(seed)
    m = sorted(sum(r.choice(xs) for _ in xs) / len(xs) for _ in range(B))
    return [round(sum(xs) / len(xs), 3), round(m[int(.025 * B)], 3), round(m[int(.975 * B)], 3)]


def main():
    rows, failed = load()
    per = {sid: {arm: {p: round(sum(v[p] for v in runs.values()) / len(runs), 3) for p in PARTS} | {"n_runs": len(runs)}
                 for arm, runs in arms.items()} for sid, arms in rows.items()}
    paired = {}
    for a, b in (("reception", "self"), ("reception", "memory"), ("both", "self"), ("self", "memory")):
        for p in PARTS:
            d = []
            for sid, arms in rows.items():
                if a in arms and b in arms:
                    d += [arms[a][x][p] - arms[b][y][p] for x, y in itertools.product(arms[a], arms[b])]
            if d:
                paired[f"{a}-{b}.{p}"] = boot(d)
    out = {"per_survey": per, "paired_pooled": paired, "failed_runs_excluded": sorted(failed),
           "note": "paired = all run-pairs within each survey, pooled over surveys; bootstrap over pairs (3 surveys only)"}
    json.dump(out, open(OUT / "analysis.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
