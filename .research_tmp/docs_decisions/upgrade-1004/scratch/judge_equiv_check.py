# -*- coding: utf-8 -*-
"""W6 S6 equivalence check for the new CS2 judge entry point on 5 dev questions (v9b dev answers, never test).

Usage: python judge_equiv_check.py <impl: new|old> <out_dir> [tag]
Writes <out_dir>/j5_<impl>_<tag>.json; prints per-question facet deltas vs the stored DeepSeek scores."""
import asyncio
import json
import os
import statistics as st
import subprocess
import sys

from compilescholar.eval.cs2 import judge as J
from compilescholar.eval.cs2 import scoring as S

impl, out_dir = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else "r1"
C = os.path.join(str(S.paths.legacy_bench()), "cs2")
inp = json.load(open(os.path.join(C, "judge_input_vnext_dev10_v9b_oa.json"), encoding="utf-8"))[:5]
stored = json.load(open(os.path.join(C, "direct_scores_vnext_dev10_v9b_oa_ds.json"), encoding="utf-8"))
ip = os.path.join(out_dir, "j5_in.json")
op = os.path.join(out_dir, f"j5_{impl}_{tag}.json")
json.dump(inp, open(ip, "w", encoding="utf-8"), ensure_ascii=False)
if os.path.exists(op):
    os.remove(op)
if impl == "new":
    asyncio.run(J.judge_file(ip, op, "dev", J.JudgeAdapter.legacy(), parallel=5, quiet_scores=True))
else:
    subprocess.run([sys.executable, "-W", "ignore", "direct_judge.py", ip, op, "5"], cwd=C,
                   env={**os.environ, "CS2_SPLIT": "dev", "PYTHONUTF8": "1"}, capture_output=True)
new = json.load(open(op, encoding="utf-8"))
dg = []
for r in inp:
    a, b = S.facets(stored.get(r["qid"])), S.facets(new.get(r["qid"]))
    if a is None or b is None:
        print(r["qid"][:12], "missing facets")
        continue
    dg.append(abs(S.official_global(a) - S.official_global(b)))
    print(r["qid"][:12], f"|dG|={dg[-1]:.3f}", " ".join(f"{k[:6]}={b[k] - a[k]:+.3f}" for k in S.FACETS))
n_err = sum(1 for v in new.values() if isinstance(v, dict) and ("error" in v or "_errors" in v))
print(f"[{impl} {tag}] judge_error_rows {n_err} | mean |dG| {st.mean(dg):.3f}")
