# -*- coding: utf-8 -*-
"""Incremental post-pass for the OUR arm: converts native answers
(answers_pilot_multi.json, written per-question by the harness) into the
official-format answers_ours.json every cycle, so the judge loop can score
our arm question-by-question instead of waiting for the run to finish
(user directive: answer one, judge one).

Idempotent: reprocesses every native row each cycle (cheap, deterministic);
the judge loop's scored-set prevents double-scoring.
"""
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

import multi_ours_run as M  # noqa: E402
from multi_baseline_common import load_id_mapping  # noqa: E402

NATIVE = os.path.join(M.ARM, "answers_pilot_multi.json")
OFFICIAL = os.path.join(M.ARM, "answers_ours.json")


def one_pass() -> int:
    if not os.path.exists(NATIVE):
        return 0
    rows = json.load(open(NATIVE, encoding="utf-8"))
    idm = load_id_mapping()
    done = {}
    if os.path.exists(OFFICIAL):
        done = {r["qid"]: r for r in json.load(open(OFFICIAL, encoding="utf-8"))}
    for r in rows:
        if r.get("answer"):
            done[r["id"]] = M.to_official_row(r, idm)
    _tmp = OFFICIAL + ".tmp"
    with open(_tmp, "w", encoding="utf-8") as _f:
        json.dump(list(done.values()), _f, ensure_ascii=False, indent=1)
    os.replace(_tmp, OFFICIAL)  # N11 atomic write
    return len(done)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    while True:
        try:
            n = one_pass()
            print(f"[ours-postpass] {n} official rows", flush=True)
        except Exception as e:
            print(f"[ours-postpass] error: {e}", flush=True)
        time.sleep(interval)
