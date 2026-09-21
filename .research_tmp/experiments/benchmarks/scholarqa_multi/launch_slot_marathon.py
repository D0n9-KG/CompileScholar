# -*- coding: utf-8 -*-
"""Gate-passed slot marathon launcher (overnight plan stage 1).

Merges the 10 smoke papers from kb/records_smoke.json into
kb/records_slot.json (same payload structure, pid-keyed -> marathon's
per-pid resume skips them = 10 papers of re-extraction saved), then hands
off to run_stage.py --name slot (local:Qwen3.8-27B, allowlist gate,
LOCAL_MAX_CONCURRENT=32, purity assertion at end).

Run ONLY after the stage-0 gate verdict is PASS (three-part judgment in
OVERNIGHT-PLAN.md).
"""
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(BASE, "kb")
SMOKE = os.path.join(KB, "records_smoke.json")
OUT = os.path.join(KB, "records_slot.json")


def main():
    smoke = json.load(open(SMOKE, encoding="utf-8"))
    if len(smoke) != 10:
        print(f"[gate] ABORT: smoke has {len(smoke)}/10 papers — not complete")
        sys.exit(2)
    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT, encoding="utf-8"))
    added = 0
    for pid, payload in smoke.items():
        if pid not in out:
            out[pid] = payload
            added += 1
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[merge] records_slot.json: {len(out)} papers "
          f"(+{added} from smoke)")

    cmd = [sys.executable, os.path.join(BASE, "run_stage.py"), "--name", "slot",
           "--",
           sys.executable, "-m", "kb_compiler.records.slot",
           "--texts", os.path.join(BASE, "corpus", "texts"),
           "--manifest", os.path.join(BASE, "corpus", "manifest.json"),
           "--cards", os.path.join(KB, "cards.json"),
           "--registry", os.path.join(KB, "registry.json"),
           "--vocab", os.path.join(KB, "dim_vocab_v1.json"),
           "--out", OUT,
           "--model", "local:Qwen3.8-27B",
           # concurrency fix (09-22 02:00 diagnosis): per-paper chunk calls
           # were SERIAL (chunk-threads=1 default) -- smoke measured 32k
           # tok/min vs the probed 384k peak @32 in-flight. 11 papers x 3
           # chunk threads = 33 ~= the 32-call semaphore ceiling; single-paper
           # wall clock shrinks ~3x. The 110k absence pass is a single call
           # (physical budget, not parallelizable) and is unaffected.
           "--workers", "11", "--chunk-threads", "3"]
    print("[marathon] launching:", " ".join(cmd[-8:]), flush=True)
    rc = subprocess.run(cmd, cwd=BASE).returncode
    print(f"[marathon] run_stage rc={rc}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
