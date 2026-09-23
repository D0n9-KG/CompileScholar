# -*- coding: utf-8 -*-
"""watchdog.py — line health sentinel (user directive 2026-09-22: the biggest
lesson of the night was processes dying SILENTLY and me noticing hours late).

Protocol per line, checked every N minutes:
  ALIVE    = ledger/progress advanced since last check
  WEDGED   = process exists but NO forward progress in STALL_LIMIT
  DEAD     = process gone but work remains (output incomplete)
  DONE     = work complete (happy path)

Each line declares a PROBE: a cheap function returning a monotonic progress
number (physical artifact state — file counts, ledger rows, processed docs —
NEVER in-memory counts or API return values: today's false 'complete'
signals all came from trusting volatile state).

Usage: run in background; prints state changes only (quiet when healthy).
"""
import json
import os
import subprocess
import time

KB = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/scholarqa_multi"
INTERVAL = 300      # 5 min between checks
STALL_LIMIT = 2400  # 40 min without progress = WEDGED

LINES = {
    "build-chain": {
        # probe: any ok-row advance in the shared build ledger (run_id-agnostic
        # — run ids change per relaunch, a filtered probe reads 0 and false-
        # alarms WEDGED on every fresh run)
        "probe": lambda: _ledger_rows(f"{KB}/kb/ledger_build.jsonl"),
        "done": lambda: os.path.exists(f"{KB}/kb/views.json"),
        "match": "registry|build.py|notation|table",
    },
    "lightrag": {
        # 2026-09-23 (2nd false alarm): state-count probes read FLAT when docs
        # flow between states at equal rates (3 enter parsing while 3 leave).
        # Monotonic probe instead: ok rows in the arm ledger — every completed
        # LLM call appends, never decreases, cannot cancel out.
        "probe": lambda: _ledger_rows(f"{KB}/baselines/lightrag/ledger_lightrag.jsonl"),
        "done": lambda: _processed_docs() >= 430,
        "match": "lightrag",
    },
    "paperqa": {
        "probe": lambda: _pqa_index_count() * 10 + _pqa_answer_rows(),
        "done": lambda: _pqa_answer_rows() >= 108,
        "match": "paperqa",
    },
    # 2026-09-23: the answering-phase lines. ours probe = answer rows x 100 +
    # ledger ok rows (answers only flush per question; ledger calls flow
    # continuously — combining both catches both stalls: no calls AND no
    # completions)
    "ours-answers": {
        "probe": lambda: _ours_answer_rows() * 100
                       + _ledger_rows(f"{KB}/baselines/ours/ledger_ours_multi.jsonl"),
        "done": lambda: _ours_answer_rows() >= 108,
        "match": "multi_ours_run",
    },
    # judge loop: GLM-5.3 verdicts (Paratera, independent of GPUStack)
    "judge": {
        "probe": lambda: _ledger_rows(f"{KB}/judge/ledger_judge.jsonl"),
        "done": lambda: False,  # never "done" — it idles between arms by design
        "match": "multi_judge_incremental",
    },
}


def _ledger_rows(path, run_id=None):
    n = 0
    try:
        for l in open(path, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("ok") and (run_id is None or r.get("run_id") == run_id):
                n += 1
    except FileNotFoundError:
        pass
    return n


def _doc_status():
    try:
        return json.load(open(f"{KB}/baselines/lightrag/work/kv_store_doc_status.json",
                              encoding="utf-8"))
    except Exception:
        return {}


def _processed_docs():
    # 2026-09-23: count pipeline-progress states, not just terminal 'processed'
    # — a doc in parsing/processing/analyzing is FORWARD progress, but the
    # processed-only probe read flat across a 40-min window and false-alarmed
    # WEDGED while extraction calls were actively flowing
    return sum(1 for v in _doc_status().values()
               if v.get("status") in ("processed", "parsing", "processing",
                                      "analyzing"))


def _ours_answer_rows():
    try:
        d = json.load(open(f"{KB}/baselines/ours/answers_pilot_multi.json",
                           encoding="utf-8"))
        return sum(1 for r in d if r.get("answer"))
    except Exception:
        return 0


def _failed_docs():
    return sum(1 for v in _doc_status().values() if v.get("status") == "failed")


def _pqa_index_count():
    import zlib, pickle
    idx = f"{KB}/baselines/paperqa/index"
    try:
        d = os.listdir(idx)[0]
        return len(pickle.loads(zlib.decompress(
            open(os.path.join(idx, d, "files.zip"), "rb").read())))
    except Exception:
        return 0


def _pqa_answer_rows():
    try:
        d = json.load(open(f"{KB}/baselines/paperqa/answers_paperqa.json",
                           encoding="utf-8"))
        return sum(1 for r in d if r.get("raw_answer"))
    except Exception:
        return 0


def _proc_alive(match):
    r = subprocess.run(
        ["powershell", "-Command",
         f"Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         f"Where-Object {{ $_.CommandLine -match '{match}' }} | "
         f"Measure-Object | Select-Object -ExpandProperty Count"],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip()) > 0
    except Exception:
        return False


def main():
    last_probe, last_change = {}, {}
    while True:
        for name, spec in LINES.items():
            if spec["done"]():
                state = "DONE"
            else:
                alive = _proc_alive(spec["match"])
                p = spec["probe"]()
                if not alive:
                    state = "DEAD"
                elif name in last_probe and p == last_probe[name]:
                    if time.time() - last_change.get(name, time.time()) > STALL_LIMIT:
                        state = "WEDGED"
                    else:
                        state = "ALIVE(no-progress)"
                else:
                    state = "ALIVE"
                    last_change[name] = time.time()
                last_probe[name] = p
            if state != last_change.get(name + ":state"):
                last_change[name + ":state"] = state
                print(f"[watchdog {time.strftime('%H:%M')}] {name}: {state} "
                      f"(probe={last_probe.get(name, '?')})", flush=True)
                if state in ("WEDGED", "DEAD"):
                    print(f"[watchdog] !!! {name} {state} — 需要人工介入 "
                          f"(重启/取证)", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
