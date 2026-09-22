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
        "probe": lambda: _ledger_rows(f"{KB}/kb/ledger_build.jsonl",
                                       run_id="multi-registry_growth"),
        "done": lambda: os.path.exists(f"{KB}/kb/views.json"),
        "match": "registry|build.py|notation|table",
    },
    "lightrag": {
        "probe": lambda: _processed_docs(),
        "done": lambda: _processed_docs() >= 430,
        "match": "lightrag",
    },
    "paperqa": {
        "probe": lambda: _pqa_index_count() * 10 + _pqa_answer_rows(),
        "done": lambda: _pqa_answer_rows() >= 108,
        "match": "paperqa",
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
    return sum(1 for v in _doc_status().values() if v.get("status") == "processed")


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
