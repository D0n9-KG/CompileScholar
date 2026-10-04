# -*- coding: utf-8 -*-
"""W2 P0-3: proposer extraction over every non-survey KB paper with an abstract (local 27B, thinking off, temp 0).
Resumable: data/state/methods.jsonl gets one line per validated proposal; data/state/methods_done.txt lists papers
already processed (including those that propose nothing)."""
import concurrent.futures as cf
import json
import os
import sys
import threading

from compilescholar.compile.skeleton import proposes as P
from compilescholar.core import paths

KB = paths.legacy_bench() / "cs2" / "base_kb_v2"
STATE = paths.data() / "state"


def main(workers: int = 16):
    os.environ.setdefault("LOCAL_MAX_CONCURRENT", "48")
    papers = json.load(open(KB / "papers.json", encoding="utf-8"))
    done_p = STATE / "methods_done.txt"
    done = set(open(done_p, encoding="utf-8").read().split()) if done_p.exists() else set()
    todo = sorted(pid for pid, p in papers.items() if p["layer"] != "survey" and (p.get("abstract") or "").strip()
                  and pid not in done)
    print(f"[P0-3] {len(todo)} to do, {len(done)} done", flush=True)
    lock = threading.Lock()
    n = found = 0
    with open(STATE / "methods.jsonl", "a", encoding="utf-8") as fm, open(done_p, "a", encoding="utf-8") as fd, \
            cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(P.extract, pid, papers[pid]["title"], papers[pid]["abstract"]): pid for pid in todo}
        for f in cf.as_completed(futs):
            pid = futs[f]
            try:
                rows = f.result()
            except Exception as e:
                print(f"  {pid} error {type(e).__name__}: {str(e)[:100]}", flush=True)
                continue
            with lock:
                for r in rows:
                    fm.write(json.dumps(r, ensure_ascii=False) + "\n")
                fd.write(pid + "\n")
                fm.flush()
                fd.flush()
                n += 1
                found += bool(rows)
                if n % 100 == 0:
                    print(f"  {n}/{len(todo)} papers, {found} with a validated proposal", flush=True)
    print(f"[P0-3] done: {n} papers, {found} with a validated proposal", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
