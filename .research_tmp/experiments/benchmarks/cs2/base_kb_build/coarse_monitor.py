# -*- coding: utf-8 -*-
"""Coarse-extraction efficiency monitor (user directive 09-27: 并发拉满 +
观察效率/问题/冗余/可优化点).

Reads the coarse-extraction ledger and reports: throughput, failure rate,
latency distribution, token economics, and redundancy signals.

Usage: python coarse_monitor.py [ledger_path]
"""
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

LEDGER = (Path(sys.argv[1]) if len(sys.argv) > 1 else
          Path(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp"
               r"\experiments\benchmarks\cs2\base_kb\ledger_coarse.jsonl"))


def parse_ts(s):
    return datetime.fromisoformat(s).timestamp()


def main():
    rows = []
    for line in open(LEDGER, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        print("no ledger rows yet")
        return
    ok = [r for r in rows if r.get("ok")]
    fail = [r for r in rows if not r.get("ok")]

    tss = [parse_ts(r["ts"]) for r in rows if r.get("ts")]
    span_s = (max(tss) - min(tss)) if tss else 0
    ptoks = sum(r.get("prompt_tokens") or 0 for r in ok)
    ctoks = sum(r.get("completion_tokens") or 0 for r in ok)
    durs = sorted((r.get("latency_ms") or 0) for r in ok)

    print(f"=== coarse-extraction monitor ({len(rows)} calls, "
          f"{span_s/60:.1f} min span) ===")
    print(f"ok={len(ok)} fail={len(fail)} "
          f"({len(fail)/max(1,len(rows))*100:.1f}% fail)")
    if durs:
        print(f"latency ms: p50={durs[len(durs)//2]} "
              f"p90={durs[int(len(durs)*0.9)]} max={durs[-1]}")
    if ok:
        ct = [r.get("completion_tokens") or 0 for r in ok]
        ct.sort()
        print(f"completion tokens: p50={ct[len(ct)//2]} "
              f"p90={ct[int(len(ct)*0.9)]} max={ct[-1]} "
              f"(cap=2000; at-cap={sum(1 for x in ct if x>=1900)})")
    if span_s > 0:
        rate = len(ok) / (span_s / 60)
        est_min = (855 - len(ok)) / max(rate, 0.01)
        print(f"throughput: {rate:.1f} papers/min | "
              f"{ptoks+ctoks and int((ptoks+ctoks)/span_s*60) or 0} tok/min | "
              f"ETA remaining {est_min:.0f} min (of 855)")
    if fail:
        errs = Counter()
        for r in fail:
            m = re.search(r"HTTP (\d+)", str(r))
            errs[m.group(1) if m else "other"] += 1
        print(f"fail modes: {dict(errs)}")
    # attempt 分布（重试=效率损失信号）
    att = Counter(r.get("attempt") for r in rows)
    if len(att) > 1 or (0 not in att):
        print(f"attempt distribution: {dict(att)} (attempt>0 = retries)")


if __name__ == "__main__":
    main()
