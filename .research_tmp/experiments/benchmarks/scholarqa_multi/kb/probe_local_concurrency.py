# -*- coding: utf-8 -*-
"""Local-GPU concurrency saturation probe (user directive 09-21: 稳定前提下
并发尽量拉高，本地不考虑成本，能多快就多快).

Fires a slot-chunk-shaped prompt (~2.5k in / 600 out) at escalating
concurrency levels; measures aggregate tokens/min, latency p50/p95, and
FAILURE COUNT per level. Stops when errors appear or throughput falls off
the knee. Recommendation = highest zero-fail level within 5% of peak.

Raw urllib bypasses _LOCAL_SEM entirely — the ThreadPool size IS the
concurrency under test. Run on an otherwise idle GPU.

Usage: python probe_local_concurrency.py [--levels 8,16,24,32,48,64]
           [--calls-per-level 32]
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")

PROMPT_HEAD = ("下面是论文的一个文本块。抽取所有实验事实为 JSON records。"
               "规则：quote-first，逐字抄原文，禁止改写。\n\n文本：\n")


def make_prompt(seed: int) -> str:
    para = ("We evaluate the proposed method on three benchmarks using a "
            "standard protocol. The model is trained for 100k steps with a "
            "learning rate of 3e-4 and batch size 128. Ablations show that "
            "each component contributes measurably: removing the auxiliary "
            "loss degrades accuracy by 2.1 points, while removing the "
            "replay buffer increases variance across seeds. ")
    body = para * ((seed % 3) + 10)
    return (PROMPT_HEAD + body
            + f"\n\n变体编号 {seed}：输出 JSON {{\"records\": []}} 并在结尾注释该编号。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="8,16,24,32,48,64")
    ap.add_argument("--calls-per-level", type=int, default=32)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import kb_infra.llm as llm
    base_url = (llm.ENV.get("LOCAL_BASE_URL") or "").rstrip("/")
    key = llm.ENV.get("LOCAL_API_KEY", "local")

    def call(seed):
        payload = {"model": "Qwen3.8-27B",
                   "messages": [{"role": "user", "content": make_prompt(seed)}],
                   "temperature": 0.0, "max_tokens": 600,
                   "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(
            base_url + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                resp = json.load(r)
        except Exception as e:
            return time.time() - t0, 0, 0, repr(e)[:90]
        u = resp.get("usage") or {}
        return (time.time() - t0, u.get("prompt_tokens") or 0,
                u.get("completion_tokens") or 0, "")

    levels = [int(x) for x in args.levels.split(",")]
    print(f"probe: levels={levels} calls/level={args.calls_per_level} "
          f"(slot-chunk-shaped prompt)", flush=True)
    rows, peak, peak_level = [], 0.0, 0
    for level in levels:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=level) as ex:
            results = list(ex.map(call, range(args.calls_per_level)))
        wall = time.time() - t0
        fails = [r[3] for r in results if r[3]]
        oks = [r for r in results if not r[3]]
        toks = sum(p + c for _, p, c, _ in oks)
        rate = toks / wall / 1000 if wall > 0 else 0.0
        lats = sorted(r[0] for r in oks) or [0.0]
        p95 = lats[min(int(len(lats) * 0.95), len(lats) - 1)]
        print(f"  conc={level:>3}: {rate:>6.1f}k tok/min | "
              f"lat p50={lats[len(lats)//2]:>6.0f}s p95={p95:>6.0f}s | "
              f"fail {len(fails)}/{len(results)} | wall {wall:.0f}s", flush=True)
        if fails:
            print(f"      first errors: {fails[:2]}", flush=True)
        rows.append({"level": level, "rate_k": round(rate, 1),
                     "fails": len(fails), "p50": lats[len(lats) // 2],
                     "p95": p95})
        if not fails and rate > peak:
            peak, peak_level = rate, level
        if fails or (peak > 0 and rate < peak * 0.92 and level > peak_level):
            print(f"  -> stability/throughput limit at conc={level}", flush=True)
            break
    safe = [r["level"] for r in rows
            if r["fails"] == 0 and r["rate_k"] >= peak * 0.95]
    rec = max(safe) if safe else peak_level
    print(f"\nPEAK {peak:.1f}k tok/min @ conc={peak_level} | "
          f"RECOMMENDED LOCAL_MAX_CONCURRENT={rec} (zero-fail, >=95% peak)")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "local_concurrency_probe.json")
    json.dump({"rows": rows, "peak_k": round(peak, 1),
               "peak_level": peak_level, "recommended": rec},
              open(out, "w"), indent=1)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
