# -*- coding: utf-8 -*-
"""Phase 0.5 P9 probe: single-stream + concurrent throughput on GPUStack
Qwen3.8-27B. Replicates the 09-21 probe methodology (zero-fail conc levels,
aggregate tok/min) to decide whether the P9 replica-suspected degradation
reproduces. Writes results JSON next to this file.

Usage: python concurrency_probe.py [levels] e.g. "1,4,8,16,32"
"""
import json
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

ENV = {}
for line in open(r"C:\Users\D0n9\Desktop\CompileScholar\.env", encoding="utf-8"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        ENV[k.strip()] = v.strip()

BASE = ENV.get("LOCAL_BASE_URL", "http://192.168.199.73/v1").rstrip("/")
KEY = ENV.get("LOCAL_API_KEY", "local")
MODEL = "Qwen3.8-27B"

GEN_TOKENS = 256
PROMPT = ("Write a continuous technical paragraph about transformer "
          "architecture. Do not stop early, do not repeat yourself.")


def one_stream(out: list, idx: int):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": GEN_TOKENS,
        "temperature": 0.7,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    ntok = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    j = json.loads(line[6:])
                except Exception:
                    continue
                delta = (j.get("choices") or [{}])[0].get("delta", {})
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    ntok += 1
        out[idx] = {"ok": True, "tok": ntok, "ttft_s": round(ttft or -1, 2),
                    "dur_s": round(time.time() - t0, 1)}
    except Exception as e:
        out[idx] = {"ok": False, "err": str(e)[:150],
                    "dur_s": round(time.time() - t0, 1)}


def probe(level: int) -> dict:
    out = [None] * level
    threads = [threading.Thread(target=one_stream, args=(out, i))
               for i in range(level)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    ok = [r for r in out if r and r["ok"]]
    toks = sum(r["tok"] for r in ok)
    ttfts = sorted(r["ttft_s"] for r in ok)
    return {
        "level": level,
        "ok_streams": len(ok),
        "failed": level - len(ok),
        "wall_s": round(wall, 1),
        "total_tokens": toks,
        "tok_per_min": round(toks / wall * 60),
        "tok_per_s_per_stream": round(toks / wall / max(1, len(ok)), 1),
        "ttft_median_s": ttfts[len(ttfts) // 2] if ttfts else None,
    }


def main():
    levels = [int(x) for x in
              (sys.argv[1] if len(sys.argv) > 1 else "1,4,8,16,32").split(",")]
    results = []
    for lv in levels:
        print(f"[probe] conc={lv} ...", flush=True)
        r = probe(lv)
        results.append(r)
        print(f"  ok={r['ok_streams']}/{lv} wall={r['wall_s']}s "
              f"total={r['total_tokens']} tok/min={r['tok_per_min']} "
              f"per-stream={r['tok_per_s_per_stream']} tok/s "
              f"ttft_med={r['ttft_median_s']}s", flush=True)
        time.sleep(3)
    out = Path(__file__).with_name("concurrency_probe_results.json")
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
