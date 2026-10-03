# -*- coding: utf-8 -*-
"""本地 27B（GPUStack）并发余量压测：在现有负载之上，再加 N 路并发请求，测单请求延迟与总吞吐随 N 的变化。
请求形态贴近答题管线写作调用：~3k token 输入、~400 token 输出、关思考。
注意：测的是"在当前后台负载之上的增量余量"，结果随后台任务数变化。
用法：python bench_local_concurrency.py [N1 N2 ...]   （缺省 4 8 16 24 32）"""
import concurrent.futures as cf
import json
import os
import re
import statistics as st
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ENV = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", ".env"))
cfg = {}
for line in open(ENV, encoding="utf-8"):
    m = re.match(r"^\s*(LOCAL_BASE_URL|LOCAL_API_KEY)\s*=\s*(.+?)\s*$", line)
    if m:
        cfg[m.group(1)] = m.group(2)
BASE = cfg["LOCAL_BASE_URL"].rstrip("/")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
FILLER = " ".join(f"Evidence item {i}: a verbatim excerpt about retrieval-augmented generation, citation grounding and "
                  f"evaluation of long-form scientific answers." for i in range(110))
PROMPT = FILLER + "\n\nWrite a 300-word paragraph summarizing the evidence above, citing items as [Ei]."


def one(_):
    body = json.dumps({"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": 400, "temperature": 0.2,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {cfg['LOCAL_API_KEY']}"})
    t = time.time()
    try:
        r = json.loads(OPENER.open(req, timeout=600).read())
        u = r.get("usage") or {}
        return time.time() - t, u.get("completion_tokens", 0), u.get("prompt_tokens", 0), None
    except Exception as e:
        return time.time() - t, 0, 0, str(e)[:80]


def main():
    levels = [int(x) for x in sys.argv[1:]] or [4, 8, 16, 24, 32]
    print(f"{'N':>4} {'ok':>4} {'lat_med':>8} {'lat_p90':>8} {'out_tok/s(total)':>17} {'in_tok':>7}")
    for n in levels:
        t0 = time.time()
        with cf.ThreadPoolExecutor(n) as ex:
            res = list(ex.map(one, range(n)))
        wall = time.time() - t0
        ok = [r for r in res if r[3] is None]
        lat = sorted(r[0] for r in ok)
        tps = sum(r[1] for r in ok) / wall if wall else 0
        errs = [r[3] for r in res if r[3]]
        print(f"{n:>4} {len(ok):>4} {st.median(lat) if lat else 0:>8.1f} {lat[int(0.9 * (len(lat) - 1))] if lat else 0:>8.1f} "
              f"{tps:>17.1f} {ok[0][2] if ok else 0:>7} {errs[:1] if errs else ''}", flush=True)


if __name__ == "__main__":
    main()
