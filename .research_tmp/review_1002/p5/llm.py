# -*- coding: utf-8 -*-
"""Minimal Paratera DeepSeek client with call/usage ledger (key never printed)."""
import json, os, re, time, threading
from openai import OpenAI

ENV = r"C:\Users\D0n9\Desktop\CompileScholar\.env"
LEDGER = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\review_1002\p5\calls.jsonl"
MODEL = "DeepSeek-V4.1-Flash"
_lock = threading.Lock()


def _env(name):
    for line in open(ENV, encoding="utf-8"):
        m = re.match(rf"^{name}\s*=\s*(.+)$", line.strip())
        if m:
            return m.group(1).strip()
    raise KeyError(name)


_client = OpenAI(api_key=_env("PARATERA_API_KEY"), base_url=_env("PARATERA_BASE_URL"))


def chat(messages, tag, max_tokens=8192, temperature=0.0, retries=4):
    last = None
    for a in range(retries):
        try:
            t0 = time.time()
            r = _client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature,
                                                max_completion_tokens=max_tokens, timeout=300)
            txt = r.choices[0].message.content or ""
            u = r.usage
            rec = {"tag": tag, "attempt": a, "ok": True, "sec": round(time.time() - t0, 1),
                   "prompt_tokens": getattr(u, "prompt_tokens", None),
                   "completion_tokens": getattr(u, "completion_tokens", None),
                   "finish": r.choices[0].finish_reason}
            with _lock:
                open(LEDGER, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
            return txt
        except Exception as e:
            last = str(e)[:200]
            with _lock:
                open(LEDGER, "a", encoding="utf-8").write(json.dumps({"tag": tag, "attempt": a, "ok": False, "err": last}) + "\n")
            time.sleep(3 * (a + 1))
    raise RuntimeError(last)
