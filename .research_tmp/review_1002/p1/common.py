# -*- coding: utf-8 -*-
"""P1 shared helpers: DeepSeek (Paratera) chat + Sciverse search, with call counters."""
import json
import os
import re
import sys
import threading
import time

import httpx

ROOT = r"C:\Users\D0n9\Desktop\CompileScholar"
for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

for p in (os.path.join(ROOT, "src"),
          os.path.join(ROOT, r".research_tmp\experiments\benchmarks\_shared\tools"),
          r"C:\Users\D0n9\Desktop\sci-evo-extract\src"):
    if p not in sys.path:
        sys.path.insert(0, p)

P1 = os.path.join(ROOT, r".research_tmp\review_1002\p1")
COUNT_F = os.path.join(P1, "calls.json")
_lock = threading.Lock()


def _bump(key, n=1):
    with _lock:
        d = json.load(open(COUNT_F)) if os.path.exists(COUNT_F) else {}
        d[key] = d.get(key, 0) + n
        json.dump(d, open(COUNT_F, "w"))
        return d[key]


def calls():
    return json.load(open(COUNT_F)) if os.path.exists(COUNT_F) else {}


_BASE = os.environ["PARATERA_BASE_URL"].rstrip("/")
_KEY = os.environ["PARATERA_API_KEY"]
MODEL = "DeepSeek-V4.1-Flash"
LLM_CAP = 800
SEARCH_CAP = 500


def chat(prompt, system=None, temperature=0.0, max_tokens=2000, json_mode=True,
         retries=3, thinking=True):
    if calls().get("llm", 0) >= LLM_CAP:
        raise RuntimeError("LLM budget exhausted")
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    body = {"model": MODEL, "messages": msgs, "temperature": temperature,
            "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if not thinking:
        body["thinking"] = {"type": "disabled"}
    url = _BASE + ("/chat/completions" if _BASE.endswith("/v1") else "/v1/chat/completions")
    last = None
    for a in range(retries):
        _bump("llm")
        try:
            with httpx.Client(timeout=180, trust_env=False,
                              headers={"Connection": "close"}) as c:
                r = c.post(url, json=body,
                           headers={"Authorization": f"Bearer {_KEY}"})
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            if not json_mode:
                return txt
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0) if m else txt)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + 3 * a)
    raise RuntimeError(f"chat failed: {last}")


_svc = None
_svc_lock = threading.Lock()


def search(query, k=10):
    """Sciverse semantic (tiered engine) -> list of {title, year, abstract, doi}."""
    global _svc
    if calls().get("search", 0) >= SEARCH_CAP:
        raise RuntimeError("search budget exhausted")
    with _svc_lock:
        if _svc is None:
            os.environ["SCIVERSE_API_TOKEN"] = os.environ.get("SCIVERSE_API_TOKEN", "")
            from sci_evo_extract.library.search_service import SearchService
            _svc = SearchService(limit=k)
    _bump("search")
    for a in range(3):
        try:
            from external_tools import _cand_row
            r = _svc.search(query, mode="agent", limit=k)
            out = []
            for c in r.candidates[:k]:
                row = _cand_row(c)
                out.append({"title": row.get("title"), "year": row.get("year"),
                            "abstract": (row.get("abstract") or "")[:1200],
                            "doi": row.get("doi")})
            return out
        except Exception as e:  # noqa: BLE001
            time.sleep(3 + 3 * a)
            last = e
    return [{"error": str(last)}]
