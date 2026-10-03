# -*- coding: utf-8 -*-
"""P4 shared helpers: DeepSeek (Paratera) chat + arXiv HTML fetch, with call counters/caps."""
import json
import os
import re
import threading
import time

import httpx

ROOT = r"C:\Users\D0n9\Desktop\CompileScholar"
for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

P4 = os.path.join(ROOT, r".research_tmp\review_1002\p4")
COUNT_F = os.path.join(P4, "calls.json")
HTML_DIR = os.path.join(P4, "html")
os.makedirs(HTML_DIR, exist_ok=True)
_lock = threading.Lock()
LLM_CAP = 600
FETCH_CAP = 150


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


def chat(prompt, system=None, max_tokens=6000, retries=3, thinking=False):
    if calls().get("llm", 0) >= LLM_CAP:
        raise RuntimeError("LLM budget exhausted")
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    body = {"model": MODEL, "messages": msgs, "temperature": 0.0,
            "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
    if not thinking:
        body["thinking"] = {"type": "disabled"}
    url = _BASE + ("/chat/completions" if _BASE.endswith("/v1") else "/v1/chat/completions")
    last = None
    for a in range(retries):
        _bump("llm")
        try:
            with httpx.Client(timeout=240, trust_env=False,
                              headers={"Connection": "close"}) as c:
                r = c.post(url, json=body, headers={"Authorization": f"Bearer {_KEY}"})
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0) if m else txt)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + 3 * a)
    raise RuntimeError(f"chat failed: {last}")


_last_fetch = [0.0]


def fetch_html(aid):
    """arXiv HTML (arxiv.org/html then ar5iv) -> plain text; cached on disk."""
    aid = re.sub(r"v\d+$", "", aid)
    cp = os.path.join(HTML_DIR, aid.replace("/", "_") + ".txt")
    if os.path.exists(cp):
        return open(cp, encoding="utf-8").read()
    txt = ""
    m = re.match(r"(\d{2})(\d{2})\.", aid)
    old = (not m) or (int(m.group(1)) * 100 + int(m.group(2)) < 2312)
    urls = [f"https://arxiv.org/html/{aid}", f"https://ar5iv.labs.arxiv.org/html/{aid}"]
    if old:
        urls.reverse()  # ar5iv covers older papers; arxiv.org/html only ~Dec 2023+
    for url in urls:
        if calls().get("fetch", 0) >= FETCH_CAP:
            break
        wait = 1.1 - (time.time() - _last_fetch[0])
        if wait > 0:
            time.sleep(wait)
        _last_fetch[0] = time.time()
        _bump("fetch")
        try:
            with httpx.Client(timeout=60, follow_redirects=True,
                              headers={"User-Agent": "Mozilla/5.0 (research pilot)"}) as c:
                r = c.get(url)
            if r.status_code != 200 or len(r.text) < 5000:
                continue
            h = r.text
            # ar5iv redirects to abs page when no HTML exists
            if "ltx_document" not in h and "ltx_page_main" not in h:
                continue
            h = re.sub(r"(?is)<(script|style|annotation|annotation-xml)[^>]*>.*?</\1>", " ", h)
            h = re.sub(r"(?is)<math[^>]*alttext=\"([^\"]*)\"[^>]*>.*?</math>", r" \1 ", h)
            h = re.sub(r"(?is)<math[^>]*>.*?</math>", " ", h)
            h = re.sub(r"(?i)<h(\d)[^>]*>", "\n§§ ", h)
            h = re.sub(r"(?i)</(p|div|tr|h\d|li|table|caption|section)>", "\n", h)
            h = re.sub(r"(?i)<t[dh][^>]*>", " | ", h)
            h = re.sub(r"<[^>]+>", " ", h)
            h = re.sub(r"&nbsp;|&#160;", " ", h)
            h = re.sub(r"&amp;", "&", h)
            h = re.sub(r"[ \t]+", " ", h)
            h = re.sub(r"\n\s*\n+", "\n", h)
            txt = h.strip()
            break
        except Exception:  # noqa: BLE001
            continue
    open(cp, "w", encoding="utf-8").write(txt)
    return txt


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(p, 3), round(max(0, c - h), 3), round(min(1, c + h), 3))
