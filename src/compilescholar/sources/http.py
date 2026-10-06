# -*- coding: utf-8 -*-
"""One HTTP layer for every external source (INTEGRATED-SYSTEM-1005 §5 `sources/`; FITNESS-LIBRARY-ACQUIRE: OpenAlex,
S2, Crossref, Sciverse and arXiv had 2-4 clients each, and refgraph cached failures as answers).

  get(source, url, params=None, headers=None, cache=True) -> Response(status, content, headers, from_cache)
  - pacing per source across processes (a file lock and a timestamp under cache/http/pace/): GAP seconds between
    request starts, so parallel builders share one budget per host;
  - 429 / 503: wait Retry-After (seconds or an HTTP date) or back off exponentially, then retry;
  - 5xx, timeouts and connection errors: back off and retry, recorded in the source's circuit (sources.circuit);
    when the circuit is open the call fails fast;
  - definitive answers (2xx, 404, 410) are returned and, with cache=True, stored under cache/http/<source>/; transient
    failures are never cached — after `tries` attempts Transient is raised;
  - the system proxy is not used (trust_env=False: a registry proxy intercepted localhost and api calls, measured).
Response bodies are bytes; .text() and .json() decode."""
from __future__ import annotations

import email.utils
import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from filelock import FileLock

from ..core import paths
from .circuit import SourceCircuit

UA = "CompileScholar-research (scholarly metadata; paced per source)"
GAP = {"arxiv_api": 3.1, "arxiv_oai": 3.0, "arxiv_html": 15.5, "openalex": 0.11, "crossref": 0.25, "s2": 1.1,
       "gcs": 0.0, "openreview": 0.5}
DEFINITIVE = {404, 410}
RETRY = {429, 500, 502, 503, 504}
CIRCUIT = SourceCircuit(failure_threshold=5, window_s=300, cooldown_s=300)
_clients: dict[str, httpx.Client] = {}
_lock = threading.Lock()


class Transient(RuntimeError):
    """The source did not give a definitive answer (retries exhausted or circuit open)."""


@dataclass
class Response:
    status: int
    content: bytes
    headers: dict
    from_cache: bool = False

    def text(self) -> str:
        return self.content.decode("utf-8", "replace")

    def json(self):
        return json.loads(self.content)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


def _client(timeout: float) -> httpx.Client:
    k = f"{timeout}"
    with _lock:
        if k not in _clients:
            _clients[k] = httpx.Client(timeout=httpx.Timeout(timeout, connect=20), follow_redirects=True,
                                       trust_env=False, headers={"User-Agent": UA})
        return _clients[k]


def _pace(source: str) -> None:
    gap = GAP.get(source, 1.0)
    if gap <= 0:
        return
    d = paths.cache() / "http" / "pace"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{source}.json"
    with FileLock(str(p) + ".lock"):
        try:
            last = json.loads(p.read_text())["last"]
        except (OSError, ValueError, KeyError):
            last = 0.0
        wait = gap - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        p.write_text(json.dumps({"last": time.time()}))


def _retry_after(h: str | None) -> float | None:
    if not h:
        return None
    if h.strip().isdigit():
        return float(h)
    try:
        return max(0.0, email.utils.parsedate_to_datetime(h).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def _cache_path(source: str, method: str, url: str, params, body) -> Path:
    key = json.dumps([method, url, sorted((params or {}).items()), body], sort_keys=True, default=str)
    h = hashlib.sha1(key.encode()).hexdigest()
    return paths.cache() / "http" / source / h[:2] / f"{h}.json"


def request(source: str, method: str, url: str, *, params: dict | None = None, headers: dict | None = None,
            json_body=None, cache: bool = True, timeout: float = 60, tries: int = 5) -> Response:
    cp = _cache_path(source, method, url, params, json_body) if cache else None
    if cp is not None and cp.exists():
        rec = json.loads(cp.read_text(encoding="utf-8"))
        return Response(rec["status"], rec["content"].encode("latin-1"), rec["headers"], from_cache=True)
    if not CIRCUIT.allow(source):
        raise Transient(f"{source}: circuit open")
    last_err = ""
    for att in range(tries):
        _pace(source)
        try:
            r = _client(timeout).request(method, url, params=params, headers=headers, json=json_body)
        except httpx.HTTPError as e:
            CIRCUIT.record(source, False)
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(120.0, 2 ** att + random.random()))
            continue
        if r.status_code in RETRY:
            if r.status_code >= 500:
                CIRCUIT.record(source, False)
            last_err = f"http {r.status_code}"
            wait = _retry_after(r.headers.get("Retry-After"))
            time.sleep(min(300.0, wait if wait is not None else 2 ** (att + 1) + random.random()))
            continue
        CIRCUIT.record(source, True)
        resp = Response(r.status_code, r.content, dict(r.headers))
        if cp is not None and (resp.ok or resp.status in DEFINITIVE):
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_suffix(f".tmp{threading.get_ident()}")
            tmp.write_text(json.dumps({"status": resp.status, "content": resp.content.decode("latin-1"),
                                       "headers": {k: v for k, v in resp.headers.items()
                                                   if k.lower() in ("content-type", "retry-after")}}),
                           encoding="utf-8")
            tmp.replace(cp)
        return resp
    raise Transient(f"{source}: {last_err} after {tries} attempts ({url})")


def get(source: str, url: str, **kw) -> Response:
    return request(source, "GET", url, **kw)
