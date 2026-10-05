# -*- coding: utf-8 -*-
"""arXiv HTML (LaTeXML) fetcher with an on-disk cache: cache/arxiv_html/<arxiv_id>.html.

arxiv.org asks crawlers for one request every 15 s (robots.txt Crawl-delay); one process-wide pacer enforces it.
Papers without an HTML rendering (no LaTeX source, or conversion failed) get a negative marker file
cache/arxiv_html/<id>.none so they are not retried."""
from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..core import paths

UA = "CompileScholar-research (arXiv HTML for citation contexts; 15 s pacing)"
DELAY = 15.0
_lock = threading.Lock()
_last = [0.0]


def cache_dir() -> Path:
    return paths.cache() / "arxiv_html"


def cached(arxiv_id: str) -> Path | None:
    p = cache_dir() / f"{arxiv_id}.html"
    return p if p.exists() else None


def fetch(arxiv_id: str) -> Path | None:
    """Cached page path, or None when arXiv has no HTML rendering for this paper."""
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    p, none = d / f"{arxiv_id}.html", d / f"{arxiv_id}.none"
    if p.exists():
        return p
    if none.exists():
        return None
    with _lock:
        wait = DELAY - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(f"https://arxiv.org/html/{arxiv_id}", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                none.write_text("404")
                return None
            raise
    if b"ltx_bibitem" not in body and b"ltx_cite" not in body:
        none.write_text("no latexml citations")
        return None
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(body)
    tmp.replace(p)
    return p
