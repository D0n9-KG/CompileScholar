# -*- coding: utf-8 -*-
"""Parse library assets into parse products (library.parses), the two tiers of INTEGRATED-SYSTEM-1005 v2.2:

  fast     GROBID 0.9.1-full on localhost (configs/local.yaml paths.grobid), TEI with sentence segmentation and
           coordinates for sentences, references and bibliography entries; plus the PDF's own hyperref citation links
           (PyMuPDF) — every acquired PDF. Measured on 61 cs v1 papers against arXiv v1 HTML: citation-sentence recall
           0.934, precision 0.919, bibliography entries 0.994 (experiments/documents/fasttier_compare.py).
  careful  MinerU on the task router (paths.mineru_file_parse; POST /tasks, poll, GET result zip with markdown and
           content_list), throttled to `slots` in flight (6 measured safe; the service is shared) and backing off when
           the router queue is long — the deep-extraction subset only.
Products are stored by PDF content hash under data/library/parsed/<parser>/<sha[:2]>/<sha>.<ext>: fast = .tei.xml.gz
plus .links.json; careful = .mineru.zip with base64 images removed (only markdown and content_list kept). A parse
that fails is retried on the next run, up to MAX_ATTEMPTS."""
from __future__ import annotations

import gzip
import io
import json
import threading
import time
import zipfile

import httpx

from ..acquire.run import read_asset
from ..core import paths
from ..dfc.store import parallel
from ..library import store

MAX_ATTEMPTS = 3
GROBID_VERSION = "0.9.1-full"
MINERU_FORM = {"backend": "vlm-vllm-async-engine", "parse_method": "auto", "lang_list": "en", "formula_enable": "true",
               "table_enable": "true", "return_md": "true", "return_content_list": "true",
               "return_middle_json": "false", "return_model_output": "false", "return_images": "false",
               "response_format_zip": "true"}


def product_path(parser: str, sha: str, ext: str):
    return paths.library() / "parsed" / parser / sha[:2] / f"{sha}.{ext}"


def _url(key: str) -> str:
    return str(paths._local_paths()[key]).rstrip("/")


def _urls(key: str) -> list[str]:
    """A config value may hold several comma-separated endpoints (e.g. the local GROBID plus one on the P40
    box over an SSH tunnel); items are spread over them deterministically by content hash."""
    return [u.strip().rstrip("/") for u in str(paths._local_paths()[key]).split(",") if u.strip()]


def _atomic(p, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(p)


# ---------------------------------------------------------------- fast tier

def grobid(data: bytes, url: str, timeout: float = 600) -> bytes:
    for att in range(8):
        r = httpx.post(f"{url}/api/processFulltextDocument", files={"input": ("paper.pdf", data, "application/pdf")},
                       data={"segmentSentences": "1", "includeRawCitations": "1", "consolidateHeader": "0",
                             "consolidateCitations": "0", "consolidateFunders": "0",
                             "teiCoordinates": ["s", "ref", "biblStruct", "head", "figure", "formula"]},
                       timeout=timeout, trust_env=False)
        if r.status_code == 503:                       # GROBID's pool is full
            time.sleep(1 + att)
            continue
        r.raise_for_status()
        return r.content
    raise RuntimeError("grobid busy")


def cite_links(data: bytes) -> list:
    """hyperref citation links: [page, [x0, y0, x1, y1], dest_page, dest_x, dest_y_top] for cite.* destinations."""
    import pymupdf
    out = []
    with pymupdf.open(stream=data, filetype="pdf") as d:
        names = d.resolve_names()
        for pg in d:
            for l in pg.get_links():
                nm = l.get("nameddest") or l.get("name") or ""
                dest = names.get(nm) if str(nm).startswith("cite.") else None
                if not dest or dest.get("page") is None or dest["page"] < 0 or not dest.get("to"):
                    continue
                tp = dest["page"]
                x, y = dest["to"]
                out.append([pg.number, list(l["from"]), tp, x, d[tp].rect.height - y])
    return out


def _fast(sha: str, data: bytes, url: str) -> str:
    tei = grobid(data, url)
    _atomic(product_path("grobid", sha, "links.json"), json.dumps(cite_links(data)).encode())
    p = product_path("grobid", sha, "tei.xml.gz")
    _atomic(p, gzip.compress(tei, 6))
    return str(p.relative_to(paths.library()))


# ---------------------------------------------------------------- careful tier

def _strip_zip(z: bytes) -> bytes:
    """Keep markdown and content_list(s); drop images and anything else; base64 images inside markdown removed."""
    import re
    src = zipfile.ZipFile(io.BytesIO(z))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
        for n in src.namelist():
            if n.endswith(".md"):
                md = re.sub(r"!\[[^\]]*\]\(data:image/[^)]*\)", "", src.read(n).decode("utf-8"))
                out.writestr(n.split("/")[-1], md)
            elif n.endswith("_content_list.json") or n.endswith("_content_list_v2.json"):
                out.writestr(n.split("/")[-1], src.read(n))
    return buf.getvalue()


def mineru(data: bytes, base: str, client: httpx.Client, max_queue: int = 24) -> bytes:
    for att in range(3):
        backoff = 2.0
        while True:
            try:
                h = client.get(f"{base}/health").json()
                if h.get("queued_tasks", 0) > max_queue:   # shared service: wait while others' queue is long
                    time.sleep(30)
                    continue
            except (httpx.HTTPError, ValueError):
                time.sleep(30)
                continue
            r = client.post(f"{base}/tasks", files={"files": ("paper.pdf", data, "application/pdf")}, data=MINERU_FORM)
            if r.status_code in (409, 429, 502, 503):
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)
                continue
            r.raise_for_status()
            break
        tid = r.json()["task_id"]
        while True:
            st = client.get(f"{base}/tasks/{tid}").json()
            if st["status"] in ("completed", "failed"):
                break
            time.sleep(3)
        if st["status"] == "completed":
            z = client.get(f"{base}/tasks/{tid}/result")
            z.raise_for_status()
            return _strip_zip(z.content)
    raise RuntimeError(f"mineru task failed 3 times: {st.get('error')}")


# ---------------------------------------------------------------- runner

def run(tier: str, paper_ids=None, workers: int | None = None, log=print, path=None) -> dict:
    """Parse every asset (of `paper_ids`, or all) that has no ok parse of this tier yet."""
    parser = {"fast": "grobid", "careful": "mineru"}[tier]
    con = store.connect(path, threads=True)
    rows = con.execute("SELECT paper_id, sha256, pointer FROM assets").fetchall()
    if paper_ids is not None:
        want = set(paper_ids)
        rows = [r for r in rows if r[0] in want]
    skip = {s for s, st, n in con.execute("SELECT sha256, status, attempts FROM parses WHERE parser=?", (parser,))
            if st == "ok" or n >= MAX_ATTEMPTS}
    todo = [(s, json.loads(p)) for s, p in {sha: ptr for _, sha, ptr in rows}.items() if s not in skip]
    log(f"[parse:{tier}] {len(todo):,} PDFs to parse")
    lock, n = threading.Lock(), {"ok": 0, "failed": 0}
    urls = _urls("grobid") if tier == "fast" else [_url("mineru_file_parse")]
    client = httpx.Client(timeout=httpx.Timeout(60, read=900), trust_env=False) if tier == "careful" else None

    def one(item):
        sha, ptr = item
        try:
            data = read_asset(ptr)
            rel = (_fast(sha, data, urls[int(sha[:8], 16) % len(urls)]) if tier == "fast"
                   else _careful(sha, data, urls[0], client))
            status, detail = "ok", ""
        except Exception as e:
            rel, status, detail = None, "failed", f"{type(e).__name__}: {e}"[:400]
        with lock:
            con.execute("INSERT INTO parses VALUES (?,?,?,?,?,?,?,1,?) ON CONFLICT(sha256, parser) DO UPDATE SET "
                        "status=excluded.status, path=excluded.path, detail=excluded.detail, "
                        "attempts=parses.attempts+1, created_at=excluded.created_at",
                        (sha, parser, GROBID_VERSION if tier == "fast" else "router", tier, status, rel, detail,
                         time.strftime("%Y-%m-%dT%H:%M:%S")))
            con.commit()
            n[status] += 1
    parallel(one, todo, workers or (14 if tier == "fast" else 6), log=log, every=500, label=f"parse:{tier}")
    con.close()
    log(f"[parse:{tier}] {n}")
    return n


def _careful(sha: str, data: bytes, base: str, client) -> str:
    p = product_path("mineru", sha, "mineru.zip")
    _atomic(p, mineru(data, base, client))
    return str(p.relative_to(paths.library()))


def load_fast(sha: str) -> tuple[bytes, list]:
    """(TEI bytes, hyperref links) of a fast-tier parse."""
    tei = gzip.decompress(product_path("grobid", sha, "tei.xml.gz").read_bytes())
    lp = product_path("grobid", sha, "links.json")
    return tei, (json.loads(lp.read_text()) if lp.exists() else [])
