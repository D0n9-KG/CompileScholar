# -*- coding: utf-8 -*-
"""acquire(paper_ids): for each paper and wanted version, try its channels in order, check identity, record.

Versions (INTEGRATED-SYSTEM-1005 §2.3): an arXiv paper gets v1 and, when it has later versions, its latest version;
intermediate versions are skipped. A DOI-only paper gets its version of record (version 0).
Writes (one writer thread; workers only fetch and verify):
  assets    one row per (paper, version, channel) that passed the identity check — a pointer for NAS / Sci-Hub, a file
            under data/library/pdf/<sha[:2]>/<sha>.pdf (written as .part, then renamed) for downloads
  attempts  every try, including missing / mismatch / error; a mismatched PDF is kept as
            data/library/quarantine/<sha>.pdf for review, never deleted
A (paper, version) that already has an asset is skipped. exclude: channels not to use in this run (CS main experiments
run with exclude={"scihub_local"})."""
from __future__ import annotations

import json
import queue
import threading
import time

from ..core import paths
from ..dfc.store import parallel
from ..library import identity, store
from ..sources import http
from . import channels as CH
from . import verify as V

ARXIV_ORDER = ("arxiv_nas", "arxiv_gcs")
DOI_ORDER = ("oa_pdf", "scihub_local")


def pdf_path(sha: str):
    return paths.library() / "pdf" / sha[:2] / f"{sha}.pdf"


def quarantine_path(sha: str):
    return paths.library() / "quarantine" / f"{sha}.pdf"


def _write(p, data: bytes) -> None:
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(p)


def plan(con, pid: str) -> list[tuple[CH.Want, tuple]]:
    """[(want, channel order)] for one paper."""
    pid = identity.canonical(con, pid)
    ax = con.execute("SELECT value FROM identifiers WHERE paper_id=? AND scheme='arxiv'", (pid,)).fetchone()
    doi = con.execute("SELECT value FROM identifiers WHERE paper_id=? AND scheme='doi' ORDER BY role != 'self' LIMIT 1",
                      (pid,)).fetchone()
    doi = doi[0] if doi else None
    if ax:
        vs = [v for (v,) in con.execute("SELECT version FROM dates WHERE paper_id=? AND kind='arxiv_v' ORDER BY version",
                                        (pid,))]
        want = sorted({1, max(vs)}) if vs else [1]
        return [(CH.Want(pid, ax[0], v, doi), ARXIV_ORDER) for v in want]
    if doi:
        return [(CH.Want(pid, None, 0, doi), DOI_ORDER)]
    return []


def _record(con, pid: str):
    r = con.execute("SELECT title FROM records WHERE paper_id=? ORDER BY source != 'arxiv' LIMIT 1", (pid,)).fetchone()
    names = con.execute("SELECT names FROM authors WHERE paper_id=? ORDER BY source != 'arxiv' LIMIT 1",
                        (pid,)).fetchone()
    sur = [a.get("surname") or "" for a in json.loads(names[0])] if names else []
    idents = [f"{s}:{v}" for s, v in con.execute("SELECT scheme, value FROM identifiers WHERE paper_id=? AND "
                                                  "scheme IN ('doi','arxiv')", (pid,))]
    return (r[0] if r else ""), sur, idents


def acquire(paper_ids, workers: int = 16, exclude=(), llm: bool = True, log=print, path=None) -> dict:
    con = store.connect(path, threads=True)
    jobs = []
    for pid in paper_ids:
        for w, order in plan(con, pid):
            have = con.execute("SELECT 1 FROM assets WHERE paper_id=? AND version=?", (w.paper_id, w.version)).fetchone()
            if not have:
                title, sur, idents = _record(con, w.paper_id)
                jobs.append((w, tuple(c for c in order if c not in exclude), title, sur, idents))
    log(f"[acquire] {len(jobs):,} (paper, version) to fetch")
    out_q: queue.Queue = queue.Queue()

    def one(job):
        w, order, title, sur, idents = job
        for ch in order:
            try:
                f = CH.CHANNELS[ch](w)
            except (http.Transient, OSError, ValueError) as e:
                out_q.put(("attempt", w, ch, "error", f"{type(e).__name__}: {e}"[:300], None))
                continue
            if f is None:
                out_q.put(("attempt", w, ch, "missing", "", None))
                continue
            sha = CH.sha256(f.data)
            res = V.check(f.data, title, sur, w.arxiv, w.doi)
            if res["verdict"] == "unsure" and llm:
                res = V.llm_verdict(res, title, sur, idents)
            if res["verdict"] in ("ok", "ok_llm"):
                if ch not in ("arxiv_nas", "scihub_local"):
                    _write(pdf_path(sha), f.data)
                    f.pointer = {**f.pointer, "file": str(pdf_path(sha).relative_to(paths.library()))}
                out_q.put(("asset", w, ch, res, f, sha))
                return
            _write(quarantine_path(sha), f.data)
            out_q.put(("attempt", w, ch, "mismatch", res["why"], sha))
        out_q.put(("done", w, None, None, None, None))

    n = {"assets": 0, "missing": 0, "mismatch": 0, "error": 0, "none": 0}
    stop = threading.Event()

    def writer():
        while not (stop.is_set() and out_q.empty()):
            try:
                kind, w, ch, a, b, sha = out_q.get(timeout=0.5)
            except queue.Empty:
                continue
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            if kind == "asset":
                res, f = a, b
                con.execute("INSERT OR REPLACE INTO assets VALUES (?,?,?,?,?,?,?,?,?)",
                            (w.paper_id, w.version, ch, json.dumps(f.pointer), sha, len(f.data), res["verdict"],
                             res["why"], now))
                con.execute("INSERT INTO attempts(paper_id, version, channel, status, detail, sha256, at) "
                            "VALUES (?,?,?,?,?,?,?)", (w.paper_id, w.version, ch, "ok", res["why"], sha, now))
                n["assets"] += 1
            elif kind == "attempt":
                con.execute("INSERT INTO attempts(paper_id, version, channel, status, detail, sha256, at) "
                            "VALUES (?,?,?,?,?,?,?)", (w.paper_id, w.version, ch, a, b, sha, now))
                n[a] += 1
            else:
                n["none"] += 1
            if sum(n.values()) % 500 == 0:
                con.commit()
        con.commit()
    wt = threading.Thread(target=writer)
    wt.start()
    try:
        parallel(one, jobs, workers, log=log, every=1000, label="acquire")
    finally:
        stop.set()
        wt.join()
        con.close()
    n["papers_without_text"] = n.pop("none")
    log(f"[acquire] {n}")
    return n
