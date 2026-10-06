# -*- coding: utf-8 -*-
"""The candidate corpora of the fixed-library benchmarks -> the registry (INTEGRATED-SYSTEM-1005 v2.1 item 1: the
library of a benchmark is its candidate corpus). Only membership and metadata are imported; queries, roles and labels
stay in the release files and are read by the benchmark adapters.

  scholarcatalyst  data/external/scholarcatalyst/corpus.jsonl  ids arxiv_<id> | oa_W<n> | s2_<sha>; `published` (day)
  prescience       data/external/prescience/test.parquet       S2 corpus_id + arxiv_id per row; `date` (day)
  masterset        data/external/masterset/.../train.parquet   internal UUID, title, year, venue, authors (no arXiv /
                                                               DOI): a paper of its own unless the same identifier
                                                               set already exists; title matches go to the queue
  ideaforecast     data/external/ideaforecast/<month>.parquet  arxiv_id (full texts stay there, latest version)
A benchmark's date is a secondary date (kind=published / venue_year): it never overrides an arXiv version date."""
from __future__ import annotations

import ast
import json
import time
from pathlib import Path

from ..core import ids, paths
from ..core.asof import Date
from . import identity, store

MASTERSET = Path("data/train_eval_set/v1.0/train.parquet")


def _names(raw) -> list[dict]:
    out = []
    for n in raw or []:
        n = (n or "").strip()
        if n:
            out.append({"name": n, "surname": n.split()[-1].strip(".,")})
    return out


def _list(v) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        try:
            return list(ast.literal_eval(v))
        except (ValueError, SyntaxError):
            return []
    return []


def scholarcatalyst():
    for line in open(paths.external("scholarcatalyst") / "corpus.jsonl", encoding="utf-8"):
        r = json.loads(line)
        key = r["id"]
        kind, _, val = key.partition("_")
        if kind == "arxiv" and "_" in val:                     # old-style id with "/" written as "_": cs_0408007
            val = val.replace("_", "/", 1)
        own = {"arxiv": ("arxiv", ids.normalize_arxiv(val)), "oa": ("openalex", val), "s2": ("s2", val)}.get(kind)
        if not own or not own[1]:
            continue
        cats = _list(r.get("categories"))
        yield identity.Incoming(source="scholarcatalyst", ids=[(*own, "self")], title=r.get("title") or "",
                                abstract=r.get("text") or "", categories=cats if kind == "arxiv" else [],
                                dates=[("published", 0, Date.parse(r.get("published")))],
                                member=("scholarcatalyst", key))


def prescience():
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(paths.external("prescience") / "test.parquet")
    for b in pf.iter_batches(batch_size=20000, columns=["corpus_id", "arxiv_id", "date", "title", "abstract",
                                                         "categories"]):
        for r in b.to_pylist():
            ax = ids.normalize_arxiv(r["arxiv_id"])
            own = [("s2_corpus", str(r["corpus_id"]), "self")]
            if ax:
                own.append(("arxiv", ax, "self"))
            yield identity.Incoming(source="prescience", ids=own, title=r.get("title") or "",
                                    abstract=r.get("abstract") or "", categories=_list(r.get("categories")),
                                    dates=[("published", 0, Date.parse(r.get("date")))],
                                    member=("prescience", str(r["corpus_id"])))


def masterset():
    import pyarrow.parquet as pq
    t = pq.read_table(paths.external("masterset") / MASTERSET,
                      columns=["paper_id", "title", "abstract", "year", "venue", "authors"])
    for r in t.to_pylist():
        try:
            authors = _names(json.loads(r.get("authors") or "[]"))
        except ValueError:
            authors = []
        y = r.get("year")
        yield identity.Incoming(source="masterset", ids=[("masterset", r["paper_id"], "self")],
                                title=r.get("title") or "", abstract=r.get("abstract") or "",
                                venue=f"{r.get('venue') or ''} {y or ''}".strip(), authors=authors,
                                dates=[("venue_year", 0, Date.parse(int(y)) if y else None)],
                                member=("masterset", r["paper_id"]))


def ideaforecast():
    import pyarrow.parquet as pq
    for f in sorted(paths.external("ideaforecast").glob("*.parquet")):
        for aid, month, title in zip(*(pq.read_table(f, columns=["arxiv_id", "month", "title"]).column(c).to_pylist()
                                       for c in ("arxiv_id", "month", "title"))):
            ax = ids.normalize_arxiv(aid)
            if ax:
                yield identity.Incoming(source="ideaforecast", ids=[("arxiv", ax, "self")], title=title or "",
                                        dates=[("published", 0, Date.parse(month))], member=("ideaforecast", ax))


BENCHMARKS = {"scholarcatalyst": scholarcatalyst, "prescience": prescience, "masterset": masterset,
              "ideaforecast": ideaforecast}


def run(names=tuple(BENCHMARKS), log=print, path=None) -> dict:
    out = {}
    with store.lock(path):
        con = store.connect(path)
        for name in names:
            t0 = time.time()
            w = identity.Writer(con, name)
            n = 0
            for inc in BENCHMARKS[name]():
                w.add(inc)
                n += 1
            out[name] = {"rows": n, **w.close(), "seconds": round(time.time() - t0)}
            log(f"[library] {name}: {out[name]}")
            con.execute("INSERT INTO imports(source, inputs, counts, started_at, finished_at) VALUES (?,?,?,?,?)",
                        (name, json.dumps({"dir": str(paths.external(name))}), json.dumps(out[name]),
                         time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)), time.strftime("%Y-%m-%dT%H:%M:%S")))
            con.commit()
        out["undated"] = identity.recompute_first_public(con)
        con.close()
    return out
