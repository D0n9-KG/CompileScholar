# -*- coding: utf-8 -*-
"""ScholarCatalyst protocol: read-only data access + the deterministic submission rules.

Mirrors the official semantics byte for byte where they are structural (third_party/scholarcatalyst/
src/evaluation/retrieve.py and agentic/ranking.py):

  - temporal filter (official `retrieve.temporal_filter`): MONTH-granular. A doc whose (year, month) is strictly
    later than the query's source-paper month is dropped; same month passes; a doc with an unparseable date
    passes; a query with an unparseable date disables the filter. `published` empty falls back to the arXiv id
    (YYMM.NNNNN).
  - source-paper withholding (official `drop_paper` + `utils.source_alias_map`): the query's own paper never
    appears in a submitted ranking. We withhold by registry identity: every SC corpus key whose paper_id equals
    the source paper's paper_id (measured 1:1 on this corpus — no key shares a paper_id with another).
  - assembly (official `agentic/ranking.assemble`): agent ids first, then up to TRAJECTORY_CAP=25 ids the agent
    saw but did not rank (discovery order), then the bm25 backfill of the original question; dedup, drop ids
    outside the corpus and the source paper, temporal filter, cut at DEPTH=100. Each item records its segment.
  - runs jsonl line (official `agentic/runner.py`): {"query_id", "ranking": [{"doc_id", "source"}], "seen",
    "segments", "summary"} — scored unchanged by the official evaluate.py.

Our own as-of discipline is stricter and plugs into the same rule: the tools are called with
as_of = END of the source paper's publication month, so a day-dated doc is visible to our index exactly when the
official month filter would pass it (docs published later in the same month are the intended inclusion); a
coarser-precision date is only visible at its hi, i.e. never earlier than the official rule would allow.

Read-only surfaces: data/external/scholarcatalyst (corpus/queries), data/library/registry.sqlite (the
doc_id <-> paper_id map from library.import_benchmarks). Rels are NEVER read here — scoring belongs to the
official evaluate.py.
"""
from __future__ import annotations

import calendar
import datetime as _dt
import json
import re
import sqlite3
import threading
from pathlib import Path

from ...core import paths

DEPTH = 100                 # official agentic/ranking.DEPTH
TRAJECTORY_CAP = 25         # official agentic/ranking.TRAJECTORY_CAP

QUERY_TYPES = ("core_query", "subfield_query")


# ---------------------------------------------------------------- queries (structural fields + question text)

def load_queries(bench_dir: Path, query_type: str) -> list[dict]:
    """The queries of one type, in file order. Fields: id, paper_id, question, type, paper_published,
    paper_domain. Rels are deliberately not joined — the runner must stay question-blind everywhere except the
    agent prompt itself, and scoring is the official evaluate.py's job."""
    out = []
    with open(Path(bench_dir) / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("type") == query_type:
                out.append(d)
    return out


def query_ids_of_type(bench_dir: Path, query_type: str) -> set[str]:
    return {q["id"] for q in load_queries(bench_dir, query_type)}


# ---------------------------------------------------------------- corpus (lazy: offsets + dates/titles in RAM)

class Corpus:
    """The closed pool. One scan keeps id -> byte offset plus the small fields (title, published) in memory;
    abstracts are read on demand (277 MB jsonl must not materialise as python dicts). Thread-safe reads."""

    def __init__(self, bench_dir: Path):
        self.path = Path(bench_dir) / "corpus.jsonl"
        self._off: dict[str, int] = {}
        self.titles: dict[str, str] = {}
        self.dates: dict[str, str] = {}
        off = 0
        with open(self.path, "rb") as f:
            for raw in f:
                d = json.loads(raw)
                i = d["id"]
                self._off[i] = off
                self.titles[i] = d.get("title") or ""
                self.dates[i] = d.get("published") or ""
                off += len(raw)
        self._fh = open(self.path, "rb")
        self._lock = threading.Lock()

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._off

    def __len__(self) -> int:
        return len(self._off)

    def ids(self):
        return self._off.keys()

    def record(self, doc_id: str) -> dict:
        with self._lock:
            self._fh.seek(self._off[doc_id])
            return json.loads(self._fh.readline())

    def snippet(self, doc_id: str, n: int = 300) -> str:
        text = self.record(doc_id).get("text") or ""
        text = re.sub(r"\s+", " ", text).strip()
        return text[:n]

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------- doc_id <-> paper_id (registry members)

class IdMap:
    """SC corpus key ("arxiv_2507.06542" | "oa_W.." | "s2_..") <-> registry paper_id ("arxiv:.." | "title:.."),
    from the benchmark import's membership rows (read-only)."""

    def __init__(self, registry: Path | None = None):
        con = sqlite3.connect(f"file:{registry or paths.library() / 'registry.sqlite'}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT key, paper_id FROM members WHERE benchmark='scholarcatalyst'").fetchall()
        finally:
            con.close()
        self.doc_to_paper: dict[str, str] = dict(rows)
        self.paper_to_docs: dict[str, list[str]] = {}
        for k, p in rows:
            self.paper_to_docs.setdefault(p, []).append(k)
        for v in self.paper_to_docs.values():
            v.sort()          # arxiv_ < oa_ < s2_ — the arXiv key (the dominant positive form) comes first

    def docs_of(self, paper_id: str) -> list[str]:
        return self.paper_to_docs.get(paper_id, [])

    def paper_of(self, doc_id: str) -> str | None:
        return self.doc_to_paper.get(doc_id)


# ---------------------------------------------------------------- time rules

def parse_ym(date_str: str, doc_id: str = "") -> tuple[int, int] | None:
    """Official retrieve.parse_ym: (year, month) from ISO-ish strings, year-only -> (y, 0), arXiv-id fallback."""
    if not date_str:
        m = re.match(r"^arxiv_(\d{2})(\d{2})\.\d+", doc_id)
        if m:
            yy, mm = int(m.group(1)), int(m.group(2))
            return (2000 + yy if yy <= 30 else 1900 + yy), mm
        return None
    m = re.match(r"^(\d{4})-(\d{2})", date_str)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{4})$", date_str)
    if m:
        return int(m.group(1)), 0
    return None


def temporal_filter(ranking: list[dict], query_date: str, corpus_dates: dict[str, str]) -> list[dict]:
    """Official retrieve.temporal_filter: drop docs published in a strictly later month; undated pass through."""
    q_ym = parse_ym(query_date)
    if q_ym is None:
        return ranking
    q_year, q_month = q_ym
    out = []
    for item in ranking:
        d_ym = parse_ym(corpus_dates.get(item["doc_id"], ""), item["doc_id"])
        if d_ym is None:
            out.append(item)
            continue
        d_year, d_month = d_ym
        if d_year < q_year:
            out.append(item)
        elif d_year == q_year and (d_month == 0 or q_month == 0 or d_month <= q_month):
            out.append(item)
    return out


def as_of_for(paper_published: str) -> str:
    """The tool cutoff for one query: the LAST day of the source paper's publication month (inclusive ISO day).
    A day-dated doc passes our index filter exactly when the official month filter passes it."""
    y, m = parse_ym(paper_published)
    if y is None or m in (0, None):
        raise ValueError(f"unparseable paper_published {paper_published!r}")
    return _dt.date(y, m, calendar.monthrange(y, m)[1]).isoformat()


def withhold_ids(query: dict, idmap: IdMap) -> set[str]:
    """Every corpus key of the query's own paper (registry identity; the official rule additionally drops
    title aliases — evaluate.py applies that at scoring time, so a residual alias can never score)."""
    src = query.get("paper_id") or ""
    ids = {src} if src else set()
    pid = idmap.paper_of(src)
    if pid:
        ids.update(idmap.docs_of(pid))
    return ids


# ---------------------------------------------------------------- assembly (official agentic/ranking.assemble)

def assemble(agent_ids: list[str], seen_ids: list[str], backfill_ids: list[str], *,
             known_ids, withheld: set[str], query_date: str, corpus_dates: dict[str, str],
             depth: int = DEPTH) -> list[dict]:
    out, used = [], set(withheld)
    for source, ids, cap in (("agent", agent_ids, None), ("trajectory", seen_ids, TRAJECTORY_CAP),
                             ("backfill", backfill_ids, None)):
        taken = 0
        for doc_id in ids:
            if doc_id in used or doc_id not in known_ids or (cap is not None and taken >= cap):
                continue
            used.add(doc_id)
            taken += 1
            out.append({"doc_id": doc_id, "source": source})
    return temporal_filter(out, query_date, corpus_dates)[:depth]


# ---------------------------------------------------------------- backfill source (the official bm25 run)

def load_backfill(bench_dir: Path, query_type: str) -> dict[str, list[str]]:
    """{query_id: [doc_id]} from the official bm25 run (runs/sparse/bm25/<qt>.jsonl — already date-filtered
    and source-dropped by the official retrieve.py). The paper's agentic rows use Gemini-Emb-2 for the backfill
    (their R@100 carries a dagger); the repo default and our reproduction use bm25 — noted in the report."""
    path = Path(bench_dir) / "runs" / "sparse" / "bm25" / f"{query_type}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"bm25 run missing at {path} — run the official BM25 arm first: "
            f"python third_party/scholarcatalyst/src/evaluation/retrieve.py --model bm25 --bench-dir {bench_dir}")
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["query_id"]] = [r["doc_id"] for r in d["ranking"]]
    return out


# ---------------------------------------------------------------- run file (append + resume)

def run_path(bench_dir: Path, pipeline: str, query_type: str) -> Path:
    return Path(bench_dir) / "runs" / "agentic" / pipeline / f"{query_type}.jsonl"


def failed_path(bench_dir: Path, pipeline: str, query_type: str) -> Path:
    return Path(bench_dir) / "runs" / "agentic" / pipeline / f"{query_type}.failed.jsonl"


def done_query_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.add(json.loads(line)["query_id"])
    return out
