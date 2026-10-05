# -*- coding: utf-8 -*-
"""L0: the papers table (SQLite) — every arXiv paper we know, with first-version date, plus a title index used to
resolve bibliography entries to arXiv ids.

Sources, union by arxiv_id (later sources fill missing fields, never overwrite a v1_date):
  1. the local OAI snapshot (arxiv-metadata-oai-snapshot.json, ends at 2404.03658): versions[0].created = v1 date;
  2. the OAI-PMH harvest (sources.arxiv_oai, arXivRaw, 2024-03 onwards).
Scope: all papers whose categories include one of SCOPE_CATS (cross-listings count), any date.

Schema
  papers(arxiv_id TEXT PRIMARY KEY, v1_date TEXT, title TEXT, norm_title TEXT, abstract TEXT,
         authors TEXT  -- JSON list of surnames, first author first
         , first_surname TEXT, categories TEXT -- space-separated, primary first
         , primary_cat TEXT, doi TEXT, source TEXT)
  title_prefix(prefix40 TEXT, arxiv_id TEXT)   -- normalized-title 40-char prefix, for resolution
"""
from __future__ import annotations

import email.utils
import json
import re
import sqlite3
from pathlib import Path

from ..core import paths
from ..sources import arxiv_oai
from ..sources.arxiv_snapshot import norm, snapshot_path

SCOPE_CATS = ("cs.", "stat.ML", "eess.AS", "eess.IV", "eess.SP")


def db_path() -> Path:
    return paths.data() / "corpus" / "papers.sqlite"


def in_scope(categories: list[str]) -> bool:
    return any(c.startswith(SCOPE_CATS[0]) or c in SCOPE_CATS[1:] for c in categories)


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("""CREATE TABLE IF NOT EXISTS papers(arxiv_id TEXT PRIMARY KEY, v1_date TEXT, title TEXT,
                   norm_title TEXT, abstract TEXT, authors TEXT, first_surname TEXT, categories TEXT,
                   primary_cat TEXT, doi TEXT, source TEXT)""")
    con.execute("CREATE TABLE IF NOT EXISTS title_prefix(prefix40 TEXT, arxiv_id TEXT)")
    return con


def _snapshot_row(rec: dict) -> tuple | None:
    cats = (rec.get("categories") or "").split()
    if not in_scope(cats):
        return None
    v = (rec.get("versions") or [{}])[0].get("created") or ""
    try:
        v1 = email.utils.parsedate_to_datetime(v).date().isoformat()
    except (TypeError, ValueError):
        v1 = ""
    ap = rec.get("authors_parsed") or []
    surn = [a[0] for a in ap if a and a[0]]
    title = re.sub(r"\s+", " ", rec.get("title") or "").strip()
    doi = rec.get("doi")
    return (rec["id"], v1, title, norm(title), re.sub(r"\s+", " ", rec.get("abstract") or "").strip(),
            json.dumps(surn, ensure_ascii=False), norm(surn[0]).split(" ")[-1] if surn else "",
            " ".join(cats), cats[0] if cats else "", doi if doi and doi != "None" else None, "snapshot")


def _oai_row(r: dict) -> tuple | None:
    cats = r.get("categories") or []
    if not in_scope(cats):
        return None
    title = r.get("title") or ""
    surn = r.get("authors") or []
    return (r["arxiv_id"], r.get("v1_date") or "", title, norm(title), r.get("abstract") or "",
            json.dumps(surn, ensure_ascii=False), norm(surn[0]).split(" ")[-1] if surn else "",
            " ".join(cats), cats[0] if cats else "", r.get("doi"), "oai")


UPSERT = """INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(arxiv_id) DO UPDATE SET
  v1_date = CASE WHEN papers.v1_date = '' THEN excluded.v1_date ELSE papers.v1_date END,
  title = CASE WHEN papers.title = '' THEN excluded.title ELSE papers.title END,
  norm_title = CASE WHEN papers.norm_title = '' THEN excluded.norm_title ELSE papers.norm_title END,
  abstract = CASE WHEN papers.abstract = '' THEN excluded.abstract ELSE papers.abstract END,
  doi = COALESCE(papers.doi, excluded.doi)"""


def build(path: Path | None = None, log=print) -> dict:
    """(Re)build the table from the snapshot + every harvested OAI-PMH window. Idempotent."""
    con = connect(path)
    n_snap = n_oai = 0
    batch = []
    with open(snapshot_path(), "rb") as f:
        for line in f:
            try:
                row = _snapshot_row(json.loads(line))
            except (ValueError, KeyError):
                continue
            if row:
                batch.append(row)
                n_snap += 1
            if len(batch) >= 20000:
                con.executemany(UPSERT, batch)
                batch = []
                log(f"[papers] snapshot {n_snap:,}")
    con.executemany(UPSERT, batch)
    batch = []
    for r in arxiv_oai.iter_records():
        row = _oai_row(r)
        if row:
            batch.append(row)
            n_oai += 1
    con.executemany(UPSERT, batch)
    con.execute("DELETE FROM title_prefix")
    con.execute("INSERT INTO title_prefix SELECT substr(norm_title, 1, 40), arxiv_id FROM papers WHERE norm_title != ''")
    con.execute("CREATE INDEX IF NOT EXISTS ix_prefix ON title_prefix(prefix40)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_date ON papers(v1_date)")
    con.commit()
    total = con.execute("SELECT count(*) FROM papers").fetchone()[0]
    con.close()
    return {"snapshot_rows": n_snap, "oai_rows": n_oai, "papers": total}


class Papers:
    """Read-only view: lookups by id and title resolution (exact-prefix + year ±1 + first-author surname)."""

    def __init__(self, path: Path | None = None):
        self.con = sqlite3.connect(f"file:{path or db_path()}?mode=ro", uri=True, check_same_thread=False)

    def get(self, arxiv_id: str) -> dict | None:
        r = self.con.execute("SELECT * FROM papers WHERE arxiv_id=?", (arxiv_id,)).fetchone()
        if not r:
            return None
        cols = [c[0] for c in self.con.execute("SELECT * FROM papers LIMIT 0").description]
        d = dict(zip(cols, r))
        d["authors"] = json.loads(d["authors"] or "[]")
        return d

    def v1_date(self, arxiv_id: str) -> str | None:
        r = self.con.execute("SELECT v1_date FROM papers WHERE arxiv_id=?", (arxiv_id,)).fetchone()
        return r[0] if r else None

    def resolve_title(self, title: str, year: int | None = None, raw_entry: str = "") -> str | None:
        """arXiv id when exactly one paper matches: same normalized-title 40-char prefix and the first 60 chars agree,
        v1 year within ±1 of the cited year (when both known — preprints precede the venue year), and the first-author
        surname appears in the raw entry (when given). Ambiguous or unmatched -> None (never a guess)."""
        t = norm(title)
        if len(t) < 12:
            return None
        rows = self.con.execute(
            "SELECT p.arxiv_id, p.norm_title, p.v1_date, p.first_surname FROM title_prefix tp "
            "JOIN papers p ON p.arxiv_id = tp.arxiv_id WHERE tp.prefix40 = ?", (t[:40],)).fetchall()
        raw = norm(raw_entry).split()
        out = []
        for aid, full, v1, sur in rows:
            if len(t) >= 40 and not (full.startswith(t[:60]) or t.startswith(full[:60])):
                continue
            if len(t) < 40 and full != t:
                continue
            surname_ok = bool(raw and sur and len(sur) > 1 and sur in raw)
            if raw and sur and len(sur) > 1 and not surname_ok:
                continue
            if year and v1:
                gap = int(year) - int(v1[:4])  # cited year minus preprint year; journal versions come later
                if gap < -1 or gap > (4 if surname_ok else 1):
                    continue
            out.append(aid)
        out = list(dict.fromkeys(out))
        return out[0] if len(out) == 1 else None
