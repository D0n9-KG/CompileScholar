# -*- coding: utf-8 -*-
"""Registry file, schema and connections.

Tables
  papers(paper_id PK, status, canonical_doi, title, first_lo, first_hi, first_precision, first_source, first_kind,
         created_at)
      paper_id: doi:<doi> | arxiv:<id> | title:<norm_title>|<year>[|n]; never reassigned. status active | merged
      (a merged id stays resolvable through aliases). first_*: the first public date (core.asof interval), recomputed
      from `dates` after every import (identity.recompute_first_public): the earliest primary date (arXiv v1, journal
      online / issued); secondary dates (a benchmark's or S2's publication date, a venue year) only when no primary
      date exists — when sources disagree the later, verified date is the leak-safe one.
  identifiers(scheme, value, paper_id, source, role)   PK(scheme, value): one value names one paper
      schemes: doi, arxiv, s2 (S2 paperId), s2_corpus, openalex, masterset, openreview, pmid, pmcid, sciverse
      roles: self (the record's own id) | published_version (a journal DOI listed on an arXiv record)
  dates(paper_id, source, kind, version, lo, hi, precision)   kind: arxiv_v (version N) | online | issued |
      received | published (a secondary source's date: S2, a benchmark) | venue_year
  records(paper_id, source, version, date_hi, title, title_key, abstract, categories, venue)   one metadata record
      per source; version / date_hi: the text version the title and abstract belong to (arXiv metadata is the latest
      version); title_key (core.ids.title_key) proposes title matches across sources
  authors(paper_id, source, names)   names: JSON [{"name", "surname", "id"?}]
  members(benchmark, key, paper_id)   a benchmark's candidate corpus, by the benchmark's own id. Membership only:
      benchmark roles, queries and labels (gold) never enter the library — adapters read them from the release files
  aliases(old_id PK, new_id, reason, at)
  merge_queue(id PK, a, b, kind, evidence, status, verdict, decided_by, decided_at, created_at)  UNIQUE(a, b, kind)
      kind: shared_doi | identifier_conflict | title_match; status open | merged | rejected
  imports(id PK, source, inputs, counts, started_at, finished_at)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from filelock import FileLock

from ..core import paths

BUSY_MS = 30000

DDL = """
CREATE TABLE IF NOT EXISTS papers(paper_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'active',
  canonical_doi TEXT, title TEXT, first_lo TEXT, first_hi TEXT, first_precision TEXT, first_source TEXT,
  first_kind TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_papers_first ON papers(first_hi);
CREATE TABLE IF NOT EXISTS identifiers(scheme TEXT NOT NULL, value TEXT NOT NULL, paper_id TEXT NOT NULL,
  source TEXT, role TEXT, PRIMARY KEY(scheme, value));
CREATE INDEX IF NOT EXISTS ix_ident_paper ON identifiers(paper_id);
CREATE TABLE IF NOT EXISTS dates(paper_id TEXT NOT NULL, source TEXT NOT NULL, kind TEXT NOT NULL,
  version INT NOT NULL DEFAULT 0, lo TEXT NOT NULL, hi TEXT NOT NULL, precision TEXT NOT NULL,
  PRIMARY KEY(paper_id, source, kind, version));
CREATE TABLE IF NOT EXISTS records(paper_id TEXT NOT NULL, source TEXT NOT NULL, version INT, date_hi TEXT,
  title TEXT, title_key TEXT, abstract TEXT, categories TEXT, venue TEXT, PRIMARY KEY(paper_id, source));
CREATE INDEX IF NOT EXISTS ix_records_tkey ON records(title_key);
CREATE TABLE IF NOT EXISTS authors(paper_id TEXT NOT NULL, source TEXT NOT NULL, names TEXT NOT NULL,
  PRIMARY KEY(paper_id, source));
CREATE TABLE IF NOT EXISTS members(benchmark TEXT NOT NULL, key TEXT NOT NULL, paper_id TEXT NOT NULL,
  PRIMARY KEY(benchmark, key));
CREATE INDEX IF NOT EXISTS ix_members_paper ON members(paper_id);
CREATE TABLE IF NOT EXISTS aliases(old_id TEXT PRIMARY KEY, new_id TEXT NOT NULL, reason TEXT, at TEXT);
CREATE TABLE IF NOT EXISTS merge_queue(id INTEGER PRIMARY KEY AUTOINCREMENT, a TEXT NOT NULL, b TEXT NOT NULL,
  kind TEXT NOT NULL, evidence TEXT, status TEXT NOT NULL DEFAULT 'open', verdict TEXT, decided_by TEXT,
  decided_at TEXT, created_at TEXT, UNIQUE(a, b, kind));
CREATE INDEX IF NOT EXISTS ix_queue_status ON merge_queue(status);
CREATE TABLE IF NOT EXISTS imports(id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, inputs TEXT, counts TEXT,
  started_at TEXT, finished_at TEXT);
"""


def db_path() -> Path:
    return paths.library() / "registry.sqlite"


def lock(path: Path | None = None) -> FileLock:
    """One writer at a time (imports, merges)."""
    p = (path or db_path()).parent / "locks"
    p.mkdir(parents=True, exist_ok=True)
    return FileLock(str(p / "registry.lock"))


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or db_path()
    if str(p).startswith(("\\\\", "//")):
        raise RuntimeError(f"the registry must be on a local disk (SQLite WAL), got {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(f"PRAGMA busy_timeout={BUSY_MS}")
    con.executescript(DDL)
    return con
