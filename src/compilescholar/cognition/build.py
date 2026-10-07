# -*- coding: utf-8 -*-
"""Stage `cognition` (phase D①): the materialised tables of INTEGRATED-SYSTEM-1005 §8 — deterministic ones first.

Passes in this first cut (all deterministic, zero LLM; the five LLM adjudications land as their own passes, each
piloted on 200 items with the dual-model agreement gate — §12 D②):
  cocite        (a, b, day, n): pairs co-cited in one citation sentence, aggregated per day — the as-of strength
                is SUM(n) WHERE day <= T (bitemporal: no snapshot needed)
  reception     reception_daily(cited, day, n): full daily citation counts — the count is never truncated by the
                other pass's sampled subset (v2.4 item 2)
  authors       author_link(author_key, paper_id): surname+initial keys over the registry's active papers, for
                the independence checks (facts) and author-aware tools
  comparisons   comparison_edge from the other pass's outcome meta (citing_better / cited_better / mixed), dated
                by the citing sentence's text-version date

Tables for the later passes (mention_link, lineage_edge/hyper, category_canon/daily, fact_member,
fact_status_event, shift_event, family_snapshot) are created up front so their schemas are stable; they stay
empty until their passes land.

Work items: one item ("all") per pass, fingerprinted by the upstream manifests' data fingerprints — any growth
in citations/extract/registry re-opens the affected aggregate (they are cheap full recomputes by SQL)."""
from __future__ import annotations

import json

from ..core import paths
from ..dfc import store

DDL = """CREATE TABLE IF NOT EXISTS cocite(a TEXT NOT NULL, b TEXT NOT NULL, day TEXT NOT NULL, n INT NOT NULL,
  PRIMARY KEY(a, b, day));
CREATE INDEX IF NOT EXISTS ix_cocite_a ON cocite(a, day);
CREATE INDEX IF NOT EXISTS ix_cocite_b ON cocite(b, day);
CREATE TABLE IF NOT EXISTS reception_daily(cited TEXT NOT NULL, day TEXT NOT NULL, n INT NOT NULL,
  PRIMARY KEY(cited, day));
CREATE INDEX IF NOT EXISTS ix_reception_day ON reception_daily(day);
CREATE TABLE IF NOT EXISTS author_link(author_key TEXT NOT NULL, paper_id TEXT NOT NULL,
  PRIMARY KEY(author_key, paper_id));
CREATE INDEX IF NOT EXISTS ix_author_paper ON author_link(paper_id);
CREATE TABLE IF NOT EXISTS comparison_edge(stmt_id INT PRIMARY KEY, a TEXT, b TEXT, outcome TEXT, date TEXT);
CREATE INDEX IF NOT EXISTS ix_comp_b ON comparison_edge(b, date);
CREATE TABLE IF NOT EXISTS mention_link(id INTEGER PRIMARY KEY, name TEXT, sid TEXT, citing TEXT, date TEXT,
  candidates TEXT, paper_id TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS lineage_edge(stmt_id INT, child TEXT, parent TEXT, relation TEXT, speaker TEXT,
  date TEXT, valid_from TEXT, kind TEXT);
CREATE TABLE IF NOT EXISTS lineage_hyper(stmt_id INT, member TEXT, relation TEXT, speaker TEXT, date TEXT,
  valid_from TEXT);
CREATE TABLE IF NOT EXISTS category_canon(phrase TEXT PRIMARY KEY, canonical TEXT, umbrella INT, decided_by TEXT);
CREATE TABLE IF NOT EXISTS category_daily(category TEXT NOT NULL, day TEXT NOT NULL, n INT,
  PRIMARY KEY(category, day));
CREATE TABLE IF NOT EXISTS fact_member(fact_id TEXT, statement_id INT, role TEXT, PRIMARY KEY(fact_id, statement_id));
CREATE TABLE IF NOT EXISTS fact_status_event(fact_id TEXT, date TEXT, status TEXT, evidence TEXT);
CREATE TABLE IF NOT EXISTS shift_event(id INTEGER PRIMARY KEY, subject TEXT, facet TEXT, window_start TEXT,
  window_end TEXT, direction TEXT, evidence TEXT, decided_by TEXT);
CREATE TABLE IF NOT EXISTS family_snapshot(snapshot TEXT, family_id TEXT, name TEXT, members TEXT, named_by TEXT,
  PRIMARY KEY(snapshot, family_id));"""


def _manifest_fp(stage: str) -> str | None:
    m = store.read_manifest(stage) or {}
    return m.get("fingerprint")


def _author_key(surname: str, name: str) -> str | None:
    sur = (surname or "").strip().lower()
    if not sur:
        return None
    toks = (name or "").split()
    ini = toks[0][0].lower() if toks and toks[0][:1].isalpha() else ""
    return f"{sur}|{ini}"


def build(workers: int | None = None, rebuild: bool = False, log=print, cit=None, ext=None, reg=None) -> dict:
    params = {"v": 1}
    with store.Run("cognition", params, rebuild=rebuild) as run:
        con = run.con
        con.executescript(DDL)
        mine = []
        if cit is None:
            cit = store.connect("citations", readonly=True)
            mine.append(cit)
        if ext is None:
            ext = store.connect("extract", readonly=True)
            mine.append(ext)
        if reg is None:
            reg = store.read_only(paths.library() / "registry.sqlite")
            mine.append(reg)
        try:
            cit_fp, ext_fp = _manifest_fp("citations"), _manifest_fp("extract")
            reg_fp = store.sha(reg.execute("SELECT count(*) FROM authors").fetchone()[0],
                               reg.execute("SELECT count(*) FROM papers WHERE status='active'").fetchone()[0])

            # ---- cocite: pairs co-cited in one sentence, aggregated per day
            w = run.work("cocite")
            if w.todo([("all", store.sha(cit_fp))]):
                con.execute("DELETE FROM cocite")
                cur = cit.execute("SELECT a.cited, b.cited, a.date, count(*) FROM cites a JOIN cites b "
                                  "ON a.sentence_id = b.sentence_id AND a.cited < b.cited "
                                  "WHERE a.date IS NOT NULL GROUP BY a.cited, b.cited, a.date")
                n = 0
                while True:
                    rows = cur.fetchmany(50000)
                    if not rows:
                        break
                    con.executemany("INSERT OR REPLACE INTO cocite VALUES (?,?,?,?)", rows)
                    n += len(rows)
                con.commit()
                w.ok("all")
                log(f"[cognition] cocite: {n:,} (pair, day) rows")

            # ---- reception: full daily citation counts
            w = run.work("reception")
            if w.todo([("all", store.sha(cit_fp))]):
                con.execute("DELETE FROM reception_daily")
                cur = cit.execute("SELECT cited, date, count(*) FROM cites WHERE date IS NOT NULL "
                                  "GROUP BY cited, date")
                n = 0
                while True:
                    rows = cur.fetchmany(50000)
                    if not rows:
                        break
                    con.executemany("INSERT OR REPLACE INTO reception_daily VALUES (?,?,?)", rows)
                    n += len(rows)
                con.commit()
                w.ok("all")
                log(f"[cognition] reception: {n:,} (paper, day) rows")

            # ---- authors: surname+initial keys of every active paper
            w = run.work("authors")
            if w.todo([("all", store.sha(reg_fp))]):
                con.execute("DELETE FROM author_link")
                active = {p for (p,) in reg.execute("SELECT paper_id FROM papers WHERE status='active'")}
                rows, n = [], 0
                for pid, names in reg.execute("SELECT paper_id, names FROM authors"):
                    if pid not in active:
                        continue
                    try:
                        lst = json.loads(names)
                    except (TypeError, ValueError):
                        continue
                    for a in lst:
                        k = _author_key(a.get("surname"), a.get("name"))
                        if k:
                            rows.append((k, pid))
                    if len(rows) >= 50000:
                        con.executemany("INSERT OR IGNORE INTO author_link VALUES (?,?)", rows)
                        n += len(rows)
                        rows = []
                if rows:
                    con.executemany("INSERT OR IGNORE INTO author_link VALUES (?,?)", rows)
                    n += len(rows)
                con.commit()
                w.ok("all")
                log(f"[cognition] author_link: {n:,} rows")

            # ---- comparisons: the other pass's qualitative outcomes
            w = run.work("comparisons")
            if w.todo([("all", store.sha(ext_fp))]):
                con.execute("DELETE FROM comparison_edge")
                cur = ext.execute("SELECT id, speaker, about, json_extract(meta, '$.outcome'), date "
                                  "FROM statements WHERE kind='other' AND date IS NOT NULL "
                                  "AND json_extract(meta, '$.outcome') IS NOT NULL")
                n = 0
                while True:
                    rows = cur.fetchmany(50000)
                    if not rows:
                        break
                    con.executemany("INSERT OR REPLACE INTO comparison_edge VALUES (?,?,?,?,?)", rows)
                    n += len(rows)
                con.commit()
                w.ok("all")
                log(f"[cognition] comparison_edge: {n:,} rows")

            q = lambda s: con.execute(s).fetchone()[0] or 0        # noqa: E731
            counts = {"cocite": q("SELECT count(*) FROM cocite"),
                      "reception": q("SELECT count(*) FROM reception_daily"),
                      "author_link": q("SELECT count(*) FROM author_link"),
                      "comparison_edge": q("SELECT count(*) FROM comparison_edge")}
            run.finish(counts)
        finally:
            for c in mine:
                c.close()
    return counts
