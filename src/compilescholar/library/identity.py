# -*- coding: utf-8 -*-
"""Identity: resolve an incoming metadata record to a paper_id, write it, queue what needs judgement, merge.

Resolution of one record (INTEGRATED-SYSTEM-1005 §3 "先解析，再写入，合并走队列"):
  1. its own identifier (ids[0]) already names a paper -> that paper (a re-import or an update of the same record);
  2. otherwise each paper named by one of its other identifiers, in order, unless that paper holds a different value
     of an identifier scheme the record also has (two arXiv ids, two disjoint DOI sets...: different works);
  3. otherwise a new paper. Every other paper the record's identifiers name is queued (shared_doi /
     identifier_conflict) and those identifiers stay with their owner.
Titles never merge at write time: propose_title_matches() queues candidate pairs (same title_key, years within
YEAR_WINDOW, a shared author surname when both have authors, no conflicting identifier, not a generic title) for the
LLM verdict (B-4); merge() applies a verdict.
paper_id: doi:<doi> when the record carries a DOI nobody owns, else arxiv:<id>, else title:<norm_title>|<year>
(|2, |3 on collision); never reassigned (a merged id becomes an alias)."""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..core import ids
from ..core.asof import Date

YEAR_WINDOW = 2
MAX_GROUP = 6
_ID_RANK = {"doi": 0, "arxiv": 1}


@dataclass
class Incoming:
    source: str
    ids: list                                   # [(scheme, value, role)]; ids[0] is the record's own identifier
    title: str = ""
    abstract: str = ""
    categories: list = field(default_factory=list)
    venue: str = ""
    version: int | None = None                  # the text version title / abstract belong to
    version_hi: str | None = None
    dates: list = field(default_factory=list)   # [(kind, version, Date)]
    authors: list = field(default_factory=list)  # [{"name", "surname", "id"?}]
    member: tuple | None = None                 # (benchmark, key)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def ids_of(con: sqlite3.Connection, pid: str) -> dict:
    out = defaultdict(set)
    for s, v in con.execute("SELECT scheme, value FROM identifiers WHERE paper_id=?", (pid,)):
        out[s].add(v)
    return out


def _conflict(rec: dict, have: dict) -> bool:
    """A scheme both sides carry with no value in common: two different works."""
    return any(s in have and vs and not (vs & have[s]) for s, vs in rec.items())


class Writer:
    """Writes incoming records of one source through one connection, committing every `batch` records."""

    def __init__(self, con: sqlite3.Connection, source: str, batch: int = 20000):
        self.con, self.source, self.batch = con, source, batch
        self.counts: Counter = Counter()
        self._n = 0
        con.execute("BEGIN")

    def lookup(self, scheme: str, value: str) -> str | None:
        r = self.con.execute("SELECT paper_id FROM identifiers WHERE scheme=? AND value=?", (scheme, value)).fetchone()
        return r[0] if r else None

    def _new_id(self, inc: Incoming) -> str:
        cand = None
        for s, v, _ in inc.ids:
            if s == "doi" and self.lookup("doi", v) is None:
                cand = f"doi:{v}"
                break
        if cand is None:
            ax = next((v for s, v, _ in inc.ids if s == "arxiv"), None)
            if ax and self.lookup("arxiv", ax) is None:
                cand = f"arxiv:{ax}"
        if cand is None:
            ys = [d.lo.year for _, _, d in inc.dates if d is not None]
            cand = f"title:{ids.norm_title(inc.title)[:200]}|{min(ys) if ys else ''}"
        pid, n = cand, 1
        while self.con.execute("SELECT 1 FROM papers WHERE paper_id=?", (pid,)).fetchone():
            n += 1
            pid = f"{cand}|{n}"
        return pid

    def add(self, inc: Incoming) -> str:
        own = self.lookup(*inc.ids[0][:2])
        hits = {}
        for s, v, _ in inc.ids:
            p = self.lookup(s, v)
            if p:
                hits.setdefault(p, []).append(s)
        target = own
        if target is None:
            rec = defaultdict(set)
            for s, v, _ in inc.ids:
                rec[s].add(v)
            target = next((p for p in hits if not _conflict(rec, ids_of(self.con, p))), None)
        if target is None:
            target = self._new_id(inc)
            self.con.execute("INSERT INTO papers(paper_id, title, created_at) VALUES (?,?,?)",
                             (target, inc.title, _now()))
            self.counts["new"] += 1
        else:
            self.counts["matched"] += 1
        for p, schemes in hits.items():
            if p != target:
                kind = "shared_doi" if set(schemes) == {"doi"} else "identifier_conflict"
                queue(self.con, target, p, kind, {"source": self.source, "shared": schemes,
                                                  "record": inc.ids[0][1]})
                self.counts[kind] += 1
        for s, v, role in inc.ids:
            if self.lookup(s, v) is None:
                self.con.execute("INSERT INTO identifiers VALUES (?,?,?,?,?)", (s, v, target, self.source, role))
                if s == "doi":
                    self.con.execute("UPDATE papers SET canonical_doi=? WHERE paper_id=? AND canonical_doi IS NULL",
                                     (v, target))
        if inc.title or inc.abstract:
            self.con.execute(
                "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(paper_id, source) DO UPDATE SET "
                "version=excluded.version, date_hi=excluded.date_hi, title=excluded.title, "
                "title_key=excluded.title_key, abstract=excluded.abstract, categories=excluded.categories, "
                "venue=excluded.venue WHERE records.version IS NULL OR excluded.version IS NULL "
                "OR excluded.version >= records.version",
                (target, self.source, inc.version, inc.version_hi, inc.title, ids.title_key(inc.title), inc.abstract,
                 " ".join(inc.categories), inc.venue))
        for kind, ver, d in inc.dates:
            if d is not None:
                self.con.execute("INSERT OR REPLACE INTO dates VALUES (?,?,?,?,?,?,?)",
                                 (target, self.source, kind, ver or 0, d.lo_iso, d.hi_iso, d.precision))
        if inc.authors:
            self.con.execute("INSERT OR REPLACE INTO authors VALUES (?,?,?)",
                             (target, self.source, json.dumps(inc.authors, ensure_ascii=False)))
        if inc.member:
            self.con.execute("INSERT OR REPLACE INTO members VALUES (?,?,?)", (*inc.member, target))
        self._n += 1
        if self._n % self.batch == 0:
            self.con.execute("COMMIT")
            self.con.execute("BEGIN")
        return target

    def close(self) -> dict:
        self.con.execute("COMMIT")
        return dict(self.counts)


def queue(con: sqlite3.Connection, a: str, b: str, kind: str, evidence: dict) -> None:
    a, b = sorted((a, b))
    con.execute("INSERT OR IGNORE INTO merge_queue(a, b, kind, evidence, created_at) VALUES (?,?,?,?,?)",
                (a, b, kind, json.dumps(evidence, ensure_ascii=False), _now()))


def recompute_first_public(con: sqlite3.Connection) -> int:
    """papers.first_* = the earliest primary date (arXiv version, journal online / issued); a secondary date (a
    benchmark's or S2's publication date, a venue year) only when no primary one exists; on equal hi the more precise.
    `received` never decides visibility."""
    con.execute("""
WITH d AS (SELECT paper_id, source, kind, lo, hi, precision,
                  CASE WHEN kind IN ('arxiv_v','online','issued') THEN 0 ELSE 1 END AS tier
           FROM dates WHERE kind != 'received'),
b AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY paper_id ORDER BY tier, hi,
             CASE precision WHEN 'day' THEN 0 WHEN 'month' THEN 1 ELSE 2 END, source) AS rn FROM d)
UPDATE papers SET first_lo=b.lo, first_hi=b.hi, first_precision=b.precision, first_source=b.source,
                  first_kind=b.kind
FROM b WHERE b.paper_id = papers.paper_id AND b.rn = 1""")
    con.commit()
    return con.execute("SELECT count(*) FROM papers WHERE first_hi IS NULL AND status='active'").fetchone()[0]


def _surnames(con: sqlite3.Connection, pid: str) -> set:
    out = set()
    for (names,) in con.execute("SELECT names FROM authors WHERE paper_id=?", (pid,)):
        out |= {ids.norm_title(a.get("surname") or "") for a in json.loads(names)} - {""}
    return out


def propose_title_matches(con: sqlite3.Connection, year_window: int = YEAR_WINDOW, max_group: int = MAX_GROUP,
                          log=print) -> dict:
    """Queue title_match pairs for the LLM verdict. Returns counts, including groups skipped as too large."""
    rows = con.execute(
        "SELECT r.title_key, r.paper_id, r.title FROM records r JOIN papers p ON p.paper_id = r.paper_id "
        "WHERE p.status = 'active' AND r.title_key IN (SELECT title_key FROM records WHERE length(title_key) >= 12 "
        "GROUP BY title_key HAVING count(DISTINCT paper_id) > 1)").fetchall()
    groups = defaultdict(dict)
    for k, pid, title in rows:
        groups[k].setdefault(pid, title)
    con.execute("BEGIN")
    n = Counter()
    for k, papers in groups.items():
        if len(papers) < 2:
            continue
        if len(papers) > max_group:
            n["groups_too_large"] += 1
            continue
        if any(ids.title_is_generic(t) for t in papers.values()):
            n["generic"] += 1
            continue
        info = {}
        for pid in papers:
            fh = con.execute("SELECT first_hi FROM papers WHERE paper_id=?", (pid,)).fetchone()[0]
            info[pid] = (ids_of(con, pid), int(fh[:4]) if fh else None, _surnames(con, pid))
        pids = sorted(papers)
        for i, a in enumerate(pids):
            for b in pids[i + 1:]:
                (ia, ya, sa), (ib, yb, sb) = info[a], info[b]
                if _conflict(ia, ib):
                    n["identifier_conflict"] += 1
                    continue
                if ya and yb and abs(ya - yb) > year_window:
                    n["years_apart"] += 1
                    continue
                if sa and sb and not (sa & sb):
                    n["no_shared_author"] += 1
                    continue
                queue(con, a, b, "title_match", {"titles": [papers[a], papers[b]], "years": [ya, yb],
                                                 "shared_surnames": sorted(sa & sb)[:5],
                                                 "authors_known": [bool(sa), bool(sb)]})
                n["queued"] += 1
    con.execute("COMMIT")
    log(f"[library] title matches: {dict(n)}")
    return dict(n)


def canonical(con: sqlite3.Connection, pid: str) -> str:
    """Follow aliases to the surviving paper_id."""
    seen = set()
    while pid not in seen:
        seen.add(pid)
        r = con.execute("SELECT new_id FROM aliases WHERE old_id=?", (pid,)).fetchone()
        if not r:
            return pid
        pid = r[0]
    raise RuntimeError(f"alias cycle at {pid}")


def resolve(con: sqlite3.Connection, scheme: str, value: str) -> str | None:
    if scheme == "doi":
        value = ids.normalize_doi(value) or value
    elif scheme == "arxiv":
        value = ids.normalize_arxiv(value) or value
    r = con.execute("SELECT paper_id FROM identifiers WHERE scheme=? AND value=?", (scheme, value)).fetchone()
    return r[0] if r else None


def survivor(a: str, b: str) -> tuple[str, str]:
    """(keep, drop) for a merge: the id with the stronger scheme (doi: > arxiv: > title:) survives."""
    ra, rb = _ID_RANK.get(a.split(":", 1)[0], 9), _ID_RANK.get(b.split(":", 1)[0], 9)
    return (a, b) if (ra, a) <= (rb, b) else (b, a)


def merge(con: sqlite3.Connection, a: str, b: str, reason: str, decided_by: str = "", recompute: bool = True) -> str:
    """Merge two papers in one transaction: every row keyed by the dropped id moves to the kept one (rows the kept
    paper already has for the same source stay as they are), the dropped id becomes an alias, aliases that pointed at
    it are re-pointed, first_public is recomputed. Returns the kept id."""
    keep, drop = survivor(canonical(con, a), canonical(con, b))
    if keep == drop:
        return keep
    con.execute("BEGIN")
    try:
        con.execute("UPDATE identifiers SET paper_id=? WHERE paper_id=?", (keep, drop))
        con.execute("UPDATE members SET paper_id=? WHERE paper_id=?", (keep, drop))
        for t in ("dates", "records", "authors", "assets"):
            con.execute(f"UPDATE OR IGNORE {t} SET paper_id=? WHERE paper_id=?", (keep, drop))
            con.execute(f"DELETE FROM {t} WHERE paper_id=?", (drop,))
        con.execute("UPDATE attempts SET paper_id=? WHERE paper_id=?", (keep, drop))
        con.execute("UPDATE ref_resolutions SET paper_id=? WHERE paper_id=?", (keep, drop))
        con.execute("UPDATE papers SET canonical_doi = COALESCE(canonical_doi, (SELECT canonical_doi FROM papers "
                    "WHERE paper_id=?)) WHERE paper_id=?", (drop, keep))
        con.execute("UPDATE papers SET status='merged' WHERE paper_id=?", (drop,))
        con.execute("UPDATE aliases SET new_id=? WHERE new_id=?", (keep, drop))
        con.execute("INSERT OR REPLACE INTO aliases VALUES (?,?,?,?)", (drop, keep, reason, _now()))
        x, y = sorted((a, b))
        # the queue row keeps its verdict (the evidence for the merge); only its status changes
        con.execute("UPDATE merge_queue SET status='merged', decided_by=?, decided_at=? "
                    "WHERE a=? AND b=? AND status='open'", (decided_by or reason, _now(), x, y))
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    if recompute:                       # a batch of merges recomputes once at the end
        recompute_first_public(con)
    return keep


def first_public(con: sqlite3.Connection, pid: str) -> Date | None:
    r = con.execute("SELECT first_lo, first_hi, first_precision FROM papers WHERE paper_id=?",
                    (canonical(con, pid),)).fetchone()
    if not r or not r[1]:
        return None
    return Date(_dt.date.fromisoformat(r[0]), _dt.date.fromisoformat(r[1]), r[2])
