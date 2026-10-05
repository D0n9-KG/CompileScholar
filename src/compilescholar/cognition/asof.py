# -*- coding: utf-8 -*-
"""The as-of view: every read of statements, citations or papers made by the compiler and the tools goes through
AsOf(T), which only ever returns rows dated <= T (DESIGN §12a invariant "as_of monotone"). Nothing downstream of L3
touches the raw tables directly.

T is a 'YYYY-MM-DD' string (inclusive). A paper is visible at T if its v1_date <= T; a statement if its date <= T;
a citation sentence if its citing paper's date <= T."""
from __future__ import annotations

import json
import sqlite3

from ..dfc import store


class AsOf:
    def __init__(self, T: str, papers=None, cit=None, ext=None):
        if len(T) != 10:
            raise ValueError(f"as_of must be YYYY-MM-DD, got {T!r}")
        self.T = T
        self.papers = papers or store.connect("papers", readonly=True)
        self.cit = cit or store.connect("citations", readonly=True)
        self.ext = ext or store.connect("extract", readonly=True)

    # ---- papers
    def paper(self, arxiv_id: str) -> dict | None:
        r = self.papers.execute("SELECT arxiv_id, v1_date, title, abstract, authors, categories FROM papers "
                                "WHERE arxiv_id=? AND v1_date<=?", (arxiv_id, self.T)).fetchone()
        if not r:
            return None
        return {"arxiv_id": r[0], "v1_date": r[1], "title": r[2], "abstract": r[3],
                "authors": json.loads(r[4] or "[]"), "categories": (r[5] or "").split()}

    def visible(self, obj: str) -> bool:
        """'paper:<id>' visible iff published by T; stubs are always visible (no date; they exist only as cited)."""
        if obj.startswith("paper:"):
            return self.paper(obj[6:]) is not None
        return True

    # ---- statements
    def statements(self, about: str | None = None, speaker: str | None = None, kind: str | None = None,
                   facets: tuple[str, ...] | None = None) -> list[dict]:
        q, a = "SELECT * FROM statements WHERE date<=?", [self.T]
        for col, v in (("about", about), ("speaker", speaker), ("kind", kind)):
            if v is not None:
                q += f" AND {col}=?"
                a.append(v)
        if facets:
            q += f" AND facet IN ({','.join('?' * len(facets))})"
            a += list(facets)
        cur = self.ext.execute(q, a)
        cols = [c[0] for c in cur.description]
        out = []
        for r in cur.fetchall():
            d = dict(zip(cols, r))
            d["group"] = json.loads(d.pop("grp") or "[]")
            d["meta"] = json.loads(d["meta"] or "{}")
            out.append(d)
        return out

    def statements_by_id(self, ids) -> list[dict]:
        ids = list(ids)
        if not ids:
            return []
        cur = self.ext.execute(f"SELECT * FROM statements WHERE date<=? AND id IN ({','.join('?' * len(ids))})",
                               [self.T, *ids])
        cols = [c[0] for c in cur.description]
        out = []
        for r in cur.fetchall():
            d = dict(zip(cols, r))
            d["group"] = json.loads(d.pop("grp") or "[]")
            d["meta"] = json.loads(d["meta"] or "{}")
            out.append(d)
        return out

    # ---- citations
    def cited_by(self, obj: str) -> list[tuple[str, str, int]]:
        """(citing, date, sentence_id) for every citation sentence about obj dated <= T."""
        return self.cit.execute("SELECT citing, date, sentence_id FROM cites WHERE cited=? AND date<=?",
                                (obj, self.T)).fetchall()

    def references(self, arxiv_id: str) -> list[str]:
        """Objects cited by a paper (its references), if the paper itself is visible."""
        if not self.visible(f"paper:{arxiv_id}"):
            return []
        return [r[0] for r in self.cit.execute("SELECT DISTINCT cited FROM entries WHERE citing=?", (arxiv_id,))]

    def co_cited(self, sentence_id: int) -> list[str]:
        return [r[0] for r in self.cit.execute("SELECT cited FROM cites WHERE sentence_id=? AND date<=?",
                                               (sentence_id, self.T))]


def connect_all() -> tuple[sqlite3.Connection, sqlite3.Connection, sqlite3.Connection]:
    return (store.connect("papers", readonly=True), store.connect("citations", readonly=True),
            store.connect("extract", readonly=True))
