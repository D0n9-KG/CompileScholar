# -*- coding: utf-8 -*-
"""The as-of view (phase D rewiring): every read of statements, citations or papers made by the compiler and the
tools goes through AsOf(T), which only ever returns rows dated <= T ("as_of monotone"). Nothing downstream of the
extract stage touches the raw tables directly.

Phase D changes from the pre-C version:
  - papers come from the library registry (paper_id keys: doi:/arxiv:/title:/stub:), not the retired papers stage;
    visibility = papers.first_hi <= T with status='active', following aliases to the survivor of a merge;
  - statements are schema v2 (loc parsed; epistemic/condition are first-class columns);
  - citations carry version-dated sentences (v1 content at the v1 date, delta content at the latest version's
    date), so a citing sentence is visible exactly when its TEXT version is;
  - stored dates are full-ISO `hi` values everywhere (registry dates.hi, docs.text_date), so the SQL string
    comparison is exact and leak-safe: a year-precision date is only "reached" at its hi (Dec 31)."""
from __future__ import annotations

import json

from ..core import paths
from ..core.asof import AsOf as _T
from ..dfc import store


class AsOf:
    def __init__(self, T: str, reg=None, cit=None, ext=None, cog=None):
        self.T = _T(T).iso
        if self.T is None:
            raise ValueError("as_of is required")
        self.reg = reg if reg is not None else store.read_only(paths.library() / "registry.sqlite")
        self.cit = cit if cit is not None else store.connect("citations", readonly=True)
        self.ext = ext if ext is not None else store.connect("extract", readonly=True)
        # the materialised cognition tables (phase D readers); None until the stage has been built — readers
        # degrade to empty results rather than raising
        if cog is None:
            cog = store.connect("cognition", readonly=True) if store.db_path("cognition").exists() else None
        self.cog = cog

    # ---- papers (the registry)
    def canonical(self, paper_id: str) -> str:
        seen = set()
        while paper_id not in seen:
            seen.add(paper_id)
            r = self.reg.execute("SELECT new_id FROM aliases WHERE old_id=?", (paper_id,)).fetchone()
            if not r:
                return paper_id
            paper_id = r[0]
        return paper_id

    def paper(self, paper_id: str) -> dict | None:
        """The registry record of one paper when visible at T (first public date reached, active)."""
        pid = self.canonical(paper_id)
        r = self.reg.execute("SELECT status, first_hi FROM papers WHERE paper_id=?", (pid,)).fetchone()
        if not r or r[0] != "active" or not r[1] or r[1] > self.T:
            return None
        rec = self.reg.execute("SELECT title, abstract, categories FROM records WHERE paper_id=? "
                               "ORDER BY source = 'arxiv' DESC LIMIT 1", (pid,)).fetchone()
        au = self.reg.execute("SELECT names FROM authors WHERE paper_id=? ORDER BY source = 'arxiv' DESC LIMIT 1",
                              (pid,)).fetchone()
        ids = {s: v for s, v in self.reg.execute("SELECT scheme, value FROM identifiers WHERE paper_id=?", (pid,))}
        try:
            authors = json.loads(au[0]) if au else []
        except (TypeError, ValueError):
            authors = []
        return {"paper_id": pid, "date": r[1], "title": (rec[0] if rec else "") or "",
                "abstract": (rec[1] if rec else "") or "", "authors": authors,
                "categories": ((rec[2] if rec else "") or "").split(), "ids": ids}

    def visible(self, obj: str) -> bool:
        """A statement object: stubs are always visible (they exist only as cited); method:<name>@<paper> follows
        its paper; a paper_id follows the registry."""
        if not obj or obj.startswith("stub:"):
            return True
        if obj.startswith("method:"):
            tail = obj.rsplit("@", 1)[-1]
            return True if not tail or tail == obj else self.visible(tail)
        return self.paper(obj) is not None

    # ---- statements (schema v2)
    @staticmethod
    def _stmt(d: dict) -> dict:
        d["group"] = json.loads(d.pop("grp", None) or "[]")
        try:
            d["meta"] = json.loads(d.get("meta") or "{}")
        except (TypeError, ValueError):
            d["meta"] = {}
        try:
            d["loc"] = json.loads(d.get("loc") or "{}")
        except (TypeError, ValueError):
            d["loc"] = {}
        return d

    def statements(self, about: str | None = None, speaker: str | None = None, kind: str | None = None,
                   facets: tuple[str, ...] | None = None) -> list[dict]:
        q, a = "SELECT * FROM statements WHERE date<=?", [self.T]
        for col, v in (("about", about and self.canonical(about)), ("speaker", speaker and self.canonical(speaker)),
                       ("kind", kind)):
            if v is not None:
                q += f" AND {col}=?"
                a.append(v)
        if facets:
            q += f" AND facet IN ({','.join('?' * len(facets))})"
            a += list(facets)
        cur = self.ext.execute(q, a)
        cols = [c[0] for c in cur.description]
        return [self._stmt(dict(zip(cols, r))) for r in cur.fetchall()]

    def statements_by_id(self, ids) -> list[dict]:
        ids = list(ids)
        if not ids:
            return []
        cur = self.ext.execute(f"SELECT * FROM statements WHERE date<=? AND id IN ({','.join('?' * len(ids))})",
                               [self.T, *ids])
        cols = [c[0] for c in cur.description]
        return [self._stmt(dict(zip(cols, r))) for r in cur.fetchall()]

    # ---- citations
    def cited_by(self, obj: str) -> list[tuple[str, str, int]]:
        """(citing, date, sentence_id) for every citation sentence about obj dated <= T."""
        return self.cit.execute("SELECT citing, date, sentence_id FROM cites WHERE cited=? AND date<=?",
                                (obj, self.T)).fetchall()

    def references(self, paper_id: str) -> list[str]:
        """Objects cited by a paper that are visible at T, when the paper itself is visible."""
        if not self.visible(paper_id):
            return []
        pid = self.canonical(paper_id)
        return [o for (o,) in self.cit.execute("SELECT DISTINCT cited FROM entries WHERE citing=?", (pid,))
                if self.visible(o)]

    def co_cited(self, sentence_id: int) -> list[str]:
        return [r[0] for r in self.cit.execute("SELECT cited FROM cites WHERE sentence_id=? AND date<=?",
                                               (sentence_id, self.T)) if self.visible(r[0])]


def connect_all():
    from ..core import paths as _p
    return (store.read_only(_p.library() / "registry.sqlite"), store.connect("citations", readonly=True),
            store.connect("extract", readonly=True))
