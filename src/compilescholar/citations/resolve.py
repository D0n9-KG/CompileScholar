# -*- coding: utf-8 -*-
"""Bibliography entry -> arXiv id (deterministic first; the network stage is optional and batched).

Stages, first hit wins:
  1. explicit arXiv id in the entry text                          method = "explicit_arxiv"
  2. papers table title resolution (prefix + year ±1 + surname)    method = "papers_title"
  3. title contained verbatim in the entry (robust to mis-split)   method = "papers_title_in_entry"
  otherwise unresolved (a stub keyed by normalized title, kept so "cited but unknown" papers still count as
  boundary nodes; never used where paper identity is required).
Title/year extraction reuses compile.skeleton.resolve.entry_title_year (title must be a substring of the entry)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..compile.skeleton.resolve import entry_title_year
from ..corpus.papers import Papers
from ..sources.arxiv_snapshot import norm

ARXIV_ID = re.compile(r"(?:arXiv[:\s]*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
# APA: "Su, J., Vargas, D. V., & Sakurai, K. (2019). One pixel attack for fooling deep neural networks. IEEE ..."
APA = re.compile(r"\(((?:19|20)\d\d)[a-z]?\)\.\s+(.+?)(?:\.\s+(?=[A-Z])|\?\s|$)")


def title_year(raw: str) -> tuple[str, int | None]:
    """APA "(Year). Title." first (entry_title_year mis-splits it on the author initials), then the skeleton's
    heuristics. The title must be a substring of the entry either way."""
    m = APA.search(raw)
    if m and len(m.group(2)) >= 12 and norm(m.group(2)) in norm(raw):
        return m.group(2).strip(" ."), int(m.group(1))
    return entry_title_year(raw)


@dataclass
class EntryResolution:
    arxiv_id: str | None
    method: str          # explicit_arxiv | papers_title | papers_title_in_entry | stub
    title: str
    year: int | None
    stub_key: str        # normalized title (or raw prefix) for unresolved entries


class EntryResolver:
    def __init__(self, papers: Papers):
        self.papers = papers
        self._cache: dict[str, EntryResolution] = {}

    def resolve(self, raw: str) -> EntryResolution:
        raw = raw or ""
        if raw in self._cache:
            return self._cache[raw]
        title, year = title_year(raw)
        res = None
        m = ARXIV_ID.search(raw)
        if m:
            res = EntryResolution(m.group(1), "explicit_arxiv", title, year, norm(title))
        if res is None and title:
            ax = self.papers.resolve_title(title, year, raw)
            if ax:
                res = EntryResolution(ax, "papers_title", title, year, norm(title))
        if res is None:
            ax = self._title_in_entry(raw, year)
            if ax:
                res = EntryResolution(ax, "papers_title_in_entry", title, year, norm(title))
        if res is None:
            res = EntryResolution(None, "stub", title, year, norm(title) or norm(raw)[:80])
        self._cache[raw] = res
        return res

    def _title_in_entry(self, raw: str, year: int | None) -> str | None:
        """Try every 5-word window of the entry as a title start; accept only a unique paper whose full normalized
        title occurs verbatim in the entry (>= 5 words)."""
        toks = norm(raw).split()
        rn = " " + " ".join(toks) + " "
        hits = set()
        for i in range(max(0, len(toks) - 4)):
            cand = " ".join(toks[i:i + 12])
            for aid in self._prefix_candidates(cand):
                t = self._title_of(aid)
                if t and len(t.split()) >= 5 and f" {t} " in rn:
                    v1 = self.papers.v1_date(aid) or ""
                    if not year or not v1 or abs(int(v1[:4]) - int(year)) <= 1:
                        hits.add(aid)
            if len(hits) > 1:
                return None
        return hits.pop() if len(hits) == 1 else None

    def _prefix_candidates(self, text: str) -> list[str]:
        return [r[0] for r in self.papers.con.execute(
            "SELECT arxiv_id FROM title_prefix WHERE prefix40 = ?", (text[:40],)).fetchall()]

    def _title_of(self, aid: str) -> str:
        r = self.papers.con.execute("SELECT norm_title FROM papers WHERE arxiv_id=?", (aid,)).fetchone()
        return r[0] if r else ""
