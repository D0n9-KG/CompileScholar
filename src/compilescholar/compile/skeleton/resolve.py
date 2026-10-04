# -*- coding: utf-8 -*-
"""W2 P0-2: bibliography entry -> paper identity (deterministic stages first; network stages only for the rest).

Stages, first hit wins (DESIGN-W2 §1):
  1. explicit id in the entry text (arXiv id / DOI)                    method = "explicit_arxiv" | "explicit_doi"
  2. local arXiv OAI snapshot title index (prefix + year ±1 + surname)  method = "oai_snapshot"
  3. KB title index (exact normalized title)                            method = "kb_title"
  4. OpenAlex batched title search (network; optional)                  method = "openalex"
  5. Crossref bibliographic query (network; optional)                   method = "crossref"
  otherwise a name-level stub (never counted where paper identity is needed).

Title / year extraction from a raw entry is heuristic (refgraph._bib_title); the extracted title must be a substring of
the raw entry text (checked), so a mis-split can only miss, not invent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ...sources import refgraph
from ...sources.arxiv_snapshot import TitleIndex, norm

YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


@dataclass
class Resolution:
    paper: str | None          # "arxiv:<id>" | "doi:<doi>" | "kb:<paper_id>" | "openalex:<W...>" | None
    method: str                # stage name, or "stub"
    title: str
    year: int | None


QUOTED = re.compile(r"[“\"]([^“”\"]{12,300}?)[,.]?[”\"]")
# Vancouver / Elsevier: "Surname A, Surname B. Title. Venue" or "A. Surname, B. Surname, Title, Venue (2020)"
_NAME = r"(?:[A-Z][A-Za-z'\-]+(?:\s[A-Z]{1,3})?|[A-Z]\.(?:[\s\-]?[A-Z]\.)*\s(?:[a-z]+\s)?[A-Z][A-Za-z'\-]+)"
AUTH_LIST = re.compile(rf"^(?:{_NAME}(?:,\s|\sand\s|,\sand\s))*(?:{_NAME}|et al\.?)[.,]\s")


def entry_title_year(raw: str) -> tuple[str, int | None]:
    """Title + year of a raw entry. Order: quoted segment (IEEE / ACM style) -> text after a leading author list
    (Vancouver / Elsevier) -> refgraph's sentence heuristic. The title must be a substring of the entry."""
    t = ""
    m = QUOTED.search(raw)
    if m:
        t = m.group(1).strip(" ,.")
    if not t:
        a = AUTH_LIST.match(raw)
        if a:
            rest = raw[a.end():]
            # title ends at ". ", " In:", ", in:", or the comma before a venue / volume ("..., Annals of X 37 (1966)",
            # "..., Signal Processing 90 (2010)", "..., IEEE Transactions on ...")
            t = re.split(r"\.\s|\sIn:?\s|,\sin:?\s|,\s(?=(?:[A-Z][A-Za-z&\-]*\s){0,8}?(?:\d+\s?\(|vol\.|pp\.))|"
                         r",\s(?=(?:IEEE|ACM|Proc|Proceedings|Journal|Annals|Bulletin|Transactions|Advances|arXiv|"
                         r"Computers|Signal|Pattern|Neural|Machine|Artificial|Information|Knowledge|Expert|"
                         r"Nature|Science|Physical|Medical|International)\b)", rest)[0].strip(" ,.")
    if not t:
        t = refgraph.clean_title(refgraph._bib_title(raw))
    if norm(t) and norm(t) not in norm(raw):
        t = ""
    ys = [int(y) for y in YEAR.findall(raw)]
    return t, (max(ys) if ys else None)


class Resolver:
    def __init__(self, kb_papers: dict, snapshot: TitleIndex | None = None):
        self.snapshot = snapshot
        self.kb_by_title = {}
        self.kb_by_arxiv = {}
        for pid, p in kb_papers.items():
            t = norm(p.get("title") or "")
            if len(t) >= 12:
                self.kb_by_title.setdefault(t, set()).add(pid)
            if p.get("arxiv_id"):
                self.kb_by_arxiv[str(p["arxiv_id"]).lower()] = pid
        self.kb_prefix4: dict[str, list[tuple[str, str]]] = {}
        for t, pids in self.kb_by_title.items():
            w = t.split()
            if len(w) >= 5 and len(pids) == 1:
                self.kb_prefix4.setdefault(" ".join(w[:4]), []).append((t, next(iter(pids))))

    def resolve(self, entry: dict) -> Resolution:
        raw = entry.get("raw") or ""
        title, year = entry_title_year(raw)
        if entry.get("arxiv"):
            ax = entry["arxiv"].lower()
            return Resolution(f"kb:{self.kb_by_arxiv[ax]}" if ax in self.kb_by_arxiv else f"arxiv:{ax}",
                              "explicit_arxiv", title, year)
        if entry.get("doi"):
            return Resolution(f"doi:{entry['doi'].lower()}", "explicit_doi", title, year)
        if title and self.snapshot is not None:
            ax = self.snapshot.lookup(title, year, raw)
            if ax:
                ax = ax.lower()
                return Resolution(f"kb:{self.kb_by_arxiv[ax]}" if ax in self.kb_by_arxiv else f"arxiv:{ax}",
                                  "oai_snapshot", title, year)
        if title:
            hit = self.kb_by_title.get(norm(title))
            if hit and len(hit) == 1:
                return Resolution(f"kb:{next(iter(hit))}", "kb_title", title, year)
        # KB title contained verbatim in the entry (robust to a mis-split title); titles of >= 5 words only
        rn = " " + norm(raw) + " "
        found = {pid for w4, items in self._kb_prefix_hits(rn) for t, pid in items if f" {t} " in rn}
        if len(found) == 1:
            return Resolution(f"kb:{found.pop()}", "kb_title_in_entry", title, year)
        return Resolution(None, "stub", title, year)

    def _kb_prefix_hits(self, rn: str):
        toks = rn.split()
        for i in range(len(toks) - 3):
            k = " ".join(toks[i:i + 4])
            if k in self.kb_prefix4:
                yield k, self.kb_prefix4[k]
