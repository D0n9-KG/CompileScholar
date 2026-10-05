# -*- coding: utf-8 -*-
"""Harvest arXiv OAI-PMH (set=cs, arXivRaw) for every datestamp from START to today.

Every paper first submitted on/after START has a datestamp >= its v1 date, so this window catches all of them (plus
older papers updated in the window, which are harmless). Papers before START come from the local OAI snapshot
(corpus.papers builds the union). Resumable: finished windows are skipped."""
import sys
from datetime import date

from compilescholar.sources import arxiv_oai

START = date(2024, 3, 1)  # overlaps the local snapshot's end (2404.03658) by a month


def main():
    n = arxiv_oai.harvest(START, date.today(), days=7, log=lambda s: print(s, flush=True))
    print(f"[oai] done: {n} records written in this run", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
