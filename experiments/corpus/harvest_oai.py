# -*- coding: utf-8 -*-
"""Harvest arXiv OAI-PMH (arXivRaw) for every datestamp from START to today, for the given sets (default cs).

Every paper first submitted on/after START has a datestamp >= its v1 date, so this window catches all of them (plus
older papers updated in the window, which are harmless). Papers before START come from the local OAI snapshot
(library.import_arxiv builds the union). Resumable: finished windows are skipped. Sets are harvested one after the
other (one client, OAI-PMH flow control).
Usage: python harvest_oai.py [set ...]      e.g. physics math eess stat q-bio q-fin econ"""
import sys
from datetime import date

from compilescholar.sources import arxiv_oai

START = date(2024, 3, 1)  # overlaps the local snapshot's end (2404.03658) by a month


def main(sets):
    for s in sets:
        n = arxiv_oai.harvest(START, date.today(), days=7, set_spec=s, log=lambda m, s=s: print(f"[{s}] {m}", flush=True))
        print(f"[oai] {s} done: {n} records written in this run", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1:] or ["cs"])
