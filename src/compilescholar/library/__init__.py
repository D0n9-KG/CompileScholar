# -*- coding: utf-8 -*-
"""The library: the authoritative registry of papers (INTEGRATED-SYSTEM-1005 §1, §3), data/library/registry.sqlite.

Holds what cannot be rebuilt from a stage: paper identity (paper_id, never reassigned), identifiers, every source's
dates per version, per-source metadata records, author lists, benchmark membership, merge aliases and the merge queue.
Derived stages key on paper_id. Imports are bulk and idempotent (library.import_arxiv, library.import_benchmarks);
identity decisions that need judgement go to the merge queue instead of being made at write time."""
