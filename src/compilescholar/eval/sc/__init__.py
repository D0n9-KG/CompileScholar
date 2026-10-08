# -*- coding: utf-8 -*-
"""ScholarCatalyst (SC) evaluation: the slot-1 agentic retrieval runner.

SC (arXiv 2610.02202, Stanford IRIS): 894 queries (core_query 207 / subfield_query 687), each a research-direction
description generated from one 2025+ source paper; the task is to rank the prior papers of a closed 190,896-doc
pool that could have inspired it. Official scoring reads runs/<model>/<query_type>.jsonl.

Modules:
  protocol  SC data access (read-only) + the deterministic submission rules (temporal filter, source-paper
            withholding, agent/trajectory/backfill assembly) mirroring the official retrieve.py / ranking.py
  backends  retrieval backends: SystemSearch (our L6 tools over the as-of index) and the bm25 backfill reader
  agent     one agent loop per query: plan -> search -> judge -> rank (local LLM, bounded call budget)
  runner    CLI orchestrator: per-query loop, resume, runs jsonl, hand-off to the official evaluate.py
  sample    stratified question-blind pilot sampling (type x domain, fixed seed)

Discipline: external retrieval is OFF (closed-pool protocol); queries/rels are never used for selection beyond
structural fields; every store is opened read-only; writes go to the shadow bench dir only.
"""
