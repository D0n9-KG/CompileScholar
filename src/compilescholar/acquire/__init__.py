# -*- coding: utf-8 -*-
"""Acquire full texts for registry papers (INTEGRATED-SYSTEM-1005 §4): pick channels by what the paper is (arXiv id /
DOI, field, year), fetch, check that the PDF is the paper, record the asset pointer or quarantine the PDF.

  channels.py   arxiv_nas (per-version mirror) · arxiv_gcs (public bucket) · scihub_local (DOI, internal use only) ·
                oa_pdf (OpenAlex best open-access PDF)
  verify.py     identity check: deterministic first (DOI / arXiv stamp on the first page, title + author surnames),
                LLM verdict for the cases in between
  run.py        acquire(paper_ids) -> assets / attempts rows; one writer, worker threads only fetch and verify"""
