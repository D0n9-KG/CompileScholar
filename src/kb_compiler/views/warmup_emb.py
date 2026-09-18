# -*- coding: utf-8 -*-
"""A5: build-chain embedding-cache warmup (2026-09-18 batch 1).

The record-search embedding cache is built lazily on the first `search`
call — for an 18.9k-record corpus that is a ~25-minute CST cold start paid
by the first question (or, in agent-integration terms, by the user's first
query after every rebuild). Warmup belongs at the END of the build chain,
not inside the query path.

This module constructs the KB exactly as the answering runner does (same
records file, same cache path) and pre-builds the on-disk cache. The
fingerprint discipline (kb_compiler/views/tools.py IL-B6) makes this safe:
the runner's first _ensure_embeddings() finds a matching fingerprint and
loads the cache instead of rebuilding.

Usage:
  py -3.13 -m kb_compiler.views.warmup_emb --records REC.json \
      --views VIEWS.json --registry REG.json --vocab V.json \
      --manifest M.json --emb-cache path/emb_cache.bin
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def warmup(records_path, views_path, registry_path, vocab_path,
            manifest_path, emb_cache_path):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from kb_compiler.views.tools import KBTools

    records = json.load(open(records_path, encoding="utf-8"))
    records.pop("canary", None)
    views = json.load(open(views_path, encoding="utf-8"))
    registry = json.load(open(registry_path, encoding="utf-8"))
    vocab = json.load(open(vocab_path, encoding="utf-8"))
    manifest = {r["paper_id"]: r for r in
                json.load(open(manifest_path, encoding="utf-8"))}

    kb = KBTools(views, registry, vocab, manifest, records,
                 emb_cache_path=emb_cache_path)
    t0 = time.time()
    kb._ensure_embeddings()
    dt = time.time() - t0
    n = len(kb._emb_cache or [])
    provider = kb._emb_provider
    size = os.path.getsize(emb_cache_path) if os.path.exists(emb_cache_path) else 0
    print(f"emb cache warm: {n} records, provider={provider}, "
          f"{dt/60:.1f} min, {size/1e6:.1f} MB -> {emb_cache_path}")
    assert n > 0 and size > 0, "warmup produced no cache"
    return n, dt, provider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--emb-cache", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warmup(args.records, args.views, args.registry, args.vocab,
           args.manifest, args.emb_cache)


if __name__ == "__main__":
    main()
