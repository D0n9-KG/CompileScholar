# -*- coding: utf-8 -*-
"""How the costs of adding papers and of retrieval grow with the library (measured on a read-only copy of the real
registry; nothing is written to it).

Part 1, adding papers (cross-paper links of a new paper):
  - identifier point lookup (resolve a DOI / arXiv id from a reference entry) at the current size;
  - title-key lookup (records.title_key index) — the fallback for entries without ids;
  - recompute_first_public over the whole papers table (today run after every import; cost grows with N);
  - propose_title_matches cost (one GROUP BY over records.title_key; grows with N).
Part 2, retrieval:
  - tantivy BM25 over N title+abstract documents with a date filter, N = 100k, 300k, 1M (top 20);
  - brute-force binary-code vector scan (dim 1024 -> 128 bytes / vector) over N = 100k, 1M, 3M, top 100 by Hamming,
    then float re-rank of the 100 (the INTEGRATED-SYSTEM-1005 §9.1 plan).
Output: printed table; runs/scale_probe/ holds the scratch tantivy indexes (deleted at the end)."""
from __future__ import annotations

import random
import shutil
import sqlite3
import statistics
import time

import numpy as np

from compilescholar.core import ids, paths

OUT = paths.runs() / "scale_probe"


def timed(fn, n=200):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000, sorted(ts)[int(len(ts) * 0.95)] * 1000


def part1():
    src = paths.library() / "registry.sqlite"
    con = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
    n_papers = con.execute("SELECT count(*) FROM papers").fetchone()[0]
    print(f"registry: {n_papers:,} papers")
    rng = random.Random(1)
    dois = [v for (v,) in con.execute("SELECT value FROM identifiers WHERE scheme='doi' LIMIT 50000")]
    axs = [v for (v,) in con.execute("SELECT value FROM identifiers WHERE scheme='arxiv' LIMIT 200000")]
    keys = [k for (k,) in con.execute("SELECT title_key FROM records WHERE source='arxiv' LIMIT 200000")]
    q = "SELECT paper_id FROM identifiers WHERE scheme=? AND value=?"
    print("  doi point lookup     median %.3f ms  p95 %.3f ms" % timed(lambda: con.execute(q, ("doi", rng.choice(dois))).fetchone()))
    print("  arxiv point lookup   median %.3f ms  p95 %.3f ms" % timed(lambda: con.execute(q, ("arxiv", rng.choice(axs))).fetchone()))
    tq = "SELECT paper_id FROM records WHERE title_key=?"
    print("  title_key lookup     median %.3f ms  p95 %.3f ms" % timed(lambda: con.execute(tq, (rng.choice(keys),)).fetchall()))
    # a new paper's 45 references, all resolved: 45 x (2 point lookups + 1 title lookup)
    t = time.perf_counter()
    for _ in range(45):
        con.execute(q, ("doi", rng.choice(dois))).fetchone()
        con.execute(q, ("arxiv", rng.choice(axs))).fetchone()
        con.execute(tq, (rng.choice(keys),)).fetchall()
    print("  one paper, 45 references resolved: %.1f ms" % ((time.perf_counter() - t) * 1000))
    # whole-table recompute and title proposal, on a copy on disk (the registry itself stays untouched)
    OUT.mkdir(parents=True, exist_ok=True)
    cp = OUT / "registry_copy.sqlite"
    t = time.perf_counter()
    disk = sqlite3.connect(cp)
    con.backup(disk)
    disk.close()
    mem = sqlite3.connect(cp)
    print("  copy %.0f s" % (time.perf_counter() - t))
    from compilescholar.library import identity
    t = time.perf_counter()
    identity.recompute_first_public(mem)
    print("  recompute_first_public over all papers: %.1f s" % (time.perf_counter() - t))
    t = time.perf_counter()
    n = mem.execute("SELECT count(*) FROM (SELECT title_key FROM records WHERE length(title_key) >= 12 "
                    "GROUP BY title_key HAVING count(DISTINCT paper_id) > 1)").fetchone()[0]
    print("  title-key duplicate groups (the propose step's scan): %d groups in %.1f s" % (n, time.perf_counter() - t))
    rows = mem.execute("SELECT title, abstract, date_hi FROM records WHERE source='arxiv' LIMIT 1000000").fetchall()
    mem.close()
    con.close()
    return rows


def part2_text(rows):
    import tantivy
    OUT.mkdir(parents=True, exist_ok=True)
    sb = tantivy.SchemaBuilder()
    sb.add_text_field("text", stored=False, tokenizer_name="en_stem")
    sb.add_integer_field("day", indexed=True, fast=True)
    sb.add_integer_field("i", stored=True)
    schema = sb.build()
    queries = ["graph neural networks for molecule property prediction", "contrastive learning of visual representations",
               "reinforcement learning from human feedback reward model", "diffusion models for image editing",
               "federated learning communication efficiency", "retrieval augmented generation for question answering",
               "adversarial robustness certified defense", "speech recognition low resource languages",
               "large language model reasoning chain of thought", "point cloud segmentation transformer"]
    for n in (100_000, 300_000, 1_000_000):
        if n > len(rows):
            break
        d = OUT / f"tv{n}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        idx = tantivy.Index(schema, path=str(d))
        w = idx.writer(heap_size=500_000_000)
        t = time.perf_counter()
        for i, (ti, ab, day) in enumerate(rows[:n]):
            w.add_document(tantivy.Document(text=f"{ti} {ab}", day=int((day or "2000-01-01").replace("-", "")), i=i))
        w.commit()
        idx.reload()
        build = time.perf_counter() - t
        s = idx.searcher()
        rng = random.Random(2)

        def one():
            qq = idx.parse_query(rng.choice(queries), ["text"])
            rq = tantivy.Query.range_query(schema, "day", tantivy.FieldType.Integer, 0, 20230601)
            s.search(tantivy.Query.boolean_query([(tantivy.Occur.Must, qq), (tantivy.Occur.Must, rq)]), 20)
        med, p95 = timed(one, 100)
        print(f"  tantivy N={n:>9,}: build {build:5.0f} s, query + date filter top20 median {med:.1f} ms  p95 {p95:.1f} ms")
    shutil.rmtree(OUT, ignore_errors=True)


def part2_vectors():
    rng = np.random.default_rng(3)
    for n in (100_000, 1_000_000, 3_000_000):
        codes = rng.integers(0, 256, size=(n, 128), dtype=np.uint8)          # 1024-bit binary codes
        q = rng.integers(0, 256, size=128, dtype=np.uint8)
        floats = rng.standard_normal((100, 1024), dtype=np.float32)

        def one():
            ham = np.bitwise_count(np.bitwise_xor(codes, q)).sum(axis=1, dtype=np.uint16)
            top = np.argpartition(ham, 100)[:100]
            _ = floats @ rng.standard_normal(1024, dtype=np.float32)          # float re-rank of the 100
            return top
        med, p95 = timed(one, 20)
        print(f"  binary scan N={n:>9,} ({n * 128 / 1e9:.2f} GB codes): top100 + re-rank median {med:.0f} ms  p95 {p95:.0f} ms")
        del codes


if __name__ == "__main__":
    print("part 1: adding papers")
    rows = part1()
    print("part 2: retrieval")
    part2_text(rows)
    del rows
    part2_vectors()
