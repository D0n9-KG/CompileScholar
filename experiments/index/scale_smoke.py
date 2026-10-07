# -*- coding: utf-8 -*-
"""Index scale smoke (§10.5 / §12 D-gate): 100k synthetic statements + 50k papers + 50k passages through the
REAL stage builds (index/build.py), then the query-time budget measurements the tools must live under
(§9.2: a single tool call < 200 ms — the index part of it).

Synthetic on purpose: deterministic word-pool text, hash-bag fake embeddings (dim configurable), no LLM, no
network. Everything runs in a scratch CS_DATA (runs/scale_smoke_index/data) — the real derived store is
untouched. Upstream stages are stood up as bare sqlite files + manifests (the pattern the tests use); the
index build itself is the real thing (work passes, tantivy, vector export).

Measured and printed:
  build      per-pass wall time, on-disk index sizes
  queries    papers / passages / statements (BM25+dense fusion) x 100 queries: median / p95 ms
  dense      pure vector search at full and half prefix (the as_of slice) : median / p95 ms
  filter     range_query really filters (an early as_of sees nothing late-dated)

Usage:  python experiments/index/scale_smoke.py [--statements 100000] [--dim 1024] [--keep]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRATCH = REPO / "runs" / "scale_smoke_index"

WORDS = ("graph attention transformer diffusion retrieval reinforcement optimization convergence bound "
         "embedding latent sparse dense robust efficient scalable adaptive hierarchical contrastive generative "
         "discriminative supervised unsupervised transfer federated quantum molecular protein image speech text "
         "knowledge reasoning planning evidence benchmark evaluation metric baseline ablation").split()
FACETS = ("contribution", "method", "result", "setting", "limitation", "categorization")


def fake_embed_factory(dim: int):
    def embed(texts):
        import numpy as np
        out = []
        for t in texts:
            v = np.zeros(dim, dtype="float32")
            for w in t.lower().split():
                v[int(hashlib.md5(w.strip(".,;:()")[:24].encode()).hexdigest(), 16) % dim] += 1.0
            n = float(np.linalg.norm(v))
            out.append((v / n).tolist() if n else v.tolist())
        return out
    return embed


def gen_text(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def stand_up_stores(n_papers: int, n_stmts: int, n_passages: int, rng: random.Random):
    from compilescholar.dfc import store
    (store.root() / "manifests").mkdir(parents=True, exist_ok=True)

    # registry: n_papers active papers spread over 2019-2025, every 10th with a doc
    from compilescholar.library import store as LS
    reg = LS.connect()
    now = "2026-10-07T00:00:00"
    rows, recs = [], []
    for i in range(n_papers):
        pid = f"arxiv:{2000 + i // 5000:04d}.{i % 5000:05d}"
        y = 2019 + (i * 7919 // n_papers) % 7
        m = 1 + (i * 104729 // n_papers) % 12
        d = 1 + (i * 15485863 // n_papers) % 28
        hi = f"{y}-{m:02d}-{d:02d}"
        rows.append((pid, "active", None, f"Paper {i} " + gen_text(rng, 6), hi, hi, "day", "test", "arxiv_v", now))
        recs.append((pid, "arxiv", 1, hi, f"Paper {i} " + gen_text(rng, 6), f"paper{i}key",
                     gen_text(rng, 60), "cs.LG", None))
    reg.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    reg.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)", recs)
    reg.commit()
    pids = [r[0] for r in rows]
    reg.close()

    # documents: n_papers/10 papers with a fast doc of ~n_passages/(n_papers/10) sentences each
    import zlib
    from compilescholar.documents.build import DDL as DOC_DDL
    docs = store.connect("documents")
    docs.executescript(DOC_DDL)
    per = max(1, n_passages // max(1, n_papers // 10))
    drows = []
    for j in range(n_papers // 10):
        pid = pids[j]
        hi = rows[j][4]
        sents = [{"sid": f"{pid}@v1#s{t}", "unit": "u1", "text": gen_text(rng, 25)} for t in range(per)]
        fast = {"title": rows[j][3], "abstract": "", "units": [
            {"uid": "u1", "kind": "para", "section": "Introduction", "text": " ".join(s["text"] for s in sents)}],
            "sentences": sents, "entries": {}, "cites": []}
        drows.append((f"{pid}@v1", pid, 1, f"sha{j}", "arxiv_nas", hi, "day", 1, len(sents), 0, 0, 0,
                      zlib.compress(json.dumps(fast).encode()), None))
    docs.executemany("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", drows)
    docs.commit()
    docs.close()

    from compilescholar.citations.build import DDL as CIT_DDL
    cit = store.connect("citations")
    cit.executescript(CIT_DDL)
    cit.commit()
    cit.close()

    # extract: n_stmts statements over the doc papers (self, dated at their paper's date)
    from compilescholar.extract.schema import DDL as EXT_DDL
    from compilescholar.extract.build import EXTRA_DDL
    ext = store.connect("extract")
    ext.executescript(EXT_DDL + EXTRA_DDL)
    hi_by = {r[0]: r[4] for r in rows}
    batch = []
    for i in range(n_stmts):
        pid = pids[(i * 7919) % len(pids)]
        f = FACETS[i % len(FACETS)]
        text = gen_text(rng, 18)
        batch.append((pid, hi_by[pid], "self", pid, "describes", f, text, text + " quote", "stated", "",
                      json.dumps({"unit_id": "u1", "sent_id": f"{pid}@v1#s{i % per}"}), None, "[]", None,
                      "{}", 2, "t2", pid, "scale", "fake", "sha"))
        if len(batch) >= 20000:
            ext.executemany("INSERT INTO statements(speaker,date,kind,about,role,facet,text,quote,epistemic,"
                            "condition,loc,target,grp,function,meta,schema_version,pass,item,run_id,model,"
                            "prompt_sha) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
            batch = []
    if batch:
        ext.executemany("INSERT INTO statements(speaker,date,kind,about,role,facet,text,quote,epistemic,"
                        "condition,loc,target,grp,function,meta,schema_version,pass,item,run_id,model,"
                        "prompt_sha) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
    ext.executemany("INSERT INTO scope VALUES (?,?)", [(p, 1) for p in pids[: n_papers // 5]])
    ext.executescript(store.WORK_DDL)
    ext.executemany("INSERT INTO _work VALUES (?,?,?,?,?,?,?)",
                    [("t2", p, "k", "ok", 0, None, now) for p in pids])
    ext.commit()
    ext.close()

    from compilescholar.cognition.build import DDL as COG_DDL
    cog = store.connect("cognition")
    cog.executescript(COG_DDL)
    cog.commit()
    cog.close()
    for s in ("documents", "citations", "extract", "cognition"):
        store.write_manifest(s, {}, {})


def timed(fn, n=100):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), sorted(ts)[int(len(ts) * 0.95)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--statements", type=int, default=100_000)
    ap.add_argument("--papers", type=int, default=50_000)
    ap.add_argument("--passages", type=int, default=50_000)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keep", action="store_true", help="keep the scratch store for inspection")
    a = ap.parse_args()

    shutil.rmtree(SCRATCH, ignore_errors=True)
    (SCRATCH / "data").mkdir(parents=True)
    os.environ["CS_DATA"] = str(SCRATCH / "data")
    sys.path.insert(0, str(REPO / "src"))
    from compilescholar.core import paths
    importlib.reload(paths)
    from compilescholar.dfc import store
    importlib.reload(store)
    from compilescholar.library import store as LS
    importlib.reload(LS)
    from compilescholar.index import build as IB
    importlib.reload(IB)
    from compilescholar.index import search as IS
    importlib.reload(IS)

    rng = random.Random(a.seed)
    t0 = time.perf_counter()
    stand_up_stores(a.papers, a.statements, a.passages, rng)
    print(f"[gen] synthetic stores: {a.papers:,} papers / {a.statements:,} statements in "
          f"{time.perf_counter() - t0:.0f}s")

    embed = fake_embed_factory(a.dim)
    t0 = time.perf_counter()
    counts = IB.build(dense=("papers", "statements"), embed=embed, model_label=f"fake-{a.dim}",
                      log=lambda *x: None)
    t_build = time.perf_counter() - t0
    def _mb(p):
        return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 1)
    sizes = {**{f"tv_{n}": _mb(IS.index_dir() / n) for n in ("papers", "passages", "statements")},
             **{f"vec_{n}": _mb(IS.vectors_dir(n)) for n in ("papers", "statements")}}
    print(f"[build] {t_build:.0f}s  counts={counts}  sizes_mb={sizes}")

    ix = IS.Index(embed=embed, model_label=f"fake-{a.dim}")
    queries = [gen_text(rng, 6) for _ in range(100)]
    for name, fn in (("papers", lambda: ix.papers(queries[rng.randrange(100)], "2026-01-01", 20)),
                     ("passages", lambda: ix.passages(queries[rng.randrange(100)], "2026-01-01", 10)),
                     ("statements", lambda: ix.statements(queries[rng.randrange(100)], "2026-01-01", 20,
                                                          per_paper=2))):
        med, p95 = timed(fn, 100)
        print(f"[query] {name:11s} median {med:6.1f} ms  p95 {p95:6.1f} ms   (budget: tool call < 200 ms)")

    import numpy as np
    v = ix._vecstore("statements")
    qv = embed([gen_text(rng, 6)])[0]
    for label, day in (("full prefix", 20260101), ("half prefix", 20220101)):
        med, p95 = timed(lambda: v.search(qv, day, 200), 50)
        nvis = int(np.searchsorted(v.days, day, "right"))
        print(f"[dense] statements {label:11s} (n={nvis:,}) median {med:6.1f} ms  p95 {p95:6.1f} ms")

    early = ix.statements(gen_text(rng, 6), "2019-01-31", 20)
    con = store.connect("extract", readonly=True)
    n_late = con.execute("SELECT count(*) FROM statements WHERE date > '2019-01-31'").fetchone()[0]
    for sid in early:
        d = con.execute("SELECT date FROM statements WHERE id=?", (sid,)).fetchone()
        assert d and d[0] <= "2019-01-31", f"as_of leak: statement {sid} dated {d}"
    con.close()
    print(f"[filter] range filter holds: {n_late:,} statements dated after 2019-01-31, none returned "
          f"({len(early)} hits, all <= cutoff)")

    out = {"counts": counts, "build_s": round(t_build, 1), "sizes_mb": sizes, "dim": a.dim,
           "statements": a.statements, "papers": a.papers}
    (SCRATCH / "summary.json").write_text(json.dumps(out, indent=1))
    ix.close()
    if not a.keep:
        os.environ.pop("CS_DATA", None)
        shutil.rmtree(SCRATCH / "data", ignore_errors=True)
    print(f"[done] summary at {SCRATCH / 'summary.json'}")


if __name__ == "__main__":
    main()
