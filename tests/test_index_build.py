# -*- coding: utf-8 -*-
"""Phase D③: the index stage — three tantivy indexes with as_of fast-field filtering, date-sorted memmap
vectors (prefix-slice as_of, model-label header), RRF fusion with the per-paper cap applied to the FUSED list
(the §9.1 bug fix: a dense-only hit can no longer bypass the cap), and incremental re-indexing (work keys +
sweeps). Synthetic upstream stages, no network, deterministic fake embeddings."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import zlib
from collections import Counter

import numpy as np
import pytest

PA = "arxiv:1901.00001"       # active 2019-01-01; docs v1 (2019-06-01) + v2 (2020-06-01) + a delta sentence
PB = "arxiv:2001.00002"       # active 2020-01-15; docs v1 (2020-06-01)
PC = "arxiv:1901.00009"       # merged into PA — must never surface
PD = "title:year paper|2021"  # active, year precision: visible only at its hi (2021-12-31), no docs

DIM = 32


def fake_embed(texts):
    """Deterministic bag-of-words vectors: every lowercase word hashes to one dimension (stopwords included —
    the BM25 side drops them, which is exactly what the dense-only cap test needs)."""
    out = []
    for t in texts:
        v = np.zeros(DIM, dtype="float32")
        for w in re.findall(r"[a-z]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        n = float(np.linalg.norm(v))
        out.append((v / n).tolist() if n else v.tolist())
    return out


class CountingEmbed:
    def __init__(self):
        self.n = 0

    def __call__(self, texts):
        self.n += len(texts)
        return fake_embed(texts)


def _fastz(sent_pairs, section="Introduction"):
    d = {"title": "t", "abstract": "",
         "units": [{"uid": "u1", "kind": "para", "section": section,
                    "text": " ".join(t for _, t in sent_pairs)}],
         "sentences": [{"sid": s, "unit": "u1", "text": t} for s, t in sent_pairs],
         "entries": {}, "cites": []}
    return zlib.compress(json.dumps(d).encode())


def _doc(docs, key, pid, ver, date, sent_pairs):
    docs.execute("INSERT INTO docs(key,paper_id,version,sha256,source,text_date,date_precision,n_units,"
                 "n_sentences,n_entries,n_cites,n_tables,fast_z,careful_z) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 (key, pid, ver, "sha-" + key, "arxiv_nas", date, "day", 1, len(sent_pairs), 0, 0, 0,
                  _fastz(sent_pairs), None))


def _stmt(ext, i, speaker, date, kind, about, role, facet, text, item, pass_):
    ext.execute("INSERT INTO statements(id,speaker,date,kind,about,role,facet,text,quote,epistemic,condition,"
                "loc,meta,schema_version,pass,item) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (i, speaker, date, kind, about, role, facet, text, f"quote {i}", "stated", "",
                 json.dumps({"unit_id": "u1", "sent_id": f"s{i}"}), "{}", 2, pass_, item))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CS_DATA", str(tmp_path / "data"))
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

    reg = LS.connect()
    now = "2026-10-07T00:00:00"
    for pid, st, hi, pr in [(PA, "active", "2019-01-01", "day"), (PB, "active", "2020-01-15", "day"),
                            (PC, "merged", "2019-02-01", "day"), (PD, "active", "2021-12-31", "year")]:
        reg.execute("INSERT INTO papers(paper_id, status, title, first_hi, first_precision, first_source, "
                    "first_kind, created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (pid, st, f"Paper {pid}", hi, pr, "test", "arxiv_v", now))
    reg.execute("INSERT INTO aliases VALUES (?,?,?,?)", (PC, PA, "test", now))
    for pid, t, tk, ab in [(PA, "GraphFormer: attention for graphs", "graphformerattentionforgraphs",
                            "We propose GraphFormer, an attention model for graphs."),
                           (PB, "Faster graph transformers", "fastergraphtransformers",
                            "We propose FastGF, an efficient graph transformer."),
                           (PD, "Year paper on graphs", "yearpaperongraphs",
                            "A year-precision paper about graphs.")]:
        reg.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)", (pid, "arxiv", 1, None, t, tk, ab, "cs.LG", ""))
    reg.commit()
    reg.close()

    from compilescholar.documents.build import DDL as DOC_DDL
    docs = store.connect("documents")
    docs.executescript(DOC_DDL)
    _doc(docs, f"{PA}@v1", PA, 1, "2019-06-01",
         [(f"{PA}@v1#s1", "GraphFormer applies attention to graphs"),
          (f"{PA}@v1#s2", "Experiments on graphs show consistent gains")])
    _doc(docs, f"{PA}@v2", PA, 2, "2020-06-01",
         [(f"{PA}@v2#s1", "GraphFormer applies attention to graphs"),
          (f"{PA}@v2#s2", "Experiments on graphs show consistent gains")])
    dz = zlib.compress(json.dumps({"sentences": [{"sid": f"{PA}@v2#s9", "unit": None,
                                                 "text": "The appendix proves a convergence bound for the "
                                                         "attention layer"}],
                                   "revised": [], "entries": [], "cites": []}).encode())
    docs.execute("INSERT INTO deltas VALUES (?,?,?,?,?,?,?,?)", (PA, f"{PA}@v1", f"{PA}@v2", 1, 0, 0, 0, dz))
    _doc(docs, f"{PB}@v1", PB, 1, "2020-06-01", [(f"{PB}@v1#s1", "GraphNet improves graphs steadily")])
    docs.commit()
    docs.close()

    from compilescholar.citations.build import DDL as CIT_DDL
    cit = store.connect("citations")
    cit.executescript(CIT_DDL)
    cit.commit()
    cit.close()

    from compilescholar.extract.schema import DDL as EXT_DDL
    from compilescholar.extract.build import EXTRA_DDL
    ext = store.connect("extract")
    ext.executescript(EXT_DDL + EXTRA_DDL)
    _stmt(ext, 1, PB, "2020-06-01", "other", PA, "compares", "result", "B beats A on graphs", PB, "other")
    _stmt(ext, 2, PA, "2019-06-01", "self", PA, "proposes", "contribution", "PA proposes FastGF for graphs",
          PA, "t1")
    _stmt(ext, 3, PA, "2019-06-01", "self", PA, "describes", "method", "alpha variant of the graph method",
          PA, "t2")
    _stmt(ext, 4, PA, "2019-06-01", "self", PA, "describes", "method", "beta variant of the graph method",
          PA, "t2")
    _stmt(ext, 5, PA, "2021-06-01", "self", PA, "describes", "method", "gamma variant of the graph method",
          PA, "t2")
    _stmt(ext, 6, PB, "2020-06-01", "self", PB, "proposes", "contribution", "GraphNet of the new era",
          PB, "t1")
    ext.executemany("INSERT INTO scope VALUES (?,?)", [(PA, 1), (PB, 1)])
    ext.executescript(store.WORK_DDL)
    ext.executemany("INSERT INTO _work VALUES (?,?,?,?,?,?,?)",
                    [("t1", PA, "k1", "ok", 0, None, now), ("t2", PA, "k2", "ok", 0, None, now),
                     ("t1", PB, "k3", "ok", 0, None, now), ("other", PB, "k4", "ok", 0, None, now)])
    ext.commit()
    ext.close()

    for s in ("documents", "citations", "extract", "cognition"):
        store.write_manifest(s, {}, {})
    return IB, IS, store


def _build(IB):
    return IB.build(dense=("papers", "statements"), embed=fake_embed, model_label="fake")


def test_build_counts(env):
    IB, IS, store = env
    c = _build(IB)
    assert c["papers"] == 3 and c["passages"] == 4 and c["statements"] == 6
    assert c["vec_papers"] == 2 and c["vec_statements"] == 5     # scope papers; self statements only
    assert store.read_manifest("index")["complete"]
    ix = IS.Index(embed=fake_embed, model_label="fake")
    got = ix.papers("graphformer attention graphs", as_of="2019-06-01", k=10)
    assert PA in got and PB not in got and PD not in got and PC not in got


def test_papers_asof_precision_and_tiers(env):
    IB, IS, _ = env
    _build(IB)
    ix = IS.Index(embed=fake_embed, model_label="fake")
    assert PD not in ix.papers("year paper graphs", as_of="2021-06-01")     # year precision: visible at hi only
    assert PD in ix.papers("year paper graphs", as_of="2021-12-31")
    t2 = ix.papers("graphs", as_of="2022-01-01", tier="t2", k=10)
    assert PA in t2 and PB in t2 and PD not in t2                            # PD has no parsed full text
    ft = ix.papers("graphs", as_of="2022-01-01", tier="ft", k=10)
    assert ft == []                                                          # nothing careful/sv in the fixture


def test_passages_dates_and_cap(env):
    IB, IS, _ = env
    _build(IB)
    ix = IS.Index(embed=fake_embed, model_label="fake")
    early = ix.passages("convergence bound appendix", as_of="2019-12-31", k=5)
    assert not any("convergence" in h["text"] for h in early)                # the delta is dated 2020-06-01
    late = ix.passages("convergence bound appendix", as_of="2020-12-31", k=5)
    assert any(h["paper"] == PA and h["date"] == "2020-06-01" and h["kind"] == "delta" for h in late)
    v1 = ix.passages("attention graphs", as_of="2019-12-31", k=5)
    assert v1 and all(h["date"] == "2019-06-01" for h in v1 if h["paper"] == PA)
    assert v1[0]["section"] == "Introduction" and v1[0]["uid"]
    capped = ix.passages("graphs", as_of="2022-01-01", k=10, per_paper=1)
    assert max(Counter(h["paper"] for h in capped).values()) == 1
    only_pa = ix.passages("graphs", as_of="2022-01-01", k=10, paper=PA)
    assert only_pa and all(h["paper"] == PA for h in only_pa)


def test_statements_filters_and_asof(env):
    IB, IS, _ = env
    _build(IB)
    ix = IS.Index(embed=fake_embed, model_label="fake")
    ids = ix.statements("beats graphs", as_of="2022-01-01", kind="other", k=10)
    assert ids and set(ids) == {1}
    ids = ix.statements("proposes FastGF", as_of="2022-01-01", facet="contribution", k=10)
    assert ids[0] == 2
    assert ix.statements("proposes FastGF", as_of="2019-01-01") == []        # every statement is later
    ids = ix.statements("variant graph method", as_of="2022-01-01", about=PA, k=10)
    assert set(ids) <= {3, 4, 5} and ids


def test_per_paper_cap_covers_dense_only_hits(env):
    """§9.1 regression: 'of the' is pure stopwords — BM25 finds nothing, every hit comes from the vector side;
    the cap must still apply (the old code let dense-only hits bypass it because `about` was unknown)."""
    IB, IS, _ = env
    _build(IB)
    ix = IS.Index(embed=fake_embed, model_label="fake")
    ids = ix.statements("of the", as_of="2022-01-01", k=10, per_paper=2)
    assert len([i for i in ids if i in (3, 4, 5)]) == 2       # PA capped at 2 of its 3
    assert len([i for i in ids if i == 6]) == 1                # PB keeps its 1
    assert 1 not in ids and 2 not in ids                       # other-kind / zero-similarity stay out


def test_vectors_as_of_is_a_prefix_slice(env):
    IB, IS, _ = env
    _build(IB)
    ix = IS.Index(embed=fake_embed, model_label="fake")
    ids = ix.statements("of the", as_of="2020-01-01", k=10, per_paper=5)
    assert 5 not in ids and 6 not in ids                       # dated 2021-06 / 2020-06: after the cutoff
    assert {3, 4} <= set(ids) <= {2, 3, 4}                     # only the 2019 rows are in the prefix slice


def test_vector_model_label_mismatch_raises(env):
    IB, IS, _ = env
    _build(IB)
    bad = IS.Index(embed=fake_embed, model_label="another-model")
    with pytest.raises(RuntimeError, match="embedding model"):
        bad.statements("of the", as_of="2022-01-01")


def test_incremental_reindex_and_sweep(env):
    IB, IS, store = env
    emb = CountingEmbed()
    c1 = IB.build(dense=("papers", "statements"), embed=emb, model_label="fake")
    assert c1["statements"] == 6 and c1["vec_statements"] == 5

    emb.n = 0
    c2 = IB.build(dense=("papers", "statements"), embed=emb, model_label="fake")
    assert emb.n == 0 and c2 == c1                             # nothing changed: nothing re-embedded

    ext = store.connect("extract")                             # extract re-runs item PA and one statement joins
    ext.execute("UPDATE _work SET key='k2b' WHERE pass='t2' AND item=?", (PA,))
    _stmt(ext, 7, PA, "2021-08-01", "self", PA, "describes", "method", "delta of the graph variant seven",
          PA, "t2")
    ext.commit()
    ext.close()
    emb.n = 0
    c3 = IB.build(dense=("papers", "statements"), embed=emb, model_label="fake")
    assert c3["statements"] == 7
    assert emb.n == 5                                          # only PA's self statements re-embed (2,3,4,5,7)
    ix = IS.Index(embed=fake_embed, model_label="fake")
    got = set(ix.statements("of the", as_of="2022-01-01", k=10, per_paper=99))
    assert {3, 4, 5, 6, 7} <= got <= {2, 3, 4, 5, 6, 7}        # 7 re-indexed; 2 may ride a hash-collision sim
    ix.close()                                                 # Windows: an open memmap locks the vector files

    ext = store.connect("extract")                             # PB's statements disappear upstream
    ext.execute("DELETE FROM statements WHERE item=?", (PB,))
    ext.execute("DELETE FROM _work WHERE item=?", (PB,))
    ext.commit()
    ext.close()
    c4 = IB.build(dense=("papers", "statements"), embed=emb, model_label="fake")
    assert c4["statements"] == 5 and c4["vec_statements"] == 5  # PA's 2,3,4,5,7 — PB's vector row swept
    ix = IS.Index(embed=fake_embed, model_label="fake")
    assert 6 not in ix.statements("of the", as_of="2022-01-01", k=10, per_paper=99)
