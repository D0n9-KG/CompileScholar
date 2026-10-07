# -*- coding: utf-8 -*-
"""The read side of the index stage (INTEGRATED-SYSTEM-1005 §9.1, phase D③).

Three tantivy indexes (papers / passages / statements), every query filtered by as_of through a `day_hi` fast
field (range upper bound — a year-precision date was stored at its hi, so the comparison is exact and leak-safe).
Dense retrieval is date-sorted memmap vectors: a binary sign-code scan (Hamming) takes a coarse pool, float dot
products re-rank it; as_of is a prefix slice of the sorted rows. The vector header records the embedding model
label and dimension — a query embedded by another model raises instead of silently scoring garbage.

Fusion is Reciprocal Rank Fusion (k = 60). The old per-paper cap bug is fixed here: the cap is applied to the
FUSED list and every candidate knows its owner (BM25 hits carry `about` as a stored field, dense hits carry an
aligned owners array), so a dense-only hit can no longer bypass the cap.

Layout (all under <derived>/index/, built by index/build.py):
  papers/ passages/ statements/   tantivy index directories (schemas below — make_schema is shared with the build)
  vectors/<name>/                 float32.dat (n,dim normalised) codes.u8 (n,ceil(dim/8)) days.i32 (n,)
                                  keys.txt owners.txt [facets.u8] mean.f32 (dim,) meta.json {model,dim,n,...}
tantivy field notes: identifier/filter fields use tokenizer_name="raw" (the default tokenizer would split
"arxiv:2001.00001" into three tokens and delete_documents/term_query on the whole string would silently match
nothing — measured); text fields use "en_stem"; `day_hi` is indexed+fast+stored.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..core import paths
from ..core.asof import AsOf

RRF_K = 60
COARSE_FACTOR = 4          # Hamming pool = pool * COARSE_FACTOR before the float re-rank (room for filters)


def index_dir() -> Path:
    return paths.derived() / "index"


def vectors_dir(name: str, root: Path | None = None) -> Path:
    return (root or index_dir()) / "vectors" / name


def day_int(iso: str) -> int:
    """'2020-01-31...' -> 20200131 (lexicographic == chronological for full ISO days)."""
    return int(iso[:10].replace("-", ""))


def day_iso(d: int) -> str:
    return f"{d // 10000:04d}-{(d // 100) % 100:02d}-{d % 100:02d}"


def asof_int(T) -> int:
    """The range-query upper bound for a cutoff: AsOf normalisation (YYYY-MM-DD required; None = no cutoff)."""
    iso = AsOf(T).iso
    return day_int(iso) if iso else 99999999


def make_schema(name: str):
    """The one schema definition per index — build and read must agree byte for byte."""
    import tantivy
    sb = tantivy.SchemaBuilder()
    if name == "papers":
        sb.add_text_field("pid", stored=True, tokenizer_name="raw")
        sb.add_text_field("title", tokenizer_name="en_stem")
        sb.add_text_field("abstract", tokenizer_name="en_stem")
        sb.add_text_field("tier")                       # default tokenizer: tokens t0/t1/t2/ft, term-filterable
        sb.add_integer_field("day_hi", indexed=True, fast=True, stored=True)
    elif name == "passages":
        sb.add_text_field("pid", stored=True, tokenizer_name="raw")
        sb.add_text_field("uid", stored=True, tokenizer_name="raw")
        sb.add_text_field("section", stored=True)
        sb.add_text_field("kind", stored=True, tokenizer_name="raw")
        sb.add_text_field("body", stored=True, tokenizer_name="en_stem")
        sb.add_integer_field("day_hi", indexed=True, fast=True, stored=True)
    elif name == "statements":
        sb.add_integer_field("sid", indexed=True, stored=True)
        sb.add_text_field("item", tokenizer_name="raw")            # extract work item: the delete_documents target
        sb.add_text_field("speaker", tokenizer_name="raw")
        sb.add_text_field("about", stored=True, tokenizer_name="raw")
        sb.add_text_field("kind", tokenizer_name="raw")
        sb.add_text_field("facet", tokenizer_name="raw")
        sb.add_text_field("role", tokenizer_name="raw")
        sb.add_text_field("epistemic", tokenizer_name="raw")
        sb.add_text_field("body", tokenizer_name="en_stem")        # text + quote; rows are read from extract by sid
        sb.add_integer_field("day_hi", indexed=True, fast=True, stored=True)
    else:
        raise ValueError(name)
    return sb.build()


def rrf(*ranks: list) -> list:
    """Reciprocal Rank Fusion (k = 60) over key lists, best first; ties broken by key for determinism."""
    sc: dict = {}
    for r in ranks:
        for i, k in enumerate(r):
            sc[k] = sc.get(k, 0.0) + 1.0 / (RRF_K + i + 1)
    return [k for k, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))]


class VecStore:
    """One date-sorted vector set. Rows are (day_hi, rowkey) sorted, so as_of = a prefix slice; the coarse pass
    is a Hamming scan over per-dimension sign codes (v > column mean), the re-rank a float dot product
    (§9.1: recall@10 0.995 measured with a 200-candidate pool)."""

    def __init__(self, name: str, model_label: str, dir: Path | None = None):
        d = Path(dir) if dir else vectors_dir(name)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta["model"] != model_label:
            raise RuntimeError(f"vector store {name!r} was built with embedding model {meta['model']!r} but queries "
                               f"embed with {model_label!r} — rebuild the index or fix EMBEDDING_MODEL")
        self.name, self.n, self.dim = name, int(meta["n"]), int(meta["dim"])
        self.keys: list[str] = []
        self.owners: list[str] = []
        self.floats = self.codes = self.days = self.mean = self.facets = None
        if not self.n:
            return
        self.floats = np.memmap(d / "float32.dat", dtype="float32", mode="r", shape=(self.n, self.dim))
        self.codes = np.memmap(d / "codes.u8", dtype="uint8", mode="r", shape=(self.n, (self.dim + 7) // 8))
        self.days = np.fromfile(d / "days.i32", dtype="int32")
        self.mean = np.fromfile(d / "mean.f32", dtype="float32")
        self.keys = (d / "keys.txt").read_text(encoding="utf-8").splitlines()
        self.owners = (d / "owners.txt").read_text(encoding="utf-8").splitlines()
        if meta.get("has_facets"):
            self.facets = np.fromfile(d / "facets.u8", dtype="uint8")

    def search(self, qv, as_of_day: int, pool: int = 200, facets=None, owner: str | None = None) -> list[tuple]:
        """[(rowkey, owner, score)] best-first, all rows dated <= as_of_day. `facets`: allowed FACETS indices
        (statements store); `owner`: keep one owner only."""
        if not self.n:
            return []
        nvis = int(np.searchsorted(self.days, as_of_day, "right"))
        if nvis == 0:
            return []
        qv = np.asarray(qv, dtype="float32").ravel()
        nrm = float(np.linalg.norm(qv))
        if nrm == 0.0:
            return []
        qv = qv / nrm
        if qv.shape[0] != self.dim:
            raise RuntimeError(f"query dimension {qv.shape[0]} != store dimension {self.dim}")
        qc = np.packbits(qv > self.mean)
        ham = np.bitwise_count(np.bitwise_xor(self.codes[:nvis], qc)).sum(axis=1, dtype=np.uint32)
        masked = np.uint32(1 << 20)
        if facets is not None and self.facets is not None:
            ham = np.where(np.isin(self.facets[:nvis], np.asarray(list(facets), dtype=np.uint8)), ham, masked)
        take = min(pool * COARSE_FACTOR, nvis)
        idx = np.argpartition(ham, take - 1)[:take]
        idx = idx[ham[idx] < masked]
        if owner is not None:
            idx = idx[np.fromiter((self.owners[i] == owner for i in idx), bool, idx.size)]
        if idx.size == 0:
            return []
        sims = np.asarray(self.floats[:nvis][idx]) @ qv
        order = np.argsort(-sims)
        return [(self.keys[int(idx[j])], self.owners[int(idx[j])], float(sims[j]))
                for j in order[:pool] if sims[j] > 0]        # orthogonal-or-worse is no evidence


class Index:
    """The read API the tools use (D④). Every search takes as_of and never returns a row dated after it.
    `embed` is a callable(list[str]) -> list[vector] (default: the local embedding service); without it search
    is BM25-only. Opening is lazy and light: tantivy meta + memmaps, nothing preloaded (§9.2)."""

    def __init__(self, embed=None, model_label: str | None = None, dir: Path | None = None):
        self.dir = Path(dir) if dir else index_dir()
        self.embed = embed
        self.model_label = model_label
        self._ix: dict = {}
        self._vec: dict = {}

    # ---- opening
    def _label(self) -> str:
        if self.model_label is None:
            from ..llm.embedding import model_label
            self.model_label = model_label()
        return self.model_label

    def _open(self, name: str):
        if name not in self._ix:
            import tantivy
            d = self.dir / name
            if not (d / "meta.json").exists():
                raise RuntimeError(f"index {name!r} is not built (run `compilescholar build index`)")
            ix = tantivy.Index(make_schema(name), path=str(d))
            ix.reload()
            self._ix[name] = ix
        return self._ix[name]

    def _vecstore(self, name: str) -> VecStore | None:
        if name not in self._vec:
            d = vectors_dir(name, self.dir)
            self._vec[name] = VecStore(name, self._label(), dir=d) if (d / "meta.json").exists() else None
        return self._vec[name]

    def warm(self, names=("papers", "passages", "statements"), vecs=("papers", "statements")) -> dict:
        """Open the tantivy indexes and the vector memmaps without preloading anything else (§9.2: an MCP
        session process starts light). Returns what opened; a missing index is reported, not raised."""
        opened = {"indexes": [], "vectors": [], "missing": []}
        for n in names:
            try:
                self._open(n)
                opened["indexes"].append(n)
            except RuntimeError:
                opened["missing"].append(n)
        for n in vecs:
            try:
                if self._vecstore(n) is not None:
                    opened["vectors"].append(n)
            except RuntimeError:
                opened["missing"].append(f"vectors/{n}")
        return opened

    def close(self) -> None:
        """Release the memmap handles. Windows keeps a mapped file locked, and an index rebuild replaces the
        vector files atomically — a reader that stays open across a build makes the export fail, so tools
        close (or reopen) the Index around a rebuild."""
        for v in self._vec.values():
            for attr in ("floats", "codes"):
                m = getattr(v, attr, None) if v is not None else None
                mmap_ = getattr(m, "_mmap", None)
                if mmap_ is not None:
                    try:
                        mmap_.close()
                    except (ValueError, BufferError):
                        pass
        self._ix.clear()
        self._vec.clear()

    # ---- query helpers
    def _embed_query(self, query: str):
        if self.embed is None:
            return None
        return self.embed([query])[0]

    @staticmethod
    def _boosted(ix, query: str, fields: tuple, boosts: tuple):
        import tantivy
        parts = []
        for f, b in zip(fields, boosts):
            q = ix.parse_query(query, [f])
            parts.append((tantivy.Occur.Should, tantivy.Query.boost_query(q, float(b)) if b != 1.0 else q))
        return tantivy.Query.boolean_query(parts)

    # ---- searches
    def papers(self, query: str, as_of, k: int = 20, pool: int = 200, tier: str | None = None) -> list[str]:
        """Paper-level find: BM25 over title (boosted) + abstract, fused with the dense paper vectors."""
        import tantivy
        ix = self._open("papers")
        schema = ix.schema
        day = asof_int(as_of)
        clauses = [(tantivy.Occur.Must, self._boosted(ix, query, ("title", "abstract"), (2.0, 1.0))),
                   (tantivy.Occur.Must, tantivy.Query.range_query(schema, "day_hi", tantivy.FieldType.Integer,
                                                                  0, day))]
        if tier:
            clauses.append((tantivy.Occur.Must, tantivy.Query.term_query(schema, "tier", tier)))
        s = ix.searcher()
        bm = [s.doc(a).to_dict()["pid"][0] for _, a in
              s.search(tantivy.Query.boolean_query(clauses), pool).hits]
        dense = []
        v, qv = self._vecstore("papers"), self._embed_query(query)
        if v is not None and qv is not None and tier is None:
            # the vector store carries no tier flags — with a tier filter on, dense hits would bypass it
            # (the same class of bug as the old per-paper-cap bypass), so the filtered search is BM25-only
            dense = [key for key, _o, _s in v.search(qv, day, pool)]
        return rrf(bm, dense)[:k]

    def passages(self, query: str, as_of, k: int = 10, paper: str | None = None, per_paper: int = 2,
                 pool: int = 400) -> list[dict]:
        """Evidence search over the canonical reading's sentences (BM25; the per-paper cap applies to the fused
        order — passages have a single rank, so score order)."""
        import tantivy
        ix = self._open("passages")
        schema = ix.schema
        clauses = [(tantivy.Occur.Must, ix.parse_query(query, ["body"])),
                   (tantivy.Occur.Must, tantivy.Query.range_query(schema, "day_hi", tantivy.FieldType.Integer,
                                                                  0, asof_int(as_of)))]
        if paper:
            clauses.append((tantivy.Occur.Must, tantivy.Query.term_query(schema, "pid", paper)))
        s = ix.searcher()
        out, per = [], Counter()
        for _, a in s.search(tantivy.Query.boolean_query(clauses), pool).hits:
            d = s.doc(a).to_dict()
            pid = d["pid"][0]
            if paper is None and per[pid] >= per_paper:
                continue
            per[pid] += 1
            out.append({"uid": d["uid"][0], "paper": pid, "date": day_iso(d["day_hi"][0]),
                        "section": (d.get("section") or [""])[0], "kind": (d.get("kind") or [""])[0],
                        "text": d["body"][0]})
            if len(out) >= k:
                break
        return out

    def statements(self, query: str, as_of, k: int = 20, kind: str | None = None, facet: str | None = None,
                   about: str | None = None, per_paper: int = 2, pool: int = 400) -> list[int]:
        """Sentence-level evidence: BM25 (text + quote) fused with the dense self-statement vectors; the
        per-paper cap runs over the fused list with owners known on both sides (the §9.1 bug fix)."""
        import tantivy
        from ..extract.schema import FACETS
        ix = self._open("statements")
        schema = ix.schema
        day = asof_int(as_of)
        clauses = [(tantivy.Occur.Must, ix.parse_query(query, ["body"])),
                   (tantivy.Occur.Must, tantivy.Query.range_query(schema, "day_hi", tantivy.FieldType.Integer,
                                                                  0, day))]
        for f, v in (("kind", kind), ("facet", facet), ("about", about), ):
            if v:
                clauses.append((tantivy.Occur.Must, tantivy.Query.term_query(schema, f, v)))
        s = ix.searcher()
        bm, owner = [], {}
        for _, a in s.search(tantivy.Query.boolean_query(clauses), pool).hits:
            d = s.doc(a).to_dict()
            sid = str(d["sid"][0])
            bm.append(sid)
            owner[sid] = d["about"][0]
        dense = []
        v, qv = self._vecstore("statements"), self._embed_query(query)
        if v is not None and qv is not None and kind in (None, "self"):
            fc = [FACETS.index(facet)] if facet else None       # the dense set holds self statements only
            for key, own, _s in v.search(qv, day, pool, facets=fc, owner=about):
                dense.append(key)
                owner.setdefault(key, own)
        out, per = [], Counter()
        for sid in rrf(bm, dense):
            ow = owner.get(sid)
            if ow is not None and per[ow] >= per_paper:
                continue
            per[ow] += 1
            out.append(int(sid))
            if len(out) >= k:
                break
        return out
