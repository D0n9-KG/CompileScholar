# -*- coding: utf-8 -*-
"""A7: hybrid full-text search over raw paper chunks (BM25 + vector, RRF).

The "compile-zone escape hatch": questions targeting content outside the
compiled record kinds (novel phrasings, appendix details, content the
extractor never covered) currently have NO route to the raw corpus — the
typed tools read records, search reads records. This module searches the
ORIGINAL PAPER TEXTS directly, which is also the functional answer to the
"precompilation can't cover everything" rebuttal: precompiled records
first, live full-text retrieval as the deterministic fallback.

Design (informed by survey_hybrid_retrieval_2026-09-18 verdict):
  - BM25: hand-rolled Okapi (k1=1.5, b=0.75), zero dependencies, deterministic
  - vector: same provider discipline as all embeddings (cst-qwen3 4096-dim,
    same model/dim for chunks and queries — kb_infra DISCIPLINE)
  - fusion: RRF k=60 (the safe default; typed tools carry the keyword-type
    queries upstream, this is the fallback path so RRF dilution is bounded)
  - storage: same on-disk format as the proven text_index (chunks.jsonl +
    emb.bin + meta.json), so the control-arm TextKB layout conventions carry

At 53-paper scale this runs in-process; at the 2435-paper full scale the
same interface moves onto Qdrant (user-ruled stack, 2026-09-19) without
touching the harness.

Usage:
  build:  py -3.13 -m kb_compiler.views.search_text --build \
            --texts DIR --index-dir DIR [--chunk-chars 1500] [--overlap 150]
  probe:  py -3.13 -m kb_compiler.views.search_text --probe "query" --index-dir DIR
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from array import array

CHUNK_CHARS = 1500
OVERLAP = 150
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60

_TOKEN = re.compile(r"[a-z0-9]+")
_WORDDOC_CACHE = None


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class TextSearchIndex:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.meta = json.load(open(os.path.join(index_dir, "meta.json"),
                                   encoding="utf-8"))
        self.chunks = [json.loads(l) for l in
                       open(os.path.join(index_dir, "chunks.jsonl"),
                            encoding="utf-8")]
        flat = array("f")
        with open(os.path.join(index_dir, "emb.bin"), "rb") as fh:
            flat.frombytes(fh.read())
        d = self.meta["dim"]
        self.embs = [flat[i * d:(i + 1) * d] for i in range(len(self.chunks))]
        # BM25 statistics
        self._doc_toks = [_tokens(c["text"]) for c in self.chunks]
        self._doc_len = [len(t) for t in self._doc_toks]
        self._avgdl = sum(self._doc_len) / max(1, len(self._doc_len))
        self._df = {}
        for toks in self._doc_toks:
            for term in set(toks):
                self._df[term] = self._df.get(term, 0) + 1
        self._n = len(self.chunks)

    # ---------- BM25 ----------
    def _bm25_rank(self, query: str, depth: int = 50) -> list[tuple[int, float]]:
        q_terms = _tokens(query)
        if not q_terms:
            return []
        scores = []
        for i in range(self._n):
            toks = self._doc_toks[i]
            if not toks:
                continue
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            dl = self._doc_len[i]
            for term in q_terms:
                f = tf.get(term)
                if not f:
                    continue
                idf = math.log(1 + (self._n - self._df.get(term, 0) + 0.5)
                               / (self._df.get(term, 0) + 0.5))
                s += idf * (f * (BM25_K1 + 1)) / \
                    (f + BM25_K1 * (1 - BM25_B + BM25_B * dl / self._avgdl))
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:depth]

    # ---------- vector ----------
    def _vector_rank(self, query: str, depth: int = 50) -> list[tuple[int, float]]:
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        from kb_infra.embedding import embed_cst
        from kb_infra.llm import cosine_sim
        if self.meta.get("provider") != "cst-qwen3":
            raise RuntimeError(f"provider mismatch: index={self.meta.get('provider')}")
        qe = embed_cst([query[:300]])[0]
        sims = sorted(((cosine_sim(qe, e), i) for i, e in enumerate(self.embs)),
                      key=lambda x: -x[0])[:depth]
        return [(i, s) for s, i in sims]

    # ---------- fused search ----------
    def search(self, query: str, k: int = 8) -> dict:
        """RRF fusion of BM25 + vector rankings over raw chunks."""
        lex = self._bm25_rank(query)
        vec = self._vector_rank(query)
        rrf = {}
        for rank, (i, _s) in enumerate(lex):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (i, _s) in enumerate(vec):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(rrf.items(), key=lambda x: -x[1])[:min(int(k or 8), 16)]
        hits = []
        for i, score in top:
            c = self.chunks[i]
            hits.append({
                "chunk_id": f"{c['paper_id']}#{c['char_start']}",
                "paper_id": c["paper_id"],
                "score": round(score, 4),
                "in_lexical": any(j == i for j, _ in lex[:10]),
                "in_vector": any(j == i for j, _ in vec[:10]),
                "text": re.sub(r"\s+", " ", c["text"])[:420],
            })
        return {"tool": "search_text", "n": len(hits), "hits": hits}

    # ---------- build ----------
    @classmethod
    def build(cls, texts_dir: str, index_dir: str,
              chunk_chars: int = CHUNK_CHARS, overlap: int = OVERLAP):
        os.makedirs(index_dir, exist_ok=True)
        chunks = []
        for fn in sorted(os.listdir(texts_dir)):
            if not fn.endswith(".md"):
                continue
            pid = fn[:-3]
            text = open(os.path.join(texts_dir, fn), encoding="utf-8").read()
            step = max(1, chunk_chars - overlap)
            for s in range(0, len(text), step):
                e = min(s + chunk_chars, len(text))
                seg = text[s:e]
                if len(seg.strip()) < 80:
                    continue
                chunks.append({"paper_id": pid, "char_start": s,
                                "char_end": e, "text": seg})
                if e >= len(text):
                    break
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        from kb_infra.embedding import embed_cst
        print(f"[search_text] embedding {len(chunks)} chunks "
              f"({len(os.listdir(texts_dir))} papers)...", flush=True)
        embs = embed_cst([c["text"][:3000] for c in chunks])
        with open(os.path.join(index_dir, "chunks.jsonl"), "w",
                  encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        with open(os.path.join(index_dir, "emb.bin"), "wb") as f:
            array("f", (x for e in embs for x in e)).tofile(f)
        json.dump({"n_chunks": len(chunks), "dim": len(embs[0]),
                   "provider": "cst-qwen3", "chunk_chars": chunk_chars,
                   "overlap": overlap},
                  open(os.path.join(index_dir, "meta.json"), "w",
                       encoding="utf-8"))
        print(f"[search_text] index built: {len(chunks)} chunks -> {index_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--texts", default="")
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--probe", default="")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.build:
        assert args.texts, "--texts required with --build"
        TextSearchIndex.build(args.texts, args.index_dir)
    if args.probe:
        idx = TextSearchIndex(args.index_dir)
        out = idx.search(args.probe, k=args.k)
        print(json.dumps(out, ensure_ascii=False, indent=1)[:2600])


if __name__ == "__main__":
    main()
