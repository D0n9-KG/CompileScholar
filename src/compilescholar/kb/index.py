# -*- coding: utf-8 -*-
"""记录级混合检索（REBUILD-PLAN-1003 V2/V3：取代 findings 的"词法过滤 + paper_id 字母序轮转截断"）。

两路召回 + RRF 融合：
  - BM25（rank_bm25 若可用，否则内置 Okapi 实现）：记录文本 = subject + claim + quote + 方法/对象名 + 论文标题；
  - 向量：与现有 emb cache 同一模型同一维度（local qwen3-embedding-8b），复用 KBTools 的缓存格式；
  - 融合：Reciprocal Rank Fusion（k=60），再按论文做 MMR 式去冗余（同一论文最多 per_paper 条），保证跨论文多样性
    但**不按 paper_id 字母序**（旧实现的系统性偏置）。
过滤是"真过滤"：kind / claim_type 给定时只返回匹配记录；claim_type=None 的记录不再放行（旧 V3）。
空/弱命中语义显式：返回 {"hits": [...], "match": "strong"|"weak"|"none"}；weak = 只有向量召回、BM25 零命中。
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_TOK = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_STOP = frozenset("""a an the of and or to in for on with by is are was were be been being as that this these those
from at its their it into via using based can than more most such which we our also have has had not no but if
then there here they them he she his her you your i me my do does did done how what when where why who whom""".split())


def tokenize(s: str) -> list[str]:
    return [t for t in _TOK.findall((s or "").lower()) if t not in _STOP and len(t) > 1]


def record_text(r: dict, paper_title: str = "") -> str:
    def nm(ref):
        if isinstance(ref, dict):
            return ref.get("canonical") or ref.get("surface") or ""
        return str(ref or "")
    parts = [r.get("subject") or r.get("claims_about") or "", r.get("claim") or r.get("missing") or "",
             r.get("quote") or "", nm(r.get("method_ref")), nm(r.get("from_method_ref")), nm(r.get("to_method_ref")),
             r.get("relation") or "", r.get("conditions") or "", paper_title]
    return " ".join(str(p) for p in parts if p)


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.dl = [len(d) for d in docs]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.tf = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.post[t].append(i)

    def scores(self, q: list[str]) -> dict[int, float]:
        out = defaultdict(float)
        for t in set(q):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.post[t]:
                f = self.tf[i][t]
                out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return out


class HybridIndex:
    """records_by_paper: {pid: {"records": [...]}}；papers: {pid: {"title": ...}}；
    embed_fn(texts)->list[vec]（可为 None=只 BM25）；doc_vecs 与 self.items 同序（可预载缓存）。"""

    def __init__(self, records_by_paper: dict, papers: dict, embed_fn=None, doc_vecs=None):
        self.items = []
        for pid, p in records_by_paper.items():
            title = (papers.get(pid) or {}).get("title") or ""
            for r in (p.get("records") if isinstance(p, dict) else p) or []:
                self.items.append((pid, r, record_text(r, title)))
        self.bm25 = BM25([tokenize(t) for _, _, t in self.items])
        self.embed_fn = embed_fn
        self.doc_vecs = doc_vecs
        if self.doc_vecs is not None:
            self._norm_docs()

    def _norm_docs(self):
        import numpy as np
        M = np.asarray(self.doc_vecs, dtype="float32")
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        self._M = M

    def ensure_vectors(self, batch: int = 64):
        if self.doc_vecs is None and self.embed_fn is not None:
            texts = [t[:1200] for _, _, t in self.items]
            self.doc_vecs = self.embed_fn(texts)
            self._norm_docs()

    def search(self, query: str, k: int = 30, kind=None, claim_type=None, paper_id=None,
               per_paper: int = 3, pool: int = 300, rrf_k: int = 60, pid_filter=None) -> dict:
        """pid_filter(paper_id) -> bool: drop records of papers it rejects before ranking (knowledge cutoff, W1-12).
        None keeps the original behaviour."""
        def ok(i):
            pid, r, _ = self.items[i]
            if paper_id and pid != paper_id:
                return False
            if pid_filter is not None and not pid_filter(pid):
                return False
            if kind and r.get("kind") not in (kind if isinstance(kind, (list, tuple, set)) else (kind,)):
                return False
            if claim_type and r.get("claim_type") not in (claim_type if isinstance(claim_type, (list, tuple, set)) else (claim_type,)):
                return False
            return True

        bm = self.bm25.scores(tokenize(query))
        bm_rank = [i for i, _ in sorted(bm.items(), key=lambda x: -x[1]) if ok(i)][:pool]
        vec_rank = []
        if self.doc_vecs is not None and self.embed_fn is not None:
            import numpy as np
            qv = np.asarray(self.embed_fn([query])[0], dtype="float32")
            qv /= (np.linalg.norm(qv) + 1e-8)
            sims = self._M @ qv
            order = np.argsort(-sims)
            vec_rank = [int(i) for i in order[: pool * 3] if ok(int(i))][:pool]
        fused = defaultdict(float)
        for rank, i in enumerate(bm_rank):
            fused[i] += 1.0 / (rrf_k + rank + 1)
        for rank, i in enumerate(vec_rank):
            fused[i] += 1.0 / (rrf_k + rank + 1)
        hits, per = [], Counter()
        for i, s in sorted(fused.items(), key=lambda x: -x[1]):
            pid, r, _ = self.items[i]
            if per[pid] >= per_paper:
                continue
            per[pid] += 1
            hits.append({"paper_id": pid, "record": r, "score": round(s, 5), "bm25": i in bm})
            if len(hits) >= k:
                break
        match = "none" if not hits else ("strong" if any(h["bm25"] for h in hits) else "weak")
        return {"hits": hits, "match": match, "n_bm25": len(bm_rank), "n_vec": len(vec_rank)}
