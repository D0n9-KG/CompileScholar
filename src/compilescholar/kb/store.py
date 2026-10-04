# -*- coding: utf-8 -*-
"""Compiled knowledge base: records + hybrid index + field-state channel (moved from answer_pipeline.KB; loading,
search and state_search behave identically).

`embed` is injected (default: compilescholar.llm.embedding.embed_local). Pass `embed=None` for BM25 only."""
from __future__ import annotations

import json
import os

from .index import HybridIndex

_DEFAULT = object()


class KB:
    def __init__(self, kb_dir: str, use_vectors: bool = True, embed=_DEFAULT):
        self.papers = json.load(open(os.path.join(kb_dir, "papers.json"), encoding="utf-8"))
        records = json.load(open(os.path.join(kb_dir, "records.json"), encoding="utf-8"))
        # the embedder is active only when precomputed record vectors exist (no vectors -> BM25 only, no state channel)
        vecs, active = None, None
        vp, mp = os.path.join(kb_dir, "record_vecs.f32"), os.path.join(kb_dir, "record_vecs.meta.json")
        if use_vectors and os.path.exists(vp) and os.path.exists(mp):
            if embed is _DEFAULT:
                from ..llm.embedding import embed_local as embed
            if embed is not None:
                import numpy as np
                meta = json.load(open(mp))
                vecs = np.fromfile(vp, dtype="float32").reshape(meta["n"], meta["dim"])
                active = embed
        embed = active
        self.index = HybridIndex(records, self.papers, embed_fn=embed, doc_vecs=vecs)
        if vecs is not None and len(self.index.items) != vecs.shape[0]:
            raise RuntimeError("record_vecs and records.json differ in order/size — re-embed the KB")
        # field-state view (method families + family-level properties / limitations / comparisons)
        self.families, self._fam_vecs, self._embed = [], None, embed
        sp = os.path.join(kb_dir, "state_merged.json")
        if os.path.exists(sp):
            self.families = [f for f in json.load(open(sp, encoding="utf-8"))["families"]
                             if f["props"] or f["limits"] or f["compares"]]
            if embed is not None and self.families:
                import numpy as np
                M = np.asarray(embed([" / ".join(f["names"][:3]) for f in self.families]), dtype="float32")
                self._fam_vecs = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

    def state_search(self, q: str, top_fam: int = 3, per_fam: int = 6, min_sim: float = 0.55) -> list[dict]:
        """Query -> nearest method families (embedding cosine) -> their cross-survey properties / limitations /
        comparisons. Each item is still a verbatim excerpt, but grouped by family."""
        if self._fam_vecs is None:
            return []
        import numpy as np
        qv = np.asarray(self._embed([q])[0], dtype="float32")
        qv /= (np.linalg.norm(qv) + 1e-8)
        sims = self._fam_vecs @ qv
        out = []
        for j in np.argsort(-sims)[:top_fam]:
            if sims[j] < min_sim:
                break
            f = self.families[int(j)]
            fam_name = f["names"][0]
            items = ([("property", x) for x in f["props"]] + [("limitation", x) for x in f["limits"]]
                     + [("comparison", x) for x in f["compares"]])
            seen = set()
            for role, x in items:
                key = (x["paper_id"], (x.get("quote") or str(x.get("text") or ""))[:120])
                if key in seen:
                    continue
                seen.add(key)
                snip = (x.get("quote") or x.get("text") or "").strip()
                if not snip:
                    continue
                m = self.papers.get(x["paper_id"]) or {}
                out.append({"src": "state", "paper_key": f"kb:{x['paper_id']}", "title": m.get("title") or x["title"],
                            "year": m.get("year"), "arxiv": m.get("arxiv_id"),
                            "snippet": f"[{role} of '{fam_name}'] " + snip[:800],
                            "family": fam_name, "role": role, "family_sim": round(float(sims[j]), 3)})
                if sum(1 for o in out if o["family"] == fam_name) >= per_fam:
                    break
        return out

    def search(self, q: str, k: int = 12) -> list[dict]:
        res = self.index.search(q, k=k, per_paper=2)
        out = []
        for h in res["hits"]:
            r, pid = h["record"], h["paper_id"]
            m = self.papers.get(pid) or {}
            snippet = (r.get("quote") or r.get("claim") or "").strip()
            if not snippet:
                continue
            out.append({"src": "kb", "paper_key": f"kb:{pid}", "title": m.get("title") or pid,
                        "year": m.get("year"), "arxiv": m.get("arxiv_id"),
                        "snippet": snippet[:900], "match": res["match"],
                        "record_id": r.get("id"), "kind": r.get("kind")})
        return out
