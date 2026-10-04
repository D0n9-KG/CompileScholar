# -*- coding: utf-8 -*-
"""GOLDCOV step 3: judge each gold point's coverage in the RECORD layer.

Two-tier design (guard against measuring retrieval instead of extraction):
  tier 1: point -> RRF top-20 candidate records -> LLM verdict
  tier 2 (only if tier-1 says missing): top-50 candidates -> re-judge
Verdicts: covered | partial | missing (+ evidence record ids + reason).

8 workers (terminal-run B2 measured 16 workers OK on Paratera; 8 is
conservative for judge-grade latency).

Usage:
  py -3.13 judge_kb.py            # resume-safe, writes kb_cov.jsonl
"""
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
D2 = os.path.dirname(HERE)
SRC = "C:/Users/D0n9/Desktop/CompileScholar/src"
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "kb_compiler", "views"))
from search_text import TextSearchIndex  # noqa: E402
from kb_infra.llm import call_paratera  # noqa: E402

IDX_DIR = os.path.join(HERE, "rec_index")
OUT = os.path.join(HERE, "kb_cov.jsonl")
MODEL = "GLM-5.3"
WORKERS = 16


def patch_numpy_rank():
    """numpy matrix rewrite of TextSearchIndex._vector_rank (local-only
    monkeypatch: 19k x 4096 cosine per query is ~30-60s in pure Python —
    measured 2026-09-19; the shared module stays untouched)."""
    import numpy as np

    def _vector_rank_np(self, query, depth=50):
        from kb_infra.embedding import embed_cst
        if self.meta.get("provider") != "cst-qwen3":
            raise RuntimeError(f"provider mismatch: index={self.meta.get('provider')}")
        qe = np.asarray(embed_cst([query[:300]])[0], dtype=np.float32)
        qe /= (np.linalg.norm(qe) + 1e-9)
        sims = self._emb_np @ qe
        idx = np.argpartition(-sims, min(depth, len(sims) - 1))[:depth]
        return [(int(i), float(sims[i])) for i in idx[np.argsort(-sims[idx])]]

    TextSearchIndex._vector_rank = _vector_rank_np

    def _load_np(self):
        import numpy as np
        flat = np.fromfile(os.path.join(self.index_dir, "emb.bin"), dtype=np.float32)
        m = np.vstack(flat.reshape(len(self.chunks), -1))
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        self._emb_np = m / (norms + 1e-9)

    orig_init = TextSearchIndex.__init__

    def init_with_np(self, index_dir):
        orig_init(self, index_dir)
        _load_np(self)

    TextSearchIndex.__init__ = init_with_np


patch_numpy_rank()

PROMPT = """You are auditing whether a knowledge base (extracted from research papers) contains the information needed to support ONE atomic key point.

Key point: {point}

Candidate knowledge-base records below (format: RECORD <id> [<kind>, paper <pid>]: text). Each was extracted from a paper; verbatim quotes come from the paper.

{candidates}

Decide:
- "covered": at least one record directly supports the point (the fact/number/comparison/trend is present, numbers matching).
- "partial": records support PART of the point but a key element (a specific number, a specific method, the comparison direction...) is absent or differs.
- "missing": no record supports the point's substance.

Be strict about numbers: a record supporting "accuracy improved" does NOT cover "accuracy improved from 74.44 to 81.74". But do not demand the point's exact phrasing — semantic equivalence counts.

Return ONLY JSON: {{"verdict": "covered|partial|missing", "evidence": ["<record id>", ...], "reason": "<one sentence>"}}"""


def parse_verdict(raw):
    t = (raw or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    body = m.group(0)
    try:
        obj = json.loads(body)
    except Exception:
        try:
            import json5
            obj = json5.loads(body)
        except Exception:
            return None
    v = obj.get("verdict")
    if v not in ("covered", "partial", "missing"):
        return None
    ev = obj.get("evidence")
    if not isinstance(ev, list):
        ev = []
    return {"verdict": v, "evidence": [str(x) for x in ev][:5],
            "reason": str(obj.get("reason", ""))[:300]}


def render_candidates(idx, rmap, point, k):
    res = idx.search(point, k=k)
    lines, ids = [], []
    for h in res["hits"]:
        rid = h["paper_id"]
        m = rmap.get(rid, {})
        lines.append(f"RECORD {rid} [{m.get('kind')}, paper {m.get('paper_id')}]: {h['text'][:550]}")
        ids.append(rid)
    return lines, ids


def judge_point(idx, rmap, point_text):
    """Tier1 k=20; if missing, tier2 k=50 (bigger pool guards retrieval)."""
    last = None
    for k, tag in ((20, "t1"), (50, "t2")):
        lines, _ = render_candidates(idx, rmap, point_text, k)
        if not lines:
            return {"verdict": "missing", "evidence": [], "reason": "no candidates",
                    "tier": tag}
        prompt = PROMPT.format(point=point_text, candidates="\n\n".join(lines))
        v = None
        for _ in range(3):
            raw = call_paratera(prompt, MODEL, max_tokens=800,
                                enable_thinking=False, temperature=0.0)
            v = parse_verdict(raw)
            if v:
                break
        if not v:
            continue  # parse fail thrice at this tier -> bigger pool retry
        v["tier"] = tag
        last = v
        if tag == "t1" and v["verdict"] == "missing":
            continue  # escalate to tier 2
        return v
    if last:
        return last
    return {"verdict": "missing", "evidence": [], "reason": "judge_parse_fail",
            "tier": "t2"}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    idx = TextSearchIndex(IDX_DIR)
    rmap = json.load(open(os.path.join(HERE, "rec_map.json"), encoding="utf-8"))
    points = json.load(open(os.path.join(HERE, "points.json"), encoding="utf-8"))
    done = {}
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done[r["key"]] = r
    todo = []
    for qid, q in points.items():
        for p in q["points"]:
            key = f"{qid}#{p['point_id']}"
            if key not in done and "__DECOMPOSE_FAIL__" not in p["text"]:
                todo.append((qid, q, p, key))
    print(f"{len(done)} judged, {len(todo)} to go (workers={WORKERS})", flush=True)
    lock = threading.Lock()
    out_f = open(OUT, "a", encoding="utf-8")
    n_done = [0]

    def work(item):
        qid, q, p, key = item
        v = judge_point(idx, rmap, p["text"])
        rec = {"key": key, "qid": qid, "prompt_type": q["prompt_type"],
               "point_id": p["point_id"], "ptype": p["ptype"], "point": p["text"],
               **v}
        with lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            n_done[0] += 1
            if n_done[0] % 25 == 0:
                print(f"[{n_done[0]}/{len(todo)}]", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, t) for t in todo]
        for _ in as_completed(futs):
            pass
    from collections import Counter
    c = Counter()
    for l in open(OUT, encoding="utf-8"):
        if l.strip():
            c[json.loads(l)["verdict"]] += 1
    print(f"DONE: {dict(c)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
