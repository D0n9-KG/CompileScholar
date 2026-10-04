# -*- coding: utf-8 -*-
"""Generate characterization goldens from the OLD answer path (run once, before the move; never re-run on new code).
The old modules now live in legacy/; regenerating requires a checkout of tag cs2-test-v9b-final.

Usage:  python tests/fixtures/characterize/make_goldens.py
Writes tests/fixtures/characterize/goldens.json and kb_tiny/record_vecs.f32(+meta) built with the fake embedder.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _characterize_impl as C  # noqa: E402


def compute(impl: str) -> dict:
    """Every characterized behaviour, as plain JSON-able data. Shared by make_goldens and the tests."""
    os.environ["KNOWLEDGE_CUTOFF"] = "2025-05"
    ns = C.load(impl)
    AP, RG, CUT, HY = ns.AP, ns.RG, ns.CUT, ns.HY
    out: dict = {}

    # 1. cutoff decision table
    cases = [(2024, None), (2025, None), (2026, None), (None, None), ("2023", None), ("x", None),
             (None, "2025-04"), (None, "2025-05"), (None, "2025-04-30"), (None, "2025"), (2026, "2024-12"), (None, "bad")]
    out["cutoff"] = [[y, d, CUT.allowed(y, d)] for y, d in cases]
    out["cutoff_explicit"] = [CUT.allowed(2025, None, cut=(2025, 13)), CUT.allowed(2024, None, cut=None)]
    CUT.set_thread_cutoff("2023-06")
    out["cutoff_thread"] = [CUT.raw_cutoff(), CUT.allowed(2023, "2023-05"), CUT.allowed(2023, "2023-06")]
    CUT.set_thread_cutoff(None)

    # 2. hybrid index on the tiny KB (BM25 only, and BM25 + fake vectors)
    kbd = C.kb_fixture_dir()
    recs = json.load(open(kbd / "records.json", encoding="utf-8"))
    paps = json.load(open(kbd / "papers.json", encoding="utf-8"))
    queries = ["graph neural networks for molecules", "limitations of large language models",
               "reinforcement learning reward", "evaluation benchmark", "zzzz qqqq"]
    hx = HY.HybridIndex(recs, paps)
    out["hybrid_bm25"] = {q: [[h["paper_id"], h["record"].get("id"), h["score"], h["bm25"]] for h in r["hits"]] + [r["match"]]
                          for q in queries for r in [hx.search(q, k=10, per_paper=2)]}
    hv = HY.HybridIndex(recs, paps, embed_fn=C.fake_embed)
    hv.ensure_vectors()
    out["hybrid_vec"] = {q: [[h["paper_id"], h["record"].get("id"), h["score"], h["bm25"]] for h in r["hits"]] + [r["match"]]
                         for q in queries for r in [hv.search(q, k=10, per_paper=2)]}
    out["tokenize"] = HY.tokenize("The GNN-based model's F1 is 0.9 on QM9; it can't fail.")

    # 3. KB loader + search + state_search (fake embedder patched where the loader resolves it)
    ns.EMB_MOD.embed_local = C.fake_embed
    kb = AP.KB(str(kbd))
    out["kb_n_families"] = len(kb.families)
    out["kb_search"] = {q: kb.search(q, 8) for q in queries}
    out["kb_state"] = {q: kb.state_search(q, min_sim=0.0) for q in queries}

    # 4. external search with the fake Sciverse (cutoff 2025-05 filters 2025+ years)
    AP._sciverse = lambda: C.FakeSciverse()
    out["ext_search"] = {q: AP.ext_search(q, 8) for q in queries[:3]}

    # 5. citation expansion with fake references
    RG.references = C.fake_references
    RG.resolve_abstracts = C.fake_resolve_abstracts
    RG.prefetch = lambda titles, deadline=None, chunk=6: None
    out["cite_expand"] = AP.cite_expand(["Seed A", "Seed B", "Seed C"], k=5, max_seeds=6)

    # 6. interleave on a real evidence order from test r1
    ev = json.load(open(C.FIX / "interleave_evidence.json", encoding="utf-8"))
    out["interleave"] = [e["eid"] for e in AP._interleave([e for e in ev if 0 in e["sections"]])]

    # 7. refgraph pure helpers
    import re as _re
    html = open(C.FIX / "arxiv_html_bib_sample.html", encoding="utf-8").read()
    items = _re.findall(r'<li[^>]*class="ltx_bibitem"[^>]*>(.*?)</li>', html, _re.S)
    import html as _html
    raws = [_re.sub(r"\s+", " ", _re.sub(r"<[^>]+>", " ", _html.unescape(x))).strip() for x in items]
    out["bib_title"] = [RG._bib_title(r) for r in raws]
    out["rg_norm"] = [RG.norm(s) for s in ("  Attention Is All You Need!  ", "BERT: Pre-training", "")]
    out["rg_clean_title"] = [RG.clean_title(s) for s in ('“C3: Cross-modal,”', "Plain title.", "'quoted'")]

    # 8. full answer() with fake LLM / retrieval: every prompt sent + the result (timings stripped)
    prompts: list[str] = []
    import threading
    lock = threading.Lock()

    def chat(prompt, max_tokens=6000, temperature=0.2):
        with lock:
            prompts.append(prompt)
        return C.fake_chat(prompt, max_tokens, temperature)
    AP.chat = chat
    res = {}
    for name, kw in (("budget1000_cite", dict(use_cite=True, word_budget=1000)),
                     ("nolimit_nocite", dict(use_cite=False, word_budget=None)),
                     ("noprobe_nostate", dict(use_cite=False, use_probe=False, use_state=False, word_budget=1000)),
                     ("noscreen", dict(use_cite=False, use_screen=False, word_budget=1000))):
        prompts.clear()
        r = AP.answer("How do graph neural networks handle molecular property prediction?", kb, use_ext=True, **kw)
        res[name] = {"result": C.strip_timing(r), "prompts": sorted(prompts)}
    prompts.clear()
    r = AP.answer("Related work for a paper on reward models.", kb, use_ext=True, use_cite=False, cutoff="2023-06",
                  task_context={"title": "Reward models", "text": "We propose a reward model. " * 20}, word_budget=None)
    res["task_context_cutoff"] = {"result": C.strip_timing(r), "prompts": sorted(prompts)}
    out["answer"] = res
    return json.loads(json.dumps(out, ensure_ascii=False, default=str))


def build_vectors():
    import numpy as np
    sys.path.insert(0, str(C.REPO / "src"))
    from kb_compiler.retrieve.hybrid import HybridIndex
    kbd = C.kb_fixture_dir()
    recs = json.load(open(kbd / "records.json", encoding="utf-8"))
    paps = json.load(open(kbd / "papers.json", encoding="utf-8"))
    hx = HybridIndex(recs, paps)
    M = np.asarray(C.fake_embed([t[:1200] for _, _, t in hx.items]), dtype="float32")
    M.tofile(kbd / "record_vecs.f32")
    json.dump({"n": M.shape[0], "dim": M.shape[1], "note": "fake embedder (tests/_characterize_impl.fake_embed)"},
              open(kbd / "record_vecs.meta.json", "w"))


if __name__ == "__main__":
    build_vectors()
    g = compute("old")
    json.dump(g, open(C.FIX / "goldens.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, sort_keys=True)
    print("goldens written:", ", ".join(sorted(g)))
