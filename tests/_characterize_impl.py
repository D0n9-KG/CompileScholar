# -*- coding: utf-8 -*-
"""Implementation selector + deterministic fakes for the characterization tests (upgrade W6 step S2).

The answer path is moving from `.research_tmp/experiments/benchmarks/_shared/tools` into `src/compilescholar`.
Goldens are generated once from the OLD code (tests/fixtures/characterize/make_goldens.py); the tests then run
against every implementation listed in IMPLS and must reproduce the goldens byte for byte.
No network, no LLM: retrieval, LLM and embeddings are replaced by the deterministic fakes below.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "characterize"
OLD_TOOLS = REPO / ".research_tmp" / "experiments" / "benchmarks" / "_shared" / "tools"

IMPLS = ["old"]


def load(impl: str) -> types.SimpleNamespace:
    """Return the modules one implementation uses: AP (answer pipeline), RG (refgraph), CUT (cutoff), HY (hybrid),
    EMB_MOD (module whose `embed_local` the KB loader resolves at call time)."""
    if impl == "old":
        if str(OLD_TOOLS) not in sys.path:
            sys.path.insert(0, str(OLD_TOOLS))
        ap = importlib.import_module("answer_pipeline")
        rg = importlib.import_module("refgraph")
        cut = importlib.import_module("cutoff")
        hy = importlib.import_module("kb_compiler.retrieve.hybrid")
        emb = importlib.import_module("kb_infra.embedding")
        return types.SimpleNamespace(AP=ap, RG=rg, CUT=cut, HY=hy, EMB_MOD=emb)
    raise ValueError(impl)


# ---------------------------------------------------------------- deterministic fakes
def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


DIM = 32


def fake_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for t in texts:
        v = [0.0] * DIM
        for w in _tok(t):
            v[_h(w) % DIM] += 1.0
        if not any(v):
            v[0] = 1.0
        out.append(v)
    return out


class FakeCand:
    def __init__(self, status, title, year, abstract, err=None):
        self.status, self.title, self.year, self.error_summary = status, title, year, err
        self.raw = {"abstract": abstract}


class FakeSciverse:
    """semantic_search(q) → 12 deterministic candidates: mixed years (some past the 2025-05 cutoff), one short
    abstract, one failed row — exercises ext_search's filtering."""

    def semantic_search(self, q: str, limit: int = 10):
        h = _h(q)
        words = _tok(q) or ["topic"]
        out = []
        for i in range(12):
            w = words[i % len(words)]
            year = 2018 + ((h >> i) % 9)
            ab = (f"We study {w} and related problems. " * 3) if i != 5 else "short"
            status = "failed" if i == 7 else "ready"
            out.append(FakeCand(status, f"Paper {(h >> (2 * i)) % 97} on {w}", year, ab, err="boom" if status == "failed" else None))
        return out[:max(limit, 10) + 2]


REF_POOL = [f"Reference work number {i} on learning" for i in range(15)]


def fake_references(title: str, deadline=None, use_arxiv_html=False):
    h = _h(title)
    refs = []
    for i in range(8):
        j = (h >> (3 * i)) % len(REF_POOL)
        refs.append({"title": REF_POOL[j], "year": 2015 + (j % 11), "date": None,
                     "abstract": (f"Abstract of reference {j}. " * 6) if j % 3 == 0 else "", "ids": {"arxiv": None}})
    return refs, "fake"


def fake_resolve_abstracts(rows, deadline=None):
    for r in rows:
        if _h(r["title"]) % 2 == 0:
            r["abstract"] = f"Resolved abstract for {r['title']}. " * 4


def fake_chat(prompt: str, max_tokens: int = 6000, temperature: float = 0.2) -> str:
    """Deterministic stand-in for the 27B: a pure function of the prompt text."""
    if prompt.startswith("You are planning a literature-grounded research report"):
        q = re.search(r"Question: (.*)\n", prompt).group(1)
        w = _tok(q)[:6] or ["topic"]
        return json.dumps({"sections": [
            {"title": f"Methods for {' '.join(w[:2])}", "goal": "establish the main methods",
             "queries": [" ".join(w[:3]), " ".join(w[2:5]) or w[0]]},
            {"title": f"Limitations of {' '.join(w[1:3])}", "goal": "establish limitations",
             "queries": [" ".join(w[3:6]) or w[0]]}]})
    if prompt.startswith("You are screening evidence"):
        ids = re.findall(r"^\[(E\d+)\]", prompt.split("Excerpts:", 1)[1], re.M)
        return json.dumps({"off_topic": [i for i in ids if int(i[1:]) % 7 == 3]})
    if prompt.startswith("Write one section"):
        ids = re.findall(r"^\[(E\d+)\]", prompt.split("EVIDENCE", 1)[1], re.M)
        if not ids:
            return ""
        s = [f"First claim about the topic [{ids[0]}].",
             f"A combined claim [{ids[0]}][{ids[min(1, len(ids) - 1)]}].",
             "An uncited framing sentence.",
             f"A claim with a missing id [E9999] and a real one [{ids[-1]}].",
             f"Repeated id [{ids[-1]}][{ids[-1]}]."]
        return " ".join(s[:3]) + "\n\n" + " ".join(s[3:])
    return ""


def kb_fixture_dir() -> Path:
    return FIX / "kb_tiny"


def strip_timing(obj):
    """Drop wall-clock fields from an answer() result so it is comparable across runs."""
    t = obj.get("trace") or {}
    for k in ("t_plan_s", "t_write_s", "elapsed_s", "t_cite_s"):
        t.pop(k, None)
    return obj


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)
