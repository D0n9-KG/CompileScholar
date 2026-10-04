# -*- coding: utf-8 -*-
"""Old kb_compiler.views.field_state (+ records.coarse_extract) vs compilescholar.compile.state: with the same fake
LLM, compile_field_state must return the same field state and send the same prompts (multiset)."""
from __future__ import annotations

import json
import re
import threading

import _characterize_impl as C

PAPERS = [{"key": f"k{i}", "title": f"Paper {i} on {['graph nets', 'transformers', 'diffusion'][i % 3]}",
           "year": 2019 + i % 5,
           "abstract": (f"We propose method M{i} for {['molecules', 'text', 'images'][i % 3]}. "
                        f"It improves accuracy on benchmark B{i % 4}. However it fails on long inputs and needs "
                        f"labels. Prior methods such as M{(i + 1) % 9} are slow. ") * 2}
          for i in range(9)]


def fake_llm(prompt: str, **kw) -> str:
    if prompt.startswith("You extract structured records from ONE paper abstract"):
        t = re.search(r'Abstract of "(.*?)"', prompt).group(1)
        i = int(re.search(r"Paper (\d+)", t).group(1))
        return json.dumps({"records": [
            {"kind": "method", "subject": f"M{i}", "claim": f"The paper proposes M{i}.", "quote": f"We propose method M{i}.",
             "mentions": [f"M{(i + 1) % 9}"]},
            {"kind": "finding", "subject": f"M{i}", "claim": "Improves accuracy.", "claim_type": "observation",
             "quote": f"It improves accuracy on benchmark B{i % 4}."},
            {"kind": "limitation", "subject": f"M{i}", "claim": "Fails on long inputs.", "quote": "However it fails on long inputs."},
            {"kind": "finding", "subject": f"M{(i + 1) % 9}", "claim": "Prior methods are slow.", "claim_type": "criticism",
             "quote": f"Prior methods such as M{(i + 1) % 9} are slow.", "mentions": [f"M{(i + 1) % 9}"]}]})
    if prompt.startswith("You are organizing the literature"):
        ids = re.findall(r"^(p\d+):", prompt, re.M)
        return json.dumps({"families": [{"name": "Family A", "definition": "first half", "members": ids[: len(ids) // 2]},
                                        {"name": "Family B", "definition": "second half", "members": ids[len(ids) // 2:-1]},
                                        {"name": "Other", "definition": "", "members": ids[-1:]}]})
    if prompt.startswith("Below are method families proposed separately"):
        ids = re.findall(r"^(f\d+):", prompt, re.M)
        return json.dumps({"groups": [{"ids": ids[:2], "name": "Merged", "definition": "merged"}]})
    if prompt.startswith("You are writing the family-level facts"):
        ids = re.findall(r"^(e\d+):", prompt, re.M)
        return json.dumps({"properties": [{"text": "Members share an approach.", "evidence": ids[:3]}],
                           "limitations": [{"text": "They fail on long inputs.", "evidence": ids[2:5]},
                                           {"text": "Unsupported claim.", "evidence": ["e999"]}]})
    if prompt.startswith("Below are statements, taken verbatim"):
        ids = re.findall(r"^(s\d+):", prompt, re.M)
        return json.dumps({"problems": [{"text": "Long inputs are unsolved.", "evidence": ids[::2]}]})
    return ""


def _run(impl: str):
    import sys
    sent = []
    lock = threading.Lock()

    def llm(prompt, **kw):
        with lock:
            sent.append(prompt)
        return fake_llm(prompt, **kw)
    if impl == "old":
        import kb_compiler.records.common as common
        import kb_compiler.views.field_state as F
        common.call_local = llm
        F.call_local = llm
    else:
        import compilescholar.compile.state.coarse as Co
        import compilescholar.compile.state.field_state as F
        Co.call_local = llm
        F.call_local = llm
    out = F.compile_field_state(PAPERS, workers=4)
    return out, sorted(sent)


def test_field_state_equivalent():
    a, pa = _run("old")
    b, pb = _run("new")
    assert C.canon(a) == C.canon(b)
    assert pa == pb
    assert a["stats"]["families"] >= 1 and a["problems"]


def test_field_state_batched_merge_equivalent(monkeypatch):
    """More papers than one PROPOSE batch -> the MERGE stage runs."""
    import compilescholar.compile.state.field_state as Fn
    import kb_compiler.views.field_state as Fo
    for F in (Fo, Fn):
        monkeypatch.setattr(F, "induce_families", lambda p, r, batch=3, _f=F.induce_families: _f(p, r, batch=batch))
    a, pa = _run("old")
    b, pb = _run("new")
    assert C.canon(a) == C.canon(b) and pa == pb
    assert any(p.startswith("Below are method families proposed separately") for p in pa)
