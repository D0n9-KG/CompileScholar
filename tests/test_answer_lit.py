# -*- coding: utf-8 -*-
"""Phase D⑤: answer_lit — the built-in consumer of the literature layer (§9.3). Other-statement evidence hangs
under the SPEAKER ("Y says X", snippet = Y's own sentence, X in the citation metadata), identity is the registry
paper_id with display identifiers from the card, probe and every sub-query run local + external in parallel,
and the trace writes the evidence table plus used_eids. The frozen v9b legacy path is characterized separately
(test_characterize_answer_path) and must stay byte-identical — nothing here touches it."""
from __future__ import annotations

import json
import re

import pytest

from compilescholar.answer import pipeline as AP

P1 = "arxiv:2001.00001"      # the speaker (Y)
P2 = "arxiv:2002.00002"      # the described work (X)
EXT = "External work on graphs"
OTHER_QUOTE = "GraphNet is slower than FastGF on large graphs."


class FakeTools:
    def __init__(self):
        self.calls = []

    def search_papers(self, query, as_of, k=8):
        self.calls.append(("search_papers", query))
        return [{"id": P1, "title": "Speaker paper", "date": "2020-01-01",
                 "self": {"display": "We propose FastGF.", "quote": "We propose FastGF, an efficient transformer."}},
                {"id": P2, "title": "Described paper", "date": "2019-01-01",
                 "self": {"display": "GraphNet.", "quote": "We propose GraphNet."}}]

    def search_external(self, query, as_of, k=8):
        self.calls.append(("search_external", query))
        return {"local": [], "external": [{"external": True, "title": EXT, "year": 2018,
                                           "display": "chunk text", "quote": "External chunk verbatim."}],
                "errors": []}

    def find_evidence(self, claim, as_of, k=8):
        self.calls.append(("find_evidence", claim))
        return [{"kind": "other", "by": P1, "about": P2, "date": "2021-01-01", "facet": "result",
                 "role": "compares", "epistemic": "stated", "display": "P1 on P2",
                 "quote": OTHER_QUOTE},
                {"kind": "passage", "by": P2, "date": "2019-06-01", "section": "Intro",
                 "display": "intro", "quote": "GraphNet applies message passing."}]

    def paper_card(self, pid, as_of):
        self.calls.append(("paper_card", pid))
        return {"id": pid, "title": {P1: "Speaker paper", P2: "Described paper"}[pid],
                "date": {P1: "2020-01-01", P2: "2019-01-01"}[pid],
                "ids": {"doi": f"10.1/{pid[-3:]}", "arxiv": pid[6:]},
                "contributions": [{"quote": f"Contribution quote of {pid}.", "display": "d"}]}

    def field_map(self, topic, as_of):
        self.calls.append(("field_map", topic))
        return {"snapshot": "2022-01-01", "before_grid": False,
                "families": [{"name": "graph family",
                              "members": [{"id": P1, "title": "Speaker paper", "date": "2020-01-01"}],
                              "facts": [{"display": "GraphNet is slow", "quote": "q",
                                         "status": "established", "n_independent": 2}]}]}

    def expand_citations(self, seeds, as_of, k=8):
        self.calls.append(("expand_citations", tuple(seeds)))
        return {"local": [{"id": P2, "title": "Described paper", "n_cocite": 3, "via": ["cocite"]}],
                "external": []}


@pytest.fixture()
def fake_chat(monkeypatch):
    prompts = []

    def chat(prompt, max_tokens=6000, temperature=0.2):
        prompts.append(prompt)
        if prompt.startswith("You are planning"):
            return json.dumps({"sections": [{"title": "Sec one", "goal": "the goal", "queries": ["q1"]}]})
        if prompt.startswith("You are screening"):
            return json.dumps({"off_topic": []})
        if prompt.startswith("Write one section"):
            eids = re.findall(r"^\[(E\d+)\]", prompt, re.M)
            assert eids, "the write prompt must carry numbered evidence"
            pick = eids[:3]
            return " ".join(f"Sentence {i} cites [{e}]." for i, e in enumerate(pick, 1))
        raise AssertionError(f"unrouted prompt: {prompt[:80]}")

    monkeypatch.setattr(AP, "chat", chat)
    return prompts


def test_answer_lit_end_to_end(fake_chat):
    tools = FakeTools()
    r = AP.answer_lit("How do graph transformers scale?", cutoff="2022-06", tools=tools)
    tr = r["trace"]
    assert tr["mode"] == "lit" and tr["as_of"] == "2022-05-31"        # the CS2 month convention -> day before
    # the probe ran local + external, and the plan got the field block
    assert ("search_papers", "How do graph transformers scale?") in tools.calls
    assert ("search_external", "How do graph transformers scale?") in tools.calls
    assert any(p.startswith("You are planning") and "graph family" in p for p in fake_chat)
    # every sub-query ran both channels; the expansion channel ran for the section
    assert ("find_evidence", "q1") in tools.calls and ("search_external", "q1") in tools.calls
    assert any(c[0] == "expand_citations" for c in tools.calls)
    # other-statement evidence hangs under the SPEAKER with the described work in metadata
    others = [e for e in tr["evidence"] if e["src"] == "lit_other"]
    assert others and others[0]["paper_id"] == P1 and others[0]["snippet"] == OTHER_QUOTE
    assert others[0]["about"] == {"id": P2, "title": "Described paper"}
    # external hits ride along as their own evidence, quote verbatim
    exts = [e for e in tr["evidence"] if e["src"] == "lit_external"]
    assert exts and exts[0]["title"] == EXT and exts[0]["snippet"] == "External chunk verbatim."
    # trace writes evidence + used_eids (§9.3)
    assert tr["used_eids"] and set(tr["used_eids"]) <= {e["eid"] for e in tr["evidence"]}
    # one paper, one citation; display identifiers from the registry card
    sec = r["sections"][0]
    speaker_cites = [c for c in sec["citations"] if c["title"] == "Speaker paper"]
    assert len(speaker_cites) == 1
    meta = speaker_cites[0]["metadata"]
    assert meta["ids"] == {"doi": "10.1/001", "arxiv": "2001.00001"}
    assert {"id": P2, "title": "Described paper"} in meta.get("describes", [])
    assert OTHER_QUOTE in speaker_cites[0]["snippets"]              # the evidence snippet is Y's own sentence
    # the writer phrased attribution into the section text through the evidence line format
    assert any("says:" in p for p in fake_chat if p.startswith("Write one section"))


def test_answer_lit_without_external(fake_chat):
    tools = FakeTools()
    r = AP.answer_lit("q?", cutoff=None, tools=tools, use_ext=False)
    assert not any(c[0] == "search_external" for c in tools.calls)
    assert r["trace"]["as_of"] and all(e["src"] != "lit_external" for e in r["trace"]["evidence"])


def test_identity_prefers_paper_id():
    e = {"paper_id": "doi:10.9/z", "arxiv": "2001.99999", "title": "T", "paper_key": "doi:10.9/z"}
    assert AP._identity(e) == "doi:10.9/z"
    legacy = {"arxiv": "2001.99999v2", "title": "T", "paper_key": "kb:x"}
    assert AP._identity(legacy) == "arxiv:2001.99999"               # the frozen rule, untouched
