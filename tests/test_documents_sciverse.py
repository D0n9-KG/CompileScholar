# -*- coding: utf-8 -*-
"""The Sciverse content layer (documents.sciverse, v2.5): the markdown adapter and its substring discipline,
version determination by each version's own sentences, per-sentence dates, and the fetch marathon's discipline
(idempotent skip, attempts -> gave_up). The Sciverse API is faked; the sv work pass inside the documents stage is
covered in test_dfc_endtoend.py."""
from __future__ import annotations

import importlib
import json
import sqlite3
import zlib

import pytest

from compilescholar.core import ids
from compilescholar.documents import sciverse as SV
from compilescholar.sources import sciverse as S

# ---------------------------------------------------------------- adapter
MD = """# 1 Introduction

We propose FastGF for graphs. It is fast and accurate.

## 2.1 Encoder

The encoder stacks transformer layers here.

<table><tr><td>0.9</td></tr></table>

$$E = mc^2$$

Figure 1: the architecture overview of the proposed method.

# References

[1] Some reference. 2020.
[2] Another reference. 2021.
[3] A third reference. 2019.
"""


def test_doc_adapter_units_sentences_refs():
    d = SV.doc("arxiv:2101.00002", MD)
    kinds = [u["kind"] for u in d["units"]]
    for want in ("section", "para", "table", "formula", "caption"):
        assert want in kinds
    assert d["has_refs"] is True
    assert not any("Some reference" in u["text"] for u in d["units"])     # the reference block is cut
    enc = next(u for u in d["units"] if u["kind"] == "para" and "stacks" in u["text"])
    assert enc["section"] == "1 Introduction > 2.1 Encoder"               # numbered heading depth
    f = next(u for u in d["units"] if u["kind"] == "formula")
    assert f["text"] == "$$E = mc^2$$"                                    # verbatim
    cap = next(u for u in d["units"] if u["kind"] == "caption")
    assert cap["text"].startswith("Figure 1:")
    sids = [s["sid"] for s in d["sentences"]]
    assert sids == [f"arxiv:2101.00002@sv#s{i + 1}" for i in range(len(sids))]
    um = {u["uid"]: u["text"] for u in d["units"]}
    assert all(s["text"] in um[s["unit"]] for s in d["sentences"])        # substring discipline (final check)
    texts = " ".join(s["text"] for s in d["sentences"])
    assert "We propose FastGF for graphs." in texts
    assert "0.9" not in texts and "mc^2" not in texts                     # tables/formulas are not prose


def test_split_sentences():
    t = "We propose a method. It trains in two days. Training uses e.g. Adam and GPUs. The result holds."
    ss = SV.split_sentences(t)
    assert ss == ["We propose a method.", "It trains in two days.",
                  "Training uses e.g. Adam and GPUs.", "The result holds."]
    assert all(s in t for s in ss)
    # a single token before the boundary (an initial, a digit) never splits — under-splitting keeps quotes
    # substring-safe, which is the direction the final check tolerates
    assert SV.split_sentences("Results by A. Smith and 8. Others followed.") == \
        ["Results by A. Smith and 8. Others followed."]
    t2 = "The bound is $a. B = 2$ in the paper. It holds generally."
    assert SV.split_sentences(t2) == ["The bound is $a. B = 2$ in the paper.", "It holds generally."]


# ---------------------------------------------------------------- version + dates
def _fast(sents):
    return {"sentences": [{"sid": f"s{i}", "text": s} for i, s in enumerate(sents)], "entries": {}, "cites": []}


V1S = ["We propose FastGF an efficient graph transformer for large graphs",
       "Training takes two days on one GPU with eight cores",
       "GraphFormer applies attention to graphs and molecules"]
V2S = V1S[:1] + ["Training takes three days on one GPU with eight cores"] + V1S[2:] + \
    ["We also prove a convergence bound for the attention layer here"]


def test_held_version():
    v1, v2 = _fast(V1S), _fast(V2S)
    assert SV.held_version(V2S, v1, v2)[0] == "latest"        # the latest version's own sentence is present
    assert SV.held_version(V1S, v1, v2)[0] == "ambiguous"     # append+revise: the v1 side has no own sentences
    assert SV.held_version(["totally different text about other topics entirely"], v1, v2)[0] == "ambiguous"
    assert SV.held_version(V1S, v1, None)[0] == "single"


def test_date_sentences():
    v1 = _fast(V1S)
    delta = {"sentences": [{"text": V2S[3]}],                                  # the latest-only sentence
             "revised": [{"text": V2S[1], "v1_text": V1S[1]}]}                 # "two days" -> "three days"
    sents = [{"sid": "a", "unit": "u", "text": V1S[0]},                                  # verbatim v1
             {"sid": "b", "unit": "u", "text": "Training takes three days on one GPU with eight cores"},
             {"sid": "c", "unit": "u", "text": V2S[3]}]                                   # latest-only
    out = SV.date_sentences(sents, v1, "2021-01-12", "2021-06-01", delta=delta)
    assert [o["date"] for o in out] == ["2021-01-12", "2021-06-01", "2021-06-01"]
    # the same revised sentence WITHOUT the delta screen: the bigram rule alone still dates it latest (0.78),
    # but a formatting-noise rendering of a v1 sentence must not fuzzy-match into v1 unscreened...
    out = SV.date_sentences(sents[1:], v1, "2021-01-12", "2021-06-01")
    assert [o["date"] for o in out] == ["2021-06-01", "2021-06-01"]
    # ...while with the delta screen it does: MinerU and GROBID render math differently ("8" vs "eight"),
    # bigram coverage 7/9 < 0.8, token_set 97 >= 85 -> v1 (the fix for the measured 46% artifact rate)
    noisy = [{"sid": "x", "unit": "u", "text": "Training takes two days on one GPU with 8 cores"}]
    out = SV.date_sentences(noisy, v1, "2021-01-12", "2021-06-01", delta=delta)
    assert out[0]["date"] == "2021-01-12"
    out = SV.date_sentences(noisy, v1, "2021-01-12", "2021-06-01")             # no delta -> stays latest
    assert out[0]["date"] == "2021-06-01"
    out = SV.date_sentences(sents, None, None, "2021-06-01")   # v1 never parsed: latest known date (leak-safe)
    assert all(o["date"] == "2021-06-01" for o in out)


# ---------------------------------------------------------------- fetch marathon
TITLES = {"arxiv:1": "Graph attention networks", "arxiv:2": "Missing in sciverse paper",
          "arxiv:3": "Broken text paper", "arxiv:4": "Short stub page paper"}
LONG = ("We propose a method for graph attention. " * 80) + \
    "\n# References\n[1] First reference. 2020.\n[2] Second reference. 2021.\n[3] Third reference. 2019.\n"


@pytest.fixture()
def fenv(tmp_path, monkeypatch):
    monkeypatch.setenv("CS_DATA", str(tmp_path / "data"))
    from compilescholar.core import paths
    importlib.reload(paths)
    from compilescholar.library import store as LS
    importlib.reload(LS)
    import compilescholar.documents.sciverse as SVm
    importlib.reload(SVm)
    reg = LS.connect()
    for pid, t in TITLES.items():
        reg.execute("INSERT INTO papers(paper_id, status, title, created_at) VALUES (?,'active',?,'now')", (pid, t))
        reg.execute("INSERT INTO records(paper_id, source, title, title_key) VALUES (?,?,?,?)",
                    (pid, "arxiv", t, ids.title_key(t)))
    reg.commit()
    reg.close()
    return SVm, tmp_path / "texts.sqlite"


def _fake(monkeypatch, calls, content_fails=(), short=()):
    def fake(method, path, *, payload=None, query=None, timeout_seconds=None, token=None, base_url=None):
        calls.append((method, path, payload or query))
        if path == "/agentic-search":
            t = payload["query"]
            pid = next(p for p, x in TITLES.items() if x == t)
            if pid == "arxiv:2":
                return {"hits": [{"doc_id": "DX", "title": "Some other paper entirely"}]}
            return {"hits": [{"doc_id": f"D-{pid}", "title": f"title:{t}",
                              "publication_published_date": "2021-01-01"}]}
        if path == "/content":
            doc = query["doc_id"]
            if doc in content_fails:
                raise S.SourceAdapterError("boom")
            if doc in short:
                return {"text": "too short", "more": "False"}
            return {"text": LONG, "more": "False"}
        raise AssertionError(path)
    monkeypatch.setattr("compilescholar.sources.sciverse.request_json", fake)


def test_fetch_ok_absent_error_gave_up(fenv, monkeypatch):
    SVm, tp = fenv
    calls = []
    _fake(monkeypatch, calls, content_fails={"D-arxiv:3"}, short={"D-arxiv:4"})
    n = SVm.fetch(sorted(TITLES), workers=2, log=lambda *a: None, path=tp)
    assert n["ok"] == 1 and n["absent"] == 1 and n["error"] == 2
    con = sqlite3.connect(tp)
    row = con.execute("SELECT doc_id, chars, has_refs, text_z FROM texts WHERE paper_id='arxiv:1'").fetchone()
    assert row[0] == "D-arxiv:1" and row[1] == len(LONG) and row[2] == 1
    assert zlib.decompress(row[3]).decode() == LONG
    st = dict(con.execute("SELECT paper_id, status FROM texts").fetchall())
    assert st == {"arxiv:1": "ok", "arxiv:2": "absent", "arxiv:3": "error", "arxiv:4": "error"}
    con.close()

    calls.clear()
    SVm.fetch(sorted(TITLES), workers=2, log=lambda *a: None, path=tp)     # ok/absent skipped, errors retried
    searched = {c[2]["query"] for c in calls if c[1] == "/agentic-search"}
    assert searched == {"Broken text paper", "Short stub page paper"}      # only the two failures were retried
    con = sqlite3.connect(tp)
    att = dict(con.execute("SELECT paper_id, attempts FROM texts WHERE paper_id IN ('arxiv:3','arxiv:4')"))
    assert att == {"arxiv:3": 2, "arxiv:4": 2}
    con.close()

    calls.clear()
    SVm.fetch(sorted(TITLES), workers=2, log=lambda *a: None, path=tp)     # 3rd attempt -> gave_up
    con = sqlite3.connect(tp)
    st = dict(con.execute("SELECT paper_id, status FROM texts").fetchall())
    assert st["arxiv:3"] == "gave_up" and st["arxiv:4"] == "gave_up"
    con.close()

    calls.clear()
    out = SVm.fetch(sorted(TITLES), workers=2, log=lambda *a: None, path=tp)
    assert out == {"todo": 0} and calls == []                              # fully idempotent


def test_materialize_from_cached_text(fenv, monkeypatch):
    SVm, tp = fenv
    calls = []
    _fake(monkeypatch, calls)
    SVm.fetch(["arxiv:1"], workers=1, log=lambda *a: None, path=tp)
    con = sqlite3.connect(tp)
    row = SVm.materialize(con, "arxiv:1", _fast(V1S), "2021-01-12", _fast(V2S), "2021-06-01")
    assert row is not None
    pid, doc_id, held, chars, n_units, n_sents, n_v1, sv_z = row
    assert (pid, doc_id, held) == ("arxiv:1", "D-arxiv:1", "ambiguous")    # the fake text matches neither version
    payload = json.loads(zlib.decompress(sv_z).decode())
    assert payload["has_refs"] is True and len(payload["sentences"]) == n_sents
    assert {s["date"] for s in payload["sentences"]} == {"2021-06-01"}     # nothing matches v1 -> latest date
    con.close()
    assert SVm.materialize(sqlite3.connect(tp), "arxiv:2", None, None, None, None) is None
