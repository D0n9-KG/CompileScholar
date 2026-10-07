# -*- coding: utf-8 -*-
"""Phase C④: the per-paper reading (tier priority, sentence dates, chunks) and the four passes as pure functions
(LLM stubbed at the call boundary — the prompts' JSON contract, the deterministic gates and the statement shapes
are what's under test). The stage wiring is covered by test_dfc_endtoend's extract section."""
from __future__ import annotations

import json

from compilescholar.extract import passes as PS
from compilescholar.extract import reading as RD

PID = "arxiv:2101.00002"


class FakeD:
    """The slice of documents.build.Documents the reading layer uses."""

    def __init__(self, docs=None, sv=None, deltas=None):
        self.docs, self._sv, self.deltas = docs or {}, sv, deltas or {}

    def versions(self, pid):
        return sorted({int(k.rsplit("@v", 1)[1]) for k in self.docs if k.startswith(pid + "@v")})

    def get(self, key):
        return self.docs.get(key)

    def sv(self, pid):
        return self._sv

    def delta(self, pid):
        return self.deltas.get(pid)


class FakeReg:
    def __init__(self, rows=None):
        self.rows = rows or {}

    def execute(self, sql, params=()):
        return FakeCur(self.rows.get((sql.split("FROM")[0].strip(), params), []))


class FakeCur:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


V1_TEXT = "We propose FastGF an efficient graph transformer for large graphs"
V2_NEW = "We also prove a convergence bound for the attention layer here"


def _doc(key, sents, date, units=None, fast=True):
    d = {"key": key, "text_date": date, "fast": None, "careful": None}
    if fast:
        d["fast"] = {"abstract": "", "units": units if units is not None else
                     [{"uid": f"{key}#para1", "kind": "para", "section": "1 Intro", "text": " ".join(sents)}],
                     "sentences": [{"sid": f"{key}#s{i + 1}", "unit": f"{key}#para1", "text": t}
                                   for i, t in enumerate(sents)],
                     "entries": {}, "cites": []}
    return d


# ---------------------------------------------------------------- reading
def test_full_text_grobid_with_delta():
    v1 = _doc(f"{PID}@v1", [V1_TEXT, "Training takes two days on one GPU"], "2021-01-12")
    v2 = _doc(f"{PID}@v2", [V1_TEXT, "Training takes three days on one GPU", V2_NEW], "2021-06-01")
    delta = {"sentences": [{"sid": f"{PID}@v2#s3", "unit": "u", "text": V2_NEW}],
             "revised": [{"sid": f"{PID}@v2#s2", "unit": "u", "text": "Training takes three days on one GPU",
                          "v1_text": "Training takes two days on one GPU"}],
             "entries": {}, "cites": []}
    D = FakeD({f"{PID}@v1": v1, f"{PID}@v2": v2}, deltas={PID: delta})
    full = RD.full_text(D, PID)
    assert full["source"] == "grobid"
    dates = {s["text"]: s["date"] for s in full["sentences"]}
    assert dates[V1_TEXT] == "2021-01-12"
    assert dates["Training takes three days on one GPU"] == "2021-06-01"   # the delta's revised sentence
    assert dates[V2_NEW] == "2021-06-01"
    assert sum(s["in_delta"] for s in full["sentences"]) == 2


def test_full_text_prefers_sciverse_and_skips_dup_delta():
    v1 = _doc(f"{PID}@v1", [V1_TEXT], "2021-01-12")
    v2 = _doc(f"{PID}@v2", [V1_TEXT, V2_NEW], "2021-06-01")
    sv = {"doc_id": "D1", "held": "latest", "detail": {}, "has_refs": True,
          "units": [{"uid": f"{PID}@sv#para1", "kind": "para", "section": "", "text": V1_TEXT + " " + V2_NEW}],
          "sentences": [{"sid": f"{PID}@sv#s1", "unit": f"{PID}@sv#para1", "text": V1_TEXT, "date": "2021-01-12"},
                        {"sid": f"{PID}@sv#s2", "unit": f"{PID}@sv#para1", "text": V2_NEW, "date": "2021-06-01"}]}
    delta = {"sentences": [{"sid": f"{PID}@v2#s2", "unit": "u", "text": V2_NEW}], "revised": [],
             "entries": {}, "cites": []}
    D = FakeD({f"{PID}@v1": v1, f"{PID}@v2": v2}, sv=sv, deltas={PID: delta})
    full = RD.full_text(D, PID)
    assert full["source"] == "sciverse" and len(full["sentences"]) == 2   # held=latest: no delta append
    sv["held"] = "v1"
    full = RD.full_text(D, PID)
    assert len(full["sentences"]) == 2                                    # the delta sentence is already in the text


def test_full_text_careful_splits_sentences():
    careful_units = [{"uid": "c#para1", "kind": "para", "section": "1 Intro",
                      "text": "We propose FastGF for graphs. It trains in two days on one GPU."},
                     {"uid": "c#table1", "kind": "table", "section": "", "text": "Table 1: x",
                      "html": "<table><tr><td>a</td></tr></table>"}]
    base = {"key": f"{PID}@v1", "text_date": "2021-01-12", "fast": None,
            "careful": {"units": careful_units, "references": []}}
    D = FakeD({f"{PID}@v1": base})
    full = RD.full_text(D, PID)
    assert full["source"] == "mineru"
    assert [s["text"] for s in full["sentences"]] == ["We propose FastGF for graphs.",
                                                      "It trains in two days on one GPU."]
    assert all(s["date"] == "2021-01-12" for s in full["sentences"])


def test_chunks_exclude_and_number():
    units = [{"uid": "u0", "kind": "section", "section": "", "text": "1 Method"},
             {"uid": "u1", "kind": "para", "section": "1 Method", "text": "x"},
             {"uid": "u2", "kind": "section", "section": "", "text": "Acknowledgements"},
             {"uid": "u3", "kind": "para", "section": "1 Method > Acknowledgements", "text": "y"},
             {"uid": "u4", "kind": "formula", "section": "1 Method", "text": "$$E=mc^2$$"},
             {"uid": "u5", "kind": "table", "section": "1 Method", "text": "<table></table>"},
             {"uid": "u6", "kind": "para", "section": "2 Experiments", "text": "z"}]
    sentences = [{"sid": "s1", "unit": "u1", "text": "First sentence here.", "date": "d1", "in_delta": 0},
                 {"sid": "s3", "unit": "u3", "text": "Thanks to everyone.", "date": "d1", "in_delta": 0},
                 {"sid": "s4", "unit": "u6", "text": "Last sentence here.", "date": "d2", "in_delta": 1},
                 {"sid": "s9", "unit": None, "text": "Delta sentence from another tier.", "date": "d2",
                  "in_delta": 1}]
    ch = RD.chunks(units, sentences)
    text = "\n".join(c["text"] for c in ch)
    assert "## 1 Method" in text and "$$E=mc^2$$" in text
    assert "Thanks to everyone." not in text and "<table>" not in text
    assert "[1] First sentence here." in text and "[later-version content]" in text
    n_by_sid = {s["sid"]: s["n"] for s in sentences if s.get("n")}
    assert n_by_sid["s1"] == 1 and n_by_sid["s9"] == 3            # the excluded sentence does not consume a number
    assert all(c["sid_by_n"] for c in ch)


class R2(FakeReg):
    def execute(self, sql, params=()):
        if "SELECT title FROM records" in sql:
            return FakeCur([("Faster graph transformers",)])
        if "SELECT abstract FROM records" in sql:
            return FakeCur([("Registry abstract text that is long enough to split.",)])
        if "SELECT first_hi FROM papers" in sql:
            return FakeCur([("2021-01-12",)])
        return FakeCur([])


def test_t1_input_sources():
    v1 = _doc(f"{PID}@v1", ["body"], "2021-01-12")
    v1["fast"]["abstract"] = "We propose FastGF. It is fast."
    D = FakeD({f"{PID}@v1": v1})
    inp = RD.t1_input(D, R2(), PID)
    assert inp["source"] == "grobid_v1" and inp["date"] == "2021-01-12"
    assert [s["n"] for s in inp["sentences"]] == [1, 2]
    inp2 = RD.t1_input(FakeD({}), R2(), PID)                        # no parse: registry abstract, marked, v1 date
    assert inp2["source"] == "registry" and inp2["date"] == "2021-01-12"


# ---------------------------------------------------------------- passes (stubbed LLM)
class StubChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, prompt, **kw):
        self.calls.append((prompt, kw))
        return self.replies.pop(0) if self.replies else "{}"


def _t1_input():
    return {"paper_id": PID, "title": "Faster graph transformers", "date": "2021-01-12", "source": "grobid_v1",
            "sentences": [
                {"n": 1, "text": "We propose FastGF, an efficient graph transformer for large graphs."},
                {"n": 2, "text": "FastGF trains in two days on one GPU, unlike prior methods."},
                {"n": 3, "text": "We use the Adam optimizer with a learning rate of 3e-4."}]}


def test_t1_items_names_and_validation():
    reply = json.dumps({"items": [
        {"n": 1, "facet": "contribution", "role": "proposes", "epistemic": "stated", "condition": "",
         "text": "proposes FastGF for large graphs"},
        {"n": 2, "facet": "result", "role": "describes", "epistemic": "demonstrated", "condition": "on one GPU",
         "text": "trains in two days"},
        {"n": 9, "facet": "result", "role": "describes", "epistemic": "stated", "condition": "", "text": "x"},
        {"n": 3, "facet": "vibes", "role": "describes", "epistemic": "stated", "condition": "", "text": "x"}],
        "names": [
            {"name": "FastGF", "aliases": [], "artefact": "method", "relation": "proposes", "n": 1},
            {"name": "Adam", "aliases": [], "artefact": "method", "relation": "uses", "n": 3},
            {"name": "NotInText", "aliases": [], "artefact": "method", "relation": "uses", "n": 2}]})
    chat = StubChat([reply])
    stmts, names, st = PS.t1(_t1_input(), chat=chat)
    assert st["bad_n"] == 1 and st["bad_vocab"] == 1 and st["name_not_literal"] == 1
    assert len(stmts) == 4                                           # 2 items + 2 names
    s0 = stmts[0]
    assert (s0.quote, s0.date, s0.about) == (_t1_input()["sentences"][0]["text"], "2021-01-12", PID)
    assert s0.loc["sent_id"] == f"{PID}@abs1" and s0.validate() == []
    prop = next(s for s in stmts if s.meta.get("name") == "FastGF")
    assert prop.role == "proposes" and prop.epistemic == "stated"
    uses = next(s for s in stmts if s.meta.get("name") == "Adam")
    assert uses.role == "uses"
    assert names == ["FastGF"]                                       # uses-names are not own methods
    assert chat.calls[0][1]["max_tokens"] == 3000


def test_t1_parse_failure_is_item_failure():
    stmts, names, st = PS.t1(_t1_input(), chat=StubChat(["not json at all {{{"]))
    assert stmts is None and st["parse_failed"] == 1


def _full():
    units = [{"uid": "u1", "kind": "para", "section": "1 Intro", "text": "x"},
             {"uid": "u2", "kind": "para", "section": "3 Experiments", "text": "y"}]
    sentences = [{"sid": "s1", "unit": "u1", "text": "We propose FastGF for large graphs here.",
                  "date": "2021-01-12", "in_delta": 0},
                 {"sid": "s2", "unit": "u2", "text": "With a batch size of 32, accuracy reaches 91.2%.",
                  "date": "2021-06-01", "in_delta": 1}]
    return {"paper_id": PID, "source": "sciverse", "units": units, "sentences": sentences}


def test_t2_anchors_dates_config_and_own_methods():
    reply = json.dumps({"items": [
        {"n": 1, "facet": "contribution", "role": "proposes", "epistemic": "stated", "condition": "",
         "text": "proposes FastGF", "mentions": [{"name": "Transformer", "relation": "extends"},
                                                 {"name": "GhostNet", "relation": "uses"}]},
        {"n": 2, "facet": "result", "role": "describes", "epistemic": "demonstrated", "condition": "",
         "text": "accuracy 91.2%"},
        {"n": 1, "facet": "result", "role": "describes", "epistemic": "stated", "condition": "",
         "text": "anchored outside its chunk? no — same chunk"}],
        "config": [{"item": "batch size", "value": "32", "unit": "", "applies_to": "training", "n": 2},
                   {"item": "lr", "value": "1e-9", "unit": "", "applies_to": "", "n": 2}],
        "own_methods": ["FastGF", "HallucinatedNet"]})
    chat = StubChat([reply])
    stmts, own, st = PS.t2(PID, "Faster graph transformers", _full(), seed_own=["Seed"], chat=chat)
    assert stmts is not None
    by_text = {s.text: s for s in stmts}
    s1 = by_text["proposes FastGF"]
    assert s1.date == "2021-01-12" and s1.meta["in_delta"] == 0 and s1.loc["sent_id"] == "s1"
    assert s1.meta["mentions"] == []                                  # neither name occurs literally in s1
    s2 = by_text["accuracy 91.2%"]
    assert s2.date == "2021-06-01" and s2.meta["in_delta"] == 1
    assert s2.quote == "With a batch size of 32, accuracy reaches 91.2%."
    cfg = by_text["batch size = 32"]
    assert cfg.facet == "config" and cfg.meta["config"]["item"] == "batch size" and cfg.validate() == []
    assert "lr = 1e-9" not in by_text                                 # value not in the sentence -> dropped
    assert st["config_value_unsupported"] == 1
    assert own == ["FastGF", "Seed"]                                  # hallucinated name is not in the text
    assert all(s.validate() == [] for s in stmts)


def test_results_gate_values_and_quotes():
    html = ("<table><tr><td>Method</td><td>Acc</td><td>Time</td></tr>"
            "<tr><td>FastGF</td><td>91.2%</td><td rowspan='2'>2d</td></tr>"
            "<tr><td>Baseline</td><td>88.0%</td></tr></table>")
    tb = [{"uid": "t1", "html": html, "caption": "Table 1: main results", "date": "2021-06-01",
           "context": ["FastGF beats the baseline."]}]
    axes = {"usable": True, "object_axis": "row", "object_index": 0, "header_rows": [0],
            "conditions": [], "measures": [{"index": 1, "metric": "accuracy", "unit": "%", "direction": "higher"}],
            "skip_rows": []}
    chat = StubChat([json.dumps(axes)])
    stmts, st = PS.results(PID, tb, own_methods=["FastGF"], chat=chat)
    assert len(stmts) == 2 and st["cells"] == 2 and st["tables_written"] == 1
    mine = next(s for s in stmts if s.meta["object"] == "FastGF")
    other = next(s for s in stmts if s.meta["object"] == "Baseline")
    assert mine.epistemic == "demonstrated" and other.epistemic == "cited"
    assert mine.meta["value"] == "91.2%" and "91.2%" in mine.quote    # the number comes from the cell
    assert mine.quote == "FastGF 91.2% 2d"                            # pre-expansion cell texts, no rowspan dup
    assert mine.date == "2021-06-01" and mine.loc["sent_id"] == f"{PID}@t1:r1"
    assert mine.validate() == []
    # gates: unusable / column-object / no valid measure
    for bad, key in (({"usable": False}, "unusable"),
                     ({**axes, "object_axis": "column"}, "gate_column_object"),
                     ({**axes, "measures": [{"index": 99, "metric": "x"}]}, "gate_no_measures")):
        _, st2 = PS.results(PID, tb, chat=StubChat([json.dumps(bad)]))
        assert st2.get(key) == 1, key


def test_other_batch_pairs_signals_and_self_cite():
    items = [{"s": 1, "sentence": "GraphFormer [1] applies attention to graphs, unlike our method.",
              "date": "2021-06-01", "citing_key": f"{PID}@v2", "sid": f"{PID}@v2#s3", "in_delta": 1,
              "cites": [{"key": "b0", "cited": "arxiv:2001.00001", "title": "GraphFormer", "raw": "gf",
                         "self_cite": 1}]},
             {"s": 2, "sentence": "We build on the transformer architecture [2].", "date": "2021-01-12",
              "citing_key": f"{PID}@v1", "sid": f"{PID}@v1#s1", "in_delta": 0,
              "cites": [{"key": "b1", "cited": "stub:attention is all you need", "title": "", "raw": "vaswani",
                         "self_cite": 0}]}]
    reply = json.dumps({"pairs": [
        {"s": 1, "key": "b0", "function": "contrast", "role": "compares",
         "about": "GraphFormer applies attention to graphs", "facet": "method",
         "epistemic": "cited", "category": None, "limitation": None, "name": "GraphFormer",
         "outcome": None, "builds_on": None, "builds_on_relation": None},
        {"s": 2, "key": "b1", "function": "basis", "role": "uses", "about": "something fully invented here",
         "facet": "contribution", "epistemic": "stated", "category": None, "limitation": None, "name": None,
         "outcome": "citing_better", "builds_on": None, "builds_on_relation": None},
        {"s": 2, "key": "bX", "function": "basis", "role": "uses", "about": "x", "facet": "contribution",
         "epistemic": "stated"}]})
    chat = StubChat([reply])
    stmts, st = PS.other_batch(PID, items, chat=chat)
    assert st["parsed"] == 2 and st.get("missing", 0) == 0             # the unknown key bX is ignored, not written
    s0 = next(s for s in stmts if s.about == "arxiv:2001.00001")
    assert s0.kind == "other" and s0.role == "compares" and s0.function == "contrast"
    assert s0.meta["self_cite"] == 1 and s0.meta["in_delta"] == 1 and s0.meta["supported"] is True
    assert s0.quote == items[0]["sentence"] and s0.date == "2021-06-01"
    assert s0.loc == {"unit_id": f"{PID}@v2", "sent_id": f"{PID}@v2#s3"}
    assert s0.group == ("arxiv:2001.00001",) and s0.validate() == []
    s1 = next(s for s in stmts if s.about.startswith("stub:"))
    assert s1.meta["supported"] is False                             # 0.6 support downgraded to a signal
    assert "something fully invented here" in s1.text                # ...and the text is kept
    assert s1.meta["outcome"] is None                                # no comparative wording in sentence 2
    assert PS.other_batch(PID, items, chat=StubChat(["garbage"]))[0] is None


def test_deep_read_composes():
    v1 = _doc(f"{PID}@v1", [V1_TEXT], "2021-01-12")
    v1["fast"]["abstract"] = "We propose FastGF, an efficient graph transformer for large graphs."
    D = FakeD({f"{PID}@v1": v1})

    class Reg:
        def execute(self, sql, params=()):
            if "SELECT title FROM records" in sql:
                return FakeCur([("Faster graph transformers",)])
            return FakeCur([])

    t1_reply = json.dumps({"items": [{"n": 1, "facet": "contribution", "role": "proposes", "epistemic": "stated",
                                      "condition": "", "text": "proposes FastGF"}],
                           "names": [{"name": "FastGF", "aliases": [], "artefact": "method",
                                      "relation": "proposes", "n": 1}]})
    t2_reply = json.dumps({"items": [{"n": 1, "facet": "method", "role": "describes", "epistemic": "stated",
                                      "condition": "", "text": "an efficient graph transformer"}],
                           "config": [], "own_methods": []})
    out = PS.deep_read(D, Reg(), PID, chat=StubChat([t1_reply, t2_reply]))
    assert out["source"] == "grobid"
    assert {s.pass_name for s in out["statements"]} == {"t1", "t2"}
    assert out["own_methods"] == ["FastGF"]


def test_mineru_figure_units():
    from compilescholar.documents import mineru as MI
    blocks = [{"type": "image", "image_caption": ["Figure 3: the architecture."], "page_idx": 2,
               "bbox": [10, 20, 300, 400]},
              {"type": "chart", "chart_caption": [], "page_idx": 5, "bbox": [0, 0, 100, 100]}]
    units, refs = MI.units(blocks, "doc")
    figs = [u for u in units if u.kind == "figure"]
    assert len(figs) == 2                                            # the captionless chart is kept as an anchor
    assert figs[0].text == "Figure 3: the architecture." and figs[0].page == 2 and figs[0].bbox == [10, 20, 300, 400]
    assert any(u.kind == "caption" for u in units)
