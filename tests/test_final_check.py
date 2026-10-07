# -*- coding: utf-8 -*-
"""Phase C⑤: the match view, the unified final check (one repair then discard with violation code, fuzzy for
reporting only) and the deterministic halves of the sentinels (the two pipeline traps, the scorers on canned
statements, the chemistry adapter shape). The LLM halves of the sentinels run through `cli sentinel`."""
from __future__ import annotations

import json

from compilescholar.documents import matchview as MV
from compilescholar.documents import sciverse as SV
from compilescholar.extract import final_check as FC
from compilescholar.extract import sentinels as SN
from compilescholar.extract.schema import Statement

PID = "arxiv:2101.00002"
SRC = ("We propose FastGF, an efficient graph transformer. Training takes two days on one GPU. "
       "Accuracy reaches 91.2% on BenchX, unlike the 88.0 of the baseline [12].")
SENTS = {"s1": "We propose FastGF, an efficient graph transformer.",
         "s2": "Training takes two days on one GPU.",
         "s3": "Accuracy reaches 91.2% on BenchX, unlike the 88.0 of the baseline [12]."}


def _st(text, quote, sent_id="s1", **over):
    base = dict(speaker=PID, date="2021-01-12", kind="self", about=PID, role="proposes", facet="contribution",
                text=text, quote=quote, loc={"unit_id": "u1", "sent_id": sent_id}, pass_name="t2", item=PID)
    return Statement(**{**base, **over})


# ---------------------------------------------------------------- match view
def test_view_folds_renderings_together():
    a = MV.view("The value $\\zeta_{k}$ is 4.2×10⁻⁵ mol [3, 7].")
    b = MV.view("The value ζ k is 4.2×10⁻⁵ mol.")
    assert a == b                                   # latex vs unicode, citation markers, whitespace, case
    assert "87.5" in MV.view("<td>87.5</td>")       # html tags out, the decimal point kept
    assert MV.view("a ± b ≤ c") == MV.view("a \\pm b \\leq c")


def test_view_does_not_eat_math_angle_brackets():
    """Regression: a generic <[^>]+> paired a math '<' in one sentence with a '>' in the next and swallowed
    everything between — 15% of a real math paper's statements were falsely discarded as QUOTE_NOT_IN_SOURCE."""
    s1 = "We may recover f with n < N samples."
    s2 = "This extends over previous works that have been limited to infinite width networks."
    s3 = "For d > 1 the rows stay orthogonal."
    source = " ".join([s1, s2, s3])
    assert MV.view(s2) in MV.view(source)          # compositionality across the join
    assert MV.view("n < N") in MV.view(source)     # the inequality survives
    assert MV.view("<td>87.5</td>") == MV.view("87.5")          # real tags are still stripped
    assert MV.view("<mml:math>x</mml:math>") == MV.view("x")


def test_locate_maps_back_to_original_span():
    sent = "We use <b>FastGF</b> on graphs."
    span = MV.locate("FastGF", sent)
    assert sent[span[0]:span[1]] == "FastGF"        # tags between matched chars would be spanned; none here
    sent2 = "We use FastGF on graphs."
    assert MV.locate("FastGF on graphs", sent2) == (7, 23)
    assert MV.locate("not here", sent) is None


# ---------------------------------------------------------------- final check
def test_clean_statements_pass_and_loc_fills():
    kept, disc, st = FC.run([_st("proposes FastGF for graphs", SENTS["s1"])], SRC, SENTS,
                            chat=lambda *a, **k: None, item=PID)
    assert len(kept) == 1 and not disc and st["clean"] == 1
    assert kept[0].loc["char_start"] == 0 and kept[0].loc["char_end"] == len(SENTS["s1"])


def test_quote_not_in_source_discarded_with_code_and_fuzzy_report():
    bad = _st("proposes FastGF for graphs and robotics", "We propose FastGF, an efficient graph transformer XY.")
    kept, disc, st = FC.run([bad], SRC, SENTS, chat=lambda *a, **k: None, item=PID)
    assert not kept and len(disc) == 1
    assert disc[0][1][0] == FC.QUOTE_NOT_IN_SOURCE
    assert st["v_QUOTE_NOT_IN_SOURCE"] == 1 and st["fuzzy_samples"]      # reported, never passed


def test_rewritten_number_discarded():
    bad = _st("accuracy reaches 99.9% on BenchX", SENTS["s3"], sent_id="s3")
    kept, disc, st = FC.run([bad], SRC, SENTS, chat=lambda *a, **k: None, item=PID)
    assert not kept and disc[0][1][0].startswith("NUMBER_REWRITTEN")
    ok = _st("accuracy reaches 91.2% on BenchX", SENTS["s3"], sent_id="s3")
    kept, disc, st = FC.run([ok], SRC, SENTS, chat=lambda *a, **k: None, item=PID)
    assert len(kept) == 1 and not disc


def test_number_in_citation_marker_is_not_a_rewrite():
    """The match view strips "[25]" (correct for quote location), but the number check reads the RAW quote —
    a text mentioning the citation's number is verbatim-sourced, not rewritten."""
    src = "We extend the results of [25] to the finite case."
    sents = {"s1": src}
    ok = _st("extends the results of reference 25 to the finite case", src)
    kept, disc, st = FC.run([ok], src, sents, chat=lambda *a, **k: None, item=PID)
    assert len(kept) == 1 and not disc


def test_one_repair_then_keep_or_discard():
    bad_quote = _st("proposes FastGF for graphs", "We propose FastGF, an efficient graph transformer!!")
    bad_beyond = _st("claims a robotics extension", "FastGF does robotics.", sent_id="s2")
    calls = []

    def repair_chat(prompt, **kw):
        calls.append(prompt)
        return json.dumps({"fixed": [{"i": 1, "drop": False, "text": "proposes FastGF for graphs",
                                      "quote": SENTS["s1"], "facet": "contribution", "role": "proposes",
                                      "epistemic": "stated", "condition": ""},
                                     {"i": 2, "drop": True}]})

    kept, disc, st = FC.run([bad_quote, bad_beyond], SRC, SENTS, chat=repair_chat, item=PID)
    assert len(calls) == 1                                            # exactly one repair round for the item
    assert len(kept) == 1 and kept[0].meta.get("repaired") is True
    assert kept[0].quote == SENTS["s1"] and kept[0].loc["char_end"] == len(SENTS["s1"])
    assert len(disc) == 1 and st["discarded"] == 1 and st["repaired"] == 1


def test_schema_violation_not_repairable_into_existence():
    bad = _st("x", SENTS["s1"], facet="vibes")
    kept, disc, st = FC.run([bad], SRC, SENTS, chat=lambda *a, **k: None, item=PID)
    assert not kept and disc[0][1][0].startswith("SCHEMA")


# ---------------------------------------------------------------- sentinels (deterministic halves)
def test_pipeline_traps_never_fire():
    g = SN._gate_traps()
    assert g == {"T7_quote_not_substring_caught": True, "T8_number_rewritten_caught": True}


def _res(obj, val, metric="", cond="", epistemic="demonstrated"):
    return Statement(speaker=SN.CANARY_PID, date=SN.DATE, kind="self", about=SN.CANARY_PID, role="describes",
                     facet="result", text=f"{obj}: {metric}: {val}", quote=f"{obj} {val}", epistemic=epistemic,
                     loc={"unit_id": "t", "sent_id": "t:r1"},
                     meta={"object": obj, "metric": metric, "value": val}, pass_name="results")


def test_canary_scorer_facts_and_traps():
    good = [_res("NOVA", "87.5", "score"), _res("UNIFORM", "64.2", "score"),
            _res("NOVA", "74.9", "CLS Swin-T accuracy"),
            Statement(speaker=SN.CANARY_PID, date=SN.DATE, kind="self", about=SN.CANARY_PID, role="describes",
                      facet="config", text="replay buffer size = 2M", quote="NOVA maintains a replay buffer of "
                      "size 2M transitions.", loc={"unit_id": "u", "sent_id": "x"},
                      meta={"config": {"item": "replay buffer size", "value": "2M", "unit": "", "applies_to": ""}},
                      pass_name="t2"),
            Statement(speaker=SN.CANARY_PID, date=SN.DATE, kind="self", about=SN.CANARY_PID, role="extends",
                      facet="method", text="NOVA extends LUNA with an overlap-aware aggregator",
                      quote="NOVA extends LUNA (Doe et al., 2019) with an overlap-aware aggregator.",
                      loc={"unit_id": "u", "sent_id": "y"},
                      meta={"mentions": [{"name": "LUNA", "relation": "extends"}]}, pass_name="t2"),
            Statement(speaker=SN.CANARY_PID, date=SN.DATE, kind="self", about=SN.CANARY_PID, role="describes",
                      facet="absence", text="does not evaluate on continuous control tasks",
                      quote="We do not evaluate on continuous control tasks.",
                      loc={"unit_id": "u", "sent_id": "z"}, pass_name="t2"),
            Statement(speaker=SN.CANARY_PID, date=SN.DATE, kind="other", about="stub:trace2020", role="criticizes",
                      facet="result", text="TRACE argues prioritized sampling introduces bias and corrects it "
                      "with per-sample importance weights", quote="TRACE (Kim et al., 2020) argues that ...",
                      function="background", loc={"unit_id": "u", "sent_id": "w"}, pass_name="other")]
    sc = SN.score_canary(good)
    assert all(sc["facts"].values()), sc["facts"]
    assert not any(sc["traps"].values()), sc["traps"]
    # each trap fires on its own poisoned statement
    poisoned = {
        "T2_epistemic_flip": [Statement(speaker=SN.CANARY_PID, date=SN.DATE, kind="self", about=SN.CANARY_PID,
                                        role="describes", facet="result", text="NOVA scores 91.3 on BenchY",
                                        quote="q", epistemic="demonstrated", loc={"unit_id": "u", "sent_id": "s"},
                                        pass_name="t2")],
        "T4_number_binding": [_res("NOVA", "79.1", "score")],
        "T5_widetable_misbinding": [_res("NOVA", "74.9", "QA RoBERTa-L accuracy")],
        "T6_own_epistemic_flip": [Statement(speaker=SN.CANARY_PID, date=SN.DATE, kind="self", about=SN.CANARY_PID,
                                            role="describes", facet="result", text="NOVA achieves 87.5 on BenchX",
                                            quote="q", epistemic="cited", loc={"unit_id": "u", "sent_id": "s"},
                                            pass_name="t2")],
        "T3_condition_drop": [Statement(speaker=SN.CANARY_PID, date=SN.DATE, kind="self", about=SN.CANARY_PID,
                                        role="describes", facet="result",
                                        text="the improvement over UNIFORM is significant only in sparse settings",
                                        quote="q", epistemic="demonstrated", condition="",
                                        loc={"unit_id": "u", "sent_id": "s"}, pass_name="t2")],
    }
    for trap, extra in poisoned.items():
        sc = SN.score_canary(good + extra)
        assert sc["traps"][trap], trap


def test_chem_adapter_shape():
    d = SV.doc(SN.CHEM_PID, SN.CHEM_TEXT)
    assert d["has_refs"] is True                                      # the UNTITLED trailing list is cut
    assert not any("iVBOR" in u["text"] for u in d["units"])          # base64 figure never reaches prose
    assert not any("J. Colloid" in s["text"] for s in d["sentences"])
    assert any(u["kind"] == "table" and "4.2×10⁻⁵" in u["text"] for u in d["units"])
    assert any("Ca²⁺" in s["text"] or "Ca2+" in s["text"] for s in d["sentences"])   # superscripts survive
