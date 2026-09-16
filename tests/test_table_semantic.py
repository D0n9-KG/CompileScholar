# -*- coding: utf-8 -*-
"""F35 semantic table layer unit tests — deterministic gate + correction +
trigger ID. LLM is mocked (no calls). Run:
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_table_semantic.py -q
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kb_compiler.records import table_semantic as TS


# ---------------------------------------------------------------- fixtures
def _rec(metric, surf, value, subject=None, th="<tr><td>Dataset</td><td>RCE</td></tr>"):
    r = {"kind": "result", "chunk_id": "p1#table", "table_header": th,
         "measure": {"metric": metric, "value": value, "unit": "", "direction": "",
                     "aggregation": "", "timepoint": ""},
         "method_ref": {"surface": surf, "canonical": None, "entity_id": None},
         "dims": {}, "quote": f"<tr><td>{surf}</td><td>{value}</td></tr>",
         "paper_id": "p1", "chunk_char_start": 100, "provenance": "table_channel_v2"}
    if subject:
        r["dims_new"] = {"dims.subject": [subject]}
    return r


REG = {"entities": [
    {"entity_id": "e1", "canonical": "TD3+BC", "aliases": ["TD3+BC", "td3bc"]},
    {"entity_id": "e2", "canonical": "ReCOIL", "aliases": ["ReCOIL"]},
]}
LOOKUP = TS._build_entity_lookup(REG)


# ---------------------------------------------------------------- trigger ID
def test_trigger_no_subject():
    assert TS.is_trigger_record(_rec("RCE", "halfcheetah", "55.2")) == "no_subject"


def test_not_trigger_with_subject_clean_metric():
    r = _rec("accuracy", "TD3+BC", "55.2", subject="image classification")
    assert TS.is_trigger_record(r) == ""


def test_trigger_cond_label_metric():
    r = _rec("Base", "LLaMA-2", "6.7", subject="x")   # subject present but metric=condition
    assert TS.is_trigger_record(r) == "cond_label_metric"


def test_trigger_fused_metric():
    r = _rec("IHS [14] > IoU", "PSPNet", "0.5", subject="x")
    assert TS.is_trigger_record(r) == "fused_metric"


def test_group_tables_flags_trigger():
    rb = {"p1": {"records": [_rec("RCE", "halfcheetah", "55.2"),
                             _rec("ORIL", "halfcheetah", "60.1"),
                             _rec("accuracy", "TD3+BC", "70", subject="cls")]}}
    tabs = TS.group_tables(rb)
    # first two share table_header -> one trigger table; third has subject -> its own
    trig = {k: t for k, t in tabs.items() if t["trigger"]}
    assert len(trig) >= 1
    assert any(t["trigger"] == "no_subject" for t in trig.values())


# ---------------------------------------------------------------- table_repr
def test_table_repr_axes():
    recs = [_rec("RCE", "halfcheetah", "55.2"), _rec("ORIL", "hopper", "60.1")]
    rep = TS.table_repr(recs)
    assert rep["column_headers"] == ["RCE", "ORIL"]      # currently metric == methods
    assert rep["row_labels"] == ["halfcheetah", "hopper"]  # currently surface == datasets


# ---------------------------------------------------------------- gate
def test_gate_accept_valid():
    rep = TS.table_repr([_rec("RCE", "halfcheetah", "55.2")])
    v, p2, why = TS.gate({"method_axis": "column", "metric_name": "return",
                          "confidence": "high"}, rep,
                         "Table 3: average episodic return across datasets.",
                         "return over 100 evaluations", LOOKUP)
    assert v == "accept"
    assert p2["metric_name"] == "return"   # G4 passes (return in caption)


def test_gate_g3_rejects_benchmark_metric():
    rep = TS.table_repr([_rec("RCE", "halfcheetah", "55.2")])
    v, p2, why = TS.gate({"method_axis": "column", "metric_name": "imagenet accuracy",
                          "confidence": "high"}, rep, "Table 3: ...", "...", LOOKUP)
    assert v == "reject"
    assert any("G3" in r for r in why)


def test_gate_g4_degrades_ungrounded_metric():
    rep = TS.table_repr([_rec("RCE", "halfcheetah", "55.2")])
    v, p2, why = TS.gate({"method_axis": "column", "metric_name": "flurboscore",
                          "confidence": "high"}, rep, "Table 3: average performance.",
                         "some context without the word", LOOKUP)
    assert v == "accept"            # degrade, not reject
    assert p2["metric_name"] == ""  # G4 dropped it
    assert any("G4" in r for r in why)


def test_gate_g6_low_confidence_drops_metric():
    rep = TS.table_repr([_rec("RCE", "halfcheetah", "55.2")])
    v, p2, why = TS.gate({"method_axis": "column", "metric_name": "return",
                          "confidence": "low"}, rep, "Table 3: return.", "return", LOOKUP)
    assert v == "accept"
    assert p2["metric_name"] == ""  # G6: low conf -> keep F32 metric
    assert any("G6" in r for r in why)


def test_gate_g1_rejects_column_axis_no_headers():
    rep = {"column_headers": [], "row_labels": ["a"], "header_html": "", "row_quote_sample": ""}
    v, p2, why = TS.gate({"method_axis": "column", "metric_name": "", "confidence": "high"},
                         rep, "", "", LOOKUP)
    assert v == "reject"
    assert any("G1" in r for r in why)


# ---------------------------------------------------------------- correction
def test_apply_correction_role_reversal():
    recs = [_rec("TD3+BC", "halfcheetah-medium", "55.2"),
            _rec("ReCOIL", "halfcheetah-medium", "60.1")]
    prop = {"method_axis": "column", "metric_name": "return", "confidence": "high"}
    out, n = TS.apply_correction(recs, prop, LOOKUP)
    assert n == 2
    r0 = out[0]
    # method <- old column header (TD3+BC), re-linked to registry e1
    assert r0["method_ref"]["surface"] == "TD3+BC"
    assert r0["method_ref"]["entity_id"] == "e1"
    # subject <- old row label (dataset)
    assert r0["dims_new"]["dims.subject"] == ["halfcheetah-medium"]
    # metric <- recovered true metric
    assert r0["measure"]["metric"] == "return"
    assert r0["provenance"] == "f35_semantic"
    # value preserved
    assert r0["measure"]["value"] == "55.2"


def test_apply_correction_row_axis_noop():
    recs = [_rec("accuracy", "TD3+BC", "55.2")]
    out, n = TS.apply_correction(recs, {"method_axis": "row", "metric_name": "",
                                        "confidence": "high"}, LOOKUP)
    assert n == 0
    assert out[0]["provenance"] == "table_channel_v2"   # untouched


def test_apply_correction_unlinked_method_keeps_surface():
    recs = [_rec("SomeUnknownMethod", "somedataset", "55.2")]
    out, n = TS.apply_correction(recs, {"method_axis": "column", "metric_name": "",
                                        "confidence": "high"}, LOOKUP)
    assert out[0]["method_ref"]["surface"] == "SomeUnknownMethod"
    assert out[0]["method_ref"]["entity_id"] is None     # never fabricate
    assert out[0]["dims_new"]["dims.subject"] == ["somedataset"]


def test_correction_does_not_mutate_input():
    recs = [_rec("TD3+BC", "halfcheetah", "55.2")]
    TS.apply_correction(recs, {"method_axis": "column", "metric_name": "return",
                               "confidence": "high"}, LOOKUP)
    assert recs[0]["provenance"] == "table_channel_v2"   # original untouched (deep copy)
    assert recs[0]["measure"]["metric"] == "TD3+BC"
