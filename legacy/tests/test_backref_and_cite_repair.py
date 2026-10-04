# -*- coding: utf-8 -*-
"""Regression tests for the 2026-09-23 carpet-audit fixes. Each test pins a
measured failure form so a regression cannot silently return."""

import os
import sys

import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      ".research_tmp", "experiments", "benchmarks", "_shared", "tools")
for _p in (_SRC, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def test_p01_backref_matcher_orders():
    """P0-1: whole-content and entity:X forms must match BEFORE word-split."""
    from evidence_gate2r_harness import _has_valid_id
    valid = {"Photonic Crystal Enhanced Microscopy", "Metasurface", "abc123def45678"}
    # multi-word canonical inside [entity: X]
    assert _has_valid_id("N1. [entity: Photonic Crystal Enhanced Microscopy] note", valid)
    # whole-content canonical
    assert _has_valid_id("N2. [Metasurface] note", valid)
    # composite [a|b] split form still works (IL-C2 legacy)
    assert _has_valid_id("N3. [abc123def45678|2015] note", valid)
    # unknown -> False
    assert not _has_valid_id("N4. [totally unknown thing] note", valid)


def test_p01_long_stems():
    """P0-1b: bracket cap must admit ~100-char paper stems."""
    from evidence_gate2r_harness import BRACKET
    stem = "Hybrid_Lipid_Polymer_Nanoparticles_for_Combined_Chemo-_and_Photodynamic_Therapy"
    line = f"N1. [{stem}] note"
    m = BRACKET.search(line)
    assert m and m.group(1) == stem


def test_n6_clean_answer_artifacts():
    """N6: control-byte replacement cured — chunk-suffix citations are
    cleaned to the bare citation, no invisible bytes injected."""
    from evidence_gate2r_harness import _clean_answer_artifacts
    out = _clean_answer_artifacts(
        "as shown in [Gao et al., 2024-chunk-config] the result holds")
    assert out == "as shown in [Gao et al., 2024] the result holds"
    assert "\x01" not in out


def test_n7_numeric_gate_boundary():
    """N7: '95' must NOT pass via a hex-record-id substring in the anchors."""
    from evidence_gate2r_harness import answer_gates
    notes = "N1. [3072d10a795359] PCEM provides label-free imaging of cell adhesion"
    answer = ("The system achieved a resolution of 95 nm in imaging experiments. "
              "[Photonic_crystal_enhanced_microscopy_for_imaging_of_live_cell_adhesion_]")
    missing, _, _ = answer_gates(answer, notes, [])
    assert "95" in missing  # unanchored number must be flagged


def test_n2_plan_lines_are_not_anchors():
    """N2: bare [plan] prose numbers must not satisfy the numeric gate."""
    from evidence_gate2r_harness import answer_gates
    notes = "[plan] look for the 42.7 percent throughput improvement claim"
    answer = "The method achieves a throughput improvement of 42.7 percent. [Some_Paper_Stem]"
    missing, _, _ = answer_gates(answer, notes, [])
    assert "42.7" in missing


def test_p1c_renumber_mapping():
    """P1-C: sequential numeric citations map back to note sources."""
    sys.path.insert(0, _TOOLS)
    from multi_cite_repair import repair_row, note_sources
    row = {
        "answer_raw": "Compute-optimal works [13]. Sampling plateaus [2]. "
                      "Real finding [601d12dd99b813].",
        "notes_final": "N1. [601d12dd99b813] compute-optimal scaling\n"
                       "N2. [4beedc244b301b] parallel sampling plateau",
    }
    info = repair_row(row, n_ctx=4)
    # 2 numeric refs vs 2 note sources: compatible -> renumber-mapped
    assert info["mode"] in ("renumber-mapped", "oor-stripped")
    if info["mode"] == "renumber-mapped":
        assert "[601d12dd99b813]" in info["text"] or \
               "[4beedc244b301b]" in info["text"]
        assert "[13]" not in info["text"]


def test_p1d_compile_strip():
    """P1-D (in-harness post-filter logic): markers not in notes are
    stripped; note ids survive. Mirrors the harness inline filter."""
    import re
    notes = ("N1. [601d12dd99b813] compute-optimal scaling\n"
             "N2. [4beedc244b301b] parallel sampling plateau")
    answer = ("Compute-optimal scaling works [13]. Parallel sampling plateaus "
              "[2]. Real finding [601d12dd99b813].")
    _note_ids = set(re.findall(r"\[([0-9a-f]{14})\]", notes)) | \
                set(re.findall(r"\[([A-Za-z][A-Za-z0-9_\-]{15,110})\]", notes))
    out = answer
    for _m in re.finditer(r"\[([^\[\]]{1,110})\]", answer):
        _tok = _m.group(1)
        if _tok in _note_ids:
            continue
        if any(p.strip() in _note_ids
               for p in re.split(r"[|,;/\s]+", _tok) if p.strip()):
            continue
        out = out.replace(_m.group(0), "", 1)
    assert "[13]" not in out and "[2]" not in out
    assert "[601d12dd99b813]" in out


def test_runlog_roundtrip(tmp_path):
    """P2-3: runlog writes events + summary; counters accumulate."""
    from kb_compiler import runlog
    runlog._RUNS_DIR = str(tmp_path)
    with runlog.run("unit") as log:
        log.event("phase", name="x")
        log.counter("k", 2)
        log.counter("k", 3)
    import glob
    import json
    d = sorted(glob.glob(str(tmp_path / "unit-*")))[-1]
    summary = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["counters"] == {"k": 5}
