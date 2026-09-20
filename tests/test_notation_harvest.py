# -*- coding: utf-8 -*-
"""notation_harvest unit tests — formula scanning (both delimiter forms),
structural gates (symbol-in-quote, verbatim formula containment, dedup,
prime distinctness). LLM is mocked (no calls). Run:
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_notation_harvest.py -q
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kb_compiler.records import notation_harvest as NH


# ---------------------------------------------------------------- scanning
def test_scan_finds_double_dollar_blocks():
    text = "intro\n$$ E = mc^2 $$\nmid\n$$ L(\\theta) = \\sum_i x_i $$\noutro"
    fs = NH.scan_formulas(text)
    assert len(fs) == 2
    assert fs[0]["formula"] == "E = mc^2"
    assert "intro" in fs[0]["context"] and "mid" in fs[0]["context"]


def test_scan_skips_trivial_double_dollar():
    text = "noise $$a$$ real $$ f(x) = x^2 + 3 $$ end"
    fs = NH.scan_formulas(text)
    assert len(fs) == 1 and "x^2" in fs[0]["formula"]


def test_scan_finds_substantial_single_dollar_outside_blocks():
    # NEFTUNE family: single-$ display math incl. \begin{array}
    text = ("We define $ X _ { \\mathrm { e m b } } + ( \\frac { \\alpha } "
            "{ \\sqrt { L d } } ) \\epsilon $ as the noised embedding. "
            "Scalar $\\alpha$ tunes it. $$ E = mc^2 $$ blocks stay primary.")
    fs = NH.scan_formulas(text)
    assert len(fs) == 2
    assert any("frac" in f["formula"] for f in fs)      # the long inline form
    assert any(f["formula"] == "E = mc^2" for f in fs)  # the $$ form
    assert not any(f["formula"] == "\\alpha" for f in fs)  # bare variable skipped


# ---------------------------------------------------------------- gates
def _batch():
    return [{"formula": "X = a + b", "char_start": 100, "context": "ctx", "context_start": 0}]


def test_gate_g1_symbol_must_be_in_quote():
    rec = {"symbol": "$q$", "quote": "the value of $z$ is given", "definition": "d", "formula_idx": 0}
    ok, reason = NH.gate(rec, _batch(), set())
    assert not ok and "G1" in reason


def test_gate_g2_formula_must_be_verbatim_in_quote():
    rec = {"symbol": "$X$", "quote": "X is defined as something else entirely",
           "definition": "d", "formula_idx": 0}
    ok, reason = NH.gate(rec, _batch(), set())
    assert not ok and "G2" in reason


def test_gate_g2_passes_on_whitespace_normalized_containment():
    # mineru inserts spaces inside LaTeX; containment is ws-normalized
    rec = {"symbol": "$X$", "quote": "we have  X = a   + b  in this regime",
           "definition": "the sum", "formula_idx": 0}
    ok, _ = NH.gate(rec, _batch(), set())
    assert ok


def test_gate_g3_duplicate_symbol_rejected():
    rec = {"symbol": "$X$", "quote": "X = a + b", "definition": "d", "formula_idx": 0}
    ok, reason = NH.gate(rec, _batch(), {"x"})
    assert not ok and "G3" in reason


def test_gate_g4_missing_definition_rejected():
    rec = {"symbol": "$X$", "quote": "X = a + b", "definition": "", "formula_idx": 0}
    ok, reason = NH.gate(rec, _batch(), set())
    assert not ok and "G4" in reason


# ---------------------------------------------------------------- prime fix
def test_math_strip_keeps_prime_distinct():
    # live-run lesson: X' (derived) vs X (base) must NOT collapse
    a = NH._math_strip("$X _ { \\mathrm { e m b } } ^ { \\prime }$")
    b = NH._math_strip("$X _ { \\mathrm { e m b } }$")
    assert a != b
    assert a.replace("'", "").strip() == b


def test_math_strip_folds_layout_noise():
    # macro wrapping and spacing fold; greek-macro vs unicode is NOT folded
    # (mineru emits the macro form consistently, cross-form dedup unneeded)
    assert NH._math_strip("$\\mathrm{emb}$") == NH._math_strip("$ emb $")
    assert NH._math_strip("$ X $") == NH._math_strip("$X$")
