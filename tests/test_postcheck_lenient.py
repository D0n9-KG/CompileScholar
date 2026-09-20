# -*- coding: utf-8 -*-
"""Unit tests for the opt-in lenient quote channels (AirQA IL-5):
tag-stripped (table-rendered quotes) + de-hyphenated (PDF line-break) lookup.
Default OFF must be byte-identical to the frozen PS protocol behavior."""
import sys

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
from kb_compiler.records.postcheck import run_postcheck  # noqa: E402

TEXT = ("# Paper\n\n## Experiments\n"
        "<table><tr><td>Method</td><td>Score</td></tr>"
        "<tr><td>SFGC</td><td>GEOM</td><td>Whole Dataset</td></tr></table>\n\n"
        "We dis- covered that the widget improves cover- age greatly.\n")


def _rec(quote, rid="t1"):
    return {"id": rid, "kind": "finding", "claim": "observation about the method",
            "strength": "stated", "epistemic": "stated", "claim_type": "observation",
            "quote": quote}


def _run(quote, lenient):
    rb = {"p1": {"records": [_rec(quote)], "overflow": [], "entity_queue": []}}
    checked, dropped, warnings, stats = run_postcheck(
        rb, {"p1": TEXT}, {}, "unused-model", dry=True,
        lenient_quote_channels=lenient)
    passed = bool(checked.get("p1", {}).get("records"))
    return passed, dropped


def test_table_rendered_quote_off_drops():
    passed, dropped = _run("SFGC GEOM Whole Dataset", lenient=False)
    assert not passed
    assert any("quote_not_in_text" in str(d.get("violations")) for d in dropped)


def test_table_rendered_quote_on_passes():
    passed, _ = _run("SFGC GEOM Whole Dataset", lenient=True)
    assert passed


def test_hyphenation_off_drops():
    passed, _ = _run("discovered that the widget improves coverage", lenient=False)
    assert not passed


def test_hyphenation_on_passes():
    passed, _ = _run("discovered that the widget improves coverage", lenient=True)
    assert passed


def test_fabricated_quote_still_dropped_with_lenient():
    passed, dropped = _run("a completely fabricated sentence about nothing",
                           lenient=True)
    assert not passed
    assert any("quote_not_in_text" in str(d.get("violations")) for d in dropped)


def test_legit_verbatim_quote_passes_both_modes():
    # contiguous prose quote: passes in BOTH modes (lenient must not regress
    # the normal channel)
    for lenient in (False, True):
        passed, _ = _run("that the widget improves", lenient=lenient)
        assert passed, f"lenient={lenient}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
