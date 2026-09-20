# -*- coding: utf-8 -*-
"""Unit tests for skeleton._repair_json_escapes (AirQA HuCurl case:
LaTeX-in-caption invalid \\escape killed a whole card at temp 0)."""
import json
import sys

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
from kb_compiler.records.skeleton import _repair_json_escapes  # noqa: E402


def test_invalid_latex_escape_repaired():
    raw = r'{"caption": "Ent (sp) n^{\gg} indicates the curriculum"}'
    fixed, n = _repair_json_escapes(raw)
    obj = json.loads(fixed)
    assert obj["caption"] == r"Ent (sp) n^{\gg} indicates the curriculum"
    assert n == 1


def test_invalid_u_escape_repaired():
    raw = r'{"x": "\underbrace{y}"}'
    fixed, n = _repair_json_escapes(raw)
    obj = json.loads(fixed)
    assert obj["x"] == r"\underbrace{y}"
    assert n == 1


def test_valid_double_backslash_untouched():
    # the regex-lookahead regression: '\\S3.4' is VALID JSON (escaped
    # backslash + S) and must survive unchanged
    raw = '{"x": "see \\\\S3.4"}'
    fixed, n = _repair_json_escapes(raw)
    assert fixed == raw, fixed
    assert n == 0
    assert json.loads(fixed)["x"] == "see \\S3.4"


def test_valid_escapes_untouched():
    raw = '{"a": "line\\nbreak", "b": "\\u0041", "c": "tab\\there", "d": "q\\"q"}'
    fixed, n = _repair_json_escapes(raw)
    assert fixed == raw
    assert n == 0
    obj = json.loads(fixed)
    assert obj["b"] == "A"


def test_mixed_valid_invalid():
    raw = r'{"a": "\\S ok", "b": "\gg bad", "c": "\n fine"}'
    fixed, n = _repair_json_escapes(raw)
    obj = json.loads(fixed)
    assert obj["a"] == "\\S ok"
    assert obj["b"] == r"\gg bad"
    assert n == 1


def test_trailing_lone_backslash():
    fixed, n = _repair_json_escapes('{"a": "x\\')
    assert fixed.endswith("\\\\")
    assert n == 1


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
