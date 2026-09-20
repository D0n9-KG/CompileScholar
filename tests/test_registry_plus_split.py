# -*- coding: utf-8 -*-
"""F34 unit tests: '+'-suffix variant split guard in registry merges."""
import sys

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
from kb_compiler.records.registry import _split_plus_variants  # noqa: E402


def _mentions(*surfaces):
    keys = [f"k{i}" for i in range(len(surfaces))]
    mentions = {k: {"surface": s} for k, s in zip(keys, surfaces)}
    return keys, mentions


def test_plus_variant_split_from_base():
    keys, mentions = _mentions("mCLIP", "mCLIP+", "M3P")
    groups = [{"members": [0, 1, 2], "canonical": 0, "entity_type": "method"}]
    out = _split_plus_variants(groups, keys, mentions)
    # mCLIP+ pulled into its own entity; others stay merged
    assert len(out) == 2
    big = next(g for g in out if len(g["members"]) == 2)
    solo = next(g for g in out if len(g["members"]) == 1)
    assert solo["members"] == [1] and solo["canonical"] == 1
    assert big["members"] == [0, 2] and big["canonical"] == 0


def test_plus_without_base_untouched():
    # "X+" with no "X" in the same group = no split (nothing conflated)
    keys, mentions = _mentions("GPT+", "UC2")
    groups = [{"members": [0, 1], "canonical": 0, "entity_type": "method"}]
    out = _split_plus_variants(groups, keys, mentions)
    assert len(out) == 1 and out[0]["members"] == [0, 1]


def test_canonical_repair_when_plus_was_canonical():
    keys, mentions = _mentions("mCLIP", "mCLIP+")
    groups = [{"members": [0, 1], "canonical": 1, "entity_type": "method"}]
    out = _split_plus_variants(groups, keys, mentions)
    big = next(g for g in out if g["members"] == [0])
    assert big["canonical"] == 0     # canonical repaired to a kept member


def test_whitespace_insensitive():
    keys, mentions = _mentions("m CLIP", "mCLIP +")
    groups = [{"members": [0, 1], "canonical": 0, "entity_type": "method"}]
    out = _split_plus_variants(groups, keys, mentions)
    assert len(out) == 2


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
