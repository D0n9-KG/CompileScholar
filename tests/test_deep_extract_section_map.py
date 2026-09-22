# -*- coding: utf-8 -*-
"""Section semantic map tests (task #12): card-label primary routing, regex
fallback, fallback counter, mismatch odometer, enum gate. LLM is mocked.
Run:
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_slot_section_map.py -q
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kb_compiler.records import deep_extract


def _text():
    return ("# T\n\n" + "i" * 700 + "\n\n## 3.3 Evaluation\n\n" + "e" * 700 +
            "\n\n## Related Work\n\n" + "r" * 700 + "\n")


def _reset():
    deep_extract.ROUTE_STATS.update({"card": 0, "regex": 0, "fallback": 0, "mismatch": 0})


def test_card_label_is_primary_route():
    _reset()
    chunks = deep_extract.chunk_text("t", _text(), section_labels=[
        {"title": "3.3 Evaluation", "label": "method"}])
    ev = next(c for c in chunks if "Evaluation" in c["section"])
    # method kinds: finding/config/lineage/notation (+finding base)
    assert "notation" in ev["kinds"]
    assert "result" not in ev["kinds"]  # NOT the regex experiment route
    assert deep_extract.ROUTE_STATS["card"] >= 1


def test_regex_fallback_when_card_misses_title():
    _reset()
    chunks = deep_extract.chunk_text("t", _text(), section_labels=[
        {"title": "A Section That Does Not Exist", "label": "method"}])
    ev = next(c for c in chunks if "Evaluation" in c["section"])
    assert "result" in ev["kinds"]  # regex experiment route fired
    assert deep_extract.ROUTE_STATS["regex"] >= 1


def test_no_card_map_preserves_legacy_regex_behavior():
    _reset()
    chunks = deep_extract.chunk_text("t", _text())
    ev = next(c for c in chunks if "Evaluation" in c["section"])
    assert "result" in ev["kinds"]
    assert deep_extract.ROUTE_STATS["card"] == 0


def test_card_regex_mismatch_counted_card_wins():
    _reset()
    chunks = deep_extract.chunk_text("t", _text(), section_labels=[
        {"title": "3.3 Evaluation", "label": "related_work"}])
    ev = next(c for c in chunks if "Evaluation" in c["section"])
    assert "result" not in ev["kinds"]          # card label won
    assert deep_extract.ROUTE_STATS["mismatch"] == 1     # odometer saw the conflict


def test_invalid_card_label_dropped_to_regex():
    _reset()
    chunks = deep_extract.chunk_text("t", _text(), section_labels=[
        {"title": "3.3 Evaluation", "label": "experimental_tactics"}])  # not in enum
    ev = next(c for c in chunks if "Evaluation" in c["section"])
    assert "result" in ev["kinds"]  # enum gate rejected the label -> regex
    assert deep_extract.ROUTE_STATS["card"] == 0


def test_label_map_normalizes_titles():
    m = deep_extract._norm_label_map([
        {"title": "  Related   Work ", "label": "related_work"},
        {"title": "x", "label": "bogus"},
        "not-a-dict",
    ])
    assert m == {"related work": "related_work"}  # normalized key, gates applied
