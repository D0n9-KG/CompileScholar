"""G1-B1 regression: compare() empty-result vocab hints + axis diagnosis.

Measured on Multi-108: 15/27 compare calls returned empty (obs=62) because
planner free-text args collided with the compiled matrix vocabulary or put
entity names on the subject axis. The hint payload lets the model re-target
in one step.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kb_compiler.views.tools import KBTools  # noqa: E402


def _cell():
    return {"value": "0.62", "band": {"setup": ["default"], "budget_bucket": ""},
            "record_id": "r1", "paper_id": "P1"}


def _kb():
    views = {"matrix": {"tables": {
        "icr mice||brain accumulation": {"gold nanoparticles": [_cell()]},
        "7b||mmlu": {"QLoRA": [_cell()], "LoRA": [_cell()]},
        "openai embeddings||beir": {"E5": [_cell()], "BERT": [_cell()]},
    }},
        "cards": {"cards": {}}}
    registry = {"entities": [
        {"entity_id": "q1", "canonical": "QLoRA", "aliases": ["QLoRA"],
         "in_corpus_paper_id": "P1"},
        {"entity_id": "g1", "canonical": "gold nanoparticles", "aliases": [],
         "in_corpus_paper_id": "P2"},
    ], "surface_index": {"qlora": "q1", "gold nanoparticles": "g1"}}
    return KBTools(views, registry, {}, {}, {})


def test_compare_miss_returns_nearby_keys():
    kb = _kb()
    # no token overlap with any compiled key -> empty + hints
    r = kb.compare(metric="tumor regression rate")
    assert r["n"] == 0
    keys = r["vocab_hint"]["nearby_keys"]
    assert any(k["subject"] == "icr mice" and
               k["metric"] == "brain accumulation" for k in keys)


def test_compare_axis_confusion_note():
    kb = _kb()
    # QLoRA is a matrix ENTITY; passing it as subject is the measured shape
    r = kb.compare(subject="QLoRA", metric="average performance")
    assert r["n"] == 0
    assert "entities=['QLoRA']" in r["vocab_hint"]["axis_note"]


def test_compare_entity_diagnosis():
    kb = _kb()
    # subject axis blocks every row -> entity diagnosis fires
    r = kb.compare(subject="unknown setup", entities=["QLoRA", "nonexistent method"])
    assert r["n"] == 0
    ed = r["vocab_hint"]["entities"]
    assert ed["in_matrix"] == ["QLoRA"]
    assert ed["not_in_matrix"] == ["nonexistent method"]


def test_compare_hit_has_no_hint():
    kb = _kb()
    r = kb.compare(subject="7b", metric="mmlu")
    assert r["n"] == 2
    assert "vocab_hint" not in r
