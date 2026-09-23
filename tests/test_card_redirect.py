"""G1-B2 regression: card() miss returns nearest in-corpus redirects.

Measured on Multi-108: 146/153 empty card observations hit out-of-corpus
entities and each was a dead step. The redirect list lets the model
re-target in one step.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kb_compiler.views.tools import KBTools  # noqa: E402


def _kb():
    views = {"cards": {"cards": {
        "e1": {"canonical": "Phage Display Library Platform",
               "aliases": ["PDLP"]},
        "e2": {"canonical": "DLCZ protocol", "aliases": ["DLCZ"]},
    }}}
    registry = {"entities": [
        {"entity_id": "e1", "canonical": "Phage Display Library Platform",
         "aliases": ["PDLP"], "in_corpus_paper_id": "P1"},
        {"entity_id": "x1", "canonical": "phage display library",
         "aliases": [], "in_corpus_paper_id": None},
        {"entity_id": "e2", "canonical": "DLCZ protocol", "aliases": ["DLCZ"],
         "in_corpus_paper_id": "P2"},
    ], "surface_index": {"phage display library": "x1",
                         "phage display library platform": "e1"}}
    return KBTools(views, registry, {}, {}, {})


def test_card_miss_on_out_of_corpus_redirects():
    kb = _kb()
    r = kb.card("phage display library")
    assert r["n"] == 0
    assert "error" in r
    # redirect names the in-corpus dossier with maximal token overlap
    assert r["nearest_in_corpus"] == ["Phage Display Library Platform"]


def test_card_hit_unaffected():
    kb = _kb()
    r = kb.card("DLCZ protocol")
    assert r["tool"] == "card"
    assert "nearest_in_corpus" not in r or not r["nearest_in_corpus"] or \
        r.get("canonical") == "DLCZ protocol"


def test_card_miss_no_overlap_returns_empty_list():
    kb = _kb()
    r = kb.card("quantum entanglement swapping")
    assert r["n"] == 0
    assert r["nearest_in_corpus"] == []
