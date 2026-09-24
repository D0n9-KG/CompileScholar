"""Integration test: extract_paper drives the full batch-2/3 loop —
coarse extraction (mocked LLM) -> mentions backflow into views -> tier
registration in the sci-evo SQLite store -> promotion-candidate surfacing.
"""

import json
import sys
import tempfile
from pathlib import Path

_CS = Path(r"C:\Users\D0n9\Desktop\CompileScholar")
sys.path.insert(0, str(_CS / "src"))
sys.path.insert(0, str(_CS / ".research_tmp" / "experiments" /
                        "benchmarks" / "_shared" / "tools"))
sys.path.insert(0, r"C:\Users\D0n9\Desktop\sci-evo-extract\src")

from external_tools import ExternalTools  # noqa: E402


def _fake_coarse(monkey_llm):
    import kb_compiler.records.coarse_extract as ce
    ce.call_json = monkey_llm


def test_extract_paper_backflow_and_tiers(tmp_path, monkeypatch):
    fake = {"records": [
        {"kind": "method", "subject": "Mamba", "claim": "Linear-time SSM.",
         "quote": "we propose Mamba",
         "mentions": ["Transformer"]},
        {"kind": "limitation", "subject": "Mamba", "claim": "Recall gap.",
         "quote": "recall tasks remain hard", "mentions": []},
    ]}
    import kb_compiler.records.coarse_extract as ce
    monkeypatch.setattr(ce, "call_json", lambda *a, **k: fake)

    registry = {
        "entities": [{"entity_id": "ccc3", "canonical": "Transformer",
                      "aliases": ["Transformer"], "entity_type": "method"}],
        "surface_index": {"transformer": "ccc3"},
    }
    views = {"genealogy": {"nodes": {
        "ccc3": {"canonical": "Transformer", "in_corpus": True}}, "edges": []}}
    ext = ExternalTools(views, {}, model="local:Qwen3.8-27B",
                        registry=registry, blocklist=None,
                        backflow_path=str(tmp_path / "bf.jsonl"),
                        tier_db=str(tmp_path / "growth.db"))

    obs = ext.extract_paper("Mamba: Linear-Time Sequence Modeling",
                            "A" * 300)
    assert obs["n"] == 2
    # backflow: mention matched, paper attached
    assert obs["attached_to_lineage"] == ["Transformer"]
    assert "Transformer" in obs["library_growth"]
    # tier: limitation record -> promotion candidate surfaced to the loop
    assert obs["promotion_candidate"] is True

    # genealogy now carries the weak edge
    edges = views["genealogy"]["edges"]
    assert len(edges) == 1 and edges[0]["relation"] == "external_mention"
    assert edges[0]["from_name"] == "Transformer"

    # second call: no re-extraction (tier store refuses), no dup edges
    obs2 = ext.extract_paper("Mamba: Linear-Time Sequence Modeling",
                             "A" * 300)
    assert obs2["tier_note"] and "no re-extraction" in obs2["tier_note"]
    assert len(views["genealogy"]["edges"]) == 1

    # tier store reflects the state; pending queue holds the candidate
    ts = ext._tier_store()
    st = ts.state(ts.find(title="Mamba: Linear-Time Sequence Modeling"))
    assert st["tier"] == "coarse" and st["promotion_candidate"]
    pend = ts.pending_promotions()
    assert len(pend) == 1 and "Mamba" in pend[0]["title"]


def test_extract_paper_no_registry_still_works(tmp_path, monkeypatch):
    """Backflow/tier are optional layers — a bare ExternalTools (no
    registry, no tier db) still returns coarse records."""
    fake = {"records": [
        {"kind": "finding", "subject": "X", "claim": "c", "quote": "q",
         "mentions": []}]}
    import kb_compiler.records.coarse_extract as ce
    monkeypatch.setattr(ce, "call_json", lambda *a, **k: fake)
    ext = ExternalTools({"genealogy": {"nodes": {}, "edges": []}}, {})
    obs = ext.extract_paper("T", "B" * 300)
    assert obs["n"] == 1 and "records" in obs
    assert "attached_to_lineage" not in obs
    assert "promotion_candidate" not in obs
