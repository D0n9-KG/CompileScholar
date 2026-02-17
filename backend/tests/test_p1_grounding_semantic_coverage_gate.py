# backend/tests/test_p1_grounding_semantic_coverage_gate.py
"""
Tests for P1 Fix: semantic coverage rate metric + configurable gate.

Root cause: hybrid grounding mode lets lexical=supported claims bypass semantic
judgment entirely, resulting in 83.9% of claims without semantic verification.
Fix: track semantic coverage rate in quality_report; add optional gate via
phase1_gate_semantic_coverage_min schema rule (default 0.0 = disabled).
"""
from __future__ import annotations

import pytest

from app.extraction.orchestrator import run_phase1_extraction
from app.ingest.models import Chunk, DocumentIR, MdSpan, PaperDraft


def _doc() -> DocumentIR:
    return DocumentIR(
        paper=PaperDraft(
            paper_source="paperA",
            md_path="C:/tmp/paperA/source.md",
            title="Paper A",
            title_alt=None,
            authors=["Alice"],
            doi="10.1000/papera",
            year=2024,
        ),
        chunks=[
            Chunk(
                chunk_id="c1",
                paper_source="paperA",
                md_path="C:/tmp/paperA/source.md",
                span=MdSpan(start_line=1, end_line=5),
                section="Method",
                kind="block",
                text="The proposed algorithm achieves state of the art performance on multiple benchmarks.",
            ),
        ],
        references=[],
        citations=[],
    )


def _schema_with_rules(rules: dict) -> dict:
    return {
        "paper_type": "research",
        "version": 1,
        "steps": [{"id": "Method", "enabled": True, "order": 0}],
        "claim_kinds": [{"id": "Result", "enabled": True}],
        "rules": {
            "phase1_gate_supported_ratio_min": 0.0,
            "phase1_gate_step_coverage_min": 0.0,
            "phase2_gate_critical_slot_coverage_min": 0.0,
            "phase2_gate_conflict_rate_max": 1.0,
            **rules,
        },
    }


def _logic_extractor(*, doc, paper_id, schema):
    return {
        "logic": {"Method": {"summary": "Improved performance", "evidence_chunk_ids": ["c1"]}},
        "step_order": ["Method"],
    }


def _claim_extractor(*, doc, paper_id, schema, step_order):
    return [
        {
            "text": "The proposed algorithm achieves state of the art performance",
            "confidence": 0.85,
            "step_type": "Method",
            "kinds": ["Result"],
            "origin_chunk_id": "c1",
            "worker_id": "w1",
        }
    ]


def _grounding_judge_lexical_only(*, claims, chunk_by_id, schema):
    """All claims pass lexical, none go to semantic."""
    return [
        {
            "canonical_claim_id": claims[0]["canonical_claim_id"],
            "support_label": "supported",
            "judge_score": 0.85,
            "reason": "lexical overlap",
            "judge_mode": "lexical",
        }
    ]


def _grounding_judge_semantic_all(*, claims, chunk_by_id, schema):
    """All claims go through semantic judgment."""
    return [
        {
            "canonical_claim_id": claims[0]["canonical_claim_id"],
            "support_label": "supported",
            "judge_score": 0.90,
            "reason": "semantic supported",
            "judge_mode": "semantic",
        }
    ]


def test_quality_report_includes_grounding_semantic_coverage_rate(tmp_path):
    """quality_report always includes grounding_semantic_coverage_rate."""
    out = run_phase1_extraction(
        doc=_doc(),
        paper_id="doi:10.1000/papera",
        cite_rec={"cites_resolved": []},
        schema=_schema_with_rules({}),
        artifacts_dir=tmp_path / "phase1",
        logic_extractor=_logic_extractor,
        claim_extractor=_claim_extractor,
        grounding_judge=_grounding_judge_lexical_only,
        allow_weak=False,
    )
    report = out["quality_report"]
    assert "grounding_semantic_coverage_rate" in report
    assert isinstance(report["grounding_semantic_coverage_rate"], float)
    assert 0.0 <= report["grounding_semantic_coverage_rate"] <= 1.0


def test_semantic_coverage_rate_is_zero_when_all_lexical(tmp_path):
    """Coverage rate = 0 when all claims are judged lexically."""
    out = run_phase1_extraction(
        doc=_doc(),
        paper_id="doi:10.1000/papera",
        cite_rec={"cites_resolved": []},
        schema=_schema_with_rules({}),
        artifacts_dir=tmp_path / "phase1",
        logic_extractor=_logic_extractor,
        claim_extractor=_claim_extractor,
        grounding_judge=_grounding_judge_lexical_only,
        allow_weak=False,
    )
    report = out["quality_report"]
    assert report["grounding_semantic_coverage_rate"] == pytest.approx(0.0)
    assert report["grounding_lexical_judged"] == 1
    assert report["grounding_semantic_judged"] == 0


def test_semantic_coverage_rate_is_one_when_all_semantic(tmp_path):
    """Coverage rate = 1.0 when all claims are judged semantically."""
    out = run_phase1_extraction(
        doc=_doc(),
        paper_id="doi:10.1000/papera",
        cite_rec={"cites_resolved": []},
        schema=_schema_with_rules({}),
        artifacts_dir=tmp_path / "phase1",
        logic_extractor=_logic_extractor,
        claim_extractor=_claim_extractor,
        grounding_judge=_grounding_judge_semantic_all,
        allow_weak=False,
    )
    report = out["quality_report"]
    assert report["grounding_semantic_coverage_rate"] == pytest.approx(1.0)
    assert report["grounding_semantic_judged"] == 1
    assert report["grounding_lexical_judged"] == 0


def test_semantic_coverage_gate_disabled_by_default(tmp_path):
    """Default gate (0.0) never fails for zero semantic coverage."""
    out = run_phase1_extraction(
        doc=_doc(),
        paper_id="doi:10.1000/papera",
        cite_rec={"cites_resolved": []},
        schema=_schema_with_rules({}),  # No gate configured
        artifacts_dir=tmp_path / "phase1",
        logic_extractor=_logic_extractor,
        claim_extractor=_claim_extractor,
        grounding_judge=_grounding_judge_lexical_only,
        allow_weak=False,
    )
    report = out["quality_report"]
    # Coverage is 0 but gate is disabled → should NOT fail on semantic_coverage
    assert "semantic_coverage" not in (report.get("gate_fail_reasons") or [])


def test_semantic_coverage_gate_fails_when_coverage_below_threshold(tmp_path):
    """Gate fails when semantic coverage is below the configured minimum."""
    out = run_phase1_extraction(
        doc=_doc(),
        paper_id="doi:10.1000/papera",
        cite_rec={"cites_resolved": []},
        schema=_schema_with_rules({
            "phase1_gate_semantic_coverage_min": 0.5,  # Require at least 50% semantic
        }),
        artifacts_dir=tmp_path / "phase1",
        logic_extractor=_logic_extractor,
        claim_extractor=_claim_extractor,
        grounding_judge=_grounding_judge_lexical_only,  # 0% semantic
        allow_weak=True,  # Don't let other gates block
    )
    report = out["quality_report"]
    assert "semantic_coverage" in (report.get("gate_fail_reasons") or [])
    assert report["gate_passed"] is False


def test_semantic_coverage_gate_passes_when_above_threshold(tmp_path):
    """Gate passes when semantic coverage meets or exceeds the configured minimum."""
    out = run_phase1_extraction(
        doc=_doc(),
        paper_id="doi:10.1000/papera",
        cite_rec={"cites_resolved": []},
        schema=_schema_with_rules({
            "phase1_gate_semantic_coverage_min": 0.5,  # Require at least 50% semantic
        }),
        artifacts_dir=tmp_path / "phase1",
        logic_extractor=_logic_extractor,
        claim_extractor=_claim_extractor,
        grounding_judge=_grounding_judge_semantic_all,  # 100% semantic
        allow_weak=False,
    )
    report = out["quality_report"]
    assert "semantic_coverage" not in (report.get("gate_fail_reasons") or [])


def test_thresholds_dict_includes_semantic_coverage_min(tmp_path):
    """quality_report.thresholds includes phase1_gate_semantic_coverage_min."""
    configured_min = 0.3
    out = run_phase1_extraction(
        doc=_doc(),
        paper_id="doi:10.1000/papera",
        cite_rec={"cites_resolved": []},
        schema=_schema_with_rules({
            "phase1_gate_semantic_coverage_min": configured_min,
        }),
        artifacts_dir=tmp_path / "phase1",
        logic_extractor=_logic_extractor,
        claim_extractor=_claim_extractor,
        grounding_judge=_grounding_judge_lexical_only,
        allow_weak=True,
    )
    report = out["quality_report"]
    thresholds = report.get("thresholds") or {}
    assert "phase1_gate_semantic_coverage_min" in thresholds
    assert thresholds["phase1_gate_semantic_coverage_min"] == pytest.approx(configured_min)
