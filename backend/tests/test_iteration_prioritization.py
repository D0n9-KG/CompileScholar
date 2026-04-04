from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.research_logic import (
    IterationPriorityPreflightError,
    build_iteration_priority_inspection,
    build_iteration_priority_summary,
    build_phase10_fallback_surface,
    load_phase8_comparison_inspection,
    load_phase8_comparison_summary,
    load_phase10_comparison_summary,
    load_phase10_evidence,
    rank_iteration_recommendations,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE8_SUMMARY_PATH = REPO_ROOT / 'tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/comparison_summary.json'
PHASE8_INSPECTION_PATH = REPO_ROOT / 'tmp/phase8_sampled_single_paper_l2/baseline-cycle-01/comparison_inspection.json'
PHASE10_VERIFICATION_PATH = REPO_ROOT / '.planning/phases/10-multi-paper-l3-and-l4-validation/10-VERIFICATION.md'
PHASE10_REPORT_PATH = REPO_ROOT / 'docs/replay/reports/phase10-multi-paper-l3-l4-validation.md'


def test_load_phase8_comparison_summary_preserves_owner_buckets() -> None:
    summary = load_phase8_comparison_summary(PHASE8_SUMMARY_PATH)

    assert summary.iteration_label == 'baseline-cycle-01'
    assert [bucket.bucket for bucket in summary.owner_buckets[:2]] == ['relation_assembly', 'slot_recovery']
    assert summary.owner_buckets[0].fixed_count == 7
    assert summary.owner_buckets[0].random_count == 2
    assert summary.owner_buckets[1].fixed_count == 6


def test_load_phase8_comparison_inspection_preserves_owner_buckets() -> None:
    inspection = load_phase8_comparison_inspection(PHASE8_INSPECTION_PATH)

    assert inspection.baseline_only is True
    assert [bucket.bucket for bucket in inspection.owner_buckets[:2]] == ['relation_assembly', 'slot_recovery']
    assert inspection.owner_buckets[0].verdict_counts == {
        'recurring_failure': 7,
        'new_edge_case': 2,
    }


def test_load_phase10_comparison_summary_reads_explicit_json_path(tmp_path: Path) -> None:
    summary_path = tmp_path / 'comparison_summary.json'
    summary_path.write_text(
        json.dumps(
            {
                'packet_id': 'packet-1',
                'cutoff_year': 2021,
                'package': {'current': {'quality_tier': 'red'}},
                'replay': {'current': {'quality_tier': 'yellow'}},
                'prior_review': {'current': {'prior_candidate_count': 0}},
                'export': {'current': {'quality_tier': 'yellow'}},
                'best_cycle_selection': {
                    'selected_iteration_label': 'phase14-candidate-01',
                    'primary_recommendation_id': 'packet_construction',
                    'recommendation_evidence_refs': [
                        'training_view:C:/tmp/phase14-candidate-01/export_bundle/outputs/training_view.json',
                        'export_summary:C:/tmp/phase14-candidate-01/export_bundle/export_summary.json',
                    ],
                },
                'blocker_queue': {
                    'package_validation': [
                        {
                            'code': 'support_cluster_too_small',
                            'message': 'Package validation reports `support_cluster_too_small`.',
                            'current': True,
                            'baseline': False,
                            'vs_baseline': 'new',
                        }
                    ]
                },
                'source_artifacts': {
                    'current_replay_inspection': 'tmp/replay/replay_inspection.json',
                    'dataset_manifest': 'tmp/final-dataset/dataset_manifest.json',
                    'stability_handoff': 'tmp/final-dataset/stability_handoff.json',
                },
                'notes': {'source': 'fixture'},
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    surface = load_phase10_comparison_summary(summary_path)

    assert surface.packet_id == 'packet-1'
    assert surface.source_refs.phase10_summary_path == str(summary_path.resolve())
    assert surface.source_refs.fallback_used is False
    assert surface.current_recommendation == 'packet_construction'
    assert surface.selected_iteration_label == 'phase14-candidate-01'
    assert surface.recommendation_evidence_refs == [
        'training_view:C:/tmp/phase14-candidate-01/export_bundle/outputs/training_view.json',
        'export_summary:C:/tmp/phase14-candidate-01/export_bundle/export_summary.json',
    ]
    assert surface.source_artifacts['dataset_manifest'] == 'tmp/final-dataset/dataset_manifest.json'
    assert surface.source_artifacts['stability_handoff'] == 'tmp/final-dataset/stability_handoff.json'
    assert surface.blocker_queue['package_validation'][0].code == 'support_cluster_too_small'


def test_build_phase10_fallback_surface_uses_report_and_verification_when_json_absent() -> None:
    fallback_surface = build_phase10_fallback_surface(PHASE10_VERIFICATION_PATH, PHASE10_REPORT_PATH)

    assert fallback_surface.source_refs.fallback_used is True
    assert fallback_surface.source_refs.phase10_verification_path == str(PHASE10_VERIFICATION_PATH.resolve())
    assert fallback_surface.source_refs.phase10_report_path == str(PHASE10_REPORT_PATH.resolve())
    assert fallback_surface.current_recommendation == 'packet_construction'
    assert fallback_surface.package['current']['quality_tier'] == 'red'
    assert fallback_surface.replay['delta']['failure_counts_by_layer_delta']['l2'] == 0
    assert fallback_surface.blocker_queue['package_validation'][0].code == 'support_cluster_too_small'


def test_load_phase10_evidence_raises_preflight_error_when_sources_missing(tmp_path: Path) -> None:
    with pytest.raises(IterationPriorityPreflightError, match='Phase 10 evidence unavailable'):
        load_phase10_evidence(
            summary_path=tmp_path / 'missing-comparison-summary.json',
            verification_path=tmp_path / 'missing-verification.md',
            report_path=tmp_path / 'missing-report.md',
        )


def test_rank_iteration_recommendations_orders_packet_construction_first_for_current_evidence() -> None:
    phase8_summary = load_phase8_comparison_summary(PHASE8_SUMMARY_PATH)
    phase10_surface = build_phase10_fallback_surface(PHASE10_VERIFICATION_PATH, PHASE10_REPORT_PATH)

    recommendations = rank_iteration_recommendations(
        phase8_summary=phase8_summary,
        phase10_surface=phase10_surface,
    )

    assert [recommendation.id for recommendation in recommendations[:3]] == [
        'packet_construction',
        'l4_aggregation',
        'l2_extraction',
    ]
    assert recommendations[0].supporting_blocker_stages == ['package_validation', 'replay']
    assert recommendations[2].supporting_owner_buckets[:2] == ['relation_assembly', 'slot_recovery']
    assert recommendations[0].evidence[-1] == 'best-cycle evidence refs (unknown): none'


def test_build_iteration_priority_summary_keeps_supporting_l2_evidence_visible_on_current_baseline() -> None:
    phase8_summary = load_phase8_comparison_summary(PHASE8_SUMMARY_PATH)
    phase8_inspection = load_phase8_comparison_inspection(PHASE8_INSPECTION_PATH)
    phase10_surface = build_phase10_fallback_surface(PHASE10_VERIFICATION_PATH, PHASE10_REPORT_PATH)

    summary = build_iteration_priority_summary(
        phase8_summary=phase8_summary,
        phase8_inspection=phase8_inspection,
        phase10_surface=phase10_surface,
    )
    inspection = build_iteration_priority_inspection(
        phase8_summary=phase8_summary,
        phase8_inspection=phase8_inspection,
        phase10_surface=phase10_surface,
    )

    assert summary.primary_recommendation_id == 'packet_construction'
    assert [bucket.bucket for bucket in summary.supporting_l2_evidence[:2]] == ['relation_assembly', 'slot_recovery']
    assert summary.phase10_blocker_queue['package_validation'][0].code == 'support_cluster_too_small'
    assert 'best_cycle_selection' in summary.phase10_stage_surfaces
    assert inspection.ranking_signals['replay_l2_delta'] == 0
    assert inspection.ranking_signals['recommendation_evidence_refs'] == []


def test_build_iteration_priority_summary_preserves_closeout_refs_from_phase10_surface(tmp_path: Path) -> None:
    summary_path = tmp_path / 'comparison_summary.json'
    summary_path.write_text(
        json.dumps(
            {
                'packet_id': 'packet-1',
                'cutoff_year': 2021,
                'package': {'current': {'quality_tier': 'red'}},
                'replay': {'current': {'quality_tier': 'yellow'}},
                'prior_review': {'current': {'prior_candidate_count': 0}},
                'export': {'current': {'quality_tier': 'yellow'}},
                'best_cycle_selection': {
                    'selected_iteration_label': 'phase16-repeat-01',
                    'primary_recommendation_id': 'packet_construction',
                    'recommendation_evidence_refs': ['best_cycle_selection:C:/tmp/phase16/export_bundle/best_cycle_selection.json'],
                },
                'blocker_queue': {'package_validation': []},
                'source_artifacts': {
                    'dataset_manifest': 'tmp/final-dataset/dataset_manifest.json',
                    'stability_handoff': 'tmp/final-dataset/stability_handoff.json',
                },
                'notes': {'source': 'fixture'},
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    phase8_summary = load_phase8_comparison_summary(PHASE8_SUMMARY_PATH)
    phase8_inspection = load_phase8_comparison_inspection(PHASE8_INSPECTION_PATH)
    phase10_surface = load_phase10_comparison_summary(summary_path)

    summary = build_iteration_priority_summary(
        phase8_summary=phase8_summary,
        phase8_inspection=phase8_inspection,
        phase10_surface=phase10_surface,
    )

    assert summary.source_refs.training_dataset_manifest_path == 'tmp/final-dataset/dataset_manifest.json'
    assert summary.source_refs.stability_handoff_path == 'tmp/final-dataset/stability_handoff.json'
