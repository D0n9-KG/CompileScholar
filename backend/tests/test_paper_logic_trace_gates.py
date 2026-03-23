from __future__ import annotations

from app.paper_logic_trace.gates import (
    build_quality_payload,
    evaluate_hot_path_gate,
    needs_lightweight_audit,
)
from app.paper_logic_trace.models import EvidenceAnchor, ResearchMove


def test_hot_path_gate_rejects_anchorless_move() -> None:
    result = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='method',
                act_type='propose_method',
                summary='Uses graph encoding',
                anchor_ids=[],
            )
        ],
        anchors=[],
    )

    assert result['passed'] is False
    assert result['invalid_move_ids'] == ['m-1']
    assert result['hard_fail_reasons'] == ['no_anchors', 'all_moves_invalid']


def test_sparse_valid_trace_can_export_with_downgraded_quality() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='method',
                act_type='propose_method',
                summary='Uses graph encoding',
                anchor_ids=['a-1'],
            )
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id='a-1',
                paper_id='paper-1',
                source_ref='chunk:1',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
        ],
    )

    quality = build_quality_payload(gate_report)

    assert gate_report['passed'] is True
    assert quality['quality_tier'] == 'yellow'
    assert quality['audit_status'] == 'eligible'
    assert quality['quality_tier_score'] > 0
    assert 'sparse_trace' in quality['quality_flags']


def test_lightweight_audit_is_quality_flag_not_export_blocker() -> None:
    gate_report = {
        'passed': True,
        'invalid_move_ids': [],
        'move_count': 1,
        'anchor_count': 1,
        'sparse_trace': True,
        'quality_tier_score': 0.61,
    }

    assert needs_lightweight_audit(gate_report) is True


def test_mixed_trace_passes_with_yellow_quality_instead_of_hard_failure() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-valid',
                sequence_no=1,
                role='method',
                act_type='propose_method',
                summary='We propose a graph encoder for retrieval tasks.',
                methods=[{'surface': 'graph encoder'}],
                research_objects=[{'surface': 'retrieval tasks'}],
                anchor_ids=['a-1'],
                slot_provenance=[
                    {'field': 'methods', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
            ResearchMove(
                move_id='m-invalid',
                sequence_no=2,
                role='result',
                act_type='report_effect',
                summary='Improves accuracy.',
                anchor_ids=[],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id='a-1',
                paper_id='paper-1',
                source_ref='chunk:1',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
        ],
    )

    quality = build_quality_payload(gate_report)

    assert gate_report['passed'] is True
    assert gate_report['invalid_move_ids'] == ['m-invalid']
    assert quality['quality_tier'] == 'yellow'
    assert quality['audit_status'] == 'eligible'
    assert 'invalid_moves_present' in quality['quality_flags']


def test_richer_trace_scores_green_without_audit() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We study particle crushing prediction under high stress conditions.',
                research_objects=[{'surface': 'particle crushing prediction'}],
                anchor_ids=['a-1'],
                slot_provenance=[{'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We propose a discrete element simulation workflow for crushing behavior.',
                methods=[{'surface': 'discrete element simulation'}],
                anchor_ids=['a-2'],
                slot_provenance=[{'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='result',
                act_type='report_effect',
                summary='The proposed workflow improves prediction accuracy against the baseline.',
                metrics=[{'surface': 'prediction accuracy'}],
                effects=[{'direction': 'improve'}],
                comparators=[{'surface': 'baseline'}],
                anchor_ids=['a-3'],
                slot_provenance=[
                    {'field': 'metrics', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'effects', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3'], start=1)
        ],
    )

    quality = build_quality_payload(gate_report)

    assert gate_report['passed'] is True
    assert quality['quality_tier'] == 'green'
    assert quality['audit_status'] == 'not_needed'
    assert quality['quality_tier_score'] >= 0.78
