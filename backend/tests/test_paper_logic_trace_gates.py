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


def test_lightweight_audit_is_quality_flag_not_export_blocker() -> None:
    gate_report = {
        'passed': True,
        'invalid_move_ids': [],
        'move_count': 1,
        'anchor_count': 1,
        'sparse_trace': True,
    }

    assert needs_lightweight_audit(gate_report) is True
