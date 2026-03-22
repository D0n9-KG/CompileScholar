from __future__ import annotations

from typing import Any

from .models import EvidenceAnchor, ResearchMove


def evaluate_hot_path_gate(
    *,
    moves: list[ResearchMove],
    anchors: list[EvidenceAnchor],
) -> dict[str, Any]:
    invalid_move_ids = [
        move.move_id
        for move in moves
        if not str(move.role or '').strip()
        or not str(move.act_type or '').strip()
        or not list(move.anchor_ids or [])
    ]
    passed = bool(moves) and not invalid_move_ids
    sparse_trace = passed and (len(moves) < 2 or len(anchors) < 2)
    return {
        'passed': passed,
        'invalid_move_ids': invalid_move_ids,
        'move_count': len(moves),
        'anchor_count': len(anchors),
        'sparse_trace': sparse_trace,
    }


def needs_lightweight_audit(gate_report: dict[str, Any]) -> bool:
    if not bool(gate_report.get('passed')):
        return False
    return bool(gate_report.get('sparse_trace'))


def build_quality_payload(gate_report: dict[str, Any]) -> dict[str, Any]:
    passed = bool(gate_report.get('passed'))
    quality_tier = 'green' if passed and not bool(gate_report.get('sparse_trace')) else 'yellow' if passed else 'red'
    audit_status = 'eligible' if needs_lightweight_audit(gate_report) else 'not_needed' if passed else 'blocked'
    return {
        'quality_tier': quality_tier,
        'hot_path_gate_report': gate_report,
        'audit_status': audit_status,
    }
