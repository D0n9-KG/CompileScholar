from __future__ import annotations

from typing import Any

from .models import EvidenceAnchor, MoveRelation, ResearchMove


_CORE_SLOT_FIELDS = (
    'research_objects',
    'methods',
    'observed_variables',
    'metrics',
    'comparators',
    'conditions',
    'effects',
    'limitation_types',
    'resource_mentions',
)
_CRITICAL_ROLES = {'problem', 'method', 'result'}
_GREEN_THRESHOLD = 0.78
_YELLOW_THRESHOLD = 0.38


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(float(numerator) / float(denominator), 1.0))


def _has_core_slot_signal(move: ResearchMove) -> bool:
    return any(bool(getattr(move, field, None)) for field in _CORE_SLOT_FIELDS)


def _summary_ready(move: ResearchMove) -> bool:
    summary = str(move.summary or '').strip()
    if len(summary) >= 18:
        return True
    return len(summary.split()) >= 3


def _anchor_density(moves: list[ResearchMove], anchors: list[EvidenceAnchor]) -> float:
    if not moves:
        return 0.0
    density = len(anchors) / max(1, len(moves))
    return max(0.0, min(density, 1.0))


def _relation_coverage(moves: list[ResearchMove], move_relations: list[MoveRelation]) -> float:
    if len(moves) < 2:
        return 1.0
    target = max(1, len(moves) - 1)
    return _safe_ratio(len(move_relations), target)


def _quality_score(
    *,
    moves: list[ResearchMove],
    anchors: list[EvidenceAnchor],
    move_relations: list[MoveRelation],
    invalid_move_ids: list[str],
) -> tuple[float, dict[str, float]]:
    valid_move_count = max(0, len(moves) - len(invalid_move_ids))
    valid_move_ratio = _safe_ratio(valid_move_count, len(moves))
    size_ratio = max(0.0, min(valid_move_count / 8.0, 1.0))
    anchor_ratio = _anchor_density(moves, anchors)
    summary_ratio = _safe_ratio(sum(1 for move in moves if _summary_ready(move)), len(moves))
    slot_ready_ratio = _safe_ratio(sum(1 for move in moves if _has_core_slot_signal(move)), len(moves))
    role_ratio = _safe_ratio(len({str(move.role) for move in moves} & _CRITICAL_ROLES), len(_CRITICAL_ROLES))
    relation_ratio = _relation_coverage(moves, move_relations)
    provenance_ratio = _safe_ratio(sum(1 for move in moves if move.slot_provenance), len(moves))

    signals = {
        'valid_move_ratio': round(valid_move_ratio, 4),
        'size_ratio': round(size_ratio, 4),
        'anchor_density': round(anchor_ratio, 4),
        'summary_ready_ratio': round(summary_ratio, 4),
        'slot_ready_ratio': round(slot_ready_ratio, 4),
        'critical_role_coverage_ratio': round(role_ratio, 4),
        'relation_coverage_ratio': round(relation_ratio, 4),
        'slot_provenance_ratio': round(provenance_ratio, 4),
    }
    score = (
        valid_move_ratio * 0.15
        + size_ratio * 0.10
        + anchor_ratio * 0.10
        + summary_ratio * 0.10
        + slot_ready_ratio * 0.30
        + role_ratio * 0.10
        + relation_ratio * 0.05
        + provenance_ratio * 0.10
    )
    return round(score, 4), signals


def _soft_flags(*, sparse_trace: bool, invalid_move_ids: list[str], signals: dict[str, float]) -> list[str]:
    flags: list[str] = []
    if sparse_trace:
        flags.append('sparse_trace')
    if invalid_move_ids:
        flags.append('invalid_moves_present')
    if signals.get('slot_ready_ratio', 0.0) < 0.45:
        flags.append('low_slot_coverage')
    if signals.get('critical_role_coverage_ratio', 0.0) < 0.67:
        flags.append('limited_role_coverage')
    if signals.get('slot_provenance_ratio', 0.0) < 0.35:
        flags.append('weak_slot_provenance')
    if signals.get('relation_coverage_ratio', 0.0) < 0.5:
        flags.append('weak_relation_stitching')
    return flags


def evaluate_hot_path_gate(
    *,
    moves: list[ResearchMove],
    anchors: list[EvidenceAnchor],
    move_relations: list[MoveRelation] | None = None,
) -> dict[str, Any]:
    move_relations = list(move_relations or [])
    invalid_move_ids = [
        move.move_id
        for move in moves
        if not str(move.role or '').strip()
        or not str(move.act_type or '').strip()
        or not list(move.anchor_ids or [])
        or not str(move.summary or '').strip()
    ]
    summary_ready_count = sum(1 for move in moves if _summary_ready(move))
    hard_fail_reasons: list[str] = []
    if not moves:
        hard_fail_reasons.append('no_moves')
    if not anchors:
        hard_fail_reasons.append('no_anchors')
    if len(invalid_move_ids) >= len(moves) and moves:
        hard_fail_reasons.append('all_moves_invalid')
    if summary_ready_count == 0 and moves:
        hard_fail_reasons.append('no_summary_signal')
    passed = not hard_fail_reasons
    sparse_trace = passed and (len(moves) < 2 or len(anchors) < 2)
    quality_tier_score, score_signals = _quality_score(
        moves=moves,
        anchors=anchors,
        move_relations=move_relations,
        invalid_move_ids=invalid_move_ids,
    )
    soft_flags = _soft_flags(
        sparse_trace=sparse_trace,
        invalid_move_ids=invalid_move_ids,
        signals=score_signals,
    )
    return {
        'passed': passed,
        'invalid_move_ids': invalid_move_ids,
        'move_count': len(moves),
        'anchor_count': len(anchors),
        'relation_count': len(move_relations),
        'sparse_trace': sparse_trace,
        'hard_fail_reasons': hard_fail_reasons,
        'soft_flags': soft_flags,
        'quality_tier_score': quality_tier_score,
        **score_signals,
    }


def needs_lightweight_audit(gate_report: dict[str, Any]) -> bool:
    if not bool(gate_report.get('passed')):
        return False
    if bool(gate_report.get('sparse_trace')):
        return True
    if bool(gate_report.get('invalid_move_ids')):
        return True
    return float(gate_report.get('quality_tier_score') or 0.0) < _GREEN_THRESHOLD


def build_quality_payload(gate_report: dict[str, Any]) -> dict[str, Any]:
    passed = bool(gate_report.get('passed'))
    score = float(gate_report.get('quality_tier_score') or 0.0)
    if not passed or score < _YELLOW_THRESHOLD:
        quality_tier = 'red'
    elif bool(gate_report.get('sparse_trace')) or score < _GREEN_THRESHOLD:
        quality_tier = 'yellow'
    else:
        quality_tier = 'green'
    audit_status = 'eligible' if needs_lightweight_audit(gate_report) else 'not_needed' if passed else 'blocked'
    return {
        'quality_tier': quality_tier,
        'quality_tier_score': round(score, 4),
        'quality_flags': list(gate_report.get('soft_flags') or []),
        'hot_path_gate_report': gate_report,
        'audit_status': audit_status,
    }
