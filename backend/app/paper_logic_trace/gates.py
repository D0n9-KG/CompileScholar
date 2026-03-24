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
_GREEN_THRESHOLD = 0.78
_YELLOW_THRESHOLD = 0.38
_QUALITY_NOISE_PREFIXES = (
    '# abstract',
    '# article info',
    '# articleinfo',
    '# credit author statement',
    'abstract',
    'article info',
    'articleinfo',
    'available online',
    'credit author statement',
    'keywords:',
    'keyword:',
    'article history',
)
_EXPECTED_ROLES_BY_PAPER_TYPE: dict[str, tuple[str, ...]] = {
    'empirical': ('problem', 'method', 'result'),
    'benchmark': ('problem', 'method', 'result'),
    'case_study': ('problem', 'method', 'result'),
    'theoretical': ('problem', 'method', 'interpretation'),
    'review': ('background', 'interpretation'),
    'software': ('problem', 'method'),
    'unknown': ('problem', 'method'),
}
_EXPECTED_SLOTS_BY_PAPER_TYPE: dict[str, tuple[str, ...]] = {
    'empirical': ('research_objects', 'methods', 'metrics', 'comparators', 'effects'),
    'benchmark': ('research_objects', 'methods', 'metrics', 'comparators', 'effects'),
    'case_study': ('research_objects', 'methods', 'conditions', 'metrics'),
    'theoretical': ('research_objects', 'methods', 'conditions'),
    'review': ('research_objects', 'methods'),
    'software': ('research_objects', 'methods', 'resource_mentions'),
    'unknown': ('research_objects', 'methods'),
}
_SPARSE_SLOT_RATIO = 0.08


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
    paper_type: str,
) -> tuple[float, dict[str, float]]:
    valid_move_count = max(0, len(moves) - len(invalid_move_ids))
    valid_move_ratio = _safe_ratio(valid_move_count, len(moves))
    size_ratio = max(0.0, min(valid_move_count / 8.0, 1.0))
    anchor_ratio = _anchor_density(moves, anchors)
    summary_ratio = _safe_ratio(sum(1 for move in moves if _summary_ready(move)), len(moves))
    slot_ready_ratio = _safe_ratio(sum(1 for move in moves if _has_core_slot_signal(move)), len(moves))
    expected_roles = set(_EXPECTED_ROLES_BY_PAPER_TYPE.get(str(paper_type or 'unknown'), _EXPECTED_ROLES_BY_PAPER_TYPE['unknown']))
    role_ratio = _safe_ratio(len({str(move.role) for move in moves} & expected_roles), len(expected_roles))
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


def _looks_like_noise_summary(summary: str) -> bool:
    normalized = str(summary or '').strip().lower()
    if not normalized:
        return False
    if any(normalized.startswith(prefix) for prefix in _QUALITY_NOISE_PREFIXES):
        return True
    return normalized.startswith('#') and len(normalized.split()) <= 6


def is_noise_summary(summary: str) -> bool:
    return _looks_like_noise_summary(summary)


def _l2_completeness_audit(
    *,
    moves: list[ResearchMove],
    move_relations: list[MoveRelation],
    paper_type: str,
) -> dict[str, Any]:
    move_count = len(moves)
    role_counts: dict[str, int] = {}
    slot_counts: dict[str, int] = {}
    slot_move_ratios: dict[str, float] = {}
    observed_roles = sorted({str(move.role) for move in moves if str(move.role or '').strip()})
    for role in observed_roles:
        role_counts[role] = sum(1 for move in moves if str(move.role) == role)
    for field in _CORE_SLOT_FIELDS:
        count = sum(1 for move in moves if bool(getattr(move, field, None)))
        slot_counts[field] = count
        slot_move_ratios[field] = round(_safe_ratio(count, move_count), 4)

    expected_roles = list(_EXPECTED_ROLES_BY_PAPER_TYPE.get(str(paper_type or 'unknown'), _EXPECTED_ROLES_BY_PAPER_TYPE['unknown']))
    expected_slots = list(_EXPECTED_SLOTS_BY_PAPER_TYPE.get(str(paper_type or 'unknown'), _EXPECTED_SLOTS_BY_PAPER_TYPE['unknown']))
    missing_expected_roles = sorted(role for role in expected_roles if role not in observed_roles)
    expected_role_set = set(expected_roles)
    critical_role_ratio = _safe_ratio(len(set(observed_roles) & expected_role_set), len(expected_role_set))
    missing_l2_slot_fields = [field for field in _CORE_SLOT_FIELDS if slot_counts.get(field, 0) == 0]
    missing_expected_slot_fields = [field for field in expected_slots if slot_counts.get(field, 0) == 0]
    sparse_expected_slot_fields = [
        field
        for field in expected_slots
        if 0 < slot_counts.get(field, 0) and slot_move_ratios.get(field, 0.0) < _SPARSE_SLOT_RATIO
    ]
    noise_move_ids = [move.move_id for move in moves if _looks_like_noise_summary(move.summary)]
    signature_ready_move_count = sum(
        1
        for move in moves
        if any(
            bool(getattr(move, field, None))
            for field in ('research_objects', 'methods', 'metrics', 'comparators', 'conditions', 'effects', 'resource_mentions')
        )
    )
    relation_ratio = _relation_coverage(moves, move_relations)
    has_outcome_signal = any(role in observed_roles for role in ('result', 'interpretation'))
    requires_outcome_signal = str(paper_type or 'unknown') not in {'software'}
    ready_for_community = signature_ready_move_count >= 3 and (slot_counts.get('methods', 0) > 0 or slot_counts.get('research_objects', 0) > 0)
    ready_for_l3 = (
        not missing_expected_roles
        and relation_ratio >= 0.4
        and critical_role_ratio >= 0.67
        and (has_outcome_signal or not requires_outcome_signal)
        and (slot_counts.get('methods', 0) > 0 or slot_counts.get('research_objects', 0) > 0)
    )
    evidence_signal_count = sum(1 for field in ('metrics', 'comparators', 'effects') if slot_counts.get(field, 0) > 0)
    context_signal_count = sum(1 for field in ('conditions', 'resource_mentions', 'limitation_types') if slot_counts.get(field, 0) > 0)
    ready_for_l4 = ready_for_l3 and evidence_signal_count >= 2 and context_signal_count >= 1

    role_score = _safe_ratio(len(expected_roles) - len(missing_expected_roles), len(expected_roles))
    expected_slot_score = _safe_ratio(len(expected_slots) - len(missing_expected_slot_fields), len(expected_slots))
    sparse_penalty = _safe_ratio(len(sparse_expected_slot_fields), len(expected_slots))
    noise_penalty = _safe_ratio(len(noise_move_ids), move_count)
    completeness_score = max(
        0.0,
        min(
            1.0,
            role_score * 0.30
            + critical_role_ratio * 0.20
            + expected_slot_score * 0.35
            + (1.0 - sparse_penalty) * 0.05
            + (1.0 - noise_penalty) * 0.05
            + relation_ratio * 0.05,
        ),
    )

    return {
        'paper_type': str(paper_type or 'unknown'),
        'move_count': move_count,
        'role_counts': role_counts,
        'observed_roles': observed_roles,
        'expected_roles': expected_roles,
        'missing_expected_roles': missing_expected_roles,
        'critical_role_coverage_ratio': round(critical_role_ratio, 4),
        'slot_counts': slot_counts,
        'slot_move_ratios': slot_move_ratios,
        'missing_l2_slot_fields': missing_l2_slot_fields,
        'expected_slot_fields': expected_slots,
        'missing_expected_slot_fields': missing_expected_slot_fields,
        'sparse_expected_slot_fields': sparse_expected_slot_fields,
        'noise_move_ids': noise_move_ids,
        'signature_ready_move_count': signature_ready_move_count,
        'relation_coverage_ratio': round(relation_ratio, 4),
        'ready_for_community': ready_for_community,
        'ready_for_l3': ready_for_l3,
        'ready_for_l4': ready_for_l4,
        'completeness_score': round(completeness_score, 4),
    }


def evaluate_hot_path_gate(
    *,
    moves: list[ResearchMove],
    anchors: list[EvidenceAnchor],
    move_relations: list[MoveRelation] | None = None,
    paper_type: str = 'unknown',
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
        paper_type=paper_type,
    )
    completeness_audit = _l2_completeness_audit(
        moves=moves,
        move_relations=move_relations,
        paper_type=paper_type,
    )
    quality_tier_score = round(
        quality_tier_score * 0.7 + float(completeness_audit.get('completeness_score') or 0.0) * 0.3,
        4,
    )
    soft_flags = _soft_flags(
        sparse_trace=sparse_trace,
        invalid_move_ids=invalid_move_ids,
        signals=score_signals,
    )
    if completeness_audit.get('missing_expected_roles'):
        soft_flags.append('missing_expected_roles')
    if completeness_audit.get('missing_expected_slot_fields'):
        soft_flags.append('missing_expected_slots')
    if completeness_audit.get('sparse_expected_slot_fields'):
        soft_flags.append('sparse_expected_slots')
    if completeness_audit.get('noise_move_ids'):
        soft_flags.append('residual_noise_moves')
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
        'l2_completeness_audit': completeness_audit,
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
    completeness_audit = dict(gate_report.get('l2_completeness_audit') or {})
    missing_expected_roles = list(completeness_audit.get('missing_expected_roles') or [])
    missing_expected_slots = list(completeness_audit.get('missing_expected_slot_fields') or [])
    noise_move_ids = list(completeness_audit.get('noise_move_ids') or [])
    critical_role_coverage = float(completeness_audit.get('critical_role_coverage_ratio') or 0.0)
    if not passed or score < _YELLOW_THRESHOLD:
        quality_tier = 'red'
    elif (
        bool(gate_report.get('sparse_trace'))
        or score < _GREEN_THRESHOLD
        or critical_role_coverage < 0.67
        or bool(missing_expected_roles)
        or len(missing_expected_slots) >= 2
        or bool(noise_move_ids)
    ):
        quality_tier = 'yellow'
    else:
        quality_tier = 'green'
    audit_status = 'eligible' if needs_lightweight_audit(gate_report) else 'not_needed' if passed else 'blocked'
    return {
        'quality_tier': quality_tier,
        'quality_tier_score': round(score, 4),
        'quality_flags': list(gate_report.get('soft_flags') or []),
        'hot_path_gate_report': gate_report,
        'l2_completeness_audit': completeness_audit,
        'audit_status': audit_status,
    }
