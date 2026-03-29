from __future__ import annotations

import re
from difflib import SequenceMatcher
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
_RESEARCH_OBJECT_REQUIRED_FOR_L4 = {'empirical', 'benchmark', 'case_study', 'theoretical', 'unknown'}
_COMPARATOR_REQUIRED_FOR_L4 = {'empirical', 'benchmark', 'case_study', 'unknown'}
_OUTCOME_COMPARATOR_ROLES = {'result', 'experiment', 'interpretation'}
_TRUSTED_EXTRACTION_MODES = {'direct', 'normalized'}
_TRUSTED_SUPPORT_STRENGTHS = {'strong', 'exact'}
_COMPARISON_SUMMARY_PATTERNS = (
    ' agreement with ',
    ' agree with ',
    ' agrees with ',
    ' compared against ',
    ' compared to ',
    ' compared with ',
    ' compare against ',
    ' compare to ',
    ' compare with ',
    ' difference between ',
    ' differences between ',
    ' higher than ',
    ' in contrast to ',
    ' lower than ',
    ' over baseline ',
    ' over the baseline ',
    ' relative to ',
    ' similar to ',
    ' unlike ',
    ' versus ',
    ' vs ',
)
_ROUTE_STATE_SEED_REQUIRED_COMPONENTS = (
    'topic_scope_candidates',
    'dominant_method_candidates',
    'supporting_evidence_ids',
    'readiness_feature_inputs.method_maturity_signals',
    'source_move_ids',
)
_ROUTE_STATE_SEED_CONTEXT_GROUPS: dict[str, tuple[str, ...]] = {
    'challenge_signal': ('challenging_evidence_ids',),
    'bottleneck_signal': (
        'known_bottleneck_candidates',
        'readiness_feature_inputs.bottleneck_signals',
    ),
    'measurement_signal': (
        'measurement_protocol_candidates',
        'readiness_feature_inputs.measurement_maturity_signals',
    ),
    'resource_signal': (
        'active_benchmark_candidates',
        'readiness_feature_inputs.data_resource_signals',
    ),
    'infrastructure_signal': (
        'toolchain_candidates',
        'readiness_feature_inputs.infrastructure_signals',
    ),
    'enabling_signal': (
        'enabling_condition_candidates',
        'alternative_route_candidates',
    ),
}
_ROUTE_STATE_SEED_MIN_CONTEXT_GROUPS = 2
_METADATA_TOKEN_RE = re.compile(r'[0-9a-z]+(?:-[0-9a-z]+)?', re.IGNORECASE)
_METADATA_GENERIC_TOKENS = {
    'and',
    'for',
    'from',
    'into',
    'model',
    'models',
    'paper',
    'study',
    'system',
    'the',
    'this',
    'with',
}


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(float(numerator) / float(denominator), 1.0))


def _is_textual_char(char: str) -> bool:
    return char.isalnum() or '\u4e00' <= char <= '\u9fff'


def _normalize_similarity_text(value: str | None) -> str:
    pieces: list[str] = []
    for char in str(value or ''):
        pieces.append(char.lower() if _is_textual_char(char) else ' ')
    return ' '.join(''.join(pieces).split())


def _text_similarity(left: str | None, right: str | None) -> float:
    normalized_left = _normalize_similarity_text(left)
    normalized_right = _normalize_similarity_text(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    return SequenceMatcher(a=normalized_left, b=normalized_right).ratio()


def _semantic_text_tokens(value: str | None) -> set[str]:
    lowered = str(value or '').lower()
    tokens = {
        token
        for token in _METADATA_TOKEN_RE.findall(lowered)
        if len(token) >= 3 and token not in _METADATA_GENERIC_TOKENS
    }
    for run in re.findall(r'[\u4e00-\u9fff]{2,}', lowered):
        tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _row_value(row: Any, field: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(field, default)
    return getattr(row, field, default)


def _slot_provenance_rows(move: ResearchMove, field: str) -> list[Any]:
    rows: list[Any] = []
    for row in list(move.slot_provenance or []):
        if str(_row_value(row, 'field', '') or '') != field:
            continue
        rows.append(row)
    return rows


def _has_trusted_slot_signal(move: ResearchMove, field: str) -> bool:
    values = list(getattr(move, field, None) or [])
    if not values:
        return False
    provenance_rows = _slot_provenance_rows(move, field)
    if not provenance_rows:
        return True
    field_level_rows = [row for row in provenance_rows if _row_value(row, 'value_index', None) is None]
    for provenance in field_level_rows:
        extraction_mode = str(_row_value(provenance, 'extraction_mode', '') or '').strip().lower()
        support_strength = str(_row_value(provenance, 'support_strength', '') or '').strip().lower()
        if extraction_mode in _TRUSTED_EXTRACTION_MODES and support_strength in _TRUSTED_SUPPORT_STRENGTHS:
            return True
    provenance_by_index = {
        int(_row_value(row, 'value_index', -1)): row
        for row in provenance_rows
        if _row_value(row, 'value_index', None) is not None
    }
    for index, _value in enumerate(values):
        provenance = provenance_by_index.get(index)
        if provenance is None:
            continue
        extraction_mode = str(_row_value(provenance, 'extraction_mode', '') or '').strip().lower()
        support_strength = str(_row_value(provenance, 'support_strength', '') or '').strip().lower()
        if extraction_mode in _TRUSTED_EXTRACTION_MODES and support_strength in _TRUSTED_SUPPORT_STRENGTHS:
            return True
    return False


def _trusted_slot_value_indices(move: ResearchMove, field: str) -> set[int]:
    values = list(getattr(move, field, None) or [])
    if not values:
        return set()
    provenance_rows = _slot_provenance_rows(move, field)
    if not provenance_rows:
        return set(range(len(values)))
    field_level_rows = [row for row in provenance_rows if _row_value(row, 'value_index', None) is None]
    for provenance in field_level_rows:
        extraction_mode = str(_row_value(provenance, 'extraction_mode', '') or '').strip().lower()
        support_strength = str(_row_value(provenance, 'support_strength', '') or '').strip().lower()
        if extraction_mode in _TRUSTED_EXTRACTION_MODES and support_strength in _TRUSTED_SUPPORT_STRENGTHS:
            return set(range(len(values)))
    indices: set[int] = set()
    for provenance in provenance_rows:
        raw_index = _row_value(provenance, 'value_index', None)
        if raw_index is None:
            continue
        extraction_mode = str(_row_value(provenance, 'extraction_mode', '') or '').strip().lower()
        support_strength = str(_row_value(provenance, 'support_strength', '') or '').strip().lower()
        if extraction_mode not in _TRUSTED_EXTRACTION_MODES or support_strength not in _TRUSTED_SUPPORT_STRENGTHS:
            continue
        try:
            index = int(raw_index)
        except Exception:
            continue
        if 0 <= index < len(values):
            indices.add(index)
    return indices


def _has_informative_effect_signal(move: ResearchMove) -> bool:
    for effect in list(move.effects or []):
        direction = str(_row_value(effect, 'direction', '') or '').strip().lower()
        if direction and direction not in {'unknown', 'none'}:
            return True
    return False


def _has_semantic_outcome_comparator_signal(move: ResearchMove) -> bool:
    if str(move.role or '') not in _OUTCOME_COMPARATOR_ROLES:
        return False
    trusted_indices = _trusted_slot_value_indices(move, 'comparators')
    if not trusted_indices:
        return False
    if str(move.act_type or '') == 'compare_baseline':
        return True
    summary = f" {str(move.summary or '').strip().lower()} "
    if any(pattern in summary for pattern in _COMPARISON_SUMMARY_PATTERNS):
        return True
    if len(trusted_indices) >= 2:
        return True
    for effect in list(move.effects or []):
        comparator_surface = str(_row_value(effect, 'comparator_surface', '') or '').strip()
        if comparator_surface:
            return True
    return False


def _trusted_slot_count(moves: list[ResearchMove], field: str, *, roles: set[str] | None = None) -> int:
    count = 0
    for move in moves:
        if roles is not None and str(move.role or '') not in roles:
            continue
        if _has_trusted_slot_signal(move, field):
            count += 1
    return count


def _has_grounded_constraint_context(move: ResearchMove) -> bool:
    return _has_trusted_slot_signal(move, 'conditions') or _has_trusted_slot_signal(move, 'limitation_types')


def _uses_theory_modeling_evidence_profile(
    *,
    paper_type_token: str,
    observed_roles: list[str],
    supported_slot_counts: dict[str, int],
) -> bool:
    if paper_type_token == 'theoretical':
        return True
    observed_role_set = set(observed_roles)
    has_unknown_theory_like_limitation_profile = (
        paper_type_token == 'unknown'
        and 'problem' in observed_role_set
        and 'method' in observed_role_set
        and 'limitation' in observed_role_set
        and 'experiment' not in observed_role_set
        and supported_slot_counts.get('methods', 0) > 0
        and supported_slot_counts.get('research_objects', 0) > 0
        and supported_slot_counts.get('limitation_types', 0) > 0
        and supported_slot_counts.get('metrics', 0) == 0
        and supported_slot_counts.get('comparators', 0) == 0
    )
    return (
        (
            'interpretation' in observed_role_set
            and 'result' not in observed_role_set
            and supported_slot_counts.get('methods', 0) > 0
            and supported_slot_counts.get('research_objects', 0) > 0
        )
        or has_unknown_theory_like_limitation_profile
    )


def _expected_role_present(*, role: str, observed_role_set: set[str], paper_type_token: str) -> bool:
    if role in observed_role_set:
        return True
    return paper_type_token == 'theoretical' and role == 'interpretation' and 'limitation' in observed_role_set


def _expected_slot_present(*, field: str, slot_counts: dict[str, int], paper_type_token: str) -> bool:
    if slot_counts.get(field, 0) > 0:
        return True
    return paper_type_token == 'theoretical' and field == 'conditions' and slot_counts.get('limitation_types', 0) > 0


def _expected_supported_slot_present(*, field: str, supported_slot_counts: dict[str, int], paper_type_token: str) -> bool:
    if supported_slot_counts.get(field, 0) > 0:
        return True
    return paper_type_token == 'theoretical' and field == 'conditions' and supported_slot_counts.get('limitation_types', 0) > 0


def _grounded_relation_count(move_relations: list[MoveRelation]) -> int:
    count = 0
    for relation in move_relations:
        relation_type = str(_row_value(relation, 'relation_type', '') or '').strip().lower()
        if relation_type == 'motivates':
            continue
        anchor_ids = list(_row_value(relation, 'anchor_ids', []) or [])
        if anchor_ids:
            count += 1
    return count


def _has_core_slot_signal(move: ResearchMove) -> bool:
    return any(_has_trusted_slot_signal(move, field) for field in _CORE_SLOT_FIELDS) or _has_informative_effect_signal(move)


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
    return _safe_ratio(_grounded_relation_count(move_relations), target)


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


def _looks_like_suspicious_metadata_title(title: str | None) -> bool:
    normalized = str(title or '').strip().lower()
    if not normalized:
        return True
    if any(normalized.startswith(prefix) for prefix in _QUALITY_NOISE_PREFIXES):
        return True
    if normalized.startswith('#') and len(normalized.split()) <= 6:
        return True
    return False


def _has_metadata_summary_mismatch(
    paper_metadata: dict[str, Any] | None,
    derived_views: dict[str, Any] | None,
) -> bool:
    metadata = dict(paper_metadata or {})
    paper_summaries = dict((derived_views or {}).get('paper_summaries') or {})
    title = str(metadata.get('title') or '').strip()
    summary = str(paper_summaries.get('one_paragraph_summary') or '').strip()
    if not title or not summary:
        return False
    if _looks_like_suspicious_metadata_title(title) or _looks_like_noise_summary(summary):
        return False
    title_tokens = _semantic_text_tokens(title)
    summary_tokens = _semantic_text_tokens(summary)
    if len(title_tokens) < 2 or len(summary_tokens) < 2:
        return False
    overlap = title_tokens & summary_tokens
    overlap_coverage = _safe_ratio(len(overlap), len(title_tokens))
    if overlap_coverage >= 0.2 or len(overlap) >= 5:
        return False
    return _text_similarity(title, summary) < 0.3


def _route_state_seed_audit(derived_views: dict[str, Any] | None) -> dict[str, Any]:
    route_state_seed = dict((derived_views or {}).get('route_state_seed') or {})
    readiness_inputs = dict(route_state_seed.get('readiness_feature_inputs') or {})
    if not route_state_seed:
        return {
            'available': False,
            'ready_for_route_compilation': False,
            'missing_seed_components': [],
            'component_counts': {},
        }

    component_specs = (
        ('topic_scope_candidates', route_state_seed.get('topic_scope_candidates')),
        ('dominant_method_candidates', route_state_seed.get('dominant_method_candidates')),
        ('active_benchmark_candidates', route_state_seed.get('active_benchmark_candidates')),
        ('known_bottleneck_candidates', route_state_seed.get('known_bottleneck_candidates')),
        ('enabling_condition_candidates', route_state_seed.get('enabling_condition_candidates')),
        ('alternative_route_candidates', route_state_seed.get('alternative_route_candidates')),
        ('measurement_protocol_candidates', route_state_seed.get('measurement_protocol_candidates')),
        ('toolchain_candidates', route_state_seed.get('toolchain_candidates')),
        ('supporting_evidence_ids', route_state_seed.get('supporting_evidence_ids')),
        ('challenging_evidence_ids', route_state_seed.get('challenging_evidence_ids')),
        ('readiness_feature_inputs.method_maturity_signals', readiness_inputs.get('method_maturity_signals')),
        ('readiness_feature_inputs.measurement_maturity_signals', readiness_inputs.get('measurement_maturity_signals')),
        ('readiness_feature_inputs.data_resource_signals', readiness_inputs.get('data_resource_signals')),
        ('readiness_feature_inputs.infrastructure_signals', readiness_inputs.get('infrastructure_signals')),
        ('readiness_feature_inputs.bottleneck_signals', readiness_inputs.get('bottleneck_signals')),
    )
    component_counts = {
        field: len(list(values or []))
        for field, values in component_specs
    }
    component_counts['source_move_ids'] = len(list(route_state_seed.get('source_move_ids') or []))
    missing_seed_components = [
        field
        for field, values in component_specs
        if not list(values or [])
    ]
    if component_counts['source_move_ids'] == 0:
        missing_seed_components.append('source_move_ids')

    missing_required_seed_components = [
        field
        for field in _ROUTE_STATE_SEED_REQUIRED_COMPONENTS
        if component_counts.get(field, 0) == 0
    ]

    context_group_coverage: dict[str, dict[str, Any]] = {}
    covered_context_groups: list[str] = []
    missing_context_groups: list[str] = []
    for group_name, fields in _ROUTE_STATE_SEED_CONTEXT_GROUPS.items():
        covered_fields = [field for field in fields if component_counts.get(field, 0) > 0]
        covered = bool(covered_fields)
        context_group_coverage[group_name] = {
            'covered': covered,
            'fields': list(fields),
            'covered_fields': covered_fields,
        }
        if covered:
            covered_context_groups.append(group_name)
        else:
            missing_context_groups.append(group_name)

    covered_context_group_count = len(covered_context_groups)
    route_compilation_blockers = list(missing_required_seed_components)
    if covered_context_group_count < _ROUTE_STATE_SEED_MIN_CONTEXT_GROUPS:
        route_compilation_blockers.append(
            f'context_groups<{_ROUTE_STATE_SEED_MIN_CONTEXT_GROUPS}'
        )

    return {
        'available': True,
        'ready_for_route_compilation': not route_compilation_blockers,
        'missing_seed_components': missing_seed_components,
        'missing_required_seed_components': missing_required_seed_components,
        'route_compilation_blockers': route_compilation_blockers,
        'context_group_coverage': context_group_coverage,
        'covered_context_groups': covered_context_groups,
        'missing_context_groups': missing_context_groups,
        'covered_context_group_count': covered_context_group_count,
        'min_context_groups_for_route_compilation': _ROUTE_STATE_SEED_MIN_CONTEXT_GROUPS,
        'component_counts': component_counts,
    }


def is_noise_summary(summary: str) -> bool:
    return _looks_like_noise_summary(summary)


def _l2_completeness_audit(
    *,
    moves: list[ResearchMove],
    move_relations: list[MoveRelation],
    paper_type: str,
) -> dict[str, Any]:
    paper_type_token = str(paper_type or 'unknown')
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
    supported_slot_counts = {
        field: _trusted_slot_count(moves, field)
        for field in _CORE_SLOT_FIELDS
    }
    supported_outcome_comparator_count = sum(1 for move in moves if _has_semantic_outcome_comparator_signal(move))

    expected_roles = list(_EXPECTED_ROLES_BY_PAPER_TYPE.get(paper_type_token, _EXPECTED_ROLES_BY_PAPER_TYPE['unknown']))
    expected_slots = list(_EXPECTED_SLOTS_BY_PAPER_TYPE.get(paper_type_token, _EXPECTED_SLOTS_BY_PAPER_TYPE['unknown']))
    observed_role_set = set(observed_roles)
    missing_expected_roles = sorted(
        role
        for role in expected_roles
        if not _expected_role_present(
            role=role,
            observed_role_set=observed_role_set,
            paper_type_token=paper_type_token,
        )
    )
    matched_expected_role_count = sum(
        1
        for role in expected_roles
        if _expected_role_present(
            role=role,
            observed_role_set=observed_role_set,
            paper_type_token=paper_type_token,
        )
    )
    critical_role_ratio = _safe_ratio(matched_expected_role_count, len(expected_roles))
    missing_l2_slot_fields = [field for field in _CORE_SLOT_FIELDS if slot_counts.get(field, 0) == 0]
    missing_expected_slot_fields = [
        field
        for field in expected_slots
        if not _expected_slot_present(
            field=field,
            slot_counts=slot_counts,
            paper_type_token=paper_type_token,
        )
    ]
    informative_effect_count = sum(1 for move in moves if _has_informative_effect_signal(move))
    missing_supported_expected_slot_fields = [
        field
        for field in expected_slots
        if (
            informative_effect_count == 0
            if field == 'effects'
            else not _expected_supported_slot_present(
                field=field,
                supported_slot_counts=supported_slot_counts,
                paper_type_token=paper_type_token,
            )
        )
    ]
    requires_comparator_signal_for_l4 = paper_type_token in _COMPARATOR_REQUIRED_FOR_L4
    if requires_comparator_signal_for_l4 and supported_outcome_comparator_count == 0 and 'comparators' not in missing_supported_expected_slot_fields:
        missing_supported_expected_slot_fields.append('comparators')
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
    requires_outcome_signal = paper_type_token not in {'software'}
    ready_for_community = signature_ready_move_count >= 3 and (slot_counts.get('methods', 0) > 0 or slot_counts.get('research_objects', 0) > 0)
    ready_for_l3 = (
        not missing_expected_roles
        and relation_ratio >= 0.4
        and critical_role_ratio >= 0.67
        and (has_outcome_signal or not requires_outcome_signal)
        and (supported_slot_counts.get('methods', 0) > 0 or supported_slot_counts.get('research_objects', 0) > 0)
    )
    evidence_signal_count = sum(
        1
        for field in ('metrics', 'comparators')
        if (
            supported_outcome_comparator_count > 0
            if field == 'comparators'
            else supported_slot_counts.get(field, 0) > 0
        )
    )
    if informative_effect_count > 0:
        evidence_signal_count += 1
    context_signal_count = sum(1 for field in ('conditions', 'resource_mentions', 'limitation_types') if supported_slot_counts.get(field, 0) > 0)
    requires_object_signal_for_l4 = paper_type_token in _RESEARCH_OBJECT_REQUIRED_FOR_L4
    empirical_ready_for_l4 = (
        ready_for_l3
        and informative_effect_count > 0
        and evidence_signal_count >= 2
        and context_signal_count >= 1
        and (supported_slot_counts.get('research_objects', 0) > 0 or not requires_object_signal_for_l4)
        and (supported_outcome_comparator_count > 0 or not requires_comparator_signal_for_l4)
    )
    theory_modeling_profile = _uses_theory_modeling_evidence_profile(
        paper_type_token=paper_type_token,
        observed_roles=observed_roles,
        supported_slot_counts=supported_slot_counts,
    )
    grounded_constraint_move_count = sum(
        1
        for move in moves
        if str(move.role or '') in {'interpretation', 'limitation', 'future_work'} and _has_grounded_constraint_context(move)
    )
    theory_context_signal_count = sum(1 for field in ('conditions', 'limitation_types') if supported_slot_counts.get(field, 0) > 0)
    theory_modeling_ready_for_l4 = (
        theory_modeling_profile
        and ready_for_l3
        and supported_slot_counts.get('research_objects', 0) > 0
        and supported_slot_counts.get('methods', 0) > 0
        and theory_context_signal_count >= 1
        and grounded_constraint_move_count > 0
    )
    ready_for_l4 = empirical_ready_for_l4 or theory_modeling_ready_for_l4

    role_score = _safe_ratio(len(expected_roles) - len(missing_expected_roles), len(expected_roles))
    expected_slot_score = _safe_ratio(len(expected_slots) - len(missing_expected_slot_fields), len(expected_slots))
    supported_expected_slot_score = _safe_ratio(
        len(expected_slots) - len(missing_supported_expected_slot_fields),
        len(expected_slots),
    )
    sparse_penalty = _safe_ratio(len(sparse_expected_slot_fields), len(expected_slots))
    noise_penalty = _safe_ratio(len(noise_move_ids), move_count)
    completeness_score = max(
        0.0,
        min(
            1.0,
            role_score * 0.25
            + critical_role_ratio * 0.20
            + expected_slot_score * 0.20
            + supported_expected_slot_score * 0.20
            + (1.0 - sparse_penalty) * 0.05
            + (1.0 - noise_penalty) * 0.05
            + relation_ratio * 0.05,
        ),
    )

    return {
        'paper_type': paper_type_token,
        'move_count': move_count,
        'role_counts': role_counts,
        'observed_roles': observed_roles,
        'expected_roles': expected_roles,
        'missing_expected_roles': missing_expected_roles,
        'critical_role_coverage_ratio': round(critical_role_ratio, 4),
        'slot_counts': slot_counts,
        'supported_slot_counts': supported_slot_counts,
        'supported_semantic_outcome_comparator_count': supported_outcome_comparator_count,
        'slot_move_ratios': slot_move_ratios,
        'missing_l2_slot_fields': missing_l2_slot_fields,
        'expected_slot_fields': expected_slots,
        'missing_expected_slot_fields': missing_expected_slot_fields,
        'missing_supported_expected_slot_fields': missing_supported_expected_slot_fields,
        'sparse_expected_slot_fields': sparse_expected_slot_fields,
        'noise_move_ids': noise_move_ids,
        'signature_ready_move_count': signature_ready_move_count,
        'relation_coverage_ratio': round(relation_ratio, 4),
        'l4_evidence_profile': 'theory_modeling' if theory_modeling_profile else 'empirical_default',
        'grounded_constraint_move_count': grounded_constraint_move_count,
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
    completeness_audit = dict(gate_report.get('l2_completeness_audit') or {})
    paper_type = str(completeness_audit.get('paper_type') or 'unknown')
    if not bool(completeness_audit.get('ready_for_l3')):
        return True
    if paper_type in _RESEARCH_OBJECT_REQUIRED_FOR_L4 and not bool(completeness_audit.get('ready_for_l4')):
        return True
    return float(gate_report.get('quality_tier_score') or 0.0) < _GREEN_THRESHOLD


def build_quality_payload(
    gate_report: dict[str, Any],
    *,
    paper_metadata: dict[str, Any] | None = None,
    derived_views: dict[str, Any] | None = None,
) -> dict[str, Any]:
    passed = bool(gate_report.get('passed'))
    score = float(gate_report.get('quality_tier_score') or 0.0)
    completeness_audit = dict(gate_report.get('l2_completeness_audit') or {})
    missing_expected_roles = list(completeness_audit.get('missing_expected_roles') or [])
    missing_expected_slots = list(completeness_audit.get('missing_expected_slot_fields') or [])
    noise_move_ids = list(completeness_audit.get('noise_move_ids') or [])
    critical_role_coverage = float(completeness_audit.get('critical_role_coverage_ratio') or 0.0)
    paper_type = str(completeness_audit.get('paper_type') or 'unknown')
    ready_for_l3 = bool(completeness_audit.get('ready_for_l3'))
    ready_for_l4 = bool(completeness_audit.get('ready_for_l4'))
    requires_l4_for_green = paper_type in _RESEARCH_OBJECT_REQUIRED_FOR_L4
    quality_flags = list(gate_report.get('soft_flags') or [])
    metadata_summary_mismatch = _has_metadata_summary_mismatch(paper_metadata, derived_views)
    route_state_seed_audit = _route_state_seed_audit(derived_views)
    route_state_seed_thin = route_state_seed_audit['available'] and not route_state_seed_audit['ready_for_route_compilation']
    if metadata_summary_mismatch and 'metadata_summary_mismatch' not in quality_flags:
        quality_flags.append('metadata_summary_mismatch')
    if route_state_seed_thin and 'route_state_seed_thin' not in quality_flags:
        quality_flags.append('route_state_seed_thin')
    if not passed or score < _YELLOW_THRESHOLD:
        quality_tier = 'red'
    elif (
        bool(gate_report.get('sparse_trace'))
        or score < _GREEN_THRESHOLD
        or critical_role_coverage < 0.67
        or bool(missing_expected_roles)
        or len(missing_expected_slots) >= 2
        or bool(noise_move_ids)
        or not ready_for_l3
        or (requires_l4_for_green and not ready_for_l4)
        or metadata_summary_mismatch
        or route_state_seed_thin
    ):
        quality_tier = 'yellow'
    else:
        quality_tier = 'green'
    audit_status = 'eligible' if needs_lightweight_audit(gate_report) else 'not_needed' if passed else 'blocked'
    if metadata_summary_mismatch and passed:
        audit_status = 'eligible'
    if route_state_seed_thin and passed:
        audit_status = 'eligible'
    return {
        'quality_tier': quality_tier,
        'quality_tier_score': round(score, 4),
        'quality_flags': quality_flags,
        'hot_path_gate_report': gate_report,
        'l2_completeness_audit': completeness_audit,
        'audit_status': audit_status,
        'route_state_seed_audit': route_state_seed_audit,
    }
