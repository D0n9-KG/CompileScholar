from __future__ import annotations

import re
from typing import Any, Iterable

from .models import MentionValue, PaperLogicTrace, ResearchMove


_WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.IGNORECASE)
_GENERIC_TOKENS = {
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'behavior',
    'for',
    'in',
    'is',
    'method',
    'methods',
    'model',
    'of',
    'on',
    'paper',
    'results',
    'study',
    'studied',
    'system',
    'that',
    'the',
    'this',
    'to',
    'under',
    'using',
    'we',
    'with',
}


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        token = str(value or '').strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _mention_token(mention: MentionValue) -> str:
    return str(mention.normalized or mention.surface or '').strip()


def _mention_tokens(mentions: list[MentionValue]) -> list[str]:
    return _unique(_mention_token(mention) for mention in mentions)


def _summary_terms(summary: str) -> list[str]:
    return [
        token.lower()
        for token in _WORD_RE.findall(str(summary or '').lower())
        if token and token.lower() not in _GENERIC_TOKENS and len(token) >= 3
    ]


def _summary_fallback_tokens(move: ResearchMove) -> dict[str, list[str]]:
    terms = _summary_terms(move.summary)
    if not terms:
        return {'method_tokens': [], 'object_tokens': [], 'condition_tokens': []}

    object_tokens = _unique(
        [' '.join(terms[i : i + 2]) for i in range(max(0, min(len(terms) - 1, 2)))]
        or terms[:2]
    )

    condition_tokens: list[str] = []
    lowered = str(move.summary or '').lower()
    for cue in ('under', 'with', 'at', 'during'):
        marker = f'{cue} '
        if marker not in lowered:
            continue
        tail = lowered.split(marker, 1)[1]
        words = [token for token in _WORD_RE.findall(tail) if token and token.lower() not in _GENERIC_TOKENS]
        if words:
            condition_tokens.append(' '.join(words[:3]).lower())

    if move.role in {'method', 'experiment'}:
        return {
            'method_tokens': object_tokens[:3],
            'object_tokens': [],
            'condition_tokens': _unique(condition_tokens)[:2],
        }
    return {
        'method_tokens': [],
        'object_tokens': object_tokens[:3],
        'condition_tokens': _unique(condition_tokens)[:2],
    }


def _slot_provenance_for(move: ResearchMove, field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in move.slot_provenance:
        if item.field != field:
            continue
        rows.append(
            {
                'field': item.field,
                'value_index': item.value_index,
                'anchor_ids': list(item.anchor_ids),
                'extraction_mode': item.extraction_mode,
                'support_strength': item.support_strength,
                'confidence': item.confidence,
                'notes': item.notes,
            }
        )
    return rows


def _slot_entries(field: str, moves: list[ResearchMove], attr_name: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for move in moves:
        mentions: list[MentionValue] = list(getattr(move, attr_name))
        for mention in mentions:
            token = _mention_token(mention)
            if not token:
                continue
            entries.append(
                {
                    'move_id': move.move_id,
                    'field': field,
                    'surface': mention.surface,
                    'normalized': mention.normalized,
                    'type': mention.type,
                    'anchor_ids': list(mention.anchor_ids),
                    'provenance': _slot_provenance_for(move, field),
                }
            )
    return entries


def build_l2_5_slot_inventory(moves: list[ResearchMove]) -> dict[str, list[dict[str, Any]]]:
    return {
        'research_objects': _slot_entries('research_objects', moves, 'research_objects'),
        'methods': _slot_entries('methods', moves, 'methods'),
        'metrics': _slot_entries('metrics', moves, 'metrics'),
        'conditions': _slot_entries('conditions', moves, 'conditions'),
        'comparators': _slot_entries('comparators', moves, 'comparators'),
        'limitation_types': _slot_entries('limitation_types', moves, 'limitation_types'),
        'resource_mentions': _slot_entries('resource_mentions', moves, 'resource_mentions'),
    }


def build_l1_bridge_hints(moves: list[ResearchMove]) -> dict[str, list[dict[str, Any]]]:
    resource_candidates = _slot_entries('resource_mentions', moves, 'resource_mentions')
    metric_candidates = _slot_entries('metrics', moves, 'metrics')

    benchmark_candidates = [
        candidate
        for candidate in resource_candidates
        if str(candidate.get('type') or '').strip().lower() == 'benchmark'
    ]

    return {
        'resource_candidates': resource_candidates,
        'benchmark_candidates': benchmark_candidates,
        'metric_candidates': metric_candidates,
        'protocol_candidates': [],
        'toolchain_candidates': [],
    }


def build_community_signatures(paper_id: str, moves: list[ResearchMove]) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for move in moves:
        fallback = _summary_fallback_tokens(move)
        method_tokens = _mention_tokens(move.methods) or list(fallback['method_tokens'])
        object_tokens = _mention_tokens(move.research_objects) or list(fallback['object_tokens'])
        condition_tokens = _mention_tokens(move.conditions) or list(fallback['condition_tokens'])
        signatures.append(
            {
                'move_id': move.move_id,
                'paper_id': paper_id,
                'role': move.role,
                'act_type': move.act_type,
                'method_tokens': method_tokens,
                'object_tokens': object_tokens,
                'metric_tokens': _mention_tokens(move.metrics),
                'condition_tokens': condition_tokens,
                'comparator_tokens': _mention_tokens(move.comparators),
                'effect_directions': _unique(effect.direction for effect in move.effects),
                'limitation_tokens': _mention_tokens(move.limitation_types),
                'resource_tokens': _mention_tokens(move.resource_mentions),
                'anchor_ids': list(move.anchor_ids),
            }
        )
    return signatures


def build_route_feature_candidates(moves: list[ResearchMove]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for move in moves:
        if move.methods:
            candidates.append(
                {
                    'move_id': move.move_id,
                    'feature_type': 'method_candidate',
                    'tokens': _mention_tokens(move.methods),
                    'anchor_ids': list(move.anchor_ids),
                }
            )
        if move.metrics:
            candidates.append(
                {
                    'move_id': move.move_id,
                    'feature_type': 'metric_candidate',
                    'tokens': _mention_tokens(move.metrics),
                    'anchor_ids': list(move.anchor_ids),
                }
            )
        if move.conditions:
            candidates.append(
                {
                    'move_id': move.move_id,
                    'feature_type': 'condition_candidate',
                    'tokens': _mention_tokens(move.conditions),
                    'anchor_ids': list(move.anchor_ids),
                }
            )
    return candidates


def build_paper_summaries(trace: PaperLogicTrace) -> dict[str, Any]:
    role_distribution: dict[str, int] = {}
    for move in trace.canonical_core.moves:
        role_distribution[move.role] = role_distribution.get(move.role, 0) + 1

    move_summaries = [move.summary.strip() for move in trace.canonical_core.moves if move.summary.strip()]
    one_paragraph_summary = ' '.join(move_summaries[:3]).strip()

    return {
        'move_role_distribution': role_distribution,
        'one_paragraph_summary': one_paragraph_summary,
        'key_method_summary': next((move.summary for move in trace.canonical_core.moves if move.role == 'method'), ''),
    }


def build_derived_views(trace: PaperLogicTrace) -> dict[str, Any]:
    moves = trace.canonical_core.moves
    return {
        'l2_5_slot_inventory': build_l2_5_slot_inventory(moves),
        'l1_bridge_hints': build_l1_bridge_hints(moves),
        'community_signatures': build_community_signatures(trace.paper_metadata.paper_id, moves),
        'route_feature_candidates': build_route_feature_candidates(moves),
        'paper_summaries': build_paper_summaries(trace),
    }
