from __future__ import annotations

import re
from typing import Any, Iterable

from .gates import is_noise_summary
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
_TRUSTED_EXTRACTION_MODES = {'direct', 'normalized'}
_TRUSTED_SUPPORT_STRENGTHS = {'strong', 'exact'}
_L1_TOOLCHAIN_RESOURCE_TYPES = {'hardware', 'instrument', 'platform', 'software', 'tool'}
_SUMMARY_ROLE_PRIORITY = {
    'result': 5,
    'method': 5,
    'experiment': 5,
    'problem': 4,
    'interpretation': 3,
    'limitation': 3,
    'future_work': 2,
    'hypothesis': 2,
    'background': 1,
}
_SUMMARY_SECTION_PRIORITY = (
    ('abstract', 4),
    ('summary', 4),
    ('conclusion', 3),
    ('discussion', 3),
    ('result', 2),
    ('method', 2),
    ('experiment', 2),
)
_TOPIC_ROLE_PRIORITY = {
    'result': 5,
    'interpretation': 4,
    'problem': 4,
    'experiment': 3,
    'method': 2,
    'background': 1,
}
_SUMMARY_NOISE_CUES = (
    'accepted ',
    'available online',
    'corresponding author',
    'received ',
)
_METHOD_SUMMARY_POSITIVE_CUES = (
    ' uses ',
    ' using ',
    ' propose',
    ' describes ',
    ' represent',
    ' model ',
    ' simulate',
    ' workflow',
)
_METHOD_SUMMARY_NEGATIVE_CUES = (
    ' gave detailed insight ',
    ' improve',
    ' improved',
    ' outperforms ',
    ' outperformed ',
    ' better accuracy ',
    ' reveals ',
    ' showed ',
    ' shows ',
    ' reports ',
    ' at most ',
)
_CONTENT_PROFILE_CURRENT_WORK_CUES = (
    ' this paper ',
    ' this study ',
    ' this work ',
    ' this article ',
    ' in this paper ',
    ' in this study ',
    ' in this work ',
    ' our approach ',
    ' our method ',
    ' our model ',
    ' we propose ',
    ' we present ',
    ' we develop ',
    ' we introduce ',
    ' we use ',
    ' we employ ',
    ' we perform ',
    ' 本文',
    ' 本研究',
    ' 本工作',
    ' 文中',
)
_CONTENT_PROFILE_PRIOR_WORK_CUES = (
    ' previous work ',
    ' prior work ',
    ' previous study ',
    ' previous studies ',
    ' prior study ',
    ' prior studies ',
    ' earlier work ',
    ' earlier studies ',
    ' work by ',
    ' studies by ',
    ' reported by ',
    ' proposed by ',
    ' developed by ',
    ' has addressed ',
    ' have addressed ',
    ' has been conducted ',
    ' have been conducted ',
    ' et al',
    ' ref. [',
    ' refs. [',
    ' in [',
    ' 前人',
    ' 已有研究',
    ' 已有工作',
    ' 前期研究',
    ' 文献',
)
_BACKGROUND_CONTEXT_CUES = (
    'describes the experimental setup',
    'experimental setup',
    'measurement techniques',
    'study was conducted',
    'system properties',
    'paper describes',
    'simulation model considers',
)
_METHOD_SIGNAL_OPERATIVE_SUMMARY_CUES = (
    ' employ ',
    ' employs ',
    ' propose',
    ' proposes ',
    ' simulate',
    ' models contact forces',
    ' determine particle velocities',
    ' determine shear rate',
    ' iterative ',
    ' iteration',
    '试验',
)
_METHOD_SIGNAL_BROAD_SUMMARY_CUES = (
    'describes the governing equations',
    'governing equations',
    'framework is described',
    'simulation model considers',
    'global step enforces',
    'parameters are selected',
    'describe their simulation system',
    'initial configuration generation method',
)
_METHOD_SIGNAL_OPERATIVE_LABEL_CUES = (
    'algorithm',
    'approach',
    'test',
    'protocol',
    'workflow',
    'procedure',
    'simulation',
    'dynamics',
    'iteration',
    '试验',
)
_METHOD_SIGNAL_GENERIC_LABEL_CUES = (
    'equation',
    'equations',
    'law',
    'laws',
    'resistance',
    'regularization',
    'parameter',
    'parameters',
    'variable',
    'variables',
    'state law',
    'state laws',
    'evolution law',
    'evolution laws',
)
_SECTION_HEADING_RE = re.compile(r'^\s*(?:\d+(?:\.\d+)*|[ivx]+)\.?\s+', re.IGNORECASE)
_PIPE_SECTION_HEADING_RE = re.compile(r'^\s*(?:section\s+)?\d+(?:\.\d+)*\s*[|:：-]\s+\S', re.IGNORECASE)
_LATEX_TITLE_NOISE_RE = re.compile(r'(?:\\(?:mathrm|text|begin|end)\b|\$)')
_GENERIC_SECTION_HEADINGS = {
    'abstract',
    'conclusion',
    'discussion',
    'experimental setup',
    'introduction',
    'materials and methods',
    'method',
    'methods',
    'results',
    'simulation procedure',
}
_SUMMARY_CJK_RE = re.compile(r'[\u4e00-\u9fff]')
_SUMMARY_LATIN_RE = re.compile(r'[A-Za-z]')


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


def _summary_anchor_sections(trace: PaperLogicTrace) -> dict[str, str]:
    sections: dict[str, str] = {}
    for anchor in trace.canonical_core.evidence_anchors:
        joined = ' > '.join(str(part or '').strip().lower() for part in (anchor.section_path or []) if str(part or '').strip())
        sections[anchor.anchor_id] = joined
    return sections


def _section_priority(section_path: str) -> int:
    lowered = str(section_path or '').lower()
    if not lowered:
        return 0
    for cue, score in _SUMMARY_SECTION_PRIORITY:
        if cue in lowered:
            return score
    return 0


def _looks_like_author_fragment(summary: str) -> bool:
    stripped = str(summary or '').strip()
    if not stripped:
        return False
    if _SUMMARY_CJK_RE.search(stripped):
        return False
    lowered = stripped.lower()
    if is_noise_summary(stripped) or lowered.startswith('©'):
        return True
    if any(cue in lowered for cue in _SUMMARY_NOISE_CUES):
        return True
    words = [token for token in re.split(r'\s+', stripped) if token]
    if len(words) > 5:
        return False
    cleaned_words = [re.sub(r'[^A-Za-z.]', '', word) for word in words]
    cleaned_words = [word for word in cleaned_words if word]
    if not cleaned_words:
        return False
    if any(word.lower().strip('.') in _GENERIC_TOKENS for word in cleaned_words):
        return False
    decorated = 0
    for word in cleaned_words:
        bare = word.strip('.')
        if not bare:
            continue
        if word.endswith('.'):
            decorated += 1
            continue
        if bare.isupper() or bare.istitle():
            decorated += 1
    return decorated == len(cleaned_words)


def _is_summary_contentful(summary: str) -> bool:
    stripped = str(summary or '').strip()
    if not stripped:
        return False
    if _looks_like_author_fragment(stripped):
        return False
    cjk_count = len(_SUMMARY_CJK_RE.findall(stripped))
    if cjk_count >= 10 and len(stripped) >= 16:
        return True
    if len(stripped) >= 36:
        return True
    return len(stripped.split()) >= 5


def _move_summary_score(move: ResearchMove, anchor_sections: dict[str, str]) -> tuple[int, int]:
    summary = str(move.summary or '').strip()
    if not _is_summary_contentful(summary):
        return (-1, 0)
    section_score = max((_section_priority(anchor_sections.get(anchor_id, '')) for anchor_id in move.anchor_ids), default=0)
    role_score = _SUMMARY_ROLE_PRIORITY.get(str(move.role or ''), 1)
    slot_bonus = 1 if any(bool(getattr(move, field, None)) for field in ('research_objects', 'methods', 'metrics', 'comparators', 'conditions', 'limitation_types', 'resource_mentions')) or bool(move.effects) else 0
    length_bonus = 1 if 40 <= len(summary) <= 280 else 0
    return (role_score + section_score * 3 + slot_bonus + length_bonus, section_score)


def _looks_like_section_heading(value: str | None) -> bool:
    normalized = ' '.join(str(value or '').strip().lower().split())
    if not normalized:
        return False
    if normalized in _GENERIC_SECTION_HEADINGS:
        return True
    if _PIPE_SECTION_HEADING_RE.match(normalized):
        return True
    if normalized.count('$') >= 2 or _LATEX_TITLE_NOISE_RE.search(normalized):
        return True
    if not _SECTION_HEADING_RE.match(normalized):
        return False
    tail = _SECTION_HEADING_RE.sub('', normalized).strip()
    if not tail:
        return True
    return tail in _GENERIC_SECTION_HEADINGS or len(tail.split()) <= 4


def _title_alignment_terms(*values: str | None) -> set[str]:
    terms: set[str] = set()
    for value in values:
        if _looks_like_section_heading(value):
            continue
        terms.update(_summary_terms(str(value or '')))
    return terms


def _title_alignment_score(summary: str, title_terms: set[str]) -> int:
    if not title_terms:
        return 0
    summary_terms = set(_summary_terms(summary))
    if not summary_terms:
        return 0
    return len(summary_terms & title_terms)


def _summary_language(summary: str) -> str:
    text = str(summary or '')
    cjk_count = len(_SUMMARY_CJK_RE.findall(text))
    latin_count = len(_SUMMARY_LATIN_RE.findall(text))
    if cjk_count >= 8 and cjk_count >= latin_count:
        return 'cjk'
    if latin_count >= 24 and latin_count >= cjk_count * 2:
        return 'latin'
    if cjk_count >= 4 and latin_count >= 8:
        return 'mixed'
    return 'unknown'


def _summary_role_bucket(move: ResearchMove) -> str:
    if move.role in {'problem', 'background'}:
        return 'opening'
    if move.role in {'method', 'experiment'}:
        return 'method'
    if move.role in {'result', 'interpretation'}:
        return 'outcome'
    if move.role in {'limitation', 'future_work'}:
        return 'extension'
    return 'other'


def _preferred_summary_language_from_pool(
    scored_moves: list[tuple[int, int, int, ResearchMove]],
) -> str | None:
    language_metrics: dict[str, dict[str, Any]] = {}
    for score, _section_score, _sequence_no, move in scored_moves:
        language = _summary_language(move.summary)
        if language not in {'cjk', 'latin'}:
            continue
        bucket = _summary_role_bucket(move)
        metrics = language_metrics.setdefault(
            language,
            {
                'core_buckets': set(),
                'extension_count': 0,
                'total_score': 0,
                'move_count': 0,
            },
        )
        if bucket in {'opening', 'method', 'outcome'}:
            metrics['core_buckets'].add(bucket)
        elif bucket == 'extension':
            metrics['extension_count'] += 1
        metrics['total_score'] += int(score)
        metrics['move_count'] += 1
    if not language_metrics:
        return None
    return sorted(
        language_metrics,
        key=lambda language: (
            -len(language_metrics[language]['core_buckets']),
            -int(language_metrics[language]['extension_count']),
            -int(language_metrics[language]['total_score']),
            -int(language_metrics[language]['move_count']),
        ),
    )[0]


def _preferred_summary_language(selected: list[ResearchMove]) -> str | None:
    counts: dict[str, int] = {}
    order: list[str] = []
    for move in selected:
        language = _summary_language(move.summary)
        if language not in {'cjk', 'latin'}:
            continue
        counts[language] = counts.get(language, 0) + 1
        if language not in order:
            order.append(language)
    if not counts:
        return None
    return sorted(counts, key=lambda language: (-counts[language], order.index(language)))[0]


def _select_summary_moves(trace: PaperLogicTrace) -> list[ResearchMove]:
    anchor_sections = _summary_anchor_sections(trace)
    title_terms = _title_alignment_terms(trace.paper_metadata.title, trace.paper_metadata.title_alt)
    title_alignment_by_move_id: dict[str, int] = {}
    scored_moves: list[tuple[int, int, int, ResearchMove]] = []
    for move in trace.canonical_core.moves:
        score, section_score = _move_summary_score(move, anchor_sections)
        if score < 0:
            continue
        title_alignment_by_move_id[move.move_id] = _title_alignment_score(move.summary, title_terms)
        scored_moves.append((score, section_score, int(move.sequence_no), move))
    if not scored_moves:
        return [move for move in trace.canonical_core.moves if str(move.summary or '').strip()][:3]
    ordered = sorted(scored_moves, key=lambda item: (-item[0], -item[1], item[2]))
    global_preferred_language = _preferred_summary_language_from_pool(ordered)
    selected: list[ResearchMove] = []
    selected_ids: set[str] = set()
    opening_front_predicate = lambda move: move.role in {'problem', 'background'} and (
        int(move.sequence_no) <= 4
        or max((_section_priority(anchor_sections.get(anchor_id, '')) for anchor_id in move.anchor_ids), default=0) >= 3
    )

    def pick(predicate: Any) -> None:
        candidates = [
            (score, section_score, sequence_no, move)
            for score, section_score, sequence_no, move in ordered
            if move.move_id not in selected_ids and predicate(move)
        ]
        if not candidates:
            return
        preferred_language = global_preferred_language or _preferred_summary_language(selected)
        if preferred_language:
            matching_candidates = [
                candidate
                for candidate in candidates
                if _summary_language(candidate[3].summary) == preferred_language
            ]
            if matching_candidates:
                candidates = matching_candidates
        candidates = sorted(
            candidates,
            key=lambda item: (
                -(item[0] + min(title_alignment_by_move_id.get(item[3].move_id, 0), 3) * 2),
                -title_alignment_by_move_id.get(item[3].move_id, 0),
                -item[1],
                item[2],
            ),
        )
        chosen = candidates[0]
        move = chosen[3]
        selected.append(move)
        selected_ids.add(move.move_id)

    if global_preferred_language:
        pick(lambda move: opening_front_predicate(move) and _summary_language(move.summary) == global_preferred_language)
    if not selected and global_preferred_language:
        pick(lambda move: move.role in {'problem', 'background'} and _summary_language(move.summary) == global_preferred_language)
    if not selected:
        pick(opening_front_predicate)
    if not selected:
        pick(
            lambda move: move.role in {'method', 'interpretation'}
            and (int(move.sequence_no) <= 4 or max((_section_priority(anchor_sections.get(anchor_id, '')) for anchor_id in move.anchor_ids), default=0) >= 3)
        )
    pick(lambda move: move.role in {'method', 'experiment'})
    pick(lambda move: move.role in {'result', 'interpretation'})
    pick(lambda move: move.role in {'limitation', 'future_work'})

    fallback_order = ordered
    if global_preferred_language:
        fallback_order = sorted(
            ordered,
            key=lambda item: (
                _summary_language(item[3].summary) != global_preferred_language,
                -title_alignment_by_move_id.get(item[3].move_id, 0),
                -item[0],
                -item[1],
                item[2],
            ),
        )

    for _score, _section_score, _sequence_no, move in fallback_order:
        if move.move_id in selected_ids:
            continue
        selected.append(move)
        selected_ids.add(move.move_id)
        if len(selected) >= 3:
            break
    return selected[:3]


def _summary_cue_count(summary: str, cues: tuple[str, ...]) -> int:
    lowered = f" {str(summary or '').strip().lower()} "
    return sum(1 for cue in cues if cue in lowered)


def _summary_has_current_work_cue(summary: str) -> bool:
    return _summary_cue_count(summary, _CONTENT_PROFILE_CURRENT_WORK_CUES) > 0


def _summary_has_prior_work_cue(summary: str) -> bool:
    return _summary_cue_count(summary, _CONTENT_PROFILE_PRIOR_WORK_CUES) > 0


def _content_profile_role_summary_score(
    move: ResearchMove,
    *,
    requested_roles: set[str],
    anchor_sections: dict[str, str],
    title_terms: set[str],
) -> tuple[int, int, int, int]:
    summary = str(move.summary or '').strip()
    base_score, section_score = _move_summary_score(move, anchor_sections)
    title_score = min(_title_alignment_score(summary, title_terms), 3)
    score = base_score + title_score * 2
    sequence_no = int(move.sequence_no)

    if requested_roles == {'method', 'experiment'}:
        if move.role == 'method':
            score += 8
        elif move.role == 'experiment':
            score += 5
        if _summary_has_current_work_cue(summary):
            score += 6
        if _summary_has_prior_work_cue(summary) and not _summary_has_current_work_cue(summary):
            score -= 14
        if move.act_type == 'adapt_method' and _summary_has_prior_work_cue(summary):
            score -= 4
    elif requested_roles == {'problem', 'background', 'hypothesis'}:
        if move.role == 'problem':
            score += 8
        elif move.role == 'hypothesis':
            score += 6
        elif move.role == 'background':
            score -= 4
            if _summary_cue_count(summary, _BACKGROUND_CONTEXT_CUES) > 0:
                score -= 4
    elif requested_roles == {'result'}:
        if move.role == 'result':
            score += 8
    elif requested_roles == {'interpretation'}:
        if move.role == 'interpretation':
            score += 6

    return (score, section_score, title_score, -sequence_no)


def _method_focus_score(move: ResearchMove) -> int:
    summary = str(move.summary or '').strip()
    if not _is_summary_contentful(summary):
        return -999
    score = 0
    if move.role == 'method':
        score += 4
    elif move.role == 'experiment':
        score += 2
    if move.act_type in {'propose_method', 'adapt_method', 'build_resource', 'run_experiment', 'set_condition'}:
        score += 2
    score += len(_trusted_mentions(move, 'methods', list(move.methods))) * 4
    if _trusted_mentions(move, 'resource_mentions', list(move.resource_mentions)):
        score += 1
    score += _summary_cue_count(summary, _METHOD_SUMMARY_POSITIVE_CUES) * 2
    score -= _summary_cue_count(summary, _METHOD_SUMMARY_NEGATIVE_CUES) * 3
    if _trusted_mentions(move, 'metrics', list(move.metrics)):
        score -= 1
    if _trusted_mentions(move, 'comparators', list(move.comparators)):
        score -= 1
    if move.effects:
        score -= 2
    return score


def _select_key_method_move(trace: PaperLogicTrace) -> ResearchMove | None:
    method_candidates = [
        move
        for move in trace.canonical_core.moves
        if move.role in {'method', 'experiment'} and _is_summary_contentful(move.summary)
    ]
    if not method_candidates:
        return None
    anchor_sections = _summary_anchor_sections(trace)
    return sorted(
        method_candidates,
        key=lambda move: (
            -(_method_focus_score(move) * 2 + _move_summary_score(move, anchor_sections)[0]),
            -_move_summary_score(move, anchor_sections)[0],
            -_method_focus_score(move),
            int(move.sequence_no),
        ),
    )[0]


def _slot_provenance_for(move: ResearchMove, field: str, *, value_index: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in move.slot_provenance:
        if item.field != field:
            continue
        if value_index is not None and item.value_index not in {None, value_index}:
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


def _trusted_mentions(move: ResearchMove, field: str, mentions: list[MentionValue]) -> list[MentionValue]:
    trusted: list[MentionValue] = []
    for index, mention in enumerate(mentions):
        if mention.inferred:
            continue
        provenance = _slot_provenance_for(move, field, value_index=index)
        if provenance:
            extraction_mode = str(provenance[0].get('extraction_mode') or '').strip().lower()
            support_strength = str(provenance[0].get('support_strength') or '').strip().lower()
            if extraction_mode not in _TRUSTED_EXTRACTION_MODES or support_strength not in _TRUSTED_SUPPORT_STRENGTHS:
                continue
        trusted.append(mention)
    return trusted


def _slot_entries(field: str, moves: list[ResearchMove], attr_name: str, *, trusted_only: bool = False) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for move in moves:
        mentions: list[MentionValue] = list(getattr(move, attr_name))
        if trusted_only:
            mentions = _trusted_mentions(move, field, mentions)
        for index, mention in enumerate(mentions):
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
                    'provenance': _slot_provenance_for(move, field, value_index=index if trusted_only else None),
                }
            )
    return entries


def _effect_entries(moves: list[ResearchMove]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for move in moves:
        for effect in move.effects:
            entries.append(
                {
                    'move_id': move.move_id,
                    'role': move.role,
                    'act_type': move.act_type,
                    'direction': effect.direction,
                    'magnitude_text': effect.magnitude_text,
                    'magnitude_numeric': effect.magnitude_numeric,
                    'unit': effect.unit,
                    'comparator_surface': effect.comparator_surface,
                    'anchor_ids': list(effect.anchor_ids),
                    'confidence': effect.confidence,
                }
            )
    return entries


def _effect_direction_entries(effect_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            'move_id': entry['move_id'],
            'role': entry['role'],
            'act_type': entry['act_type'],
            'direction': entry['direction'],
            'comparator_surface': entry['comparator_surface'],
            'anchor_ids': list(entry['anchor_ids']),
            'confidence': entry['confidence'],
        }
        for entry in effect_entries
        if str(entry.get('direction') or '').strip()
    ]


def _effect_size_entries(effect_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            'move_id': entry['move_id'],
            'role': entry['role'],
            'act_type': entry['act_type'],
            'magnitude_text': entry['magnitude_text'],
            'magnitude_numeric': entry['magnitude_numeric'],
            'unit': entry['unit'],
            'comparator_surface': entry['comparator_surface'],
            'anchor_ids': list(entry['anchor_ids']),
            'confidence': entry['confidence'],
        }
        for entry in effect_entries
        if (
            entry.get('magnitude_text') is not None
            or entry.get('magnitude_numeric') is not None
            or entry.get('unit') is not None
        )
    ]


def _slot_entries_with_move_context(
    field: str,
    moves: list[ResearchMove],
    attr_name: str,
    *,
    trusted_only: bool = False,
    roles: set[str] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for move in moves:
        if roles is not None and move.role not in roles:
            continue
        mentions: list[MentionValue] = list(getattr(move, attr_name))
        if trusted_only:
            mentions = _trusted_mentions(move, field, mentions)
        for index, mention in enumerate(mentions):
            token = _mention_token(mention)
            if not token:
                continue
            entries.append(
                {
                    'move_id': move.move_id,
                    'role': move.role,
                    'act_type': move.act_type,
                    'summary': move.summary,
                    'surface': mention.surface,
                    'normalized': mention.normalized,
                    'type': mention.type,
                    'anchor_ids': list(mention.anchor_ids),
                    'confidence': mention.confidence,
                    'provenance': _slot_provenance_for(move, field, value_index=index if trusted_only else None),
                }
            )
    return entries


def _primary_contributions(moves: list[ResearchMove]) -> list[str]:
    contributions: list[str] = []

    def _add(label: str) -> None:
        if label not in contributions:
            contributions.append(label)

    if any(move.role in {'problem', 'background', 'hypothesis'} for move in moves):
        _add('problem_framing')
    if any(_trusted_mentions(move, 'methods', list(move.methods)) for move in moves):
        _add('method_proposal')
    if any(
        move.role in {'experiment', 'result', 'interpretation'}
        and (
            _trusted_mentions(move, 'metrics', list(move.metrics))
            or move.effects
        )
        for move in moves
    ):
        _add('outcome_evidence')
    if any(
        move.role in {'experiment', 'result', 'interpretation'}
        and _trusted_mentions(move, 'comparators', list(move.comparators))
        for move in moves
    ):
        _add('comparison_evidence')
    if any(_trusted_mentions(move, 'limitation_types', list(move.limitation_types)) for move in moves):
        _add('limitation_evidence')
    if any(_trusted_mentions(move, 'resource_mentions', list(move.resource_mentions)) for move in moves):
        _add('resource_signal')
    if any(move.role == 'interpretation' for move in moves):
        _add('mechanistic_interpretation')
    if any(move.role == 'future_work' for move in moves):
        _add('future_direction')
    return contributions


def _outcome_signals(moves: list[ResearchMove]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for move in moves:
        if move.role not in {'experiment', 'result', 'interpretation'}:
            continue
        metrics = _trusted_mentions(move, 'metrics', list(move.metrics))
        comparators = _trusted_mentions(move, 'comparators', list(move.comparators))
        conditions = _trusted_mentions(move, 'conditions', list(move.conditions))
        effect_directions = _unique(effect.direction for effect in move.effects if str(effect.direction).strip())
        effect_comparators = _unique(
            str(effect.comparator_surface or '').strip().lower()
            for effect in move.effects
            if str(effect.comparator_surface or '').strip()
        )
        if not (metrics or comparators or move.effects):
            continue
        entries.append(
            {
                'move_id': move.move_id,
                'role': move.role,
                'act_type': move.act_type,
                'summary': move.summary,
                'metric_tokens': _mention_tokens(metrics),
                'comparator_tokens': _mention_tokens(comparators),
                'effect_directions': effect_directions,
                'effect_comparator_tokens': effect_comparators,
                'condition_tokens': _mention_tokens(conditions),
                'anchor_ids': list(move.anchor_ids),
                'confidence': move.confidence,
            }
        )
    return entries


def _comparison_signals(moves: list[ResearchMove]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for move in moves:
        if move.role not in {'experiment', 'result', 'interpretation'}:
            continue
        comparators = _trusted_mentions(move, 'comparators', list(move.comparators))
        if not comparators:
            continue
        conditions = _trusted_mentions(move, 'conditions', list(move.conditions))
        entries.append(
            {
                'move_id': move.move_id,
                'role': move.role,
                'act_type': move.act_type,
                'summary': move.summary,
                'comparator_tokens': _mention_tokens(comparators),
                'condition_tokens': _mention_tokens(conditions),
                'anchor_ids': list(move.anchor_ids),
                'confidence': move.confidence,
            }
        )
    return entries


def build_future_work_signals(moves: list[ResearchMove]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for move in moves:
        if move.role != 'future_work':
            continue
        target_objects = _trusted_mentions(move, 'research_objects', list(move.research_objects))
        methods = _trusted_mentions(move, 'methods', list(move.methods))
        conditions = _trusted_mentions(move, 'conditions', list(move.conditions))
        limitation_types = _trusted_mentions(move, 'limitation_types', list(move.limitation_types))
        resource_mentions = _trusted_mentions(move, 'resource_mentions', list(move.resource_mentions))
        if not (
            target_objects
            or methods
            or conditions
            or limitation_types
            or resource_mentions
            or str(move.summary or '').strip()
        ):
            continue
        entries.append(
            {
                'move_id': move.move_id,
                'role': move.role,
                'act_type': move.act_type,
                'summary': move.summary,
                'target_object_tokens': _mention_tokens(target_objects),
                'method_tokens': _mention_tokens(methods),
                'condition_tokens': _mention_tokens(conditions),
                'limitation_tokens': _mention_tokens(limitation_types),
                'resource_tokens': _mention_tokens(resource_mentions),
                'anchor_ids': list(move.anchor_ids),
                'confidence': move.confidence,
            }
        )
    return entries


def _filter_prior_work_method_signal_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not entries:
        return []
    current_or_neutral_entries = [
        entry
        for entry in entries
        if not _summary_has_prior_work_cue(str(entry.get('summary') or ''))
        or _summary_has_current_work_cue(str(entry.get('summary') or ''))
    ]
    current_or_neutral_labels = _unique(_entry_label(entry) for entry in current_or_neutral_entries if _entry_label(entry))
    if len(current_or_neutral_labels) < 2:
        return list(entries)
    return [
        entry
        for entry in entries
        if not (
            _summary_has_prior_work_cue(str(entry.get('summary') or ''))
            and not _summary_has_current_work_cue(str(entry.get('summary') or ''))
        )
    ]


def build_route_compiler_contract(
    *,
    paper_id: str,
    paper_type: str,
    moves: list[ResearchMove],
) -> dict[str, Any]:
    topic_object_roles = {'problem', 'method', 'experiment', 'result', 'interpretation'}
    method_roles = {'method', 'experiment'}
    comparison_roles = {'experiment', 'result', 'interpretation'}

    topic_objects = _slot_entries_with_move_context(
        'research_objects',
        moves,
        'research_objects',
        trusted_only=True,
        roles=topic_object_roles,
    )
    method_signals = _slot_entries_with_move_context(
        'methods',
        moves,
        'methods',
        trusted_only=True,
        roles=method_roles,
    )
    method_signals = _filter_prior_work_method_signal_entries(method_signals)
    condition_signals = _slot_entries_with_move_context(
        'conditions',
        moves,
        'conditions',
        trusted_only=True,
    )
    limitation_signals = _slot_entries_with_move_context(
        'limitation_types',
        moves,
        'limitation_types',
        trusted_only=True,
    )
    resource_signals = _slot_entries_with_move_context(
        'resource_mentions',
        moves,
        'resource_mentions',
        trusted_only=True,
    )
    outcome_entries = _outcome_signals(moves)
    comparison_entries = _comparison_signals(moves)
    future_work_entries = build_future_work_signals(moves)

    role_distribution: dict[str, int] = {}
    for move in moves:
        role_distribution[move.role] = role_distribution.get(move.role, 0) + 1

    return {
        'paper_id': paper_id,
        'paper_type': paper_type,
        'primary_contributions': _primary_contributions(moves),
        'topic_signals': {
            'objects': topic_objects,
            'methods': method_signals,
        },
        'outcome_signals': outcome_entries,
        'comparison_signals': comparison_entries,
        'future_direction_signals': future_work_entries,
        'constraint_signals': {
            'conditions': condition_signals,
            'limitations': limitation_signals,
            'resources': resource_signals,
        },
        'role_distribution': role_distribution,
        'signal_counts': {
            'topic_object_entries': len(topic_objects),
            'method_entries': len(method_signals),
            'outcome_entries': len(outcome_entries),
            'comparison_entries': len(comparison_entries),
            'future_work_entries': len(future_work_entries),
            'condition_entries': len(condition_signals),
            'limitation_entries': len(limitation_signals),
            'resource_entries': len(resource_signals),
        },
    }


def build_l2_5_slot_inventory(moves: list[ResearchMove]) -> dict[str, list[dict[str, Any]]]:
    research_objects = _slot_entries('research_objects', moves, 'research_objects')
    methods = _slot_entries('methods', moves, 'methods')
    observed_variables = _slot_entries('observed_variables', moves, 'observed_variables')
    metrics = _slot_entries('metrics', moves, 'metrics')
    conditions = _slot_entries('conditions', moves, 'conditions')
    comparators = _slot_entries('comparators', moves, 'comparators')
    limitation_types = _slot_entries('limitation_types', moves, 'limitation_types')
    resource_mentions = _slot_entries('resource_mentions', moves, 'resource_mentions')
    effects = _effect_entries(moves)

    return {
        'research_objects': research_objects,
        'research_object': list(research_objects),
        'methods': methods,
        'operation_or_method': list(methods),
        'observed_variables': observed_variables,
        'observed_variable': list(observed_variables),
        'metrics': metrics,
        'metric': list(metrics),
        'conditions': conditions,
        'condition_context': list(conditions),
        'comparators': comparators,
        'comparison_target': list(comparators),
        'effects': effects,
        'effect_direction': _effect_direction_entries(effects),
        'effect_size': _effect_size_entries(effects),
        'limitation_types': limitation_types,
        'limitation_type': list(limitation_types),
        'resource_mentions': resource_mentions,
    }


def build_l1_bridge_hints(moves: list[ResearchMove]) -> dict[str, list[dict[str, Any]]]:
    resource_candidates = _slot_entries('resource_mentions', moves, 'resource_mentions', trusted_only=True)
    metric_candidates = _slot_entries('metrics', moves, 'metrics', trusted_only=True)

    benchmark_candidates = [
        candidate
        for candidate in resource_candidates
        if str(candidate.get('type') or '').strip().lower() == 'benchmark'
    ]

    protocol_candidates: list[dict[str, Any]] = []
    toolchain_candidates: list[dict[str, Any]] = []
    for move in moves:
        methods = _trusted_mentions(move, 'methods', list(move.methods))
        metrics = _trusted_mentions(move, 'metrics', list(move.metrics))
        comparators = _trusted_mentions(move, 'comparators', list(move.comparators))
        conditions = _trusted_mentions(move, 'conditions', list(move.conditions))
        resources = _trusted_mentions(move, 'resource_mentions', list(move.resource_mentions))
        resource_tokens = _mention_tokens(resources)
        resource_types = _unique(str(mention.type or '').strip().lower() for mention in resources if str(mention.type or '').strip())
        method_tokens = _mention_tokens(methods)
        metric_tokens = _mention_tokens(metrics)
        comparator_tokens = _mention_tokens(comparators)
        condition_tokens = _mention_tokens(conditions)

        if metric_tokens and (comparator_tokens or condition_tokens or method_tokens or resource_tokens):
            protocol_candidates.append(
                {
                    'move_id': move.move_id,
                    'role': move.role,
                    'act_type': move.act_type,
                    'summary': move.summary,
                    'metric_tokens': metric_tokens,
                    'comparator_tokens': comparator_tokens,
                    'condition_tokens': condition_tokens,
                    'method_tokens': method_tokens,
                    'resource_tokens': resource_tokens,
                    'anchor_ids': list(move.anchor_ids),
                    'confidence': move.confidence,
                }
            )

        if resource_tokens and any(resource_type in _L1_TOOLCHAIN_RESOURCE_TYPES for resource_type in resource_types):
            toolchain_candidates.append(
                {
                    'move_id': move.move_id,
                    'role': move.role,
                    'act_type': move.act_type,
                    'summary': move.summary,
                    'resource_tokens': resource_tokens,
                    'resource_types': resource_types,
                    'method_tokens': method_tokens,
                    'anchor_ids': list(move.anchor_ids),
                    'confidence': move.confidence,
                }
            )

    return {
        'resource_candidates': resource_candidates,
        'benchmark_candidates': benchmark_candidates,
        'metric_candidates': metric_candidates,
        'protocol_candidates': protocol_candidates,
        'toolchain_candidates': toolchain_candidates,
    }


def _entry_label(entry: dict[str, Any]) -> str:
    return str(entry.get('normalized') or entry.get('surface') or '').strip().lower()


def _entry_labels(entries: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    labels = [_entry_label(entry) for entry in entries if _entry_label(entry)]
    return _unique(labels)[:limit]


def _rank_topic_object_entries(entries: list[dict[str, Any]], method_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_counts: dict[str, int] = {}
    label_order: dict[str, int] = {}
    for index, entry in enumerate(entries):
        label = _entry_label(entry)
        if not label:
            continue
        label_counts[label] = label_counts.get(label, 0) + 1
        label_order.setdefault(label, index)

    method_labels = {_entry_label(entry) for entry in method_entries if _entry_label(entry)}

    def _score(entry: dict[str, Any]) -> tuple[int, int, int]:
        label = _entry_label(entry)
        tokens = _WORD_RE.findall(label)
        role = str(entry.get('role') or '').strip().lower()
        semantic_score = _TOPIC_ROLE_PRIORITY.get(role, 1) * 10
        semantic_score += min(len(tokens), 4) * 2
        if len(tokens) <= 1:
            semantic_score -= 3
        if label in method_labels:
            semantic_score -= 8
        return (
            label_counts.get(label, 0),
            semantic_score,
            -label_order.get(label, 0),
        )

    return sorted(entries, key=_score, reverse=True)


def _rank_method_signal_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_counts: dict[str, int] = {}
    label_order: dict[str, int] = {}
    for index, entry in enumerate(entries):
        label = _entry_label(entry)
        if not label:
            continue
        label_counts[label] = label_counts.get(label, 0) + 1
        label_order.setdefault(label, index)

    act_priority = {
        'propose_method': 6,
        'run_experiment': 5,
        'adapt_method': 3,
        'build_resource': 2,
        'set_condition': 1,
    }

    def _score(entry: dict[str, Any]) -> tuple[int, int, int, int]:
        label = _entry_label(entry)
        summary = str(entry.get('summary') or '')
        summary_lower = summary.lower()
        tokens = _WORD_RE.findall(label)
        score = act_priority.get(str(entry.get('act_type') or '').strip().lower(), 0) * 5
        score += min(len(tokens), 4)
        if len(tokens) <= 1:
            score -= 2
        if _summary_has_current_work_cue(summary):
            score += 10
        if _summary_has_prior_work_cue(summary) and not _summary_has_current_work_cue(summary):
            score -= 12
        score += _summary_cue_count(summary, _METHOD_SIGNAL_OPERATIVE_SUMMARY_CUES) * 3
        score -= _summary_cue_count(summary, _METHOD_SIGNAL_BROAD_SUMMARY_CUES) * 4
        if any(cue in label for cue in _METHOD_SIGNAL_OPERATIVE_LABEL_CUES):
            score += 4
        if any(cue in label for cue in _METHOD_SIGNAL_GENERIC_LABEL_CUES):
            score -= 6
        if label.endswith('method') and 'improved' not in label and _summary_has_prior_work_cue(summary):
            score -= 4
        if 'framework' in label and 'propose' not in summary_lower and 'proposes' not in summary_lower:
            score -= 2
        return (
            score,
            label_counts.get(label, 0),
            -label_order.get(label, 0),
            len(tokens),
        )

    return sorted(entries, key=_score, reverse=True)


def _entry_anchor_ids(entries: list[dict[str, Any]], *, limit: int | None = None) -> list[str]:
    flattened: list[str] = []
    for entry in entries:
        flattened.extend(str(anchor_id).strip() for anchor_id in (entry.get('anchor_ids') or []) if str(anchor_id).strip())
    unique_ids = _unique(flattened)
    if limit is None:
        return unique_ids
    return unique_ids[:limit]


def _nested_token_values(rows: list[dict[str, Any]], field: str, *, limit: int = 6) -> list[str]:
    flattened: list[str] = []
    for row in rows:
        flattened.extend(str(token).strip().lower() for token in (row.get(field) or []) if str(token).strip())
    return _unique(flattened)[:limit]


def _future_direction_labels(rows: list[dict[str, Any]], *, limit: int = 6) -> list[str]:
    labels: list[str] = []
    for row in rows:
        methods = [str(token).strip().lower() for token in (row.get('method_tokens') or []) if str(token).strip()]
        targets = [str(token).strip().lower() for token in (row.get('target_object_tokens') or []) if str(token).strip()]
        resources = [str(token).strip().lower() for token in (row.get('resource_tokens') or []) if str(token).strip()]
        conditions = [str(token).strip().lower() for token in (row.get('condition_tokens') or []) if str(token).strip()]
        label = ''
        if methods and targets:
            label = f'{methods[0]} -> {targets[0]}'
        elif targets and resources:
            label = f'{targets[0]} via {resources[0]}'
        elif methods and resources:
            label = f'{methods[0]} with {resources[0]}'
        elif methods:
            label = methods[0]
        elif targets:
            label = targets[0]
        elif resources:
            label = resources[0]
        elif conditions:
            label = conditions[0]
        else:
            label = str(row.get('summary') or '').strip().lower()
        if label:
            labels.append(label)
    return _unique(labels)[:limit]


def build_route_state_seed(
    trace: PaperLogicTrace,
    *,
    route_compiler_contract: dict[str, Any] | None = None,
    l1_bridge_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    moves = trace.canonical_core.moves
    route_compiler_contract = dict(
        route_compiler_contract
        or build_route_compiler_contract(
            paper_id=trace.paper_metadata.paper_id,
            paper_type=trace.paper_metadata.paper_type,
            moves=moves,
        )
    )
    l1_bridge_hints = dict(l1_bridge_hints or build_l1_bridge_hints(moves))

    topic_object_entries = list((route_compiler_contract.get('topic_signals') or {}).get('objects') or [])
    method_entries = list((route_compiler_contract.get('topic_signals') or {}).get('methods') or [])
    limitation_entries = list((route_compiler_contract.get('constraint_signals') or {}).get('limitations') or [])
    condition_entries = list((route_compiler_contract.get('constraint_signals') or {}).get('conditions') or [])
    resource_entries = list((route_compiler_contract.get('constraint_signals') or {}).get('resources') or [])
    outcome_entries = list(route_compiler_contract.get('outcome_signals') or [])
    comparison_entries = list(route_compiler_contract.get('comparison_signals') or [])
    future_direction_entries = list(route_compiler_contract.get('future_direction_signals') or [])
    benchmark_entries = list(l1_bridge_hints.get('benchmark_candidates') or [])
    protocol_candidates = list(l1_bridge_hints.get('protocol_candidates') or [])
    toolchain_candidates = list(l1_bridge_hints.get('toolchain_candidates') or [])

    if not topic_object_entries:
        topic_object_entries = _slot_entries_with_move_context(
            'research_objects',
            moves,
            'research_objects',
            trusted_only=False,
            roles={'problem', 'method', 'experiment', 'result', 'interpretation'},
        )
    topic_object_entries = _rank_topic_object_entries(topic_object_entries, method_entries)
    ranked_method_entries = _rank_method_signal_entries(method_entries)

    supporting_evidence_ids = _unique(
        [
            *_entry_anchor_ids(topic_object_entries),
            *_entry_anchor_ids(ranked_method_entries),
            *_entry_anchor_ids(resource_entries),
            *_entry_anchor_ids(outcome_entries),
            *_entry_anchor_ids(comparison_entries),
            *_entry_anchor_ids(protocol_candidates),
            *_entry_anchor_ids(toolchain_candidates),
        ]
    )
    challenging_evidence_ids = _unique(
        [
            *_entry_anchor_ids(limitation_entries),
        ]
    )

    return {
        'paper_id': trace.paper_metadata.paper_id,
        'paper_type': trace.paper_metadata.paper_type,
        'source_trace_id': trace.trace_id,
        'cutoff_year_hint': trace.paper_metadata.year,
        'topic_scope_candidates': _entry_labels(topic_object_entries),
        'dominant_method_candidates': _entry_labels(ranked_method_entries),
        'active_benchmark_candidates': _entry_labels(benchmark_entries),
        'known_bottleneck_candidates': _entry_labels(limitation_entries),
        'enabling_condition_candidates': _entry_labels(condition_entries),
        'alternative_route_candidates': _future_direction_labels(future_direction_entries),
        'measurement_protocol_candidates': protocol_candidates,
        'toolchain_candidates': toolchain_candidates,
        'supporting_evidence_ids': supporting_evidence_ids,
        'challenging_evidence_ids': challenging_evidence_ids,
        'source_move_ids': [move.move_id for move in moves],
        'readiness_feature_inputs': {
            'method_maturity_signals': _entry_labels(ranked_method_entries),
            'measurement_maturity_signals': _nested_token_values(protocol_candidates, 'metric_tokens'),
            'data_resource_signals': _entry_labels(benchmark_entries),
            'infrastructure_signals': _nested_token_values(toolchain_candidates, 'resource_tokens'),
            'bottleneck_signals': _entry_labels(limitation_entries),
        },
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
        methods = _trusted_mentions(move, 'methods', list(move.methods))
        metrics = _trusted_mentions(move, 'metrics', list(move.metrics))
        comparators = _trusted_mentions(move, 'comparators', list(move.comparators))
        conditions = _trusted_mentions(move, 'conditions', list(move.conditions))
        limitation_types = _trusted_mentions(move, 'limitation_types', list(move.limitation_types))
        resource_mentions = _trusted_mentions(move, 'resource_mentions', list(move.resource_mentions))

        if methods:
            candidates.append(
                {
                    'move_id': move.move_id,
                    'candidate_type': 'method_candidate',
                    'tokens': _mention_tokens(methods),
                    'anchor_ids': list(move.anchor_ids),
                }
            )
        if metrics:
            candidates.append(
                {
                    'move_id': move.move_id,
                    'candidate_type': 'metric_candidate',
                    'tokens': _mention_tokens(metrics),
                    'anchor_ids': list(move.anchor_ids),
                }
            )
        if comparators:
            candidates.append(
                {
                    'move_id': move.move_id,
                    'candidate_type': 'comparison_candidate',
                    'tokens': _mention_tokens(comparators),
                    'anchor_ids': list(move.anchor_ids),
                }
            )
        if conditions:
            candidates.append(
                {
                    'move_id': move.move_id,
                    'candidate_type': 'condition_candidate',
                    'tokens': _mention_tokens(conditions),
                    'anchor_ids': list(move.anchor_ids),
                }
            )
        benchmark_tokens = [
            _mention_token(mention)
            for mention in resource_mentions
            if str(mention.type or '').strip().lower() == 'benchmark' and _mention_token(mention)
        ]
        if benchmark_tokens:
            candidates.append(
                {
                    'move_id': move.move_id,
                    'candidate_type': 'benchmark_candidate',
                    'tokens': _unique(benchmark_tokens),
                    'anchor_ids': list(move.anchor_ids),
                }
            )
        if limitation_types:
            candidates.append(
                {
                    'move_id': move.move_id,
                    'candidate_type': 'limitation_candidate',
                    'tokens': _mention_tokens(limitation_types),
                    'anchor_ids': list(move.anchor_ids),
                }
            )
    return candidates


def build_paper_summaries(trace: PaperLogicTrace) -> dict[str, Any]:
    role_distribution: dict[str, int] = {}
    for move in trace.canonical_core.moves:
        role_distribution[move.role] = role_distribution.get(move.role, 0) + 1

    summary_moves = _select_summary_moves(trace)
    move_summaries = [move.summary.strip() for move in summary_moves if move.summary.strip()]
    one_paragraph_summary = ' '.join(move_summaries[:3]).strip()
    method_move = _select_key_method_move(trace)
    key_method_summary = str(method_move.summary or '').strip() if method_move else ''

    return {
        'move_role_distribution': role_distribution,
        'one_paragraph_summary': one_paragraph_summary,
        'key_method_summary': key_method_summary,
    }


def build_paper_content_profile(trace: PaperLogicTrace) -> dict[str, Any]:
    moves = list(trace.canonical_core.moves)
    paper_summaries = build_paper_summaries(trace)
    anchor_sections = _summary_anchor_sections(trace)
    title_terms = _title_alignment_terms(trace.paper_metadata.title, trace.paper_metadata.title_alt)

    def _role_summaries(roles: set[str], *, limit: int = 3) -> list[str]:
        summaries: list[str] = []
        eligible_moves = [move for move in moves if move.role in roles]
        ordered_moves = sorted(
            eligible_moves,
            key=lambda move: _content_profile_role_summary_score(
                move,
                requested_roles=roles,
                anchor_sections=anchor_sections,
                title_terms=title_terms,
            ),
            reverse=True,
        )
        skip_prior_work_method_ids: set[str] = set()
        if roles == {'method', 'experiment'}:
            contentful_current_or_neutral = [
                move
                for move in ordered_moves
                if _is_summary_contentful(move.summary) and not _summary_has_prior_work_cue(move.summary)
            ]
            if len(contentful_current_or_neutral) >= max(2, limit - 1):
                skip_prior_work_method_ids = {
                    move.move_id
                    for move in ordered_moves
                    if _summary_has_prior_work_cue(move.summary)
                }
        for move in ordered_moves:
            if move.role not in roles:
                continue
            if move.move_id in skip_prior_work_method_ids:
                continue
            summary = str(move.summary or '').strip()
            if not _is_summary_contentful(summary) or summary in summaries:
                continue
            summaries.append(summary)
            if len(summaries) >= limit:
                break
        return summaries

    def _finding_summaries(*, limit: int = 3) -> list[str]:
        summaries = _role_summaries({'result'}, limit=limit)
        if len(summaries) >= limit:
            return summaries
        for summary in _role_summaries({'interpretation'}, limit=limit):
            if summary in summaries:
                continue
            summaries.append(summary)
            if len(summaries) >= limit:
                break
        return summaries

    limitation_statements: list[dict[str, Any]] = []
    for move in sorted(moves, key=lambda item: int(item.sequence_no)):
        if move.role != 'limitation':
            continue
        summary = str(move.summary or '').strip()
        if not _is_summary_contentful(summary):
            continue
        limitation_tokens = _mention_tokens(_trusted_mentions(move, 'limitation_types', list(move.limitation_types)))
        limitation_statements.append(
            {
                'move_id': move.move_id,
                'summary': summary,
                'limitation_tokens': limitation_tokens,
                'anchor_ids': list(move.anchor_ids),
                'confidence': move.confidence,
            }
        )

    citation_contexts = [
        {
            'citation_act_id': citation.citation_act_id,
            'source_move_id': citation.source_move_id,
            'target_paper_id': citation.target_paper_id,
            'purpose': citation.purpose,
            'polarity': citation.polarity,
            'semantic_signal': citation.semantic_signal,
            'target_scope': citation.target_scope,
            'anchor_ids': list(citation.anchor_ids),
            'confidence': citation.confidence,
        }
        for citation in trace.canonical_core.citation_acts
    ]
    figure_refs = [
        {
            'figure_id': figure.figure_id,
            'caption': figure.caption,
            'anchor_ids': list(figure.anchor_ids),
        }
        for figure in trace.canonical_core.figure_refs
    ]
    table_refs = [
        {
            'table_id': table.table_id,
            'caption': table.caption,
            'anchor_ids': list(table.anchor_ids),
        }
        for table in trace.canonical_core.table_refs
    ]
    future_work_statements = build_future_work_signals(moves)
    finding_summaries = _finding_summaries()

    return {
        'paper_id': trace.paper_metadata.paper_id,
        'paper_type': trace.paper_metadata.paper_type,
        'one_paragraph_summary': paper_summaries.get('one_paragraph_summary', ''),
        'key_method_summary': paper_summaries.get('key_method_summary', ''),
        'problem_statements': _role_summaries({'problem', 'background', 'hypothesis'}),
        'method_statements': _role_summaries({'method', 'experiment'}),
        'key_findings': finding_summaries,
        'limitation_statements': limitation_statements,
        'future_work_statements': future_work_statements,
        'citation_contexts': citation_contexts,
        'figure_refs': figure_refs,
        'table_refs': table_refs,
        'coverage': {
            'problem_statement_count': len(_role_summaries({'problem', 'background', 'hypothesis'})),
            'method_statement_count': len(_role_summaries({'method', 'experiment'})),
            'finding_count': len(finding_summaries),
            'limitation_count': len(limitation_statements),
            'future_work_count': len(future_work_statements),
            'citation_context_count': len(citation_contexts),
            'figure_count': len(figure_refs),
            'table_count': len(table_refs),
        },
    }


def build_paper_content_audit(
    trace: PaperLogicTrace,
    *,
    paper_content_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = dict(paper_content_profile or build_paper_content_profile(trace))
    coverage = dict(profile.get('coverage') or {})
    key_method_move = _select_key_method_move(trace)
    key_method_summary = str(profile.get('key_method_summary') or '').strip()
    key_method_score = _method_focus_score(key_method_move) if key_method_move else None
    method_candidates = [
        move
        for move in trace.canonical_core.moves
        if move.role in {'method', 'experiment'} and _is_summary_contentful(move.summary)
    ]
    best_method_score = max((_method_focus_score(move) for move in method_candidates), default=None)

    flags: list[str] = []
    suspicious_title_alt = bool(trace.paper_metadata.title_alt) and _looks_like_section_heading(trace.paper_metadata.title_alt)
    if suspicious_title_alt:
        flags.append('suspicious_title_alt')
    if int(coverage.get('problem_statement_count') or 0) == 0:
        flags.append('missing_problem_statements')
    if int(coverage.get('method_statement_count') or 0) == 0:
        flags.append('missing_method_statements')
    if int(coverage.get('finding_count') or 0) == 0:
        flags.append('missing_key_findings')
    if not key_method_summary:
        flags.append('missing_key_method_summary')
    elif (
        key_method_score is not None
        and best_method_score is not None
        and best_method_score - key_method_score >= 3
        and _summary_cue_count(key_method_summary, _METHOD_SUMMARY_NEGATIVE_CUES) > 0
    ):
        flags.append('method_summary_drift')

    return {
        'available': True,
        'flags': flags,
        'suspicious_title_alt': suspicious_title_alt,
        'title_alt_heading_like': _looks_like_section_heading(trace.paper_metadata.title_alt),
        'key_method_summary_outcome_heavy': _summary_cue_count(key_method_summary, _METHOD_SUMMARY_NEGATIVE_CUES) > 0,
        'coverage': coverage,
    }


def build_derived_views(trace: PaperLogicTrace) -> dict[str, Any]:
    moves = trace.canonical_core.moves
    l2_5_slot_inventory = build_l2_5_slot_inventory(moves)
    l1_bridge_hints = build_l1_bridge_hints(moves)
    future_work_signals = build_future_work_signals(moves)
    paper_content_profile = build_paper_content_profile(trace)
    route_compiler_contract = build_route_compiler_contract(
        paper_id=trace.paper_metadata.paper_id,
        paper_type=trace.paper_metadata.paper_type,
        moves=moves,
    )
    return {
        'l2_5_slot_inventory': l2_5_slot_inventory,
        'l1_bridge_hints': l1_bridge_hints,
        'community_signatures': build_community_signatures(trace.paper_metadata.paper_id, moves),
        'route_feature_candidates': build_route_feature_candidates(moves),
        'future_work_signals': future_work_signals,
        'route_compiler_contract': route_compiler_contract,
        'route_state_seed': build_route_state_seed(
            trace,
            route_compiler_contract=route_compiler_contract,
            l1_bridge_hints=l1_bridge_hints,
        ),
        'paper_summaries': build_paper_summaries(trace),
        'paper_content_profile': paper_content_profile,
        'paper_content_audit': build_paper_content_audit(
            trace,
            paper_content_profile=paper_content_profile,
        ),
    }
