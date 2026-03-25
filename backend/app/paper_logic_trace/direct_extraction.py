from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any, Callable

from app.ingest.models import Chunk, DocumentIR
from app.paper_logic_trace.models import normalize_trace_paper_type


logger = logging.getLogger(__name__)


MoveExtractorFn = Callable[..., dict[str, Any]]

_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\\-]+|\\d+|[\\u4e00-\\u9fff]+")
_STOP_TOKENS = {
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'by',
    'for',
    'from',
    'in',
    'into',
    'is',
    'of',
    'on',
    'or',
    'our',
    'paper',
    'that',
    'the',
    'their',
    'this',
    'to',
    'using',
    'we',
    'with',
}
_REFERENCE_TOKENS = {'reference', 'references', 'bibliography', 'acknowledgment', 'acknowledgement', 'appendix'}
_NOISE_SECTION_TOKENS = {
    'title',
    'authors',
    'author information',
    'highlights',
    'article info',
    'articleinfo',
    'article history',
}
_IMAGE_ONLY_RE = re.compile(r'^\s*!\[[^\]]*\]\([^)]+\)\s*$')
_MARKDOWN_HEADING_RE = re.compile(r'^\s*#{1,6}\s*(.+?)\s*$')
_FRONT_MATTER_INSTITUTION_CUES = (
    'university',
    'universite',
    'université',
    'department',
    'school of',
    'institute',
    'laboratory',
    'faculty of',
    'centre for',
    'center for',
    'college of',
    'cnrs',
    'cedex',
)
_FRONT_MATTER_METADATA_CUES = (
    'available online',
    'article history',
    'received ',
    'accepted ',
    'in revised form',
    'keywords:',
    'keyword:',
    'corresponding author',
)
_FRONT_MATTER_METADATA_RE = re.compile(
    r'^\s*(?:received|accepted|published online|available online|keywords?)\b[:\s-]*',
    re.IGNORECASE,
)
_NOISE_SUMMARY_PREFIXES = (
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
)
_PROBLEM_TEXT_PATTERNS = (
    'this paper investigates',
    'this paper studies',
    'this paper addresses',
    'this paper outlines',
    'we investigate',
    'we study',
    'we address',
    'the aim of this paper',
    'the objective of this paper',
    'the goal of this paper',
    'the work presented here',
    'in this study',
    'in this work',
    'in this paper, we present investigations',
    'in this paper we present investigations',
    'challenge',
    'research gap',
)
_PROBLEM_SUBJECT_HINTS = (
    'this paper',
    'this study',
    'this work',
    'in this paper',
    'in this study',
    'in this work',
    'the work presented here',
    'we present',
)
_PROBLEM_PURPOSE_HINTS = (
    'aims to',
    'aimed to',
    'goal is to',
    'objective is to',
    'in order to investigate',
    'in order to study',
    'to investigate',
    'to study',
    'to examine',
    'to quantify',
)
_RESULT_TEXT_PATTERNS = (
    'results show',
    'results indicate',
    'we show that',
    'we find that',
    'we found that',
    'our results show',
    'demonstrate that',
    'demonstrates that',
    'reveals that',
    'revealed that',
    'led to',
    'improved',
    'decreased',
    'increased',
)
_CONDITION_CUE_WORDS = ('under', 'with', 'at', 'during')
_METRIC_CUE_WORDS = (
    'accuracy',
    'degree',
    'deviation',
    'deviations',
    'dissolution',
    'error',
    'fraction',
    'hardness',
    'index',
    'rate',
    'ratio',
    'uniformity',
    'wear',
)
_OBSERVED_VARIABLE_CUE_WORDS = (
    'density',
    'densities',
    'displacement',
    'displacements',
    'force',
    'forces',
    'pressure',
    'pressures',
    'rotation',
    'rotations',
    'strain',
    'stresses',
    'stress',
    'strength',
    'strengths',
    'velocity',
    'velocities',
)
_OBSERVED_VARIABLE_BAD_PREFIXES = (
    'accuracy of',
    'how far the',
    'indicated by',
    'insight into',
    'precision of',
    'results show',
    'simulation of',
    'terms of',
    'they exhibit',
)
_OBSERVED_VARIABLE_BAD_TOKENS = {
    'decrease',
    'decreased',
    'exhibit',
    'exhibiting',
    'increase',
    'increased',
    'indicated',
    'show',
    'shows',
}
_STRONG_METRIC_TOKENS = {
    'accuracy',
    'degree',
    'deviation',
    'deviations',
    'dissolution',
    'error',
    'fraction',
    'hardness',
    'index',
    'rate',
    'ratio',
    'uniformity',
    'wear',
}
_METRIC_BAD_PREFIXES = (
    'details on',
    'detail on',
    'insight into',
    'insights into',
    'measurement of',
    'until the',
)
_METRIC_BAD_TOKENS = {
    'investigate',
    'investigation',
    'occurring',
    'occur',
    'exhibit',
    'exhibits',
    'exhibiting',
    'reveals',
    'reveal',
}
_RESOURCE_BACKTRACK_STOP_TOKENS = {
    *_STOP_TOKENS,
    *_METRIC_CUE_WORDS,
    'density',
    'densities',
    'pressure',
    'pressures',
    'strain',
    'stresses',
    'stress',
    'measurement',
    'measurements',
    'measured',
    'observed',
    'observation',
    'observations',
    'quantification',
    'quantified',
    'quantify',
}
_RESOURCE_CUE_WORDS = (
    'camera',
    'cell',
    'drum',
    'framework',
    'imaging',
    'machine',
    'microscope',
    'microscopy',
    'microtomography',
    'mixer',
    'platform',
    'pump',
    'scanner',
    'software',
    'tester',
    'tomography',
)
_RESOURCE_BAD_TYPES = {
    'application',
    'citation',
    'reference',
    'theory',
}
_RESOURCE_KEEP_TYPES = {
    'dataset',
    'instrument',
    'platform',
    'protocol',
    'software',
    'tool',
}
_GENERIC_LIMITATION_PHRASES = {
    'assumption',
    'methodological',
    'methodological_assumption',
    'model_assumption',
    'technique_limitation',
    'data_processing_error',
    'validity_boundary',
}
_LIMITATION_CUE_WORDS = (
    'cost',
    'costly',
    'difficult',
    'difficulty',
    'expensive',
    'limitation',
    'limitations',
    'memory',
)
_COMPARATOR_ENTITY_HINTS = {
    'baseline',
    'sample',
    'samples',
    'experiment',
    'experiments',
    'simulation',
    'simulations',
    'measurement',
    'measurements',
    'method',
    'methods',
    'region',
    'regions',
    'condition',
    'conditions',
    'case',
    'cases',
    'result',
    'results',
    'estimate',
    'estimates',
    'contact',
    'contacts',
    'group',
    'groups',
    'mixture',
    'mixtures',
    'model',
    'models',
}
_COMPARATOR_BAD_TOKENS = {
    'author',
    'authors',
    'experienced',
    'appears',
    'appeared',
    'showing',
    'showed',
    'show',
    'shows',
    'predicted',
    'observed',
}
_COMPARATOR_BAD_LEADS = {
    'applied',
    'different',
    'higher',
    'lower',
    'greater',
    'smaller',
    'larger',
}
_REPORTING_VERB_CUES = (
    'investigate',
    'investigates',
    'study',
    'studies',
    'propose',
    'proposes',
    'show',
    'shows',
    'find',
    'finds',
    'indicate',
    'indicates',
    'demonstrate',
    'demonstrates',
    'compare',
    'compares',
    'analyze',
    'analyzes',
    'analyse',
    'analyses',
    'develop',
    'develops',
    'evaluate',
    'evaluates',
    'measure',
    'measures',
)
_SECTION_ROLE_HINTS: list[tuple[tuple[str, ...], str]] = [
    (('future work', 'future directions', 'future'), 'future_work'),
    (('limitation', 'limitations', 'threats to validity'), 'limitation'),
    (('result', 'results', 'finding', 'findings'), 'result'),
    (('discussion', 'interpretation', 'analysis'), 'interpretation'),
    (('experiment', 'evaluation', 'experimental', 'benchmark', 'ablation'), 'experiment'),
    (('method', 'approach', 'framework', 'model', 'algorithm', 'implementation'), 'method'),
    (('problem', 'motivation', 'task', 'challenge', 'gap'), 'problem'),
    (('background', 'introduction', 'preliminar', 'related work'), 'background'),
]
_ROLE_TO_ACT: dict[str, str] = {
    'problem': 'define_task',
    'background': 'identify_gap',
    'hypothesis': 'formulate_hypothesis',
    'method': 'propose_method',
    'experiment': 'run_experiment',
    'result': 'report_effect',
    'interpretation': 'explain_mechanism',
    'limitation': 'state_limitation',
    'future_work': 'suggest_extension',
}
_ACT_TO_ROLE: dict[str, str] = {
    'identify_gap': 'problem',
    'define_task': 'problem',
    'formulate_hypothesis': 'hypothesis',
    'propose_method': 'method',
    'adapt_method': 'method',
    'run_experiment': 'experiment',
    'measure_outcome': 'experiment',
    'compare_baseline': 'experiment',
    'report_effect': 'result',
    'explain_mechanism': 'interpretation',
    'diagnose_failure': 'limitation',
    'state_limitation': 'limitation',
    'suggest_extension': 'future_work',
}
_ALLOWED_ROLES = tuple(_ROLE_TO_ACT.keys())
_ALLOWED_ACTS = (
    'identify_gap',
    'define_task',
    'formulate_hypothesis',
    'propose_method',
    'adapt_method',
    'build_resource',
    'set_condition',
    'run_experiment',
    'measure_outcome',
    'compare_baseline',
    'report_effect',
    'explain_mechanism',
    'diagnose_failure',
    'state_limitation',
    'suggest_extension',
)


def _normalize_space(value: object) -> str:
    return _SPACE_RE.sub(' ', str(value or '').strip())


def _normalize_role(value: object) -> str:
    token = _normalize_space(value).lower().replace(' ', '_')
    return token if token in _ALLOWED_ROLES else 'background'


def _normalize_act_type(value: object, *, role: str) -> str:
    token = _normalize_space(value).lower().replace(' ', '_')
    return token if token in _ALLOWED_ACTS else _ROLE_TO_ACT.get(role, 'define_task')


def _is_reference_section(section: object) -> bool:
    label = _normalize_space(section).lower()
    return any(token in label for token in _REFERENCE_TOKENS)


def _looks_like_author_line(text: str) -> bool:
    clean = _normalize_space(text)
    if not clean or len(clean) > 120:
        return False
    lowered = clean.lower()
    if any(marker in lowered for marker in ('paper', 'method', 'result', 'experiment', 'introduction')):
        return False
    letters = [ch for ch in clean if ch.isalpha()]
    if len(letters) < 6:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / max(1, len(letters))
    if upper_ratio >= 0.6:
        return True

    normalized_delimiters = re.sub(r'[\u00b7\u2022•|/]+', ',', clean)
    parts = [part.strip() for part in re.split(r',|\band\b', normalized_delimiters, flags=re.IGNORECASE) if part.strip()]
    if len(parts) < 2:
        return False

    name_like_parts = 0
    for part in parts:
        normalized = re.sub(r'[\*\d]+$', '', part).strip()
        normalized = re.sub(r'\b[a-z]\b', '', normalized).strip()
        tokens = [token for token in normalized.split() if token]
        if not 1 <= len(tokens) <= 4:
            continue
        if all(token[0].isupper() for token in tokens if token[0].isalpha()):
            name_like_parts += 1
    return name_like_parts >= 2


def _looks_like_affiliation_summary(text: str) -> bool:
    clean = _normalize_space(text)
    if not clean:
        return False
    lowered = clean.lower()
    has_affiliation_cue = any(cue in lowered for cue in _FRONT_MATTER_INSTITUTION_CUES)
    has_email = '@' in lowered or ' e-mail ' in f' {lowered} ' or ' email ' in f' {lowered} '
    if not has_affiliation_cue and not has_email:
        return False
    has_reporting_verb = any(verb in lowered for verb in _REPORTING_VERB_CUES)
    comma_count = clean.count(',')
    digit_count = sum(1 for ch in clean if ch.isdigit())
    if has_email and not has_reporting_verb:
        return True
    if has_affiliation_cue and comma_count >= 2 and digit_count >= 2 and not has_reporting_verb:
        return True
    return False


def _looks_like_front_matter_noise(text: str, *, section: str, paper_title: str) -> bool:
    lowered = text.lower()
    in_title_block = bool(section) and bool(paper_title) and section == paper_title
    if in_title_block and (_FRONT_MATTER_METADATA_RE.match(lowered) or any(cue in lowered for cue in _FRONT_MATTER_METADATA_CUES)):
        return True
    if in_title_block and any(cue in lowered for cue in _FRONT_MATTER_INSTITUTION_CUES):
        return True
    if in_title_block and _looks_like_author_line(text):
        return True
    return False


def _looks_like_heading_only(text: str, *, section: str) -> bool:
    match = _MARKDOWN_HEADING_RE.match(str(text or ''))
    if not match:
        return False
    heading = _normalize_space(match.group(1))
    if not heading:
        return True
    normalized_heading = re.sub(r'^(?:\d+(?:\.\d+)*)\s*', '', heading).strip(' .:-').lower()
    normalized_section = re.sub(r'^(?:\d+(?:\.\d+)*)\s*', '', _normalize_space(section)).strip(' .:-').lower()
    if normalized_section and (
        normalized_heading == normalized_section
        or normalized_heading in normalized_section
        or normalized_section in normalized_heading
    ):
        return True
    heading_words = normalized_heading.split()
    verb_markers = {
        'is',
        'are',
        'was',
        'were',
        'investigates',
        'investigate',
        'proposes',
        'propose',
        'shows',
        'show',
        'demonstrates',
        'demonstrate',
        'improves',
        'improve',
        'decreases',
        'decrease',
        'increases',
        'increase',
    }
    if len(heading_words) <= 8 and not any(word in verb_markers for word in heading_words):
        return True
    return False


def _looks_like_noise_summary(text: object) -> bool:
    clean = _normalize_space(text)
    if not clean:
        return True
    lowered = clean.lower()
    if _looks_like_heading_only(clean, section=''):
        return True
    if _looks_like_author_line(clean):
        return True
    if _looks_like_affiliation_summary(clean):
        return True
    if any(lowered.startswith(prefix) for prefix in _NOISE_SUMMARY_PREFIXES):
        return True
    if lowered.startswith('received ') or lowered.startswith('accepted ') or lowered.startswith('copyright '):
        return True
    if 'article history' in lowered and ('received ' in lowered or 'accepted ' in lowered):
        return True
    return False


def _is_noise_chunk(chunk: Chunk, *, paper_title: object) -> bool:
    section = _normalize_space(chunk.section).lower()
    text = _normalize_space(chunk.text)
    lowered = text.lower()
    if section in _NOISE_SECTION_TOKENS:
        return True
    if _IMAGE_ONLY_RE.match(str(chunk.text or '')):
        return True
    if _looks_like_heading_only(str(chunk.text or ''), section=section):
        return True
    title = _normalize_space(paper_title).lower()
    if text.startswith('#') and title and title in lowered:
        return True
    if lowered.startswith('copyright ') or lowered.startswith('preprint '):
        return True
    if _looks_like_front_matter_noise(text, section=section, paper_title=title):
        return True
    return False


def _role_for_section(section: object) -> str:
    label = _normalize_space(section).lower()
    for hints, role in _SECTION_ROLE_HINTS:
        if any(hint in label for hint in hints):
            return role
    return 'background'


def _role_for_chunk(chunk: Chunk, *, paper_title: object) -> str:
    role = _role_for_section(chunk.section)
    text = _normalize_space(chunk.text).lower()
    section = _normalize_space(chunk.section).lower()
    title = _normalize_space(paper_title).lower()
    intro_like = section.startswith('1') or 'introduction' in section or 'background' in section
    pre_section_like = not section or (title and section == title)
    if role == 'background' and (intro_like or pre_section_like):
        if _looks_like_problem_statement(text):
            return 'problem'
    if role in {'background', 'interpretation', 'experiment'}:
        if any(pattern in text for pattern in _RESULT_TEXT_PATTERNS):
            return 'result'
    return role


def _looks_like_problem_statement(text: str) -> bool:
    if any(pattern in text for pattern in _PROBLEM_TEXT_PATTERNS):
        return True
    return any(subject in text for subject in _PROBLEM_SUBJECT_HINTS) and any(
        hint in text for hint in _PROBLEM_PURPOSE_HINTS
    )


def _promote_role_from_act_type(*, role: str, act_type: str) -> str:
    promoted = _ACT_TO_ROLE.get(act_type)
    if not promoted:
        return role
    if role == promoted:
        return role
    if role in {'background', 'interpretation'}:
        return promoted
    if role == 'problem' and promoted in {'method', 'experiment', 'result'}:
        return promoted
    return role


def _window_max_chars(schema: dict[str, Any]) -> int:
    rules = dict(schema.get('rules') or {})
    try:
        value = int(rules.get('paper_logic_trace_window_chars_max') or 5000)
    except Exception:
        value = 5000
    return max(1200, min(9000, value))


def _max_moves_per_window(schema: dict[str, Any]) -> int:
    rules = dict(schema.get('rules') or {})
    try:
        value = int(rules.get('paper_logic_trace_moves_per_window_max') or 2)
    except Exception:
        value = 2
    return max(1, min(4, value))


def _keyword_mentions(text: str, *, limit: int = 3) -> list[dict[str, Any]]:
    tokens = [token.lower() for token in _WORD_RE.findall(text) if token]
    phrases: list[str] = []
    for idx in range(len(tokens)):
        unigram = tokens[idx]
        if unigram in _STOP_TOKENS or len(unigram) < 3:
            continue
        phrases.append(unigram)
        if idx + 1 < len(tokens):
            bigram = f'{unigram} {tokens[idx + 1]}'
            if tokens[idx + 1] not in _STOP_TOKENS and len(tokens[idx + 1]) >= 3:
                phrases.append(bigram)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for phrase in phrases:
        if phrase in seen:
            continue
        seen.add(phrase)
        rows.append({'surface': phrase, 'normalized': phrase})
        if len(rows) >= limit:
            break
    return rows


def _clean_phrase(phrase: str) -> str:
    tokens = [token for token in _WORD_RE.findall(_normalize_space(phrase).lower()) if token]
    while tokens and tokens[0] in _STOP_TOKENS:
        tokens.pop(0)
    while tokens and tokens[-1] in _STOP_TOKENS:
        tokens.pop()
    return ' '.join(tokens)


def _merge_raw_mention_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for row in group or []:
            surface = _normalize_space((row or {}).get('surface') or (row or {}).get('normalized') or '')
            if not surface:
                continue
            key = surface.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(row))
    return rows


def _phrase_suffix_mentions(text: str, *, cue_words: tuple[str, ...], limit: int = 3) -> list[dict[str, Any]]:
    if not text:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cue in cue_words:
        pattern = re.compile(
            rf'\b((?:[a-z0-9-]+\s+){{0,3}}{re.escape(cue)})\b',
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            phrase = _clean_phrase(match.group(1))
            if not phrase or phrase in seen:
                continue
            seen.add(phrase)
            rows.append({'surface': phrase, 'normalized': phrase})
            if len(rows) >= limit:
                return rows
    return rows


def _metric_mentions(text: str, *, limit: int = 3) -> list[dict[str, Any]]:
    return _phrase_suffix_mentions(text, cue_words=_METRIC_CUE_WORDS, limit=limit)


def _observed_variable_mentions(text: str, *, limit: int = 3) -> list[dict[str, Any]]:
    return _phrase_suffix_mentions(text, cue_words=_OBSERVED_VARIABLE_CUE_WORDS, limit=limit)


def _refine_observed_variable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[tuple[dict[str, Any], str]] = []
    for row in rows or []:
        phrase = _normalize_space(row.get('normalized') or row.get('surface') or '').lower()
        if not phrase:
            continue
        if any(phrase.startswith(prefix) for prefix in _OBSERVED_VARIABLE_BAD_PREFIXES):
            continue
        tokens = phrase.split()
        if not tokens:
            continue
        if any(token in _OBSERVED_VARIABLE_BAD_TOKENS for token in tokens):
            continue
        prepared.append((row, phrase))

    keep_indices: list[int] = []
    phrases = [phrase for _, phrase in prepared]
    for index, phrase in enumerate(phrases):
        is_subsumed = False
        for other_index, other in enumerate(phrases):
            if index == other_index or len(other) <= len(phrase):
                continue
            if re.search(rf'(^|\\b){re.escape(phrase)}($|\\b)', other):
                is_subsumed = True
                break
        if not is_subsumed:
            keep_indices.append(index)
    return [prepared[index][0] for index in keep_indices]


def _refine_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[tuple[dict[str, Any], str]] = []
    for row in rows or []:
        phrase = _normalize_space(row.get('normalized') or row.get('surface') or '').lower()
        if not phrase:
            continue
        if any(phrase.startswith(prefix) for prefix in _METRIC_BAD_PREFIXES):
            continue
        tokens = phrase.split()
        if not tokens:
            continue
        if any(token in _METRIC_BAD_TOKENS for token in tokens):
            continue
        prepared.append((row, phrase))

    keep_indices: list[int] = []
    phrases = [phrase for _, phrase in prepared]
    for index, phrase in enumerate(phrases):
        is_subsumed = False
        for other_index, other in enumerate(phrases):
            if index == other_index or len(other) <= len(phrase):
                continue
            if re.search(rf'(^|\\b){re.escape(phrase)}($|\\b)', other):
                is_subsumed = True
                break
        if not is_subsumed:
            keep_indices.append(index)
    return [prepared[index][0] for index in keep_indices]


def _reclassify_physical_metric_rows(
    *,
    metrics: list[dict[str, Any]],
    observed_variables: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observed_rows = list(observed_variables or [])
    kept_metrics: list[dict[str, Any]] = []
    observed_seen = {
        _normalize_space(row.get('normalized') or row.get('surface') or '').lower()
        for row in observed_rows
        if _normalize_space(row.get('normalized') or row.get('surface') or '')
    }
    for row in metrics or []:
        phrase = _normalize_space(row.get('normalized') or row.get('surface') or '').lower()
        tokens = phrase.split()
        if (
            tokens
            and any(token in _OBSERVED_VARIABLE_CUE_WORDS for token in tokens)
            and not any(token in _STRONG_METRIC_TOKENS for token in tokens)
        ):
            if phrase and phrase not in observed_seen:
                observed_rows.append(dict(row))
                observed_seen.add(phrase)
            continue
        kept_metrics.append(row)
    return observed_rows, kept_metrics


def _resource_mentions_from_text(text: str, *, limit: int = 3) -> list[dict[str, Any]]:
    tokens = [token.lower() for token in _WORD_RE.findall(_normalize_space(text)) if token]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, token in enumerate(tokens):
        if token not in _RESOURCE_CUE_WORDS:
            continue
        start = index
        while start > 0 and index - start < 3:
            candidate = tokens[start - 1]
            if candidate in {'and', 'or'}:
                break
            if candidate in _RESOURCE_BACKTRACK_STOP_TOKENS:
                break
            start -= 1
        phrase = _clean_phrase(' '.join(tokens[start : index + 1]))
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        rows.append({'surface': phrase, 'normalized': phrase})
        if len(rows) >= limit:
            break
    return rows


def _refine_resource_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[tuple[dict[str, Any], str]] = []
    for row in rows or []:
        phrase = _normalize_space(row.get('normalized') or row.get('surface') or '').lower()
        type_name = _normalize_space(row.get('type') or '').lower()
        if not phrase:
            continue
        if type_name in _RESOURCE_BAD_TYPES:
            continue
        if phrase.startswith('http://') or phrase.startswith('https://'):
            continue
        if '[' in phrase or ']' in phrase or ' et al' in phrase or ' doi' in phrase:
            continue
        if re.search(r'^[a-z][a-z\s.\-]*&\s*[a-z][a-z\s.\-]*\(\d{4}\)$', phrase):
            continue
        if re.search(r'^[a-z][a-z\s.\-]*&\s*[a-z][a-z\s.\-]*\d{4}$', phrase):
            continue
        tokens = phrase.split()
        has_resource_cue = any(token in _RESOURCE_CUE_WORDS for token in tokens)
        if len(tokens) == 1 and not has_resource_cue and type_name not in _RESOURCE_KEEP_TYPES:
            continue
        prepared.append((row, phrase))

    keep_indices: list[int] = []
    phrases = [phrase for _, phrase in prepared]
    for index, phrase in enumerate(phrases):
        is_subsumed = False
        for other_index, other in enumerate(phrases):
            if index == other_index or len(other) <= len(phrase):
                continue
            if re.search(rf'(^|\b){re.escape(phrase)}($|\b)', other):
                is_subsumed = True
                break
        if not is_subsumed:
            keep_indices.append(index)
    return [prepared[index][0] for index in keep_indices]


def _comparator_mentions(text: str, *, limit: int = 2) -> list[dict[str, Any]]:
    lowered = _normalize_space(text).lower()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    pair_patterns = (
        r'\bagreement between\s+([a-z0-9][a-z0-9\-\s]{2,30})\s+and\s+([a-z0-9][a-z0-9\-\s]{2,30})\b',
        r'\bagreement is found between\s+([a-z0-9][a-z0-9\-\s]{2,30})\s+and\s+([a-z0-9][a-z0-9\-\s]{2,30})\b',
        r'\b([a-z0-9][a-z0-9\-\s]{2,30})\s+and\s+([a-z0-9][a-z0-9\-\s]{2,30})\s+are\s+in\s+(?:(?:good|close|quantitative)\s+)?agreement\b',
        r'\bcompar(?:e|es|ing)\s+([a-z0-9][a-z0-9\-\s]{2,40}?)\s+with\s+([a-z0-9][a-z0-9\-\s]{2,40})\b',
    )
    single_patterns = (
        r'\bcompared with\s+([a-z0-9][a-z0-9\-\s]{2,50})',
        r'\bcompared to\s+([a-z0-9][a-z0-9\-\s]{2,50})',
        r'\bthan\s+([a-z0-9][a-z0-9\-\s]{2,50})',
        r'\bunlike\s+([a-z0-9][a-z0-9\-\s]{2,50})',
        r'\bversus\s+([a-z0-9][a-z0-9\-\s]{2,50})',
        r'\bvs\.?\s+([a-z0-9][a-z0-9\-\s]{2,50})',
        r'\bsimilar to\s+([a-z0-9][a-z0-9\-\s]{2,50})',
        r'\bagreement with\s+([a-z0-9][a-z0-9\-\s]{2,50})',
    )

    def _push_phrase(raw_phrase: str) -> bool:
        pronoun_match = re.match(r'^(?:that|those)\s+(?:for|of)\s+(.+)$', raw_phrase.strip())
        if pronoun_match:
            raw_phrase = pronoun_match.group(1)
        phrase = re.split(r'[.,;:()]', raw_phrase, maxsplit=1)[0]
        phrase = re.split(r'\b(?:during|under|while|when|for|in|at|on|using|via)\b', phrase, maxsplit=1)[0]
        phrase = _clean_phrase(' '.join(phrase.split()[:6]))
        if not phrase or phrase in seen:
            return False
        seen.add(phrase)
        rows.append({'surface': phrase, 'normalized': phrase})
        return len(rows) >= limit

    for pattern in pair_patterns:
        for match in re.finditer(pattern, lowered):
            if _push_phrase(match.group(1)) or _push_phrase(match.group(2)):
                return rows
    for pattern in single_patterns:
        for match in re.finditer(pattern, lowered):
            if _push_phrase(match.group(1)):
                return rows
    return rows


def _refine_comparator_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[tuple[dict[str, Any], str]] = []
    for row in rows or []:
        phrase = _normalize_space(row.get('normalized') or row.get('surface') or '').lower()
        tokens = phrase.split()
        if not tokens:
            continue
        if (
            len(tokens) == 1
            and tokens[0] not in _COMPARATOR_ENTITY_HINTS
            and not (len(tokens[0]) >= 5 and tokens[0].endswith('s'))
        ):
            continue
        if any(token in _COMPARATOR_BAD_TOKENS for token in tokens):
            continue
        if tokens[0] in _COMPARATOR_BAD_LEADS and not any(token in _COMPARATOR_ENTITY_HINTS for token in tokens):
            continue
        prepared.append((row, phrase))

    keep_indices: list[int] = []
    phrases = [phrase for _, phrase in prepared]
    for index, phrase in enumerate(phrases):
        is_subsumed = False
        for other_index, other in enumerate(phrases):
            if index == other_index or len(other) <= len(phrase):
                continue
            if re.search(rf'(^|\b){re.escape(phrase)}($|\b)', other):
                is_subsumed = True
                break
        if not is_subsumed:
            keep_indices.append(index)
    return [prepared[index][0] for index in keep_indices]


def _limitation_mentions(text: str, *, limit: int = 2) -> list[dict[str, Any]]:
    lowered = _normalize_space(text).lower()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    patterns = (
        r'\b([a-z0-9-]+\s+cost)\b',
        r'\b(memory limitations?)\b',
        r'\b(difficult to [a-z0-9-]+(?:\s+[a-z0-9-]+){0,2})\b',
        r'\b(expensive [a-z0-9-]+(?:\s+[a-z0-9-]+){0,2})\b',
        r'\b(limited [a-z0-9-]+(?:\s+[a-z0-9-]+){0,2})\b',
    )
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            phrase = _clean_phrase(match.group(1))
            if not phrase or phrase in seen:
                continue
            seen.add(phrase)
            rows.append({'surface': phrase, 'normalized': phrase})
            if len(rows) >= limit:
                return rows
    for cue in _LIMITATION_CUE_WORDS:
        if cue not in lowered:
            continue
        words = [token for token in _WORD_RE.findall(lowered) if token]
        for index, token in enumerate(words):
            if token != cue:
                continue
            start = max(0, index - 1)
            end = min(len(words), index + 3)
            phrase = _clean_phrase(' '.join(words[start:end]))
            if not phrase or phrase in seen:
                continue
            seen.add(phrase)
            rows.append({'surface': phrase, 'normalized': phrase})
            if len(rows) >= limit:
                return rows
    return rows


def _trim_limitation_phrase(phrase: str) -> str:
    clean = _normalize_space(phrase).strip(' ,.;:')
    clean = re.split(r'\b(?:which|that|while|when|because|but|and)\b', clean, maxsplit=1)[0]
    return _clean_phrase(clean)


def _refine_limitation_rows(rows: list[dict[str, Any]], *, text: str) -> list[dict[str, Any]]:
    lowered_text = _normalize_space(text).lower()
    prepared: list[tuple[dict[str, Any], str]] = []
    for row in rows or []:
        phrase = _normalize_space(row.get('normalized') or row.get('surface') or '').lower()
        if not phrase:
            continue
        if phrase in _GENERIC_LIMITATION_PHRASES:
            candidate = None
            assumption_match = re.search(r'assum(?:e|es|ed|ing)\s+(?:that\s+)?([a-z0-9][a-z0-9\-\s]{4,50})', lowered_text)
            if assumption_match:
                base = _trim_limitation_phrase(assumption_match.group(1))
                if base:
                    candidate = f'{base} assumption'
            if not candidate:
                validity_match = re.search(r'valid only\s+(below|above|under)\s+([a-z0-9][a-z0-9\-\s.]{2,25})', lowered_text)
                if validity_match:
                    scope = _trim_limitation_phrase(validity_match.group(2))
                    if scope:
                        candidate = f'valid only {validity_match.group(1)} {scope}'
            if not candidate:
                due_match = re.search(r'due to\s+([a-z0-9][a-z0-9\-\s]{4,40})', lowered_text)
                if due_match:
                    base = _trim_limitation_phrase(due_match.group(1))
                    if base:
                        candidate = f'due to {base}'
            if not candidate:
                req_match = re.search(r'requiring\s+([a-z0-9][a-z0-9\-\s]{4,40})', lowered_text)
                if req_match:
                    base = _trim_limitation_phrase(req_match.group(1))
                    if base:
                        candidate = f'requiring {base}'
            if candidate:
                row = {**row, 'surface': candidate, 'normalized': candidate}
                phrase = candidate
        prepared.append((row, phrase))

    seen: set[str] = set()
    refined: list[dict[str, Any]] = []
    for row, phrase in prepared:
        if phrase in seen:
            continue
        seen.add(phrase)
        refined.append(row)
    return refined


def _augment_sparse_slots(
    *,
    text: str,
    role: str,
    observed_variables: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    comparators: list[dict[str, Any]],
    limitation_types: list[dict[str, Any]],
    resource_mentions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    role_token = _normalize_role(role)
    metric_roles = {'result', 'experiment', 'interpretation'}
    observed_variable_roles = {'experiment', 'result', 'interpretation', 'method'}
    limitation_roles = {'limitation', 'interpretation', 'result', 'future_work'}
    resource_roles = {'method', 'experiment', 'result'}
    heuristic_comparators = _comparator_mentions(text, limit=3) if role_token in metric_roles else []
    return {
        'observed_variables': observed_variables or (_observed_variable_mentions(text, limit=3) if role_token in observed_variable_roles else []),
        'metrics': metrics or (_metric_mentions(text, limit=3) if role_token in metric_roles else []),
        'comparators': _merge_raw_mention_rows(comparators, heuristic_comparators),
        'limitation_types': limitation_types or (_limitation_mentions(text, limit=2) if role_token in limitation_roles else []),
        'resource_mentions': resource_mentions or (_resource_mentions_from_text(text, limit=3) if role_token in resource_roles else []),
    }


def _condition_mentions(text: str, *, limit: int = 2) -> list[dict[str, Any]]:
    lowered = _normalize_space(text).lower()
    rows: list[dict[str, Any]] = []
    for cue in _CONDITION_CUE_WORDS:
        for match in re.finditer(rf'\b{cue}\s+([a-z0-9][a-z0-9\-\s]{{4,40}})', lowered):
            phrase = _normalize_space(match.group(1)).strip(' .,;:')
            if not phrase:
                continue
            candidate = f'{cue} {phrase}'
            rows.append({'surface': candidate, 'normalized': candidate})
            if len(rows) >= limit:
                return rows
    return rows


def _summary_from_text(text: str, *, max_chars: int = 220) -> str:
    clean = _normalize_space(text)
    if not clean:
        return ''
    sentences = re.split(r'(?<=[.!?。！？])\s+', clean)
    summary = ' '.join(sentence.strip() for sentence in sentences[:2] if sentence.strip()).strip()
    if not summary:
        summary = clean
    if len(summary) <= max_chars:
        return summary
    return summary[: max_chars - 3].rstrip() + '...'


def _window_text(chunks: list[Chunk], max_chars: int) -> str:
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        text = _normalize_space(chunk.text)
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rstrip()
        parts.append(f'[{chunk.chunk_id}] {text}')
        total += len(text)
    return '\n\n'.join(parts)


def _semantic_windows(doc: DocumentIR, schema: dict[str, Any]) -> list[dict[str, Any]]:
    max_chars = _window_max_chars(schema)
    windows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for chunk in doc.chunks:
        text = _normalize_space(chunk.text)
        if not text or _is_reference_section(chunk.section) or _is_noise_chunk(chunk, paper_title=doc.paper.title):
            continue
        role_hint = _role_for_chunk(chunk, paper_title=doc.paper.title)
        if (
            current is None
            or current['role_hint'] != role_hint
            or current['char_count'] + len(text) > max_chars
        ):
            current = {
                'window_id': f'{doc.paper.paper_source}:window:{len(windows) + 1}',
                'role_hint': role_hint,
                'act_hint': _ROLE_TO_ACT.get(role_hint, 'define_task'),
                'section_path': [str(chunk.section).strip()] if str(chunk.section or '').strip() else [],
                'chunks': [],
                'char_count': 0,
            }
            windows.append(current)
        current['chunks'].append(chunk)
        current['char_count'] += len(text)
    return [window for window in windows if window.get('chunks')]


def _normalize_mention_rows(rows: list[dict[str, Any]] | None, *, anchor_ids: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows or []:
        surface = _normalize_space(row.get('surface') or row.get('normalized') or '')
        if not surface:
            continue
        out.append(
            {
                'surface': surface,
                'normalized': _normalize_space(row.get('normalized') or surface).lower() or None,
                'type': _normalize_space(row.get('type') or '') or None,
                'anchor_ids': list(anchor_ids),
                'confidence': row.get('confidence'),
                'inferred': False,
            }
        )
    return out


def _normalize_effect_rows(rows: list[dict[str, Any]] | None, *, anchor_ids: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows or []:
        direction = _normalize_space(row.get('direction') or 'unknown').lower()
        if direction not in {'increase', 'decrease', 'improve', 'worsen', 'mixed', 'none', 'unknown'}:
            direction = 'unknown'
        out.append(
            {
                'direction': direction,
                'magnitude_text': _normalize_space(row.get('magnitude_text') or '') or None,
                'comparator_surface': _normalize_space(row.get('comparator_surface') or '') or None,
                'anchor_ids': list(anchor_ids),
                'confidence': row.get('confidence'),
            }
        )
    return out


def _slot_provenance_rows(field: str, values: list[dict[str, Any]], *, anchor_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        rows.append(
            {
                'field': field,
                'value_index': index,
                'anchor_ids': list(anchor_ids),
                'extraction_mode': 'direct',
                'support_strength': 'strong',
                'confidence': value.get('confidence'),
            }
        )
    return rows


def _fallback_move_payload(window: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[Chunk] = list(window.get('chunks') or [])
    if not chunks:
        return []
    anchor_chunk_ids = [str(chunk.chunk_id).strip() for chunk in chunks[:2] if str(chunk.chunk_id).strip()]
    summary = _summary_from_text(' '.join(chunk.text for chunk in chunks))
    role = _normalize_role(window.get('role_hint'))
    methods = _keyword_mentions(summary, limit=2) if role in {'method', 'experiment'} else []
    research_objects = _keyword_mentions(summary, limit=2) if role not in {'method', 'experiment'} else []
    conditions = _condition_mentions(summary, limit=2)
    augmented = _augment_sparse_slots(
        text=summary,
        role=role,
        observed_variables=[],
        metrics=[],
        comparators=[],
        limitation_types=_keyword_mentions(summary, limit=2) if role == 'limitation' else [],
        resource_mentions=[],
    )
    return [
        {
            'role': role,
            'act_type': _ROLE_TO_ACT.get(role, 'define_task'),
            'summary': summary,
            'anchor_chunk_ids': anchor_chunk_ids,
            'research_objects': research_objects,
            'methods': methods,
            'observed_variables': augmented['observed_variables'],
            'metrics': augmented['metrics'],
            'comparators': augmented['comparators'],
            'conditions': conditions,
            'limitation_types': augmented['limitation_types'],
            'resource_mentions': augmented['resource_mentions'],
            'effects': [],
            'confidence': 0.0,
        }
    ]


def _derive_move_confidence(
    *,
    summary: str,
    anchor_chunk_ids: list[str],
    research_objects: list[dict[str, Any]],
    methods: list[dict[str, Any]],
    observed_variables: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    comparators: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    limitation_types: list[dict[str, Any]],
    resource_mentions: list[dict[str, Any]],
    raw_confidence: Any,
) -> float:
    try:
        raw_value = float(raw_confidence)
    except (TypeError, ValueError):
        raw_value = 0.0
    slot_count = sum(
        len(items)
        for items in (
            research_objects,
            methods,
            observed_variables,
            metrics,
            comparators,
            conditions,
            effects,
            limitation_types,
            resource_mentions,
        )
    )
    heuristic = 0.22
    heuristic += min(0.18, 0.07 * len(anchor_chunk_ids))
    heuristic += min(0.38, 0.06 * slot_count)
    if len(_normalize_space(summary)) >= 32:
        heuristic += 0.08
    if methods or research_objects:
        heuristic += 0.08
    if metrics or comparators or effects:
        heuristic += 0.06
    if raw_value > 0.0:
        heuristic = max(heuristic, raw_value)
    return round(min(0.98, heuristic), 4)


def _extract_window_moves_llm(
    *,
    doc: DocumentIR,
    schema: dict[str, Any],
    window: dict[str, Any],
) -> list[dict[str, Any]]:
    from app.llm.client import call_validated_json
    from app.llm.schemas import ResearchMoveWindowResponse

    chunks: list[Chunk] = list(window.get('chunks') or [])
    if not chunks:
        return []
    system, user = _build_research_move_prompt(doc=doc, schema=schema, window=window)
    try:
        validated = call_validated_json(system, user, ResearchMoveWindowResponse)
    except Exception:
        logger.debug('ResearchMove window extraction failed; using heuristic fallback', exc_info=True)
        return []
    payload = validated.model_dump(mode='json') if hasattr(validated, 'model_dump') else dict(validated or {})
    return list(payload.get('moves') or [])


def _build_research_move_prompt(
    *,
    doc: DocumentIR,
    schema: dict[str, Any],
    window: dict[str, Any],
) -> tuple[str, str]:
    chunks: list[Chunk] = list(window.get('chunks') or [])
    max_moves = _max_moves_per_window(schema)
    chunk_ids = [str(chunk.chunk_id).strip() for chunk in chunks if str(chunk.chunk_id).strip()]
    chunk_text = _window_text(chunks, _window_max_chars(schema))
    system = (
        'You extract canonical ResearchMove records for a PaperLogicTrace.\n'
        'Return strict JSON only.\n'
        'Do not mention any field that is not directly supported by the provided chunks.\n'
        'Use only the provided chunk ids in anchor_chunk_ids.\n'
        f'Allowed roles: {", ".join(_ALLOWED_ROLES)}.\n'
        f'Allowed act_type values: {", ".join(_ALLOWED_ACTS)}.\n'
        'Keep summary concise and factual.\n'
        'Normalize short phrases when obvious, but do not invent domain ontology.\n'
        'Positive example: "This paper investigates ..." or "In this paper, we investigate ..." in an abstract/introduction window usually signals a problem or define_task move.\n'
        'Positive example: "This paper outlines ... to investigate ..." or "The work presented here ... aims to ..." in an abstract/introduction window usually signals a problem or define_task move.\n'
        'Positive example: "Results show that ..." or "we find that ..." usually signals a result/report_effect move.\n'
        'Positive example: "the mixing degree is higher than the dry mixture baseline" should yield a metric like "mixing degree" and a comparator like "dry mixture baseline".\n'
        'Positive example: "using X-ray microtomography and high-speed camera observations" should populate resource_mentions.\n'
        'Positive example: "advanced tracking techniques are expensive because of computational cost" should populate limitation_types such as "computational cost" or "expensive tracking techniques".\n'
        'Negative example: an author line, affiliation line, or received date is front matter and should produce no move.\n'
        'Negative example: image-only markdown or captionless asset references should produce no move.\n'
        'Negative example: named citations, statistical laws, and application areas are not resource_mentions unless the text explicitly presents them as a tool, platform, dataset, software package, protocol, or instrument.\n'
        'Negative example: do not treat every noun phrase as a metric or resource; only extract them when the text explicitly uses them as an evaluation target, comparator, limitation, tool, platform, or instrument.\n'
    )
    user = (
        f'Paper title: {doc.paper.title or doc.paper.paper_source}\n'
        f'Role hint: {window.get("role_hint")}\n'
        f'Section path: {" > ".join(window.get("section_path") or []) or "(unknown)"}\n'
        f'Max moves: {max_moves}\n'
        f'Available chunk ids: {", ".join(chunk_ids)}\n\n'
        'Return JSON like:\n'
        '{\n'
        '  "moves": [\n'
        '    {\n'
        '      "role": "method",\n'
        '      "act_type": "propose_method",\n'
        '      "summary": "...",\n'
        '      "anchor_chunk_ids": ["c1"],\n'
        '      "research_objects": [{"surface": "...", "normalized": "...", "type": ""}],\n'
        '      "methods": [{"surface": "...", "normalized": "...", "type": ""}],\n'
        '      "observed_variables": [],\n'
        '      "metrics": [],\n'
        '      "comparators": [],\n'
        '      "conditions": [],\n'
        '      "limitation_types": [],\n'
        '      "resource_mentions": [],\n'
        '      "effects": [{"direction": "unknown", "magnitude_text": "", "comparator_surface": ""}],\n'
        '      "confidence": 0.0\n'
        '    }\n'
        '  ]\n'
        '}\n\n'
        f'Chunks:\n{chunk_text}'
    )
    return system, user


def _move_rows_from_windows(
    *,
    doc: DocumentIR,
    paper_id: str,
    schema: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in doc.chunks}
    windows = _semantic_windows(doc, schema)
    evidence_rows: list[dict[str, Any]] = []
    move_defs: list[dict[str, Any]] = []
    extracted_moves = 0

    for window_index, window in enumerate(windows, start=1):
        raw_moves = _extract_window_moves_llm(doc=doc, schema=schema, window=window)
        if not raw_moves:
            raw_moves = _fallback_move_payload(window)
        for move_offset, raw_move in enumerate(raw_moves, start=1):
            role = _normalize_role(raw_move.get('role') or window.get('role_hint'))
            act_type = _normalize_act_type(raw_move.get('act_type'), role=role)
            role = _promote_role_from_act_type(role=role, act_type=act_type)
            summary = _summary_from_text(raw_move.get('summary') or _window_text(window.get('chunks') or [], 1200))
            if _looks_like_noise_summary(summary):
                continue
            anchor_chunk_ids = [
                str(item).strip()
                for item in (raw_move.get('anchor_chunk_ids') or [])
                if str(item).strip() in chunk_by_id
            ]
            if not anchor_chunk_ids:
                anchor_chunk_ids = [str(chunk.chunk_id).strip() for chunk in (window.get('chunks') or [])[:1] if str(chunk.chunk_id).strip()]
            if not anchor_chunk_ids:
                continue

            move_id = f'{paper_id}:move:{window_index}:{move_offset}'
            research_objects = _normalize_mention_rows(raw_move.get('research_objects'), anchor_ids=anchor_chunk_ids)
            methods = _normalize_mention_rows(raw_move.get('methods'), anchor_ids=anchor_chunk_ids)
            observed_variables = _normalize_mention_rows(raw_move.get('observed_variables'), anchor_ids=anchor_chunk_ids)
            metrics = _normalize_mention_rows(raw_move.get('metrics'), anchor_ids=anchor_chunk_ids)
            comparators = _normalize_mention_rows(raw_move.get('comparators'), anchor_ids=anchor_chunk_ids)
            conditions = _normalize_mention_rows(raw_move.get('conditions'), anchor_ids=anchor_chunk_ids)
            limitation_types = _normalize_mention_rows(raw_move.get('limitation_types'), anchor_ids=anchor_chunk_ids)
            resource_mentions = _normalize_mention_rows(raw_move.get('resource_mentions'), anchor_ids=anchor_chunk_ids)
            effects = _normalize_effect_rows(raw_move.get('effects'), anchor_ids=anchor_chunk_ids)
            augmented = _augment_sparse_slots(
                text=' '.join([
                    summary,
                    _window_text(window.get('chunks') or [], 1600),
                ]),
                role=role,
                observed_variables=observed_variables,
                metrics=metrics,
                comparators=comparators,
                limitation_types=limitation_types,
                resource_mentions=resource_mentions,
            )
            observed_variables = _normalize_mention_rows(augmented['observed_variables'], anchor_ids=anchor_chunk_ids)
            observed_variables = _refine_observed_variable_rows(observed_variables)
            metrics = _normalize_mention_rows(augmented['metrics'], anchor_ids=anchor_chunk_ids)
            metrics = _refine_metric_rows(metrics)
            observed_variables, metrics = _reclassify_physical_metric_rows(
                metrics=metrics,
                observed_variables=observed_variables,
            )
            observed_variables = _refine_observed_variable_rows(observed_variables)
            comparators = _normalize_mention_rows(augmented['comparators'], anchor_ids=anchor_chunk_ids)
            comparators = _refine_comparator_rows(comparators)
            limitation_types = _normalize_mention_rows(augmented['limitation_types'], anchor_ids=anchor_chunk_ids)
            limitation_types = _refine_limitation_rows(
                limitation_types,
                text=' '.join([summary, _window_text(window.get('chunks') or [], 1600)]),
            )
            resource_mentions = _normalize_mention_rows(augmented['resource_mentions'], anchor_ids=anchor_chunk_ids)
            resource_mentions = _refine_resource_rows(resource_mentions)

            slot_provenance = [
                *_slot_provenance_rows('research_objects', research_objects, anchor_ids=anchor_chunk_ids),
                *_slot_provenance_rows('methods', methods, anchor_ids=anchor_chunk_ids),
                *_slot_provenance_rows('observed_variables', observed_variables, anchor_ids=anchor_chunk_ids),
                *_slot_provenance_rows('metrics', metrics, anchor_ids=anchor_chunk_ids),
                *_slot_provenance_rows('comparators', comparators, anchor_ids=anchor_chunk_ids),
                *_slot_provenance_rows('conditions', conditions, anchor_ids=anchor_chunk_ids),
                *_slot_provenance_rows('limitation_types', limitation_types, anchor_ids=anchor_chunk_ids),
                *_slot_provenance_rows('resource_mentions', resource_mentions, anchor_ids=anchor_chunk_ids),
            ]
            confidence = _derive_move_confidence(
                summary=summary,
                anchor_chunk_ids=anchor_chunk_ids,
                research_objects=research_objects,
                methods=methods,
                observed_variables=observed_variables,
                metrics=metrics,
                comparators=comparators,
                conditions=conditions,
                effects=effects,
                limitation_types=limitation_types,
                resource_mentions=resource_mentions,
                raw_confidence=raw_move.get('confidence'),
            )

            for anchor_index, chunk_id in enumerate(anchor_chunk_ids, start=1):
                chunk = chunk_by_id.get(chunk_id)
                if chunk is None:
                    continue
                evidence_rows.append(
                    {
                        'anchor_id': f'{move_id}:anchor:{anchor_index}',
                        'paper_id': paper_id,
                        'source_ref': chunk_id,
                        'modality': 'text',
                        'section_path': [str(chunk.section).strip()] if str(chunk.section or '').strip() else [],
                        'locator': {
                            'chunk_id': chunk.chunk_id,
                            'start_line': chunk.span.start_line,
                            'end_line': chunk.span.end_line,
                        },
                        'quote': _normalize_space(chunk.text),
                        'citation_ids': [],
                        'support_type': 'direct',
                        'weak': False,
                        'move_id': move_id,
                        'sequence_no': len(move_defs) + 1,
                        'role_hint': role,
                        'act_hint': act_type,
                        'summary': summary,
                        'confidence': confidence,
                        'research_objects': research_objects,
                        'methods': methods,
                        'observed_variables': observed_variables,
                        'metrics': metrics,
                        'comparators': comparators,
                        'conditions': conditions,
                        'effects': effects,
                        'limitation_types': limitation_types,
                        'resource_mentions': resource_mentions,
                        'slot_provenance': slot_provenance,
                    }
                )

            move_defs.append(
                {
                    'move_id': move_id,
                    'sequence_no': len(move_defs) + 1,
                    'role': role,
                    'act_type': act_type,
                    'anchor_chunk_ids': anchor_chunk_ids,
                }
            )
            extracted_moves += 1

    report = {
        'window_count': len(windows),
        'move_count': extracted_moves,
    }
    return evidence_rows, move_defs, report


def _infer_relation_type(source_role: str, target_role: str, source_act: str = '', target_act: str = '') -> str:
    source_method_like = source_role == 'method' or source_act in {'propose_method', 'adapt_method'}
    target_method_like = target_role == 'method' or target_act in {'propose_method', 'adapt_method'}
    source_experiment_like = source_role == 'experiment' or source_act in {'run_experiment', 'measure_outcome', 'compare_baseline', 'set_condition'}
    target_experiment_like = target_role == 'experiment' or target_act in {'run_experiment', 'measure_outcome', 'compare_baseline', 'set_condition'}
    source_result_like = source_role == 'result' or source_act == 'report_effect'
    target_result_like = target_role == 'result' or target_act == 'report_effect'

    if source_method_like and target_method_like:
        return 'implements'
    if source_method_like and target_experiment_like:
        return 'evaluates'
    if source_experiment_like and target_result_like:
        return 'yields'
    if source_method_like and target_result_like:
        return 'yields'

    pair = (source_role, target_role)
    mapping = {
        ('background', 'problem'): 'motivates',
        ('problem', 'method'): 'addresses',
        ('problem', 'experiment'): 'addresses',
        ('method', 'experiment'): 'evaluates',
        ('experiment', 'result'): 'yields',
        ('experiment', 'interpretation'): 'explains',
        ('result', 'interpretation'): 'explains',
        ('result', 'limitation'): 'limits',
        ('method', 'interpretation'): 'explains',
        ('interpretation', 'limitation'): 'limits',
        ('method', 'result'): 'yields',
        ('result', 'future_work'): 'extends',
        ('limitation', 'future_work'): 'extends',
    }
    return mapping.get(pair, 'motivates')


def _relation_row(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    source_move_id = str(source.get('move_id') or '')
    target_move_id = str(target.get('move_id') or '')
    relation_type = _infer_relation_type(
        str(source.get('role') or ''),
        str(target.get('role') or ''),
        str(source.get('act_type') or ''),
        str(target.get('act_type') or ''),
    )
    return {
        'relation_id': f'{source_move_id}:rel:{target_move_id}',
        'source_move_id': source_move_id,
        'target_move_id': target_move_id,
        'relation_type': relation_type,
        'anchor_ids': [],
        'confidence': 0.65,
    }


def _build_move_relation_rows(move_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = sorted(move_defs, key=lambda row: int(row.get('sequence_no') or 0))
    seen_pairs: set[tuple[str, str]] = set()
    for index in range(len(ordered) - 1):
        source = ordered[index]
        target = ordered[index + 1]
        pair = (str(source.get('move_id') or ''), str(target.get('move_id') or ''))
        if pair in seen_pairs:
            continue
        rows.append(_relation_row(source, target))
        seen_pairs.add(pair)

    max_lookahead = 3
    for index, source in enumerate(ordered):
        source_move_id = str(source.get('move_id') or '')
        if not source_move_id:
            continue
        for lookahead in range(index + 2, min(len(ordered), index + 1 + max_lookahead)):
            target = ordered[lookahead]
            pair = (source_move_id, str(target.get('move_id') or ''))
            if pair in seen_pairs:
                continue
            relation_type = _infer_relation_type(
                str(source.get('role') or ''),
                str(target.get('role') or ''),
                str(source.get('act_type') or ''),
                str(target.get('act_type') or ''),
            )
            if relation_type == 'motivates':
                continue
            rows.append(_relation_row(source, target))
            seen_pairs.add(pair)
            break
    return rows


def _build_citation_rows(cite_rec: dict[str, Any] | None, move_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not cite_rec:
        return []
    move_ids_by_chunk: dict[str, list[str]] = defaultdict(list)
    for move in move_defs:
        for chunk_id in move.get('anchor_chunk_ids') or []:
            move_ids_by_chunk[str(chunk_id)].append(str(move.get('move_id') or ''))
    rows: list[dict[str, Any]] = []
    for index, row in enumerate((cite_rec.get('cites_resolved') or []), start=1):
        cited_paper_id = str(row.get('cited_paper_id') or '').strip()
        if not cited_paper_id:
            continue
        evidence_chunk_ids = [str(item).strip() for item in (row.get('evidence_chunk_ids') or []) if str(item).strip()]
        source_move_id = next(
            (
                move_id
                for chunk_id in evidence_chunk_ids
                for move_id in move_ids_by_chunk.get(chunk_id, [])
                if move_id
            ),
            None,
        )
        rows.append(
            {
                'citation_act_id': f'{cite_rec.get("paper_id") or "paper"}:citation:{index}',
                'source_move_id': source_move_id,
                'target_paper_id': cited_paper_id,
                'purpose': None,
                'polarity': None,
                'semantic_signal': None,
                'target_scope': None,
                'anchor_ids': [],
                'confidence': 0.4,
            }
        )
    return rows


def build_trace_quality_report(trace: Any, extraction_report: dict[str, Any] | None = None) -> dict[str, Any]:
    moves = list((trace.canonical_core.moves if trace else []) or [])
    anchors = list((trace.canonical_core.evidence_anchors if trace else []) or [])
    relations = list((trace.canonical_core.move_relations if trace else []) or [])
    gate = dict((trace.quality or {}).get('hot_path_gate_report') or {})
    quality = dict((trace.quality or {}) or {})
    slot_signal_counts = {
        'research_objects': sum(len(move.research_objects) for move in moves),
        'methods': sum(len(move.methods) for move in moves),
        'metrics': sum(len(move.metrics) for move in moves),
        'conditions': sum(len(move.conditions) for move in moves),
        'comparators': sum(len(move.comparators) for move in moves),
        'limitation_types': sum(len(move.limitation_types) for move in moves),
        'resource_mentions': sum(len(move.resource_mentions) for move in moves),
    }
    roles_present = sorted({str(move.role) for move in moves if str(move.role).strip()})
    critical_roles = {'problem', 'method', 'result'}
    critical_present = critical_roles & set(roles_present)
    slot_ready_moves = [
        move.move_id
        for move in moves
        if any(
            (
                move.research_objects,
                move.methods,
                move.metrics,
                move.conditions,
                move.comparators,
                move.limitation_types,
                move.resource_mentions,
            )
        )
    ]
    report = {
        'gate_passed': bool(gate.get('passed')),
        'quality_tier': str(quality.get('quality_tier') or 'red'),
        'quality_tier_score': float(quality.get('quality_tier_score') or 0.0),
        'quality_flags': list(quality.get('quality_flags') or []),
        'audit_status': str(quality.get('audit_status') or 'blocked'),
        'move_count': len(moves),
        'anchor_count': len(anchors),
        'relation_count': len(relations),
        'roles_present': roles_present,
        'role_coverage_ratio': len(critical_present) / max(1, len(critical_roles)),
        'supported_signal_ratio': len(slot_ready_moves) / max(1, len(moves)),
        'slot_signal_counts': slot_signal_counts,
        'paper_logic_trace_mode': 'direct_move_extraction',
    }
    if extraction_report:
        report.update(
            {
                'window_count': int(extraction_report.get('window_count') or 0),
                'move_window_count': int(extraction_report.get('move_count') or 0),
            }
        )
    return report


def build_paper_logic_trace_inputs(
    *,
    doc: DocumentIR,
    paper_id: str,
    cite_rec: dict[str, Any] | None,
    schema: dict[str, Any],
    move_extractor: MoveExtractorFn | None = None,
) -> dict[str, Any]:
    extractor = move_extractor or _default_move_extractor
    try:
        extracted = extractor(
            doc=doc,
            paper_id=paper_id,
            cite_rec=cite_rec,
            schema=schema,
        )
    except TypeError:
        extracted = extractor(
            doc=doc,
            paper_id=paper_id,
            schema=schema,
        )
    payload = dict(extracted or {})
    if payload.get('paper_metadata') and payload.get('evidence_rows') is not None:
        return payload

    evidence_rows, move_defs, report = _move_rows_from_windows(doc=doc, paper_id=paper_id, schema=schema)
    paper_metadata = {
        'paper_id': paper_id,
        'canonical_doi': doc.paper.doi,
        'title': doc.paper.title or paper_id,
        'year': doc.paper.year,
        'authors': list(doc.paper.authors or []),
        'paper_type': normalize_trace_paper_type(
            doc.paper.paper_type,
            schema.get('paper_type'),
        ),
        'source_refs': [chunk.chunk_id for chunk in doc.chunks if str(chunk.chunk_id or '').strip()],
    }
    return {
        'paper_metadata': paper_metadata,
        'evidence_rows': evidence_rows,
        'figure_rows': [],
        'table_rows': [],
        'citation_rows': _build_citation_rows(cite_rec, move_defs),
        'move_relation_rows': _build_move_relation_rows(move_defs),
        'extraction_report': report,
    }


def _default_move_extractor(
    *,
    doc: DocumentIR,
    paper_id: str,
    cite_rec: dict[str, Any] | None,
    schema: dict[str, Any],
) -> dict[str, Any]:
    del cite_rec
    return {}
