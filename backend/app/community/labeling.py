from __future__ import annotations

import re
from collections import Counter
from typing import Any


_WORD_RE = re.compile(r'[a-z0-9]+(?:-[a-z0-9]+)?', re.IGNORECASE)
_GENERIC_TOKENS = {
    'a',
    'an',
    'and',
    'approach',
    'author',
    'authors',
    'based',
    'conduct',
    'conducted',
    'describes',
    'effect',
    'existing',
    'example',
    'examples',
    'experiment',
    'experiments',
    'for',
    'method',
    'methods',
    'model',
    'of',
    'paper',
    'prior',
    'problem',
    'proposes',
    'reasoning',
    'representation',
    'representations',
    'result',
    'results',
    'same',
    'study',
    'single',
    'system',
    'the',
    'this',
    'to',
    'investigate',
    'investigates',
    'performed',
    'presented',
    'presents',
    'uses',
    'using',
    'work',
}
_GENERIC_LEAD_TOKENS = {
    'author',
    'authors',
    'paper',
    'research',
    'results',
    'study',
    'this',
    'work',
}
_GENERIC_TAIL_TOKENS = {
    'single',
}


def _tokenize(text: object) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(str(text or '').lower())]


def _normalize_phrase(text: object) -> str:
    return ' '.join(_tokenize(text))


def _is_distinctive_phrase(text: object) -> bool:
    phrase = _normalize_phrase(text)
    if not phrase:
        return False
    tokens = phrase.split()
    if not tokens:
        return False
    non_generic = [token for token in tokens if token not in _GENERIC_TOKENS]
    if not non_generic:
        return False
    if tokens[0] in _GENERIC_LEAD_TOKENS and len(non_generic) < 2:
        return False
    if tokens[-1] in _GENERIC_TAIL_TOKENS:
        return False
    if len(tokens) >= 2 and tokens[0] in _GENERIC_TOKENS and len(non_generic) == 1:
        return False
    return True


def _signal_phrases(value: object) -> list[str]:
    normalized_values: list[str] = []
    for raw in value or []:
        if isinstance(raw, dict):
            candidate = raw.get('normalized') or raw.get('surface') or ''
        else:
            candidate = raw
        phrase = _normalize_phrase(candidate)
        if _is_distinctive_phrase(phrase):
            normalized_values.append(phrase)
    phrases: list[str] = []
    if normalized_values and all(len(item.split()) <= 2 for item in normalized_values):
        for index in range(len(normalized_values) - 1):
            left_tokens = normalized_values[index].split()
            right_tokens = normalized_values[index + 1].split()
            overlap_size = 0
            max_overlap = min(len(left_tokens), len(right_tokens))
            for size in range(max_overlap, 0, -1):
                if left_tokens[-size:] == right_tokens[:size]:
                    overlap_size = size
                    break
            merged = ' '.join(left_tokens + right_tokens[overlap_size:])
            if _is_distinctive_phrase(merged):
                phrases.append(merged)
    phrases.extend(normalized_values)
    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        if phrase in seen:
            continue
        seen.add(phrase)
        deduped.append(phrase)
    return deduped


def _candidate_phrases(summary: str) -> list[str]:
    words = _tokenize(summary)
    phrases: list[str] = []
    for size in (3, 2):
        for index in range(len(words) - size + 1):
            window = words[index : index + size]
            if all(token in _GENERIC_TOKENS for token in window):
                continue
            phrase = ' '.join(window)
            if _is_distinctive_phrase(phrase):
                phrases.append(phrase)
    return phrases


def label_community(core_members: list[dict[str, Any]], evidence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    signal_counter: Counter[str] = Counter()
    phrase_counter: Counter[str] = Counter()
    word_counter: Counter[str] = Counter()

    for row in core_members:
        for key, weight in (
            ('method_tokens', 4),
            ('object_tokens', 3),
            ('metric_tokens', 2),
            ('condition_tokens', 2),
            ('comparator_tokens', 2),
            ('resource_tokens', 1),
            ('limitation_tokens', 1),
        ):
            for phrase in _signal_phrases(row.get(key) or []):
                signal_counter[phrase] += weight
        summary = str(row.get('summary') or '').strip()
        for phrase in _candidate_phrases(summary):
            phrase_counter[phrase] += 1
        for token in _tokenize(summary):
            if token not in _GENERIC_TOKENS:
                word_counter[token] += 1

    title = ''
    if signal_counter:
        title = max(signal_counter.items(), key=lambda item: (item[1], len(item[0]), item[0]))[0]
    elif phrase_counter:
        title = max(phrase_counter.items(), key=lambda item: (item[1], len(item[0]), item[0]))[0]
    elif word_counter:
        title = ' '.join(token for token, _count in word_counter.most_common(3))
    else:
        title = 'cross-paper logic pattern'

    keywords = [title]
    for phrase, _count in signal_counter.most_common(5):
        if phrase not in keywords:
            keywords.append(phrase)
    for token, _count in word_counter.most_common(5):
        if token not in keywords:
            keywords.append(token)

    summary = f'Cross-paper research moves centered on {title}.'
    if evidence_rows:
        summary = f'{summary} Supported by {len(evidence_rows)} related evidence anchors.'

    return {
        'title': title,
        'summary': summary,
        'keywords': keywords[:5],
    }
