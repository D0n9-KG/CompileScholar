from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from app.crossref.client import CrossrefClient, CrossrefWork
from app.ingest.models import DocumentIR

_PIPE_SECTION_TITLE_RE = re.compile(r'^\s*(?:section\s+)?\d+(?:\.\d+)*\s*[|:：-]\s+\S', re.IGNORECASE)
_LATEX_TITLE_NOISE_RE = re.compile(r'(?:\\(?:mathrm|text|begin|end)\b|\$)')
_SECTION_TITLE_RE = re.compile(r'^\s*(?:section\s+)?\d+(?:\.\d+)*\.?\s+(?![A-Za-z]\b)\S')
_CJK_SECTION_TITLE_RE = re.compile(r'^\s*\d+(?:\.\d+){1,4}(?:\s*[《（(【\[]|[\u4e00-\u9fff])')
_TITLE_NOISE_RE = re.compile(
    r'^\s*(?:abstract|appendix|conclusion|discussion|introduction|references?|results?)\s*$',
    re.IGNORECASE,
)
_PAPER_SOURCE_PREFIX_RE = re.compile(r'^\s*\d+[_\-\s]+')
_SPACE_RE = re.compile(r'\s+')
_BROKEN_AUTHOR_RE = re.compile(r'[\${}\^]')
_NON_NAME_RE = re.compile(r'^[\W\d_]+$')
_CJK_NAME_RE = re.compile(r'^[\u4e00-\u9fff·]{2,6}$')
_SENTENCE_LIKE_AUTHOR_CUES = (
    'abstract',
    'analysis',
    'discusses',
    'introduction',
    'purpose',
    'results',
    'summary',
    '提要',
    '摘要',
    '介绍',
    '分析',
    '实施',
    '意义',
    '标准',
)
_CROSSREF_EXACT_TITLE_CONFIDENCE_THRESHOLD = 0.35
_CROSSREF_STRONG_MATCH_CONFIDENCE_THRESHOLD = 0.75
_CROSSREF_STRONG_MATCH_TITLE_SIMILARITY = 0.8


def _normalize_space(value: object) -> str:
    return _SPACE_RE.sub(' ', str(value or '').strip())


def _normalize_doi(value: str | None) -> str | None:
    doi = _normalize_space(value).lower().rstrip(').,;')
    return doi or None


def _is_textual_char(char: str) -> bool:
    return char.isalnum() or '\u4e00' <= char <= '\u9fff'


def _normalize_similarity_text(value: str | None) -> str:
    pieces: list[str] = []
    for char in str(value or ''):
        pieces.append(char.lower() if _is_textual_char(char) else ' ')
    return _normalize_space(''.join(pieces))


def _title_similarity(left: str | None, right: str | None) -> float:
    normalized_left = _normalize_similarity_text(left)
    normalized_right = _normalize_similarity_text(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    return SequenceMatcher(a=normalized_left, b=normalized_right).ratio()


def _is_crossref_title_match_credible(
    *,
    query_title: str | None,
    selected_title: str | None,
    confidence: float,
) -> bool:
    normalized_query = _normalize_similarity_text(query_title)
    normalized_selected = _normalize_similarity_text(selected_title)
    if not normalized_query or not normalized_selected:
        return False
    if normalized_query == normalized_selected:
        return confidence >= _CROSSREF_EXACT_TITLE_CONFIDENCE_THRESHOLD
    similarity = _title_similarity(normalized_query, normalized_selected)
    return (
        confidence >= _CROSSREF_STRONG_MATCH_CONFIDENCE_THRESHOLD
        and similarity >= _CROSSREF_STRONG_MATCH_TITLE_SIMILARITY
    )


def _looks_suspicious_title(title: str | None) -> bool:
    clean = _normalize_space(title)
    if not clean:
        return True
    if len(clean) < 6:
        return True
    if _SECTION_TITLE_RE.match(clean):
        return True
    if _CJK_SECTION_TITLE_RE.match(clean):
        return True
    if _PIPE_SECTION_TITLE_RE.match(clean):
        return True
    if clean.count('$') >= 2 or _LATEX_TITLE_NOISE_RE.search(clean):
        return True
    if _TITLE_NOISE_RE.match(clean):
        return True
    return False


def _sanitize_title_alt(primary_title: str | None, title_alt: str | None) -> str | None:
    clean_alt = _normalize_space(title_alt)
    if not clean_alt:
        return None
    if clean_alt == _normalize_space(primary_title):
        return None
    if _looks_suspicious_title(clean_alt):
        return None
    return clean_alt


def _looks_like_sentence_author(name: str | None) -> bool:
    clean = _normalize_space(name)
    if not clean:
        return True
    lowered = clean.lower()
    if any(cue in lowered for cue in _SENTENCE_LIKE_AUTHOR_CUES):
        return True
    if any(mark in clean for mark in ('。', '；', '：', '！', '？')):
        return True
    if len(clean) >= 24:
        return True
    if len(clean.split()) >= 6:
        return True
    cjk_runs = re.findall(r'[\u4e00-\u9fff]+', clean)
    if any(len(run) >= 8 for run in cjk_runs):
        return True
    return False


def _looks_too_short_for_author(name: str | None) -> bool:
    clean = _normalize_space(name)
    if not clean:
        return True
    if _CJK_NAME_RE.match(clean):
        return False
    return len(clean) <= 2


def _looks_suspicious_authors(authors: list[str] | None) -> bool:
    names = [_normalize_space(name) for name in (authors or []) if _normalize_space(name)]
    if not names:
        return True
    suspicious = 0
    for name in names:
        if _looks_like_sentence_author(name):
            suspicious += 1
            continue
        if _BROKEN_AUTHOR_RE.search(name):
            suspicious += 1
            continue
        if _NON_NAME_RE.match(name):
            suspicious += 1
            continue
        if _looks_too_short_for_author(name):
            suspicious += 1
    return suspicious > 0


def _looks_suspicious_year(year: int | None) -> bool:
    if year is None:
        return True
    return year < 1800 or year > datetime.now().year + 1


def _paper_source_query(paper_source: str | None) -> str:
    clean = _normalize_space(paper_source)
    clean = _PAPER_SOURCE_PREFIX_RE.sub('', clean)
    clean = clean.replace('_', ' ').replace('-', ' ')
    return _normalize_space(clean)


def _title_query_candidates(doc: DocumentIR) -> list[str]:
    candidates = [
        _normalize_space(doc.paper.title),
        _normalize_space(doc.paper.title_alt),
        _paper_source_query(doc.paper.paper_source),
    ]
    unique: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        lowered = item.lower()
        if not item or lowered in seen:
            continue
        seen.add(lowered)
        unique.append(item)

    clean = [item for item in unique if not _looks_suspicious_title(item)]
    return clean or unique


def _needs_title_search_enrichment(doc: DocumentIR) -> bool:
    return (
        _looks_suspicious_title(doc.paper.title)
        or _looks_suspicious_authors(doc.paper.authors)
        or _looks_suspicious_year(doc.paper.year)
    )


def _resolve_title_query(doc: DocumentIR) -> str | None:
    candidates = _title_query_candidates(doc)
    return candidates[0] if candidates else None


def _repair_local_metadata(doc: DocumentIR) -> tuple[DocumentIR, list[str]]:
    paper = doc.paper
    changed_fields: list[str] = []
    new_title = paper.title
    new_title_alt = paper.title_alt
    new_authors = list(paper.authors or [])

    fallback_title_candidates = [
        _normalize_space(paper.title_alt),
        _paper_source_query(paper.paper_source),
    ]
    fallback_title = next(
        (
            candidate
            for candidate in fallback_title_candidates
            if candidate and not _looks_suspicious_title(candidate) and candidate != _normalize_space(paper.title)
        ),
        None,
    )
    if _looks_suspicious_title(paper.title) and fallback_title:
        previous_title = _normalize_space(paper.title)
        new_title = fallback_title
        if previous_title and previous_title != fallback_title:
            new_title_alt = previous_title
        changed_fields.append('title')
        if new_title_alt != paper.title_alt:
            changed_fields.append('title_alt')

    if _looks_suspicious_authors(paper.authors):
        filtered_authors = [
            name
            for name in new_authors
            if not _looks_like_sentence_author(name)
            and not _BROKEN_AUTHOR_RE.search(name)
            and not _NON_NAME_RE.match(name)
            and not _looks_too_short_for_author(name)
        ]
        if filtered_authors != new_authors:
            new_authors = filtered_authors
            changed_fields.append('authors')

    sanitized_title_alt = _sanitize_title_alt(new_title, new_title_alt)
    if sanitized_title_alt != new_title_alt:
        new_title_alt = sanitized_title_alt
        changed_fields.append('title_alt')

    if not changed_fields:
        return doc, []

    updated_paper = replace(
        paper,
        title=new_title,
        title_alt=new_title_alt,
        authors=new_authors,
    )
    return replace(doc, paper=updated_paper), sorted(set(changed_fields))


def repair_local_metadata(doc: DocumentIR) -> tuple[DocumentIR, list[str]]:
    return _repair_local_metadata(doc)


def _attach_metadata_enrichment_report(doc: DocumentIR, report: dict[str, Any]) -> DocumentIR:
    if not (bool(report.get('used_crossref')) or bool(report.get('local_fallback_used'))):
        return doc
    return replace(doc, paper=replace(doc.paper, metadata_enrichment=dict(report)))


def _apply_crossref_work(
    doc: DocumentIR,
    *,
    work: CrossrefWork,
    mode: str,
) -> tuple[DocumentIR, list[str]]:
    paper = doc.paper
    changed_fields: list[str] = []
    new_title = paper.title
    new_title_alt = paper.title_alt
    new_authors = list(paper.authors or [])
    new_year = paper.year
    new_doi = paper.doi
    new_venue = getattr(paper, 'venue', None)

    if mode == 'doi_lookup':
        if work.title and work.title != paper.title:
            previous_title = _normalize_space(paper.title)
            new_title = work.title
            if previous_title and previous_title != work.title:
                new_title_alt = previous_title
            changed_fields.append('title')
            if new_title_alt != paper.title_alt:
                changed_fields.append('title_alt')
        if work.authors and work.authors != new_authors:
            new_authors = list(work.authors)
            changed_fields.append('authors')
        if work.year and work.year != new_year:
            new_year = work.year
            changed_fields.append('year')
    else:
        if work.title and _looks_suspicious_title(paper.title) and work.title != paper.title:
            previous_title = _normalize_space(paper.title)
            new_title = work.title
            if previous_title and previous_title != work.title:
                new_title_alt = previous_title
            changed_fields.append('title')
            if new_title_alt != paper.title_alt:
                changed_fields.append('title_alt')
        if work.authors and _looks_suspicious_authors(paper.authors) and list(work.authors) != new_authors:
            new_authors = list(work.authors)
            changed_fields.append('authors')
        if work.year and _looks_suspicious_year(paper.year) and work.year != new_year:
            new_year = work.year
            changed_fields.append('year')

    normalized_doi = _normalize_doi(work.doi)
    if normalized_doi and normalized_doi != _normalize_doi(new_doi):
        new_doi = normalized_doi
        changed_fields.append('doi')
    if work.venue and work.venue != new_venue:
        new_venue = work.venue
        changed_fields.append('venue')

    sanitized_title_alt = _sanitize_title_alt(new_title, new_title_alt)
    if sanitized_title_alt != new_title_alt:
        new_title_alt = sanitized_title_alt
        changed_fields.append('title_alt')

    updated_paper = replace(
        paper,
        title=new_title,
        title_alt=new_title_alt,
        authors=new_authors,
        year=new_year,
        doi=new_doi,
        venue=new_venue,
    )
    return replace(doc, paper=updated_paper), sorted(set(changed_fields))


def _is_crossref_work(value: object) -> bool:
    return isinstance(value, CrossrefWork)


def enrich_document_metadata(
    doc: DocumentIR,
    *,
    crossref: CrossrefClient | None = None,
    confidence_threshold: float = 0.55,
) -> tuple[DocumentIR, dict[str, Any]]:
    report: dict[str, Any] = {
        'mode': 'skipped_clean_metadata',
        'used_crossref': False,
        'query': None,
        'confidence': None,
        'changed_fields': [],
        'local_fallback_used': False,
        'local_fallback_changed_fields': [],
    }

    client = crossref or CrossrefClient()
    doi = _normalize_doi(doc.paper.doi)
    if doi:
        if not hasattr(client, 'get_work_by_doi'):
            report['mode'] = 'doi_lookup_unavailable'
            repaired_doc, fallback_changed_fields = _repair_local_metadata(doc)
            if fallback_changed_fields:
                report['local_fallback_used'] = True
                report['local_fallback_changed_fields'] = fallback_changed_fields
            return _attach_metadata_enrichment_report(repaired_doc, report), report
        work = client.get_work_by_doi(doi)
        if not _is_crossref_work(work):
            report['mode'] = 'doi_lookup_failed'
            repaired_doc, fallback_changed_fields = _repair_local_metadata(doc)
            if fallback_changed_fields:
                report['local_fallback_used'] = True
                report['local_fallback_changed_fields'] = fallback_changed_fields
            return _attach_metadata_enrichment_report(repaired_doc, report), report
        enriched, changed_fields = _apply_crossref_work(doc, work=work, mode='doi_lookup')
        report.update(
            {
                'mode': 'doi_lookup',
                'used_crossref': True,
                'query': doi,
                'changed_fields': changed_fields,
            }
        )
        return _attach_metadata_enrichment_report(enriched, report), report

    if not _needs_title_search_enrichment(doc):
        repaired_doc, fallback_changed_fields = _repair_local_metadata(doc)
        if fallback_changed_fields:
            report['local_fallback_used'] = True
            report['local_fallback_changed_fields'] = fallback_changed_fields
        return _attach_metadata_enrichment_report(repaired_doc, report), report

    query = _resolve_title_query(doc)
    if not query:
        report['mode'] = 'skipped_no_query_candidate'
        repaired_doc, fallback_changed_fields = _repair_local_metadata(doc)
        if fallback_changed_fields:
            report['local_fallback_used'] = True
            report['local_fallback_changed_fields'] = fallback_changed_fields
        return _attach_metadata_enrichment_report(repaired_doc, report), report

    result = client.resolve_reference(query)
    confidence = float(result.confidence) if result else 0.0
    selected = result.selected if result else None
    report['query'] = query
    report['confidence'] = confidence
    if not _is_crossref_work(selected) or confidence < confidence_threshold:
        report['mode'] = 'skipped_low_confidence'
        repaired_doc, fallback_changed_fields = _repair_local_metadata(doc)
        if fallback_changed_fields:
            report['local_fallback_used'] = True
            report['local_fallback_changed_fields'] = fallback_changed_fields
        return _attach_metadata_enrichment_report(repaired_doc, report), report
    if not _is_crossref_title_match_credible(
        query_title=query,
        selected_title=selected.title,
        confidence=confidence,
    ):
        report['mode'] = 'skipped_unreliable_title_match'
        repaired_doc, fallback_changed_fields = _repair_local_metadata(doc)
        if fallback_changed_fields:
            report['local_fallback_used'] = True
            report['local_fallback_changed_fields'] = fallback_changed_fields
        return _attach_metadata_enrichment_report(repaired_doc, report), report

    enriched, changed_fields = _apply_crossref_work(doc, work=selected, mode='title_search')
    report.update(
        {
            'mode': 'title_search',
            'used_crossref': True,
            'changed_fields': changed_fields,
        }
    )
    return _attach_metadata_enrichment_report(enriched, report), report


__all__ = ['enrich_document_metadata', 'repair_local_metadata']
