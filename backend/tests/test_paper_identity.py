from __future__ import annotations

from app.crossref.client import CrossrefResolveResult, CrossrefWork
from app.ingest.models import DocumentIR, PaperDraft
from app.ingest.paper_identity import resolve_document_identity


class _FakeCrossref:
    def __init__(self, result: CrossrefResolveResult | None) -> None:
        self._result = result
        self.queries: list[str] = []

    def resolve_reference(self, query: str):
        self.queries.append(query)
        return self._result


def _doc(*, title: str, authors: list[str] | None = None, doi: str | None = None) -> DocumentIR:
    return DocumentIR(
        paper=PaperDraft(
            paper_source='demo_paper',
            md_path='C:/tmp/demo_paper.md',
            title=title,
            title_alt=None,
            authors=list(authors or []),
            doi=doi,
            year=None,
        ),
        chunks=[],
        references=[],
        citations=[],
    )


def test_resolve_document_identity_rejects_low_similarity_title_crossref_match() -> None:
    doc = _doc(
        title='Propellants and Rocketry in the First Part of the 20th Century',
        authors=['SNPE', '75004 Paris', 'France'],
    )
    selected = CrossrefWork(
        doi='10.1039/9781782620969-00158',
        title='Special Topics in Rocketry',
        year=2016,
        venue='Solid Rocket Propellants: Science and Technology Challenges',
        authors=[],
        score=26.740997,
    )
    crossref = _FakeCrossref(
        CrossrefResolveResult(
            query=doc.paper.title or '',
            topk=[selected],
            selected=selected,
            confidence=0.26908895,
        )
    )

    identity = resolve_document_identity(doc, doi_strategy='title_crossref', crossref=crossref)

    assert identity.doi is None
    assert identity.doi_source == 'none'
    assert crossref.queries == ['Propellants and Rocketry in the First Part of the 20th Century']


def test_resolve_document_identity_accepts_high_confidence_exact_title_match() -> None:
    doc = _doc(
        title='Learning Constitutive Laws',
        authors=['Alice Smith', 'Bob Jones'],
    )
    selected = CrossrefWork(
        doi='10.1234/example-doi',
        title='Learning Constitutive Laws',
        year=2025,
        venue='Journal X',
        authors=['Alice Smith', 'Bob Jones'],
        score=88.0,
    )
    crossref = _FakeCrossref(
        CrossrefResolveResult(
            query=doc.paper.title or '',
            topk=[selected],
            selected=selected,
            confidence=0.88,
        )
    )

    identity = resolve_document_identity(doc, doi_strategy='title_crossref', crossref=crossref)

    assert identity.doi == '10.1234/example-doi'
    assert identity.doi_source == 'title_crossref'
