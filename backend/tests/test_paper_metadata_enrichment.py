from __future__ import annotations

from app.crossref.client import CrossrefResolveResult, CrossrefWork
from app.ingest.models import DocumentIR, PaperDraft
from app.ingest.paper_metadata_enrichment import enrich_document_metadata


class _FakeCrossref:
    def __init__(
        self,
        *,
        doi_work: CrossrefWork | None = None,
        title_result: CrossrefResolveResult | None = None,
    ) -> None:
        self._doi_work = doi_work
        self._title_result = title_result
        self.get_work_by_doi_calls: list[str] = []
        self.resolve_reference_calls: list[str] = []

    def get_work_by_doi(self, doi: str) -> CrossrefWork | None:
        self.get_work_by_doi_calls.append(doi)
        return self._doi_work

    def resolve_reference(self, query: str, topk: int = 5) -> CrossrefResolveResult | None:  # noqa: ARG002
        self.resolve_reference_calls.append(query)
        return self._title_result


def _doc(
    *,
    paper_source: str = '1234_learning_constitutive_laws',
    title: str | None = 'Learning Constitutive Laws',
    title_alt: str | None = None,
    authors: list[str] | None = None,
    doi: str | None = None,
    year: int | None = 2024,
) -> DocumentIR:
    return DocumentIR(
        paper=PaperDraft(
            paper_source=paper_source,
            md_path=f'C:/tmp/{paper_source}.md',
            title=title,
            title_alt=title_alt,
            authors=list(authors or []),
            doi=doi,
            year=year,
        ),
        chunks=[],
        references=[],
        citations=[],
    )


def test_enrich_document_metadata_uses_doi_lookup_to_fix_suspicious_fields() -> None:
    doc = _doc(
        paper_source='1992_论_VLW_状态方程',
        title='2.1 爆轰产物 LJ 势参数',
        authors=['龙新平 $^{1}$', '何碧 $^{1', '2}$'],
        doi='10.1000/example-doi',
        year=2023,
    )
    crossref = _FakeCrossref(
        doi_work=CrossrefWork(
            doi='10.1000/example-doi',
            title='论 VLW 状态方程',
            year=1992,
            venue='爆炸学报',
            authors=['龙新平', '何碧', '蒋小华', '吴雄'],
            score=None,
        )
    )

    enriched, report = enrich_document_metadata(doc, crossref=crossref)

    assert enriched.paper.title == '论 VLW 状态方程'
    assert enriched.paper.title_alt is None
    assert enriched.paper.authors == ['龙新平', '何碧', '蒋小华', '吴雄']
    assert enriched.paper.year == 1992
    assert enriched.paper.venue == '爆炸学报'
    assert report['used_crossref'] is True
    assert report['mode'] == 'doi_lookup'
    assert crossref.get_work_by_doi_calls == ['10.1000/example-doi']
    assert crossref.resolve_reference_calls == []


def test_enrich_document_metadata_uses_title_search_when_local_metadata_is_suspicious() -> None:
    doc = _doc(
        paper_source='992_Data-driven_computing_in_dynamics',
        title='2.1 LJ Parameters',
        title_alt='Data-driven computing in dynamics',
        authors=['Alice $^{1}$', 'Bob $^{2}$'],
        doi=None,
        year=None,
    )
    selected = CrossrefWork(
        doi='10.1016/j.jmps.2018.06.004',
        title='Data-driven computing in dynamics',
        year=2018,
        venue='Journal of the Mechanics and Physics of Solids',
        authors=['R. Ibanez', 'E. Abisset-Chavanne'],
        score=88.0,
    )
    crossref = _FakeCrossref(
        title_result=CrossrefResolveResult(
            query='Data-driven computing in dynamics',
            topk=[selected],
            selected=selected,
            confidence=0.88,
        )
    )

    enriched, report = enrich_document_metadata(doc, crossref=crossref, confidence_threshold=0.55)

    assert enriched.paper.title == 'Data-driven computing in dynamics'
    assert enriched.paper.title_alt is None
    assert enriched.paper.authors == ['R. Ibanez', 'E. Abisset-Chavanne']
    assert enriched.paper.year == 2018
    assert enriched.paper.doi == '10.1016/j.jmps.2018.06.004'
    assert report['used_crossref'] is True
    assert report['mode'] == 'title_search'
    assert crossref.resolve_reference_calls == ['Data-driven computing in dynamics']


def test_enrich_document_metadata_keeps_clean_local_metadata_without_title_search_override() -> None:
    doc = _doc(
        title='Learning Constitutive Laws',
        authors=['Alice Smith', 'Bob Jones'],
        doi=None,
        year=2024,
    )
    selected = CrossrefWork(
        doi='10.1234/other-paper',
        title='A Different Paper',
        year=2019,
        venue='Journal X',
        authors=['Other Author'],
        score=91.0,
    )
    crossref = _FakeCrossref(
        title_result=CrossrefResolveResult(
            query='Learning Constitutive Laws',
            topk=[selected],
            selected=selected,
            confidence=0.91,
        )
    )

    enriched, report = enrich_document_metadata(doc, crossref=crossref, confidence_threshold=0.55)

    assert enriched == doc
    assert report['used_crossref'] is False
    assert report['mode'] == 'skipped_clean_metadata'
    assert crossref.get_work_by_doi_calls == []
    assert crossref.resolve_reference_calls == []


def test_enrich_document_metadata_uses_local_title_fallback_when_title_search_match_is_unreliable() -> None:
    doc = _doc(
        paper_source='79_卧式双轴圆盘反应器功率特性研究',
        title='2.4 装料量及桨叶数量对搅拌功率的影响',
        authors=['陈忠辉', '王凯'],
        doi=None,
        year=2005,
    )
    selected = CrossrefWork(
        doi='10.3788/cjl201946.1001001',
        title='高功率光纤激光热光效应及模式不稳定阈值特性研究',
        year=2019,
        venue='Chinese Journal of Lasers',
        authors=['李学文', '于春雷'],
        score=82.0,
    )
    crossref = _FakeCrossref(
        title_result=CrossrefResolveResult(
            query='卧式双轴圆盘反应器功率特性研究',
            topk=[selected],
            selected=selected,
            confidence=0.5527897,
        )
    )

    enriched, report = enrich_document_metadata(doc, crossref=crossref, confidence_threshold=0.55)

    assert enriched.paper.title == '卧式双轴圆盘反应器功率特性研究'
    assert enriched.paper.title_alt is None
    assert enriched.paper.authors == ['陈忠辉', '王凯']
    assert report['used_crossref'] is False
    assert report['mode'] == 'skipped_unreliable_title_match'
    assert report['local_fallback_used'] is True
    assert sorted(report['local_fallback_changed_fields']) == ['title', 'title_alt']
    assert report['query'] == '卧式双轴圆盘反应器功率特性研究'
    assert crossref.resolve_reference_calls == ['卧式双轴圆盘反应器功率特性研究']


def test_enrich_document_metadata_prefers_clean_title_alt_over_generic_paper_source_for_local_repair() -> None:
    doc = _doc(
        paper_source='demo-paper',
        title='2.1. Non-isothermal elasto-visco-plastic behavior',
        title_alt='Data-Driven Computational Plasticity',
        authors=['Alice Smith'],
        doi=None,
        year=2024,
    )
    crossref = _FakeCrossref(title_result=None)

    enriched, report = enrich_document_metadata(doc, crossref=crossref, confidence_threshold=0.55)

    assert enriched.paper.title == 'Data-Driven Computational Plasticity'
    assert enriched.paper.title_alt is None
    assert report['used_crossref'] is False
    assert report['local_fallback_used'] is True
    assert sorted(report['local_fallback_changed_fields']) == ['title', 'title_alt']
    assert report['query'] == 'Data-Driven Computational Plasticity'


def test_enrich_document_metadata_uses_local_fallback_for_sentence_like_authors() -> None:
    doc = _doc(
        paper_source='248_《城镇污水处理厂污染物排放标准》浅释',
        title='1.2《污水综合排放标准》不适应污水处理厂建设管理需求',
        authors=[
            '提要 介绍了我国最新发布实施的《城镇污水处理厂污染物排放标准》(GB18918—2002)制定的目的、意义、原则及技术内容',
            '标准的适用范围、控制污染物的分类、标准值及制定依据',
            '标准的实施和环境效益分析等',
        ],
        doi=None,
        year=2002,
    )
    crossref = _FakeCrossref(title_result=None)

    enriched, report = enrich_document_metadata(doc, crossref=crossref, confidence_threshold=0.55)

    assert enriched.paper.title == '《城镇污水处理厂污染物排放标准》浅释'
    assert enriched.paper.title_alt is None
    assert enriched.paper.authors == []
    assert report['used_crossref'] is False
    assert report['mode'] == 'skipped_low_confidence'
    assert report['local_fallback_used'] is True
    assert sorted(report['local_fallback_changed_fields']) == ['authors', 'title', 'title_alt']
    assert report['query'] == '《城镇污水处理厂污染物排放标准》浅释'


def test_enrich_document_metadata_drops_suspicious_title_alt_when_primary_title_is_clean() -> None:
    doc = _doc(
        title='Research on performance of composite solid propellant based on a neural network model',
        title_alt='$\\mathrm{M = (err./t_test)^{*}100;}\\%$ 计算预测结果的相对误差',
        authors=['Alice Smith', 'Bob Jones'],
        doi=None,
        year=2024,
    )
    crossref = _FakeCrossref(title_result=None)

    enriched, report = enrich_document_metadata(doc, crossref=crossref, confidence_threshold=0.55)

    assert enriched.paper.title == 'Research on performance of composite solid propellant based on a neural network model'
    assert enriched.paper.title_alt is None
    assert report['used_crossref'] is False
    assert report['local_fallback_used'] is True
    assert report['local_fallback_changed_fields'] == ['title_alt']
    assert crossref.get_work_by_doi_calls == []
    assert crossref.resolve_reference_calls == []


def test_enrich_document_metadata_drops_front_matter_title_alt_when_primary_title_is_clean() -> None:
    for title_alt in (
        'Accepted Manuscript',
        'ARTICLES YOU MAY BE INTERESTED IN',
    ):
        doc = _doc(
            title='Adversarial Uncertainty Quantification in Physics-Informed Neural Networks',
            title_alt=title_alt,
            authors=['Alice Smith', 'Bob Jones'],
            doi=None,
            year=2024,
        )
        crossref = _FakeCrossref(title_result=None)

        enriched, report = enrich_document_metadata(doc, crossref=crossref, confidence_threshold=0.55)

        assert enriched.paper.title == 'Adversarial Uncertainty Quantification in Physics-Informed Neural Networks'
        assert enriched.paper.title_alt is None
        assert report['used_crossref'] is False
        assert report['local_fallback_used'] is True
        assert report['local_fallback_changed_fields'] == ['title_alt']
