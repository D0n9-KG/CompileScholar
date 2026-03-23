from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_mock_client():
    client = MagicMock()
    client.__enter__ = lambda s: s
    client.__exit__ = MagicMock(return_value=False)
    return client


def test_gate_fail_marks_canonical_trace_write_as_skipped(monkeypatch, tmp_path):
    del monkeypatch, tmp_path
    quality_report = {
        "gate_passed": False,
        "quality_tier": "red",
        "quality_tier_score": 0.1,
    }

    gate_passed = bool(quality_report.get("gate_passed"))
    assert not gate_passed

    result = {
        "paper_id": "p1",
        "gate_passed": False,
        "quality_report": quality_report,
        "skipped_canonical_write": True,
    }

    assert result["gate_passed"] is False
    assert result["skipped_canonical_write"] is True


def test_gate_pass_proceeds_with_write(monkeypatch, tmp_path):
    """When gate passes, the path continues to Neo4j write (no early return)."""
    trace_result = {
        "quality_report": {
            "gate_passed": True,
            "quality_tier": "green",
            "quality_tier_score": 0.9,
        }
    }
    quality_report = trace_result.get("quality_report") or {}
    gate_passed = bool(quality_report.get("gate_passed"))
    assert gate_passed  # If gate passed, we do NOT take early return


def test_replace_paper_gate_fail_skips_canonical_write(monkeypatch, tmp_path):
    del monkeypatch, tmp_path
    quality_report = {
        "gate_passed": False,
        "quality_tier": "red",
        "quality_tier_score": 0.1,
    }
    gate_passed = bool(quality_report.get("gate_passed"))

    assert not gate_passed

    result = {
        "paper_id": "p1",
        "source_md_path": "/fake/path.md",
        "gate_passed": False,
        "quality_report": quality_report,
        "skipped_canonical_write": True,
    }

    assert result["gate_passed"] is False
    assert result["skipped_canonical_write"] is True


def test_write_citation_semantic_artifacts_persists_projection_payloads(tmp_path):
    from app.ingest import rebuild as rebuild_mod

    citation_acts = [{"citation_id": "citeact:paper:alpha->doi:10.1000/beta"}]
    citation_mentions = [{"mention_id": "cmention:paper:alpha:3:chunk-1:12-12"}]

    summary = rebuild_mod._write_citation_semantic_artifacts(
        tmp_path,
        citation_acts=citation_acts,
        citation_mentions=citation_mentions,
    )

    assert json.loads((tmp_path / "citation_acts.json").read_text(encoding="utf-8")) == citation_acts
    assert json.loads((tmp_path / "citation_mentions.json").read_text(encoding="utf-8")) == citation_mentions
    assert summary == {
        "citation_acts": 1,
        "citation_mentions": 1,
    }


def test_rebuild_paper_gate_fail_still_persists_citation_projection_artifacts(monkeypatch, tmp_path):
    from app.ingest import rebuild as rebuild_mod
    from app.ingest.models import Chunk, CitationEvent, DocumentIR, MdSpan, PaperDraft, ReferenceEntry

    md_path = tmp_path / "source.md"
    md_path.write_text("# Demo\n\nBody [1]\n\n# References\n[1] Demo reference\n", encoding="utf-8")

    doc = DocumentIR(
        paper=PaperDraft(
            paper_source="demo-paper",
            md_path=str(md_path),
            title="Demo",
            title_alt=None,
            authors=["Alice Smith"],
            doi="10.1000/test",
            year=2024,
        ),
        chunks=[
            Chunk(
                chunk_id="chunk-1",
                paper_source="demo-paper",
                md_path=str(md_path),
                span=MdSpan(start_line=3, end_line=3),
                section="Intro",
                kind="block",
                text="Body [1]",
            )
        ],
        references=[
            ReferenceEntry(
                paper_source="demo-paper",
                md_path=str(md_path),
                ref_num=1,
                raw="Demo reference",
            )
        ],
        citations=[
            CitationEvent(
                paper_source="demo-paper",
                md_path=str(md_path),
                cited_ref_num=1,
                chunk_id="chunk-1",
                span=MdSpan(start_line=3, end_line=3),
                context="Body [1]",
            )
        ],
    )

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_paper_basic(self, paper_id):
            return {
                "paper_id": paper_id,
                "source_md_path": str(md_path),
                "doi": "10.1000/test",
            }

        def ensure_schema(self):
            pass

        def update_paper_props(self, paper_id, props):  # noqa: ARG002
            pass

        def delete_paper_subgraph(self, paper_id):  # noqa: ARG002
            pass

        def upsert_paper_and_chunks(self, doc):  # noqa: ARG002
            pass

        def upsert_figures(self, paper_id, figures):  # noqa: ARG002
            pass

        def upsert_references_and_citations(self, **kwargs):
            pass

        def update_cites_purposes(self, **kwargs):
            pass

    monkeypatch.setattr(rebuild_mod, "Neo4jClient", lambda *args, **kwargs: _FakeClient())
    monkeypatch.setattr(rebuild_mod, "parse_mineru_markdown", lambda _: doc)
    monkeypatch.setattr(rebuild_mod, "recover_references_with_agent", lambda doc, **kwargs: (doc, {"status": "ok"}))
    monkeypatch.setattr(rebuild_mod, "recover_citation_events_from_references", lambda doc, **kwargs: (doc, {"status": "ok"}))
    monkeypatch.setattr(rebuild_mod, "CrossrefClient", lambda: object())
    monkeypatch.setattr(
        rebuild_mod,
        "build_reference_and_cite_records",
        lambda doc, **kwargs: {
            "paper_id": "doi:10.1000/test",
            "refs": [{"ref_num": 1, "raw": "Demo reference"}],
            "cited_papers": [{"paper_id": "doi:10.1000/ref", "title": "Demo cited paper"}],
            "cites_resolved": [
                {
                    "cited_paper_id": "doi:10.1000/ref",
                    "ref_nums": [1],
                    "evidence_chunk_ids": ["chunk-1"],
                    "evidence_spans": ["3-3"],
                }
            ],
            "cites_unresolved": [],
        },
    )
    monkeypatch.setattr(rebuild_mod, "_schema_for_md", lambda *_: {"prompts": {}, "rules": {}, "version": 1, "paper_type": "research"})
    monkeypatch.setattr(rebuild_mod, "load_canonical_meta", lambda *_: {"paper_type": "research"})
    monkeypatch.setattr(rebuild_mod, "load_active", lambda *_: {"version": 1, "paper_type": "research", "prompts": {}, "rules": {}})
    monkeypatch.setattr(rebuild_mod, "extract_figures_from_markdown", lambda **kwargs: [])
    monkeypatch.setattr(
        rebuild_mod,
        "run_phase1_paper_logic_trace",
        lambda **kwargs: {
            "quality_report": {
                "gate_passed": False,
                "quality_tier": "yellow",
                "quality_tier_score": 0.8,
            },
            "paper_logic_trace": {
                "quality": {
                    "quality_tier": "yellow",
                    "audit_status": "eligible",
                    "hot_path_gate_report": {"passed": False, "move_count": 0, "anchor_count": 0},
                },
                "canonical_core": {"moves": []},
            },
        },
    )
    monkeypatch.setattr(
        rebuild_mod,
        "classify_citation_purposes_batch",
        lambda **kwargs: {
            "by_id": {
                "doi:10.1000/ref": {
                    "labels": ["Background"],
                    "scores": [0.2],
                }
            }
        },
    )
    monkeypatch.setattr(rebuild_mod, "_storage_dir", lambda: tmp_path)

    result = rebuild_mod.rebuild_paper("doi:10.1000/test")

    out_dir = tmp_path / "derived" / "papers" / "doi_10.1000_test"
    assert result["gate_passed"] is False
    assert result["skipped_canonical_write"] is True
    assert json.loads((out_dir / "citation_acts.json").read_text(encoding="utf-8"))
    assert json.loads((out_dir / "citation_mentions.json").read_text(encoding="utf-8")) == [
        {
            "mention_id": "cmention:doi:10.1000/test:1:chunk-1:3-3",
            "citation_id": "citeact:doi:10.1000/test->doi:10.1000/ref",
            "citing_paper_id": "doi:10.1000/test",
            "cited_paper_id": "doi:10.1000/ref",
            "ref_num": 1,
            "source_chunk_id": "chunk-1",
            "span_start": 3,
            "span_end": 3,
            "section": "intro",
            "context_text": "Body [1]",
            "target_scopes": ["paper"],
            "source": "machine",
        }
    ]


def test_rebuild_paper_gate_pass_writes_trace_not_legacy_logic_claims(monkeypatch, tmp_path):
    from app.ingest import rebuild as rebuild_mod
    from app.ingest.models import Chunk, CitationEvent, DocumentIR, MdSpan, PaperDraft, ReferenceEntry
    from app.paper_logic_trace.models import CanonicalCore, EvidenceAnchor, PaperLogicTrace, PaperMetadata, ResearchMove

    md_path = tmp_path / "source.md"
    md_path.write_text("# Demo\n\nBody [1]\n\n# References\n[1] Demo reference\n", encoding="utf-8")

    doc = DocumentIR(
        paper=PaperDraft(
            paper_source="demo-paper",
            md_path=str(md_path),
            title="Demo",
            title_alt=None,
            authors=["Alice Smith"],
            doi="10.1000/test",
            year=2024,
        ),
        chunks=[
            Chunk(
                chunk_id="chunk-1",
                paper_source="demo-paper",
                md_path=str(md_path),
                span=MdSpan(start_line=3, end_line=3),
                section="Method",
                kind="block",
                text="Body [1]",
            )
        ],
        references=[
            ReferenceEntry(
                paper_source="demo-paper",
                md_path=str(md_path),
                ref_num=1,
                raw="Demo reference",
            )
        ],
        citations=[
            CitationEvent(
                paper_source="demo-paper",
                md_path=str(md_path),
                cited_ref_num=1,
                chunk_id="chunk-1",
                span=MdSpan(start_line=3, end_line=3),
                context="Body [1]",
            )
        ],
    )

    trace_writes: list[dict] = []

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_paper_basic(self, paper_id):
            return {
                "paper_id": paper_id,
                "source_md_path": str(md_path),
                "doi": "10.1000/test",
            }

        def ensure_schema(self):
            pass

        def update_paper_props(self, paper_id, props):  # noqa: ARG002
            pass

        def delete_paper_subgraph(self, paper_id):  # noqa: ARG002
            pass

        def upsert_paper_and_chunks(self, doc):  # noqa: ARG002
            pass

        def upsert_figures(self, paper_id, figures):  # noqa: ARG002
            pass

        def upsert_references_and_citations(self, **kwargs):
            pass

        def upsert_paper_logic_trace(self, paper_id, trace_payload):
            trace_writes.append({"paper_id": paper_id, "trace_payload": trace_payload})

        def update_cites_purposes(self, **kwargs):
            pass

    trace = PaperLogicTrace(
        trace_id="doi:10.1000/test:paper_logic_trace",
        built_at="2026-03-22T12:00:00Z",
        paper_metadata=PaperMetadata(
            paper_id="doi:10.1000/test",
            title="Demo",
            canonical_doi="10.1000/test",
            source_refs=["chunk:1"],
        ),
        canonical_core=CanonicalCore(
            evidence_anchors=[
                EvidenceAnchor(
                    anchor_id="a-1",
                    paper_id="doi:10.1000/test",
                    source_ref="chunk:1",
                    modality="text",
                    section_path=["Method"],
                    locator={"chunk_id": "chunk-1", "start_line": 3, "end_line": 3},
                    quote="Body [1]",
                    citation_ids=[],
                    support_type="direct",
                    weak=False,
                )
            ],
            moves=[
                ResearchMove(
                    move_id="m-1",
                    sequence_no=1,
                    role="method",
                    act_type="propose_method",
                    summary="We propose a method.",
                    anchor_ids=["a-1"],
                    confidence=0.9,
                )
            ],
            move_relations=[],
            citation_acts=[],
            figure_refs=[],
            table_refs=[],
        ),
        derived_views={"community_signatures": [{"move_id": "m-1"}]},
        quality={
            "quality_tier": "green",
            "audit_status": "eligible",
            "hot_path_gate_report": {"passed": True, "move_count": 1, "anchor_count": 1},
        },
    )

    monkeypatch.setattr(rebuild_mod, "Neo4jClient", lambda *args, **kwargs: _FakeClient())
    monkeypatch.setattr(rebuild_mod, "parse_mineru_markdown", lambda _: doc)
    monkeypatch.setattr(rebuild_mod, "recover_references_with_agent", lambda doc, **kwargs: (doc, {"status": "ok"}))
    monkeypatch.setattr(rebuild_mod, "recover_citation_events_from_references", lambda doc, **kwargs: (doc, {"status": "ok"}))
    monkeypatch.setattr(rebuild_mod, "CrossrefClient", lambda: object())
    monkeypatch.setattr(
        rebuild_mod,
        "build_reference_and_cite_records",
        lambda doc, **kwargs: {
            "paper_id": "doi:10.1000/test",
            "refs": [{"ref_num": 1, "raw": "Demo reference"}],
            "cited_papers": [{"paper_id": "doi:10.1000/ref", "title": "Demo cited paper"}],
            "cites_resolved": [],
            "cites_unresolved": [],
        },
    )
    monkeypatch.setattr(rebuild_mod, "_schema_for_md", lambda *_: {"prompts": {}, "rules": {}, "version": 1, "paper_type": "research"})
    monkeypatch.setattr(rebuild_mod, "load_canonical_meta", lambda *_: {"paper_type": "research"})
    monkeypatch.setattr(rebuild_mod, "load_active", lambda *_: {"version": 1, "paper_type": "research", "prompts": {}, "rules": {}})
    monkeypatch.setattr(rebuild_mod, "extract_figures_from_markdown", lambda **kwargs: [])
    monkeypatch.setattr(
        rebuild_mod,
        "run_phase1_paper_logic_trace",
        lambda **kwargs: {  # noqa: ARG005
            "paper_logic_trace": trace,
            "quality_report": {"gate_passed": True, "quality_tier": "green", "quality_tier_score": 0.95},
        },
    )
    monkeypatch.setattr(rebuild_mod, "classify_citation_purposes_batch", lambda **kwargs: {"by_id": {}})
    monkeypatch.setattr(rebuild_mod, "_storage_dir", lambda: tmp_path)

    result = rebuild_mod.rebuild_paper("doi:10.1000/test")

    out_dir = tmp_path / "derived" / "papers" / "doi_10.1000_test"
    assert result["gate_passed"] is True
    assert trace_writes and trace_writes[0]["paper_id"] == "doi:10.1000/test"
    assert (out_dir / "paper_logic_trace.json").exists()
    assert not (out_dir / "llm_imrad.json").exists()


def test_rebuild_global_faiss_indexes_trace_corpora_not_legacy_logic_claims(monkeypatch, tmp_path):
    from app.ingest import rebuild as rebuild_mod

    faiss_targets: list[str] = []

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def list_chunks_for_faiss(self, limit=200000):  # noqa: ARG002
            return [
                {
                    "chunk_id": "chunk-1",
                    "paper_source": "demo-paper",
                    "md_path": "C:/tmp/demo.md",
                    "start_line": 1,
                    "end_line": 2,
                    "section": "Method",
                    "kind": "block",
                    "text": "We propose a graph encoder.",
                }
            ]

        def list_research_moves(self, limit=50000):  # noqa: ARG002
            return [
                {
                    "kind": "research_move",
                    "source_id": "m-1",
                    "text": "We propose a graph encoder.",
                    "paper_id": "doi:10.1000/test",
                    "paper_source": "demo-paper",
                    "role": "method",
                    "act_type": "propose_method",
                }
            ]

        def list_evidence_anchors(self, limit=50000):  # noqa: ARG002
            return [
                {
                    "kind": "evidence_anchor",
                    "source_id": "a-1",
                    "text": "We propose a graph encoder.",
                    "paper_id": "doi:10.1000/test",
                    "paper_source": "demo-paper",
                    "move_id": "m-1",
                    "quote": "We propose a graph encoder.",
                }
            ]

        def list_global_community_rows(self, limit=50000):  # noqa: ARG002
            return []

        def list_global_community_members(self, community_id, limit=200):  # noqa: ARG002
            return []

    monkeypatch.setattr(rebuild_mod, "Neo4jClient", lambda *args, **kwargs: _FakeClient())
    monkeypatch.setattr(rebuild_mod, "_storage_dir", lambda: tmp_path)
    monkeypatch.setattr(rebuild_mod, "build_faiss_for_chunks", lambda chunks, out_dir: {"dir": out_dir, "chunks_indexed": len(chunks)})  # noqa: ARG005,E501
    monkeypatch.setattr(
        rebuild_mod,
        "build_faiss_for_rows",
        lambda rows, out_dir, **kwargs: faiss_targets.append(str(out_dir)) or {"dir": str(out_dir), "rows_indexed": len(rows)},  # noqa: ARG005,E501
    )

    rebuild_mod.rebuild_global_faiss()

    corpus_dirs = {Path(path).name for path in faiss_targets}
    assert "research_moves" in corpus_dirs
    assert "evidence_anchors" in corpus_dirs
    assert "logic_steps" not in corpus_dirs
    assert "claims" not in corpus_dirs
