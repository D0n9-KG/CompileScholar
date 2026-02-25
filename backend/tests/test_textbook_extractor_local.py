from app.ingest.textbook_extractor_local import extract_textbook_graph_local


def test_local_extractor_returns_stable_counts_on_fixed_fixture() -> None:
    md = """
# Chapter 1

Newton second law is a relation between force and acceleration.
Momentum is defined as mass times velocity.
Energy conservation is the principle that total energy remains constant.
""".strip()

    graph = extract_textbook_graph_local(md, chapter_id="tb:demo:ch001")

    assert len(graph["nodes"]) == 3
    assert len(graph["edges"]) == 2


def test_local_extractor_edges_contain_provenance_fields() -> None:
    md = """
# Chapter 1

Momentum is defined as mass times velocity.
Energy conservation is the principle that total energy remains constant.
""".strip()

    graph = extract_textbook_graph_local(md, chapter_id="tb:demo:ch001")
    assert graph["edges"], "fixture must yield at least one edge"

    for edge in graph["edges"]:
        assert "source_chunk_id" in edge
        assert "evidence_quote" in edge
        assert "char_start" in edge
        assert "char_end" in edge
        assert "confidence" in edge
