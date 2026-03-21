from __future__ import annotations


def test_obvious_review_papers_skip_llm_call(monkeypatch):  # noqa: ANN001, ANN201
    from app.llm import client as llm_client
    from app.llm.paper_type_classifier import classify_paper_type

    calls: list[tuple[str, str]] = []

    def _fake_call_text(system: str, user: str, **kwargs) -> str:  # noqa: ARG001
        calls.append((system, user))
        return "review"

    monkeypatch.setattr(llm_client, "call_text", _fake_call_text)

    result = classify_paper_type(
        title="A systematic review of graph neural networks for scientific reasoning",
        abstract="This review summarizes existing graph neural network methods and compares their trends.",
        section_headings=["Introduction", "Survey Scope", "Comparison of Prior Work"],
        meta_paper_type=None,
    )

    assert result == "review"
    assert calls == []


def test_ambiguous_papers_still_use_llm_call(monkeypatch):  # noqa: ANN001, ANN201
    from app.llm import client as llm_client
    from app.llm.paper_type_classifier import classify_paper_type

    calls: list[tuple[str, str]] = []

    def _fake_call_text(system: str, user: str, **kwargs) -> str:  # noqa: ARG001
        calls.append((system, user))
        return "research"

    monkeypatch.setattr(llm_client, "call_text", _fake_call_text)

    result = classify_paper_type(
        title="Structured reasoning with adaptive latent plans",
        abstract="We study a new approach and evaluate it on several benchmarks.",
        section_headings=["Introduction", "Method", "Experiments"],
        meta_paper_type=None,
    )

    assert result == "research"
    assert len(calls) == 1
