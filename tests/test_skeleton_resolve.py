# -*- coding: utf-8 -*-
"""W2 P0-2 resolver: stage order, KB re-mapping of arXiv ids, ambiguity -> stub, extracted titles must be substrings."""
from __future__ import annotations

import pytest

from compilescholar.compile.skeleton.resolve import Resolver, entry_title_year
from compilescholar.sources.arxiv_snapshot import TitleIndex


def _snap(tmp_path, rows):
    p = tmp_path / "idx.tsv"
    from compilescholar.sources.arxiv_snapshot import norm
    p.write_text("".join(f"{norm(t)[:40]}\t{aid}\t{y}\t{sur}\t{norm(t)}\n" for t, aid, y, sur in rows), encoding="utf-8")
    return TitleIndex(p)


KB = {"p1": {"title": "Graph Attention Networks", "arxiv_id": "1710.10903"},
      "p2": {"title": "A unique kb-only paper title about sparse coding", "arxiv_id": None}}


def test_explicit_ids_and_kb_remap(tmp_path):
    r = Resolver(KB, _snap(tmp_path, []))
    assert r.resolve({"raw": "x", "arxiv": "1710.10903", "doi": None}).paper == "kb:p1"
    assert r.resolve({"raw": "x", "arxiv": "2001.00001", "doi": None}).paper == "arxiv:2001.00001"
    assert r.resolve({"raw": "x", "arxiv": None, "doi": "10.1/ABC"}).paper == "doi:10.1/abc"


def test_snapshot_needs_year_and_surname(tmp_path):
    snap = _snap(tmp_path, [("Deep residual learning for image recognition", "1512.03385", "2015", "he")])
    r = Resolver(KB, snap)
    raw = "K. He, X. Zhang. Deep residual learning for image recognition. In CVPR, 2016."
    res = r.resolve({"raw": raw, "arxiv": None, "doi": None})
    assert res.paper == "arxiv:1512.03385" and res.method == "oai_snapshot"
    wrong_author = "J. Smith. Deep residual learning for image recognition. In CVPR, 2016."
    assert r.resolve({"raw": wrong_author, "arxiv": None, "doi": None}).paper is None
    wrong_year = "K. He. Deep residual learning for image recognition. In CVPR, 2020."
    assert r.resolve({"raw": wrong_year, "arxiv": None, "doi": None}).paper is None


def test_snapshot_ambiguous_is_stub(tmp_path):
    snap = _snap(tmp_path, [("Attention is all you need", "1706.03762", "2017", "vaswani"),
                            ("Attention is all you need", "9999.99999", "2017", "vaswani")])
    r = Resolver({}, snap)
    res = r.resolve({"raw": "A. Vaswani. Attention is all you need. NeurIPS 2017.", "arxiv": None, "doi": None})
    assert res.paper is None and res.method == "stub"


def test_kb_title_fallback(tmp_path):
    r = Resolver(KB, _snap(tmp_path, []))
    res = r.resolve({"raw": "Z. Qian, Y. Li. A unique kb-only paper title about sparse coding. In ICML, 2019.",
                     "arxiv": None, "doi": None})
    assert res.paper == "kb:p2" and res.method in ("kb_title", "kb_title_in_entry")
    # mis-split title (initials glued on) still resolves through containment
    res = r.resolve({"raw": "Z. Q. A unique kb-only paper title about sparse coding. 2019.", "arxiv": None, "doi": None})
    assert res.paper == "kb:p2" and res.method == "kb_title_in_entry"
    # a short generic KB title never matches by containment
    r2 = Resolver({"p3": {"title": "Introduction", "arxiv_id": None}}, _snap(tmp_path, []))
    assert r2.resolve({"raw": "A. B. Introduction to things. 2019.", "arxiv": None, "doi": None}).paper is None


def test_title_must_be_substring():
    t, y = entry_title_year("K. He, X. Zhang. Deep residual learning for image recognition. In CVPR, 2016.")
    assert "deep residual learning" in t.lower() and y == 2016


@pytest.mark.parametrize("raw,want", [
    ('Y. Luo, J. Peng, and J. Ma, “When causal inference meets deep learning,” Nature Machine Intelligence , vol. 2, 2020.',
     "When causal inference meets deep learning"),
    ('S. Tian, R. Wu et al. , “Self-supervised representation learning on dynamic graphs,” in CIKM , 2021.',
     "Self-supervised representation learning on dynamic graphs"),
    ("Guo Y, Yang Y, Abbasi A. Auto-debias: Debiasing masked language models with automated biased prompts. In: Proceedings of ACL, 2022.",
     "Auto-debias: Debiasing masked language models with automated biased prompts"),
    ("D. Li, C. Du, H. He, Semi-supervised cross-modal image generation with generative adversarial networks, Pattern Recognition 100 (1) (2020) 107085.",
     "Semi-supervised cross-modal image generation with generative adversarial networks"),
])
def test_title_styles(raw, want):
    assert entry_title_year(raw)[0] == want
