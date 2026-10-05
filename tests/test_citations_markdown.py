# -*- coding: utf-8 -*-
"""citations.markdown: reference-block detection, three marker styles, LaTeX false positives, sentence grouping."""
from compilescholar.citations import markdown as M

NUMERIC = """# Introduction

Prior work on adversarial training [1] showed robustness gains. Some works [2,3] claim better initialization helps, \
while the weight $w_{2}[0]$ is not a citation. Fig. 2 shows results of [3].

# References

1. Aleksander Madry, Aleksandar Makelov. Towards deep learning models resistant to adversarial attacks. ICLR, 2018.
2. Maksym Andriushchenko and Nicolas Flammarion. Understanding and improving fast adversarial training. NeurIPS, 2020.
3. Gaurang Sriramanan, Sravanti Addepalli. Guided adversarial attack. NeurIPS, 2020.

# Appendix A

More details [1].
"""

AUTHOR_YEAR_LINES = """# Introduction

Aggregation of experts [Cesa-Bianchi and Lugosi, 2006] works well. Classical methods (ARIMA, Huang and Shih [2003], \
Chodakowska et al. [2021]) are common. Later (de Vilmarest, 2022a) extended them.

# References

N. Cesa-Bianchi and G. Lugosi. Prediction, Learning, and Games. Cambridge University Press, 2006. doi: 10.1017/CBO9780511546921.
S.-J. Huang and K.-R. Shih. Short-term load forecasting via arma model identification. IEEE Trans. Power Systems, 2003.
E. Chodakowska, J. Nazarko, and L. Nazarko. Arima models in electrical load forecasting. Energies, 14(23), 2021.
J. de Vilmarest. Modeles espace-etat pour la prevision de series temporelles. PhD thesis, 2022a.
"""

ALPHA = """# Body

As shown in [AZLS19], the bound is tight; see also [BK20, AZLS19].

# References

[AZLS19] Zeyuan Allen-Zhu, Yuanzhi Li, Zhao Song. A convergence theory for deep learning. ICML 2019.
[BK20] Some Body, Other Body. Another paper title here. NeurIPS 2020.
[CD21] Third Author. Third paper title. ICLR 2021.
"""


def test_numeric_entries_markers_and_latex_false_positive():
    bib = M.parse(NUMERIC)
    assert bib.style == "numeric"
    assert set(bib.entries) == {1, 2, 3}
    keys = [m["key"] for m in bib.markers]
    assert keys == [1, 2, 3, 3]  # w_{2}[0] dropped; appendix marker is outside the body
    assert bib.entries[2]["raw"].startswith("Maksym Andriushchenko")


def test_reference_block_ends_at_next_top_heading():
    body, refs = M.find_refs(NUMERIC)
    assert "Appendix" not in refs and "Sriramanan" in refs
    assert "Prior work" in body


def test_sentences_group_keys():
    _, cs = M.citation_sentences(NUMERIC)
    by_key = {}
    for c in cs:
        by_key.setdefault(c.key, []).append(c)
    assert len(by_key[1]) == 1 and by_key[1][0].n_keys == 1
    grouped = [c for c in cs if c.n_keys == 2]
    assert {c.key for c in grouped} == {2, 3} and grouped[0].group == (2, 3)
    assert "Fig. 2 shows results of [3]." in [c.sentence for c in cs]  # 'Fig.' does not end a sentence


def test_author_year_one_entry_per_line():
    bib = M.parse(AUTHOR_YEAR_LINES)
    assert bib.style == "author-year"
    assert ("cesa-bianchi", "2006") in bib.entries and ("vilmarest", "2022a") in bib.entries
    keys = {m["key"] for m in bib.markers}
    assert {("cesa-bianchi", "2006"), ("huang", "2003"), ("chodakowska", "2021"), ("vilmarest", "2022a")} <= keys


def test_alpha_labels():
    bib = M.parse(ALPHA)
    assert bib.style == "alpha"
    assert [m["key"] for m in bib.markers] == ["AZLS19", "BK20", "AZLS19"]


def test_first_surname():
    assert M.first_surname("N. Cesa-Bianchi and G. Lugosi. Prediction") == "Cesa-Bianchi"
    assert M.first_surname("Durmus Alp Emre Acar, Yue Zhao") == "Acar"
    assert M.first_surname("Bi, J.; Wang, Z.") == "Bi"
    assert M.first_surname("J. de Vilmarest. Modeles") == "Vilmarest"


def test_no_reference_section():
    assert M.parse("# Intro\n\nNo citations here at all.\n") is None
