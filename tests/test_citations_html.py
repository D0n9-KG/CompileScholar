# -*- coding: utf-8 -*-
"""citations.html: LaTeXML cite anchors -> bibitems, label spans stripped, math masked, grouped citations."""
from compilescholar.citations import html as H

PAGE = """<html><body>
<div class="ltx_para"><p id="S1.p1.1" class="ltx_p">Higher-order interactions occur in physical systems
<cite class="ltx_cite ltx_citemacro_citep">(<a href="#bib.bib9" class="ltx_ref">Battiston and Petri, 2022</a>)</cite>.
To connect nodes, kNN <cite class="ltx_cite ltx_citemacro_citep">(<a href="#bib.bib10" class="ltx_ref">Benko et al., 2024</a>;
<a href="#bib.bib9" class="ltx_ref">Battiston and Petri, 2022</a>)</cite> was adopted, where <math><mi>x</mi></math> is a node.</p></div>
<section class="ltx_bibliography"><ul class="ltx_biblist">
<li id="bib.bib9" class="ltx_bibitem"><span class="ltx_tag ltx_role_refnum ltx_tag_bibitem">Battiston and Petri (2022)</span>
<span class="ltx_bibblock">Federico Battiston and Giovanni Petri. 2022.</span><span class="ltx_bibblock"><em>Higher-order systems</em>.</span></li>
<li id="bib.bib10" class="ltx_bibitem"><span class="ltx_tag ltx_role_refnum ltx_tag_bibitem">Benko et al<span class="ltx_text">.</span> (2024)</span>
<span class="ltx_bibblock">Tatyana Benko and Sinan Aksoy. 2024.</span><span class="ltx_bibblock">Hypermagnet. arXiv preprint arXiv:2402.09676 (2024).</span></li>
</ul></section></body></html>"""


def test_entries_and_label_stripped():
    bib, _ = H.parse(PAGE)
    assert set(bib.entries) == {"bib9", "bib10"}
    assert bib.entries["bib10"]["raw"].startswith("Tatyana Benko")
    assert bib.entries["bib10"]["arxiv"] == "2402.09676"


def test_sentences_and_groups():
    _, cs = H.parse(PAGE)
    first = [c for c in cs if c.n_keys == 1]
    assert first and first[0].key == "bib9" and "physical systems" in first[0].sentence
    grouped = [c for c in cs if c.n_keys == 2]
    assert {c.key for c in grouped} == {"bib9", "bib10"}
    assert "[math]" in grouped[0].sentence and "<" not in grouped[0].sentence


def test_no_bibliography():
    assert H.parse("<html><p class='ltx_p'>no refs</p></html>") is None
