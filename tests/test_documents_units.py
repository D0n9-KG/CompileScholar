# -*- coding: utf-8 -*-
"""documents.units: Markdown and HTML -> the same unit kinds with section paths; references excluded."""
from compilescholar.documents import units as U

MD = """# Learning X

Abstract text that is long enough to be a paragraph of the paper.

# 1 Introduction

We study X. Prior work [1] did Y and this paragraph is long enough.

# 3 Method

## 3.2 Encoder

The encoder maps inputs to a latent space and is described here at length.

Table 1: Results on CIFAR-10 for every method we compare.

<table><tr><td>Method</td><td>Acc</td></tr><tr><td>Ours</td><td>91.2</td></tr></table>

# References

1. A. Author. Something. 2020.
2. B. Author. Other. 2021.
3. C. Author. Third. 2022.
"""

HTML = """<html><body><section><h2 class="ltx_title">1 Introduction</h2>
<p class="ltx_p">We study X and this sentence is long enough to count.</p></section>
<section><h2>3 Method</h2><h3>3.2 Encoder</h3><p class="ltx_p">The encoder <math>x</math> maps inputs to a latent space.</p>
<figure><table class="ltx_tabular"><tr><td>Ours</td><td>91.2</td></tr></table><figcaption>Table 1: Results on CIFAR-10.</figcaption></figure>
</section><section class="ltx_bibliography"><li id="bib.bib1" class="ltx_bibitem">ref</li></section></body></html>"""


def test_markdown_units():
    us = U.from_markdown("2401.1", MD)
    kinds = [u.kind for u in us]
    assert "table" in kinds and "caption" in kinds and kinds.count("section") == 4
    enc = [u for u in us if u.kind == "para" and "encoder maps" in u.text][0]
    assert enc.section == "3 Method > 3.2 Encoder"
    assert all("Something. 2020" not in u.text for u in us)          # references excluded
    assert us[0].uid == "2401.1#section1"


def test_html_units_same_shape():
    us = U.from_html("2401.1", HTML)
    enc = [u for u in us if u.kind == "para" and "encoder" in u.text][0]
    assert enc.section == "3 Method > 3.2 Encoder" and "[math]" in enc.text
    assert [u.kind for u in us].count("table") == 1 and any(u.kind == "caption" for u in us)
    assert all(u.text != "ref" for u in us)


def test_abstract_and_intro_and_method_text():
    us = U.from_markdown("2401.1", MD)
    ai = U.abstract_and_intro(us)
    assert "Abstract text" in ai and "We study X" in ai and "encoder" not in ai
    assert "encoder maps" in U.method_and_experiments(us)
