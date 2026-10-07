# -*- coding: utf-8 -*-
"""documents.tei text assembly. Regression for the 260-paper smoke finding: GROBID writes no whitespace between
<s> elements, so the itertext join fused the abstract into one unbreakable blob ("...learning.This...") and
98.4% of T1 quotes degenerated to the whole abstract — prose elements must join their sentences with a space,
while per-sentence records and bibliography entries stay byte-faithful."""
from __future__ import annotations

from compilescholar.documents import tei

NS = "http://www.tei-c.org/ns/1.0"


def _tei(abstract: str, body: str = "", figures: str = "", bibl: str = "") -> bytes:
    return f"""<TEI xmlns="{NS}"><teiHeader><fileDesc><titleStmt><title>T</title></titleStmt>
<publicationStmt><p/></publicationStmt><sourceDesc><p/></sourceDesc></fileDesc>
<profileDesc><abstract>{abstract}</abstract></profileDesc></teiHeader>
<text><body><div><head>Introduction</head>{body}{figures}</div></body>
<back><div type="references"><listBibl>{bibl}</listBibl></div></back></text></TEI>""".encode()


def test_abstract_joins_sentences_with_space():
    doc = tei.parse(_tei("<p><s>First sentence.</s><s>Second one.</s></p>"))
    assert doc.abstract == "First sentence. Second one."


def test_abstract_inline_markup_is_untouched_within_a_sentence():
    doc = tei.parse(_tei('<p><s>We use <hi rend="italic">L</hi>2 <ref type="bibr" target="#b0">[1]</ref>.</s>'
                         "<s>It works.</s></p>"))
    # within one <s> the itertext join stays byte-faithful ("L2" is not split); only boundaries gain a space
    assert doc.abstract == "We use L2 [1]. It works."


def test_abstract_without_sentences_falls_back_to_plain_text():
    doc = tei.parse(_tei("Plain abstract, no sentence tags."))
    assert doc.abstract == "Plain abstract, no sentence tags."


def test_para_unit_joins_sentences_but_sentence_records_stay_single():
    doc = tei.parse(_tei("<p><s>A.</s><s>B.</s></p>",
                         body="<p><s>Body one.</s><s>Body two cites <ref type='bibr' target='#b0'>[1]</ref>.</s></p>",
                         bibl='<biblStruct xml:id="b0"><analytic><title>Ref</title></analytic>'
                              '<note type="raw_reference">Ref. 2020.</note></biblStruct>'))
    paras = [u for u in doc.units if u.kind == "para"]
    assert paras[0].text == "Body one. Body two cites [1]."
    assert [s.text for s in doc.sentences] == ["Body one.", "Body two cites [1]."]
    assert doc.cites and doc.cites[0][1] == "b0"


def test_figdesc_joins_sentences():
    doc = tei.parse(_tei("<p><s>x</s></p>",
                         figures='<figure><head>Figure 1:</head><figDesc><s>Caption one.</s><s>Caption two.</s>'
                                 "</figDesc></figure>"))
    caps = [u for u in doc.units if u.kind == "caption"]
    assert caps[0].text == "Figure 1: Caption one. Caption two."


def test_entry_text_untouched_by_prose_join():
    doc = tei.parse(_tei("<p><s>x</s></p>",
                         bibl='<biblStruct xml:id="b0"><analytic><title>Some <hi>Title</hi> Here</title>'
                              "</analytic><note type='raw_reference'>Some Title Here. 2020.</note>"
                              "</biblStruct>"))
    assert doc.entries["b0"]["title"] == "Some Title Here"
