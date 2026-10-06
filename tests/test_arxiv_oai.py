# -*- coding: utf-8 -*-
"""sources.arxiv_oai: arXivRaw record parsing (v1 date from the version list, surnames, deleted records)."""
import xml.etree.ElementTree as ET

from compilescholar.sources import arxiv_oai as A

RECORD = """<record xmlns="http://www.openarchives.org/OAI/2.0/">
<header><identifier>oai:arXiv.org:1601.04794</identifier><datestamp>2026-09-01</datestamp></header>
<metadata><arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
<id>1601.04794</id>
<version version="v1"><date>Tue, 19 Jan 2016 04:10:52 GMT</date></version>
<version version="v2"><date>Tue, 25 Aug 2026 16:38:03 GMT</date></version>
<title>Concentration Inequalities for
  Branching Random Walks</title>
<authors>Changqing Liu and Bo Zhang (Some University)</authors>
<categories>cs.CC math.PR</categories>
<abstract>  A new framework.  </abstract>
</arXivRaw></metadata></record>"""

DELETED = """<record xmlns="http://www.openarchives.org/OAI/2.0/">
<header status="deleted"><identifier>oai:arXiv.org:x</identifier></header></record>"""


def test_parse_record_uses_v1_date_not_latest():
    r = A.parse_record(ET.fromstring(RECORD))
    assert r["arxiv_id"] == "1601.04794"
    assert r["v1_date"] == "2016-01-19" and r["versions"] == 2
    assert r["version_dates"] == {"v1": "2016-01-19", "v2": "2026-08-25"}
    assert r["authors_raw"] == "Changqing Liu and Bo Zhang (Some University)"
    assert r["title"] == "Concentration Inequalities for Branching Random Walks"
    assert r["authors"] == ["Liu", "Zhang"]
    assert r["categories"] == ["cs.CC", "math.PR"] and r["abstract"] == "A new framework."


def test_deleted_record_is_skipped():
    assert A.parse_record(ET.fromstring(DELETED)) is None


def test_surnames():
    assert A.surnames("A. Smith, B. Jones and C. Lee") == ["Smith", "Jones", "Lee"]
