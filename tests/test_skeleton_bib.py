# -*- coding: utf-8 -*-
"""W2 P0-1 bibliography / marker parsing on synthetic texts in the three LaTeXML renderings."""
from __future__ import annotations

from compilescholar.compile.skeleton import bib as B

NUMERIC = """# Survey
Transformers [1] replaced RNNs [2, 3]. See also [4–6] and [99].

## References

 [1]
A. Vaswani et al., Attention is all you need, arXiv:1706.03762, 2017.

 [2]
S. Hochreiter, Long short-term memory, doi: 10.1162/neco.1997.9.8.1735 .

 [3]
K. Cho, Learning phrase representations.

 [4]
X. Four.

 [5]
X. Five.

 [6]
X. Six.
"""

AUTHOR_YEAR = """# Survey
Prior work (Andriluka et al., 2014; Smith and Doe, 2019) and Lin et al. [2020a] differ from Lin et al. [2020b].
Kim [2021] is ambiguous.

## References

 Andriluka et al. (2014)
M. Andriluka, 2d human pose estimation, 2014.

 Smith and Doe (2019)
J. Smith, A thing, 2019.

 Lin et al. [2020a]
T. Lin, Paper A, arXiv:2001.00001.

 Lin et al. [2020b]
T. Lin, Paper B.

 Kim [2021]
First Kim.

 Kim [2021]
Second Kim.
"""


def test_numeric():
    b = B.parse(NUMERIC)
    assert b.style == "numeric" and sorted(b.entries) == [1, 2, 3, 4, 5, 6]
    assert b.entries[1]["arxiv"] == "1706.03762"
    assert b.entries[2]["doi"] == "10.1162/neco.1997.9.8.1735"
    keys = [m["key"] for m in b.markers]
    assert keys == [1, 2, 3, 4, 5, 6, 99]
    assert [m["key"] for m in b.resolved_markers()] == [1, 2, 3, 4, 5, 6]
    assert B.markers_in("as in [2] and [3, 99]", b) == [2, 3]


def test_author_year():
    b = B.parse(AUTHOR_YEAR)
    assert b.style == "author-year"
    assert ("kim", "2021") in b.ambiguous_keys
    res = [m["key"] for m in b.resolved_markers()]
    assert res == [("andriluka", "2014"), ("smith", "2019"), ("lin", "2020a"), ("lin", "2020b")]
    assert b.entries[("lin", "2020a")]["arxiv"] == "2001.00001"


def test_no_reference_section():
    assert B.parse("# Only a body [1]") is None


def test_range_cap():
    assert B._expand("1–50") == [] and B._expand("3-5") == [3, 4, 5]
