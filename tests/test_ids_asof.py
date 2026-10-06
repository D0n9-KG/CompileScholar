# -*- coding: utf-8 -*-
"""core.ids and core.asof: the spellings FITNESS-LIBRARY-ACQUIRE found split into several papers, and the precision
rule that keeps year-only dates from leaking."""
import datetime as dt

import pytest

from compilescholar.core import asof as A
from compilescholar.core import ids as I


@pytest.mark.parametrize("raw", [
    "10.1103/PhysRevLett.110.010403", "https://doi.org/10.1103/PhysRevLett.110.010403",
    "http://www.doi.org/10.1103/physrevlett.110.010403", "doi:10.1103/PhysRevLett.110.010403",
    "DOI: 10.1103/PhysRevLett.110.010403.", "(10.1103/PhysRevLett.110.010403)", "dx.doi.org/10.1103/PhysRevLett.110.010403",
    "10.1103%2FPhysRevLett.110.010403", "https://doi.org/10.1103/PhysRevLett.110.010403);",
])
def test_doi_spellings_collapse(raw):
    assert I.normalize_doi(raw) == "10.1103/physrevlett.110.010403"


def test_doi_keeps_inner_brackets_and_rejects_junk():
    assert I.normalize_doi("10.1002/(SICI)1097-4636(199601)30:1<1::AID-JBM1>3.0.CO;2-8") == \
        "10.1002/(sici)1097-4636(199601)30:1<1::aid-jbm1>3.0.co;2-8"
    assert I.normalize_doi("10.1016/j.cell.2020.01.001)") == "10.1016/j.cell.2020.01.001"
    for bad in ("", None, "arXiv:2101.00001", "doi:", "10.12/x", "not a doi"):
        assert I.normalize_doi(bad) is None


def test_arxiv_doi_is_an_arxiv_id():
    assert I.normalize_doi("10.48550/arXiv.2101.00001") is None
    assert I.arxiv_from_doi("https://doi.org/10.48550/arXiv.2101.00001") == "2101.00001"


@pytest.mark.parametrize("raw", ["2101.00001", "arXiv:2101.00001", "arXiv:2101.00001v2", "https://arxiv.org/abs/2101.00001v3",
                                 "arxiv.org/pdf/2101.00001v2.pdf", "ARXIV 2101.00001", "10.48550/arXiv.2101.00001"])
def test_arxiv_spellings_collapse(raw):
    assert I.normalize_arxiv(raw) == "2101.00001"


def test_arxiv_old_style_and_versions():
    assert I.normalize_arxiv("math.GT/0309136") == "math.GT/0309136"
    assert I.normalize_arxiv("arXiv:math.gt/0309136v2") == "math.GT/0309136"
    assert I.normalize_arxiv("hep-th/9711200") == "hep-th/9711200"
    assert I.normalize_arxiv("arXiv:cond-mat/0507321") == "cond-mat/0507321"
    assert I.arxiv_version("2101.00001v3") == 3 and I.arxiv_version("2101.00001") is None
    assert I.normalize_arxiv("foo/1234567") is None and I.normalize_arxiv("12345.6789") is None


def test_find_in_free_text():
    entry = ("Smith J, Jones K. Effect of X. N Engl J Med. 2020;382:1-9. doi:10.1056/NEJMoa2001017. "
             "Also arXiv:2101.00001v2 and hep-th/9711200 [arXiv:hep-th/9711200].")
    assert I.find_dois(entry) == ["10.1056/nejmoa2001017"]
    assert I.find_arxiv(entry) == ["2101.00001", "hep-th/9711200"]


def test_title_norm_unicode():
    assert I.norm_title("Granular Flow") == I.norm_title("granular  flow.")
    assert I.norm_title("α-Synuclein") != I.norm_title("β-Synuclein")
    assert I.norm_title("Étude des écoulements") == "étude des écoulements"
    assert I.norm_title("深度学习综述") == "深度学习综述"
    assert I.norm_title("CO2 capture in MOFs") == "co2 capture in mofs"
    assert I.norm_title("α-synuclein", greek=True) == "alpha synuclein"
    assert I.title_is_generic("Editorial") and I.title_is_generic("Reply.") and not I.title_is_generic(
        "Attention is all you need")


def test_date_precision_intervals():
    d = A.Date.parse("2021")
    assert (d.lo, d.hi, d.precision) == (dt.date(2021, 1, 1), dt.date(2021, 12, 31), "year")
    d = A.Date.parse("2020-02")
    assert d.hi == dt.date(2020, 2, 29) and d.precision == "month"
    assert A.Date.parse("2021-03-04T10:00:00Z").precision == "day"
    assert A.Date.parse(2019).precision == "year"
    assert A.Date.parse("2021-13-01").precision == "year" and A.Date.parse("2021-02-30").precision == "month"
    assert A.Date.parse("") is None and A.Date.parse("n.d.") is None and A.Date.parse(None) is None


def test_year_only_never_leaks():
    t = A.AsOf("2015-01-01")
    assert not t.visible("2015")                       # the string-comparison bug: '2015' <= '2015-01-01'
    assert t.state("2015") == "undecided" and t.state("2016") == "invisible" and t.state(None) == "undated"
    assert t.visible("2014") and t.visible("2014-12") and t.visible("2015-01-01") and not t.visible("2015-01-02")
    assert A.AsOf("2015-12-31").visible("2015")         # the whole year is in
    assert not A.AsOf("2021-06-29").visible("2021-06")  # the month is not over yet
    assert A.AsOf("2021-06-30").visible("2021-06")


def test_no_cutoff_and_month_convention():
    assert A.AsOf(None).visible(None) and A.AsOf(None).visible("2099")
    assert A.AsOf.before_month("2025-05").iso == "2025-04-30"
    assert A.AsOf.before_month("2025-01").iso == "2024-12-31"
    assert A.AsOf("2025-04-30").prefilter_year() == 2025
    with pytest.raises(ValueError):
        A.AsOf("2025/01/01")
    with pytest.raises(ValueError):
        A.AsOf("2025-13-45")


def test_earliest_prefers_smaller_hi_then_precision():
    a, b, c = A.Date.parse("2016-03-01"), A.Date.parse("2015-11"), A.Date.parse("2015")
    assert A.earliest(a, b) == b
    assert A.earliest(b, c) == b                        # 2015-11-30 < 2015-12-31
    assert A.earliest(None, None) is None


def test_legacy_month_rule_matches_v9b():
    from compilescholar.core import cutoff as C
    allowed = A.legacy_month_rule("2025-05")
    for year, date in [(2024, None), (2025, None), (2026, None), (None, "2025-04-30"), (None, "2025-05-01"),
                       (None, "2025"), (None, None)]:
        assert allowed(year, date) == C.allowed(year, date, cut=(2025, 5))
    assert A.legacy_month_rule(None)(None, None) is True


def test_title_key_joins_latex_and_plain_spellings():
    from compilescholar.core.ids import title_key
    assert title_key("$L_2$ Attacks on Nets") == title_key("L2 attacks on nets") == title_key("L-2 Attacks on Nets")
    assert title_key(r"$\alpha$-Synuclein Folding") == title_key("α-synuclein folding")
    assert title_key(r"\textbf{Deep} Nets") == title_key("Deep Nets")
    assert title_key("") == ""
