# -*- coding: utf-8 -*-
"""library: paper_id rules, resolution without write-time merging, the merge queue, first public date, merges
(INTEGRATED-SYSTEM-1005 §3; the FITNESS-LIBRARY-ACQUIRE cases C1-C11 that broke the sci-evo registry)."""
from pathlib import Path

import pytest

from compilescholar.core.asof import Date
from compilescholar.library import identity as I
from compilescholar.library import import_arxiv as A
from compilescholar.library import store


@pytest.fixture
def con(tmp_path):
    c = store.connect(tmp_path / "registry.sqlite")
    yield c
    c.close()


def _arxiv(aid, vdates, title="A Study of Things in Graphs", doi=None, authors=("Smith", "Lee")):
    return A._incoming(aid, vdates, title, "abstract", [{"name": f"X {s}", "surname": s} for s in authors],
                       ["cs.LG"], doi)


def _add(con, *incs, source=None):
    out = []
    for inc in incs:
        w = I.Writer(con, source or inc.source)
        out.append(w.add(inc))
        w.close()
    I.recompute_first_public(con)
    return out


def _queue(con, kind=None):
    q = "SELECT a, b, kind FROM merge_queue WHERE status='open'" + (" AND kind=?" if kind else "")
    return con.execute(q, (kind,) if kind else ()).fetchall()


def test_paper_id_rules(con):
    a, b = _add(con, _arxiv("arXiv:2101.00001v2", {"v1": "2021-01-01"}, doi="https://doi.org/10.1000/ABC."),
                _arxiv("2101.00002", {"v1": "2021-01-02"}, title="Another Paper About Graphs"))
    assert a == "doi:10.1000/abc" and b == "arxiv:2101.00002"
    t = _add(con, I.Incoming(source="masterset", ids=[("masterset", "u1", "self")], title="Learning to Rank Things",
                             dates=[("venue_year", 0, Date.parse(2020))]))[0]
    assert t == "title:learning to rank things|2020"


def test_arxiv_doi_in_doi_field_is_not_a_doi_and_multi_doi_split():
    inc = _arxiv("2101.00001", {"v1": "2021-01-01"}, doi="10.48550/arXiv.2101.00001")
    assert [s for s, _, _ in inc.ids] == ["arxiv"]
    inc = _arxiv("2101.00001", {"v1": "2021-01-01"}, doi="10.1109/TIT.2020.1 10.1109/ISIT.2019.2")
    assert [v for s, v, _ in inc.ids if s == "doi"] == ["10.1109/tit.2020.1", "10.1109/isit.2019.2"]


def test_reimport_is_idempotent_and_doi_spellings_are_one_paper(con):
    forms = ["10.1000/X1", "doi:10.1000/x1", "https://dx.doi.org/10.1000/x1", "http://www.doi.org/10.1000/x1",
             "10.1000%2Fx1", "(10.1000/x1)", "10.1000/x1."]
    pids = {_add(con, I.Incoming(source="crossref", ids=[("doi", A.ids.normalize_doi(f), "self")],
                                 title="One Paper With Many Spellings"))[0] for f in forms}
    assert pids == {"doi:10.1000/x1"}
    assert con.execute("SELECT count(*) FROM papers").fetchone()[0] == 1


def test_shared_doi_goes_to_the_queue_not_a_silent_merge(con):
    a, b = _add(con, _arxiv("2101.00001", {"v1": "2021-01-01"}, doi="10.1000/j"),
                _arxiv("2101.00099", {"v1": "2021-01-05"}, title="A Different Work", doi="10.1000/j"))
    assert a == "doi:10.1000/j" and b == "arxiv:2101.00099"
    assert I.resolve(con, "doi", "10.1000/J") == a      # the DOI stays with its first owner
    assert _queue(con) == [(*sorted((a, b)), "shared_doi")]


def test_foreign_identifier_does_not_pull_a_record_in(con):
    """C8: a record carrying somebody else's DOI and its own arXiv id is not merged into that paper."""
    a = _add(con, _arxiv("2101.00001", {"v1": "2021-01-01"}, doi="10.1000/owned"))[0]
    inc = _arxiv("2202.00002", {"v1": "2022-02-02"}, title="Unrelated", doi="10.1000/owned")
    b = _add(con, inc)[0]
    assert b != a and I.resolve(con, "arxiv", "2202.00002") == b and I.resolve(con, "doi", "10.1000/owned") == a


def test_second_source_attaches_by_shared_identifier(con):
    a = _add(con, _arxiv("2101.00001", {"v1": "2021-01-01"}))[0]
    b = _add(con, I.Incoming(source="prescience", ids=[("s2_corpus", "123", "self"), ("arxiv", "2101.00001", "self")],
                             title="A Study of Things in Graphs", dates=[("published", 0, Date.parse("2020-12-31"))],
                             member=("prescience", "123")))[0]
    assert a == b and I.resolve(con, "s2_corpus", "123") == a
    assert con.execute("SELECT paper_id FROM members WHERE benchmark='prescience'").fetchone()[0] == a


def test_first_public_prefers_primary_dates(con):
    a = _add(con, _arxiv("2101.00001", {"v1": "2021-01-04", "v2": "2021-06-01"}))[0]
    # a benchmark claiming an earlier date does not move the arXiv v1 date (the later, verified date is leak-safe)
    _add(con, I.Incoming(source="prescience", ids=[("arxiv", "2101.00001", "self")],
                         dates=[("published", 0, Date.parse("2020-12-20"))]))
    assert str(I.first_public(con, a)) == "2021-01-04"
    # with no primary date the secondary one is used, with its precision
    t = _add(con, I.Incoming(source="masterset", ids=[("masterset", "u9", "self")], title="Only A Venue Year",
                             dates=[("venue_year", 0, Date.parse(2019))]))[0]
    d = I.first_public(con, t)
    assert d.precision == "year" and d.hi.isoformat() == "2019-12-31"


def test_record_keeps_the_latest_version(con):
    _add(con, _arxiv("2101.00001", {"v1": "2021-01-01", "v2": "2021-02-01", "v3": "2021-03-01"}, title="Third"))
    _add(con, _arxiv("2101.00001", {"v1": "2021-01-01", "v2": "2021-02-01"}, title="Second"))
    r = con.execute("SELECT version, date_hi, title FROM records WHERE source='arxiv'").fetchone()
    assert r == (3, "2021-03-01", "Third")
    assert con.execute("SELECT count(*) FROM dates WHERE kind='arxiv_v'").fetchone()[0] == 3


def _ms(key, title, year, authors):
    return I.Incoming(source="masterset", ids=[("masterset", key, "self")], title=title,
                      authors=[{"name": a, "surname": a.split()[-1]} for a in authors],
                      dates=[("venue_year", 0, Date.parse(year))], member=("masterset", key))


def test_titles_never_merge_at_write_time_and_are_proposed_with_guards(con):
    a = _add(con, _arxiv("2101.00001", {"v1": "2021-01-01"}, title="$L_2$ Robust Training of Deep Graph Networks"))[0]
    m, other_authors, far, generic1, generic2 = _add(
        con, _ms("u1", "L2 Robust Training of Deep Graph Networks", 2022, ["Ann Smith"]),
        _ms("u2", "L-2 robust training of deep graph networks", 2021, ["Bo Chen"]),
        _ms("u3", "L2 Robust Training of Deep Graph Networks!", 2015, ["Ann Smith"]),
        _ms("u4", "Editorial", 2010, ["C D"]), _ms("u5", "Editorial", 1990, ["E F"]))
    assert len({a, m, other_authors, far}) == 4 and generic1 != generic2        # C5: two Editorials stay two
    n = I.propose_title_matches(con)
    pairs = {tuple(sorted(p[:2])) for p in _queue(con, "title_match")}
    assert tuple(sorted((a, m))) in pairs                                       # same key, year ok, shared surname
    assert tuple(sorted((a, other_authors))) not in pairs and n["no_shared_author"] >= 1
    assert not any(far in p for p in pairs)                                     # years apart
    assert not any(generic1 in p or generic2 in p for p in pairs)


def test_merge_moves_rows_aliases_and_dates(con):
    a = _add(con, _arxiv("2101.00001", {"v1": "2021-01-04"}, title="Robust Training of Deep Graph Networks"))[0]
    m = _add(con, _ms("u1", "Robust Training of Deep Graph Networks", 2020, ["Ann Smith"]))[0]
    I.propose_title_matches(con)
    keep = I.merge(con, a, m, reason="llm: same work", decided_by="test")
    assert keep == a
    assert I.canonical(con, m) == a and I.resolve(con, "masterset", "u1") == a
    assert con.execute("SELECT paper_id FROM members WHERE key='u1'").fetchone()[0] == a
    assert con.execute("SELECT status FROM papers WHERE paper_id=?", (m,)).fetchone()[0] == "merged"
    assert con.execute("SELECT status FROM merge_queue WHERE kind='title_match'").fetchone()[0] == "merged"
    assert str(I.first_public(con, m)) == "2021-01-04"                          # arXiv day date beats the venue year
    srcs = {r[0] for r in con.execute("SELECT source FROM records WHERE paper_id=?", (a,))}
    assert srcs == {"arxiv", "masterset"}


def test_registry_refuses_a_network_path():
    with pytest.raises(RuntimeError):
        store.connect(Path("//host/share/registry.sqlite"))


def test_benchmark_import_carries_no_labels(con):
    """Question-blind: an incoming record has no field for roles, queries or labels, and a benchmark import writes
    membership only."""
    assert not set(I.Incoming.__dataclass_fields__) & {"roles", "labels", "gold", "queries", "key_references"}
    cols = {r[1] for r in con.execute("PRAGMA table_info(members)")}
    assert cols == {"benchmark", "key", "paper_id"}


def test_adjudicate_needs_both_models_to_agree():
    from compilescholar.library.adjudicate import decide
    same = {"same": True, "confidence": 0.9}
    assert decide({"local": same, "paratera": same}) == "merged"
    assert decide({"local": same, "paratera": {"same": True, "confidence": 0.5}}) == "open"
    assert decide({"local": same, "paratera": {"same": False, "confidence": 0.9}}) == "open"
    assert decide({"local": {"same": False, "confidence": 0.6}, "paratera": {"same": False, "confidence": 0.7}}) \
        == "rejected"
    assert decide({"local": same}) == "open" and decide({"local": same, "paratera": None}) == "open"
    # two different arXiv ids: never merged automatically, still rejectable
    assert decide({"local": same, "paratera": same}, two_arxiv_ids=True) == "open"
    assert decide({"local": {"same": False, "confidence": 0.9}, "paratera": {"same": False, "confidence": 0.9}},
                  two_arxiv_ids=True) == "rejected"


def test_describe_shows_evidence_not_benchmark(con):
    from compilescholar.library.adjudicate import describe
    a = _add(con, _arxiv("2101.00001", {"v1": "2021-01-04"}, doi="10.1000/j"))[0]
    _add(con, I.Incoming(source="prescience", ids=[("arxiv", "2101.00001", "self")], member=("prescience", "9")))
    d = describe(con, a)
    assert "2021-01-04" in d and "doi:10.1000/j" in d and "Smith" in d and "prescience" not in d.split("[arxiv]")[0]
    assert "member" not in d and "benchmark" not in d


def test_crossref_assertion_dates():
    from compilescholar.library.import_crossref import assertion_date
    assert str(assertion_date("1 July 2015")) == "2015-07-01"
    assert str(assertion_date("July 2015")) == "2015-07" and assertion_date("July 2015").precision == "month"
    assert str(assertion_date("2015-07-01")) == "2015-07-01"
    assert assertion_date("soon") is None


def test_crossref_titles_lose_markup():
    from compilescholar.library.import_crossref import strip_markup
    t = ('Elastic properties of a paramagnet: Application to NdV<mml:math xmlns:mml="x"><mml:msub><mml:mrow>'
         '<mml:mi>O</mml:mi></mml:mrow><mml:mrow><mml:mn>4</mml:mn></mml:mrow></mml:msub></mml:math>')
    assert strip_markup(t) == "Elastic properties of a paramagnet: Application to NdVO4"
    assert strip_markup("<jats:p>A &amp; B</jats:p>") == "A & B"
