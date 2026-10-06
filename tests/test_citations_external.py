# -*- coding: utf-8 -*-
"""Stage 2 external stub resolution (citations.external): three lanes, strict validation, registry writes, and
the Resolver picking the results up after a rebuild (stage 1 by DOI, stage 2 by ref_resolutions, stage 3 by
title). The HTTP layer is faked — no network in tests; the real Crossref/OpenAlex behaviour is measured in the
smoke run."""
from __future__ import annotations

import json
import sqlite3

import pytest

from compilescholar.citations import external
from compilescholar.citations.build import DDL, Resolver, _generation
from compilescholar.core.asof import Date
from compilescholar.library import identity
from compilescholar.library import store as Lstore
from compilescholar.sources.http import Response

pytestmark = pytest.mark.usefixtures("fake_http")


def _resp(status, obj):
    return Response(status, json.dumps(obj).encode(), {})


class FakeHttp:
    """Routes the two Crossref shapes (works/<doi> point lookup, query.bibliographic search) and OpenAlex."""

    def __init__(self):
        self.search = {}    # query text -> items
        self.records = {}   # doi (lowercase) -> Crossref message
        self.oa = []        # OpenAlex results for any title.search
        self.calls = []

    def get(self, source, url, **kw):
        params = kw.get("params") or {}
        self.calls.append((source, url, dict(params)))
        if source == "crossref" and "query.bibliographic" in params:
            return _resp(200, {"message": {"items": self.search.get(params["query.bibliographic"], [])}})
        if source == "crossref" and "/works/" in url:
            m = self.records.get(url.rsplit("/works/", 1)[1].lower())
            return _resp(200, {"message": m}) if m else _resp(404, {})
        if source == "openalex":
            return _resp(200, {"results": self.oa})
        raise AssertionError(f"unexpected request {source} {url}")


@pytest.fixture
def fake_http(monkeypatch):
    f = FakeHttp()
    monkeypatch.setattr("compilescholar.sources.http.get", f.get)
    return f


def _registry(tmp_path):
    p = tmp_path / "registry.sqlite"
    Lstore.connect(p).close()
    return p


def _add_paper(reg_path, idl, title, year, source="test", surnames=("other",)):
    with Lstore.lock(reg_path):
        con = Lstore.connect(reg_path)
        w = identity.Writer(con, source)
        w.add(identity.Incoming(source=source, ids=idl, title=title, dates=[("issued", 0, Date.year(year))],
                                authors=[{"name": f"A {s.title()}", "surname": s} for s in surnames]))
        w.close()
        identity.recompute_first_public(con)
        con.commit()
        con.close()


def _citations(tmp_path, rows):
    p = tmp_path / "citations.sqlite"
    con = sqlite3.connect(p)
    con.executescript(DDL)
    con.executemany("INSERT INTO entries(citing, version, key, raw, title, year, doi, arxiv, cited, method) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return p


def _stub(citing, raw, title, year=None, doi=None):
    from compilescholar.core import ids
    tk = ids.title_key(title) or ids.norm_title(raw)[:80]
    return (citing, 1, "b0", raw, title, year, doi, None, f"stub:{tk}", "stub")


CR_WORK = {"title": ["Flow-Induced Agitations Create a Granular Fluid"],
           "author": [{"given": "K.", "family": "Nichol"}, {"given": "A.", "family": "Zanin"}],
           "issued": {"date-parts": [[2010, 8, 20]]},
           "published-online": {"date-parts": [[2010, 8, 20]]},
           "container-title": ["Physical Review Letters"]}
PRL_RAW = "K. Nichol, A. Zanin, R. Bastien, E. Wandersman and M. van Hecke, Phys. Rev. Lett., 2010, 104, 078302."
PRL_HIT = {"DOI": "10.1103/physrevlett.104.078302", "title": ["Flow-Induced Agitations Create a Granular Fluid"],
           "issued": {"date-parts": [[2010]]}, "author": [{"family": "Nichol"}],
           "type": "journal-article", "score": 83.8}


def _resolver(reg_path):
    con = sqlite3.connect(f"file:{reg_path.as_posix()}?mode=ro", uri=True)
    return Resolver(con), con


# ---------------------------------------------------------------- lane A: the entry's own DOI
def test_lane_a_doi_stub_creates_metadata_paper(tmp_path, fake_http):
    reg = _registry(tmp_path)
    fake_http.records["10.1111/abc"] = CR_WORK
    cit = _citations(tmp_path, [_stub("arxiv:1", "Some raw text with a doi 10.1111/abc.",
                                      "Phys. Rev. Lett", 2010, doi="10.1111/abc")])
    stats = external.run(cit_path=cit, reg_path=reg, log=lambda *_: None)
    assert stats["lane_a"] == 1 and stats["writer_new"] == 1
    con = Lstore.connect(reg)
    assert con.execute("SELECT 1 FROM papers WHERE paper_id='doi:10.1111/abc'").fetchone()
    assert con.execute("SELECT title_key FROM records WHERE paper_id='doi:10.1111/abc'").fetchone()[0]
    assert not con.execute("SELECT 1 FROM ref_resolutions").fetchall()   # stage 1 finds it without a ref row
    con.close()
    r, rc = _resolver(reg)
    assert r.resolve({"raw": "x", "doi": "10.1111/abc", "arxiv": None, "title": "", "year": ""}) == \
        ("doi:10.1111/abc", "entry_doi")
    rc.close()


# ---------------------------------------------------------------- lane B: query.bibliographic
def test_lane_b1_title_match_creates_paper_for_stage3(tmp_path, fake_http):
    reg = _registry(tmp_path)
    title = "Separability and geometry of object manifolds in deep neural networks"
    raw = f"U. Cohen, S. Chung, D. Lee, H. Sompolinsky. {title}. Nature Communications, 2020."
    doi = "10.1038/s41467-019-13904-w"
    fake_http.search[raw] = [{"DOI": doi, "title": [title], "issued": {"date-parts": [[2020]]},
                              "author": [{"family": "Cohen"}], "type": "journal-article", "score": 70.0}]
    fake_http.records[doi] = {"title": [title], "author": [{"given": "Uri", "family": "Cohen"}],
                              "issued": {"date-parts": [[2020, 1, 1]]}, "container-title": ["Nature Communications"]}
    cit = _citations(tmp_path, [_stub("arxiv:1", raw, title, 2020)])
    stats = external.run(cit_path=cit, reg_path=reg, log=lambda *_: None)
    assert stats["lane_b1"] == 1
    con = Lstore.connect(reg)
    assert con.execute("SELECT 1 FROM papers WHERE paper_id=?", (f"doi:{doi}",)).fetchone()
    assert not con.execute("SELECT 1 FROM ref_resolutions").fetchall()
    con.close()
    r, rc = _resolver(reg)                       # stage 3 finds the created paper by title_key
    assert r.resolve({"raw": raw, "doi": None, "arxiv": None, "title": title, "year": 2020}) == \
        (f"doi:{doi}", "registry_title")
    rc.close()


def test_lane_b2_venue_mangled_entry_gets_ref_row(tmp_path, fake_http):
    reg = _registry(tmp_path)
    fake_http.search[PRL_RAW] = [PRL_HIT]
    fake_http.records["10.1103/physrevlett.104.078302"] = CR_WORK
    cit = _citations(tmp_path, [_stub("arxiv:1", PRL_RAW, "Phys. Rev. Lett", 2010)])
    stats = external.run(cit_path=cit, reg_path=reg, log=lambda *_: None)
    assert stats["lane_b2"] == 1 and stats["ref_rows"] == 1
    con = Lstore.connect(reg)
    row = con.execute("SELECT paper_id, method FROM ref_resolutions WHERE raw_sha=?",
                      (external.raw_sha(PRL_RAW),)).fetchone()
    con.close()
    assert row == ("doi:10.1103/physrevlett.104.078302", "crossref_b2")
    r, rc = _resolver(reg)                       # stage 2: the mangled entry resolves by its raw text
    assert r.resolve({"raw": PRL_RAW, "doi": None, "arxiv": None, "title": "Phys. Rev. Lett", "year": 2010}) == \
        ("doi:10.1103/physrevlett.104.078302", "external_ref")
    rc.close()


def test_b2_rejects_low_score_wrong_year_missing_surname(tmp_path, fake_http):
    reg = _registry(tmp_path)
    low = dict(PRL_HIT, score=34.9)
    wrong_year = dict(PRL_HIT, issued={"date-parts": [[2015]]})
    no_surname = dict(PRL_HIT, author=[{"family": "Einstein"}])
    cases = [("low score", low), ("year off by 5", wrong_year), ("surname absent", no_surname)]
    for i, (name, hit) in enumerate(cases):
        raw = f"{name}: K. Nichol, Phys. Rev. Lett., 2010, 104, {i:06d}."
        fake_http.search[raw] = [hit]
    fake_http.records["10.1103/physrevlett.104.078302"] = CR_WORK
    rows = [_stub(f"arxiv:{i}", f"{name}: K. Nichol, Phys. Rev. Lett., 2010, 104, {i:06d}.", "Phys. Rev. Lett", 2010)
            for i, (name, _) in enumerate(cases)]
    cit = _citations(tmp_path, rows)
    stats = external.run(cit_path=cit, reg_path=reg, log=lambda *_: None)
    assert stats.get("lane_b2", 0) == 0 and stats["b_reject"] == 3
    con = Lstore.connect(reg)
    assert not con.execute("SELECT 1 FROM papers WHERE paper_id LIKE 'doi:%'").fetchall()
    assert not con.execute("SELECT 1 FROM ref_resolutions").fetchall()
    con.close()


def test_component_hits_are_not_papers(tmp_path, fake_http):
    reg = _registry(tmp_path)
    fake_http.search[PRL_RAW] = [dict(PRL_HIT, type="component")]
    cit = _citations(tmp_path, [_stub("arxiv:1", PRL_RAW, "Phys. Rev. Lett", 2010)])
    stats = external.run(cit_path=cit, reg_path=reg, log=lambda *_: None)
    assert stats["b_no_hits"] == 1 and stats.get("lane_b2", 0) == 0
    con = Lstore.connect(reg)
    assert not con.execute("SELECT 1 FROM papers").fetchall()
    con.close()


# ---------------------------------------------------------------- lane C: OpenAlex
def test_lane_c_openalex_title_batch(tmp_path, fake_http):
    reg = _registry(tmp_path)
    title = "ImageNet classification with deep convolutional neural networks"
    raw = f"A. Krizhevsky, I. Sutskever, G. Hinton. {title}. NIPS, 2012."
    fake_http.search[raw] = []                   # Crossref does not carry NeurIPS/Curran
    fake_http.oa = [{"id": "https://openalex.org/W2112345678", "display_name": title,
                     "publication_year": 2012, "publication_date": "2012-12-03", "doi": None,
                     "abstract_inverted_index": None,
                     "authorships": [{"author": {"display_name": "Alex Krizhevsky"}},
                                     {"author": {"display_name": "Ilya Sutskever"}}]}]
    cit = _citations(tmp_path, [_stub("arxiv:1", raw, title, 2012)])
    stats = external.run(cit_path=cit, reg_path=reg, log=lambda *_: None)
    assert stats["lane_c"] == 1 and stats["writer_new"] == 1
    con = Lstore.connect(reg)
    pid = con.execute("SELECT paper_id FROM papers").fetchone()[0]
    assert pid.startswith("title:")              # no DOI -> a title paper
    assert con.execute("SELECT 1 FROM identifiers WHERE scheme='openalex' AND value='W2112345678'").fetchone()
    assert con.execute("SELECT first_hi FROM papers").fetchone()[0].startswith("2012")
    con.close()
    r, rc = _resolver(reg)                       # stage 3 finds it by title_key + year
    cited, method = r.resolve({"raw": raw, "doi": None, "arxiv": None, "title": title, "year": 2012})
    assert (cited, method) == (pid, "registry_title")
    rc.close()
    assert any(s == "openalex" for s, _, _ in fake_http.calls)


def test_lane_c_rejects_wrong_year(tmp_path, fake_http):
    reg = _registry(tmp_path)
    title = "Some neural network method paper title"
    raw = f"A. Author. {title}. NIPS, 2012."
    fake_http.search[raw] = []
    fake_http.oa = [{"id": "https://openalex.org/W1", "display_name": title, "publication_year": 2016,
                     "publication_date": None, "doi": None, "abstract_inverted_index": None, "authorships": []}]
    cit = _citations(tmp_path, [_stub("arxiv:1", raw, title, 2012)])
    stats = external.run(cit_path=cit, reg_path=reg, log=lambda *_: None)
    assert stats["c_miss"] == 1 and stats.get("lane_c", 0) == 0
    con = Lstore.connect(reg)
    assert not con.execute("SELECT 1 FROM papers").fetchall()
    con.close()


# ---------------------------------------------------------------- idempotence / bookkeeping
def test_resolved_raws_are_skipped_on_rerun(tmp_path, fake_http):
    reg = _registry(tmp_path)
    fake_http.search[PRL_RAW] = [PRL_HIT]
    fake_http.records["10.1103/physrevlett.104.078302"] = CR_WORK
    cit = _citations(tmp_path, [_stub("arxiv:1", PRL_RAW, "Phys. Rev. Lett", 2010)])
    external.run(cit_path=cit, reg_path=reg, log=lambda *_: None)
    fake_http.calls.clear()
    fake_http.search.clear()                     # would reject if queried again
    stats = external.run(cit_path=cit, reg_path=reg, log=lambda *_: None)
    assert stats["already"] == 1 and stats["groups"] == 0
    assert fake_http.calls == []                 # nothing was queried a second time


def test_generation_digest_moves(tmp_path, fake_http):
    from compilescholar.dfc import store as dfc_store
    reg = _registry(tmp_path)
    fake_http.records["10.1111/abc"] = CR_WORK
    cit = _citations(tmp_path, [_stub("arxiv:1", "raw with 10.1111/abc", "Phys. Rev. Lett", 2010,
                                      doi="10.1111/abc")])
    ro = dfc_store.read_only(reg)
    before = _generation(ro)
    ro.close()
    external.run(cit_path=cit, reg_path=reg, log=lambda *_: None)
    ro = dfc_store.read_only(reg)
    assert _generation(ro) != before             # a citations rebuild re-opens every item
    ro.close()


def test_merge_moves_ref_resolution(tmp_path):
    reg = _registry(tmp_path)
    _add_paper(reg, [("arxiv", "1234.5678", "self")], "Paper One Title Here", 2010)
    _add_paper(reg, [("doi", "10.1/one", "self")], "Paper One Title Here", 2010)
    con = Lstore.connect(reg)
    con.execute("INSERT INTO ref_resolutions VALUES (?,?,?,?,?)",
                (external.raw_sha("some raw"), "arxiv:1234.5678", "crossref_b2", None, "now"))
    con.commit()
    identity.merge(con, "arxiv:1234.5678", "doi:10.1/one", "test")
    row = con.execute("SELECT paper_id FROM ref_resolutions WHERE raw_sha=?", (external.raw_sha("some raw"),))
    assert row.fetchone()[0] == "doi:10.1/one"   # the stronger id survives and the pointer follows
    con.close()


def test_run_without_citations_store(tmp_path):
    reg = _registry(tmp_path)
    out = external.run(cit_path=tmp_path / "missing.sqlite", reg_path=reg, log=lambda *_: None)
    assert out["error"]
