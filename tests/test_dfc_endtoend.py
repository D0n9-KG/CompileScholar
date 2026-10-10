# -*- coding: utf-8 -*-
"""End-to-end invariants of the derived pipeline on a synthetic library, no network, no LLM.

Phase C coverage note: the derived stages are being rebuilt onto registry keys (paper_id@vN) one stage at a time.
This file currently covers library -> documents -> citations (C-1: materialisation with version dates, deltas,
incremental work, failure retry, merge sweeps; C②: the resolution cascade, self-citation flags, delta cites dated
by the latest version, the stale-upstream gate). The extract / index / tools / answer invariants that lived here
before phase C return as those stages are rebuilt (C③–C⑤) and cognition / tools are re-wired (phase D); the pre-C
version of this file is in git history (tag pre-integration-20261005 .. 8594109).
"""
import gzip
import hashlib
import importlib
import json
import sqlite3
import zipfile

import pytest

from compilescholar.core import ids

NS = "http://www.tei-c.org/ns/1.0"
P1 = "arxiv:2001.00001"      # single version
P2 = "arxiv:2101.00002"      # two versions (delta) + a careful-tier parse on v1
P4 = "arxiv:9999.99999"      # gets "merged" into P1 as its v2 in one test
P5 = "arxiv:2005.00005"      # broken-product failure / retry / gave_up case
JD = "doi:10.1000/jx"        # journal paper, version of record (version 0)
# resolution targets (registry-only: no assets, never citing papers)
P0 = "arxiv:1704.01212"      # "Neural message passing for quantum chemistry"
PA = "arxiv:2010.11111"      # shares a title with PB -> ambiguity stays a stub
PB = "arxiv:2010.22222"
PC = "arxiv:2012.33333"      # year-window probes
PD = "arxiv:2013.44444"      # self-cite initial mismatch probe (Kim, Robert)
PX = "arxiv:2013.55555"      # self-cite initial mismatch probe (Kim, S.)
PNEW = "arxiv:2104.77777"    # registered mid-test: a stub resolves on the next build


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


SHA = {"p1v1": _sha("p1v1"), "p2v1": _sha("p2v1"), "p2v2": _sha("p2v2"),
       "p4v1": _sha("p4v1"), "p5v1": _sha("p5v1"), "jdv0": _sha("jdv0")}

B0 = ("b0", "Neural message passing for quantum chemistry", "2017", None, None)
B0_DOI = ("b0", "Neural message passing for quantum chemistry", "2017", "10.1000/nmp", None)
B0_ARXIV = ("b0", "Neural message passing for quantum chemistry", "2017", None, "1704.01212")
B1 = ("b1", "Convergence of attention", "2021", None, None)


def _tei(title: str, abstract: str, paras: list, bibl=()) -> bytes:
    """paras: [[(sentence text, [cited entry keys]), ...], ...]; bibl: [(key, title, year, doi, arxiv), ...]"""
    ps = []
    for para in paras:
        ss = []
        for text, keys in para:
            refs = "".join(f'<ref type="bibr" target="#{k}">[{k}]</ref>' for k in keys)
            ss.append(f"<s>{text} {refs}</s>" if refs else f"<s>{text}</s>")
        ps.append(f"<p>{''.join(ss)}</p>")
    bib = "".join(
        f'<biblStruct xml:id="{k}"><analytic><title>{t}</title></analytic>'
        + (f'<idno type="DOI">{d}</idno>' if d else "")
        + (f'<idno type="arXiv">{a}</idno>' if a else "")
        + f'<date type="published" when="{y}"/><note type="raw_reference">{t}. {y}.</note></biblStruct>'
        for k, t, y, d, a in bibl)
    return f"""<TEI xmlns="{NS}"><teiHeader><fileDesc><titleStmt><title>{title}</title></titleStmt>
<publicationStmt><p/></publicationStmt><sourceDesc><p/></sourceDesc></fileDesc>
<profileDesc><abstract>{abstract}</abstract></profileDesc></teiHeader>
<text><body><div><head>Introduction</head>{''.join(ps)}</div></body>
<back><div type="references"><listBibl>{bib}</listBibl></div></back></text></TEI>""".encode()


TEIS = {
    "p1v1": _tei("GraphFormer: attention for graphs", "We propose GraphFormer, an attention model for graphs.",
                 [[("GraphFormer applies attention to graphs", ["b0"])],
                  [("Experiments show consistent gains over strong baselines", [])]], [B0]),
    "p2v1": _tei("Faster graph transformers", "We propose FastGF, an efficient graph transformer.",
                 [[("We propose FastGF, an efficient graph transformer", [])],
                  [("Training takes two days on one GPU", [])],
                  [("GraphFormer applies attention to graphs", ["b0"])]], [B0]),
    "p2v2": _tei("Faster graph transformers", "We propose FastGF, an efficient graph transformer.",
                 [[("We propose FastGF, an efficient graph transformer", [])],
                  [("Training takes three days on one GPU", [])],
                  [("GraphFormer applies attention to graphs", ["b0"])],
                  [("We also prove a convergence bound for the attention layer", ["b1"])]], [B0, B1]),
    "p4v1": _tei("GraphFormer: attention for graphs", "We propose GraphFormer, an attention model for graphs.",
                 [[("GraphFormer applies attention to graphs", ["b0"])],
                  [("Experiments show consistent gains over strong baselines", [])],
                  [("The appendix proves additional bounds", [])]], [B0_ARXIV]),
    "p5v1": _tei("A study of something", "We study something.", [[("We study something carefully", [])]]),
    "jdv0": _tei("Journal of graph studies", "A journal paper about graphs.",
                 [[("Journal papers cite GraphFormer too", ["b0"])]], [B0_DOI]),
}
CONTENT_LIST = [
    {"type": "text", "text": "1 Introduction", "text_level": 1, "page_idx": 0, "bbox": [0, 0, 100, 10]},
    {"type": "text", "text": "Some paragraph text here.", "page_idx": 0, "bbox": [0, 10, 100, 40]},
    {"type": "table", "table_body": "<table><tr><td>0.9</td></tr></table>", "table_caption": ["Table 1: results."],
     "page_idx": 1, "bbox": [0, 0, 100, 50]},
    {"type": "list", "sub_type": "ref_text", "list_items": ["A. Message passing. 2017."], "page_idx": 2},
]

CITERS = [(P1, "GraphFormer: attention for graphs", "2020-01-10"),
          (P2, "Faster graph transformers", "2021-01-12"),
          (P4, "GraphFormer: attention for graphs", "2020-06-01"),
          (P5, "A study of something", "2020-05-05"),
          (JD, "Journal of graph studies", "2021-05-01")]
TARGETS = [(P0, "Neural message passing for quantum chemistry", "2017-07-04"),
           (PA, "Ambiguous title test", "2020-01-01"), (PB, "Ambiguous title test", "2020-06-01"),
           (PC, "Year window test", "2020-06-01"), (PD, "Initial mismatch test", "2021-01-01"),
           (PX, "Self citation probe", "2021-01-01")]
AUTH = {P0: [{"name": "Justin Gilmer", "surname": "Gilmer"}], P1: [{"name": "Alice Lee", "surname": "Lee"}],
        P2: [{"name": "J. Gilmer", "surname": "Gilmer"}], PC: [{"name": "Wei Zhang", "surname": "Zhang"}],
        PD: [{"name": "Robert Kim", "surname": "Kim"}], PX: [{"name": "S. Kim", "surname": "Kim"}],
        PA: [{"name": "Ann Author", "surname": "Author"}], PB: [{"name": "Bob Writer", "surname": "Writer"}]}


def _product(tmp_library, name: str, sha: str) -> None:
    d = tmp_library / "parsed" / "grobid" / sha[:2]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sha}.tei.xml.gz").write_bytes(gzip.compress(TEIS[name], 6))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CS_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("CS_CACHE", str(tmp_path / "cache"))
    from compilescholar.core import paths
    importlib.reload(paths)
    from compilescholar.dfc import store
    importlib.reload(store)
    from compilescholar.library import store as LS
    importlib.reload(LS)
    from compilescholar.documents import build as DB
    importlib.reload(DB)
    from compilescholar.citations import build as CB
    importlib.reload(CB)

    lib = paths.library()
    reg = LS.connect()
    now = "2026-10-07T00:00:00"
    for pid, title, lo, hi in [(p, t, d, d) for p, t, d in CITERS + TARGETS]:
        reg.execute("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (pid, "active", None, title, lo, hi, "day", "test",
                     "online" if pid == JD else "arxiv_v", now))
    for pid, ver, hi in [(P1, 1, "2020-01-10"), (P2, 1, "2021-01-12"), (P2, 2, "2021-06-01"),
                         (P4, 1, "2020-06-01"), (P5, 1, "2020-05-05")]:
        reg.execute("INSERT INTO dates VALUES (?,?,?,?,?,?,?)", (pid, "arxiv", "arxiv_v", ver, hi, hi, "day"))
    reg.execute("INSERT INTO dates VALUES (?,?,?,?,?,?,?)", (JD, "crossref", "online", 0, "2021-05-01",
                                                             "2021-05-01", "day"))
    for pid, ver, ch, sha in [(P1, 1, "arxiv_nas", SHA["p1v1"]), (P2, 1, "arxiv_nas", SHA["p2v1"]),
                              (P2, 2, "arxiv_nas", SHA["p2v2"]), (P4, 1, "arxiv_nas", SHA["p4v1"]),
                              (P5, 1, "arxiv_nas", SHA["p5v1"]), (JD, 0, "oa_pdf", SHA["jdv0"])]:
        reg.execute("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?)",
                    (pid, ver, ch, f"nas://test/{sha}.pdf", sha, 1000, "ok", None, now))
    for name, sha in SHA.items():
        reg.execute("INSERT INTO parses VALUES (?,?,?,?,?,?,?,?,?)",
                    (sha, "grobid", "0.9.1-full", "fast", "ok", f"parsed/grobid/{sha[:2]}/{sha}.tei.xml.gz", "", 1,
                     "2026-10-06T00:00:00"))
    reg.execute("INSERT INTO parses VALUES (?,?,?,?,?,?,?,?,?)",
                (SHA["p2v1"], "mineru", "router", "careful", "ok", f"parsed/mineru/{SHA['p2v1'][:2]}/x.zip", "", 1,
                 "2026-10-06T00:00:00"))
    for pid, title, hi in TARGETS:                                   # resolution targets: identity + records
        if pid.startswith("arxiv:"):
            reg.execute("INSERT INTO identifiers VALUES (?,?,?,?,?)",
                        ("arxiv", pid[len("arxiv:"):], pid, "test", "primary"))
        reg.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)",
                    (pid, "arxiv", 1, hi, title, ids.title_key(title), None, "cs.LG", None))
    reg.execute("INSERT INTO identifiers VALUES (?,?,?,?,?)", ("doi", "10.1000/nmp", P0, "test", "primary"))
    for pid, au in AUTH.items():
        reg.execute("INSERT INTO authors VALUES (?,?,?)", (pid, "arxiv", json.dumps(au)))
    reg.commit()
    reg.close()
    for name, sha in SHA.items():
        _product(lib, name, sha)
    md = lib / "parsed" / "mineru" / SHA["p2v1"][:2]
    md.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(md / f"{SHA['p2v1']}.mineru.zip", "w") as z:
        z.writestr("auto/p2v1_content_list.json", json.dumps(CONTENT_LIST))

    DB.build(workers=2, log=lambda *a: None)
    CB.build(workers=2, log=lambda *a: None)
    return store


def _reg():
    from compilescholar.core import paths
    return sqlite3.connect(paths.library() / "registry.sqlite")


def _work(env, pass_name="docs"):
    con = env.connect("documents" if pass_name in ("docs", "delta", "sv") else "citations", readonly=True)
    try:
        return {r[0]: r[1:] for r in con.execute("SELECT item, key, status, attempts FROM _work WHERE pass=?",
                                                 (pass_name,))}
    finally:
        con.close()


def _cit(env):
    return env.connect("citations", readonly=True)


# ---------------------------------------------------------------- documents (C-1)

def test_docs_materialised_with_version_dates(env):
    from compilescholar.documents.build import Documents
    D = Documents()
    d = D.get(f"{P1}@v1")
    assert d["text_date"] == "2020-01-10" and d["date_precision"] == "day" and d["source"] == "arxiv_nas"
    assert d["fast"]["abstract"] == "We propose GraphFormer, an attention model for graphs."
    sids = [s["sid"] for s in d["fast"]["sentences"]]
    assert d["fast"]["cites"] == [[sids[0], "b0", 1]]                   # ref target links sentence 1 to b0 (JSON: list)
    assert d["fast"]["entries"]["b0"]["title"] == B0[1]
    assert d["careful"] is None and D.get(f"{P1}@v2") is None
    j = D.get(f"{JD}@v0")                                               # version of record: the online date
    assert j["text_date"] == "2021-05-01" and j["fast"] is not None
    c = D.get(f"{P2}@v1")                                               # careful tier where MinerU parsed it
    assert c["careful"]["references"] == ["A. Message passing. 2017."]
    kinds = [u["kind"] for u in c["careful"]["units"]]
    assert "table" in kinds and c["careful"]["units"][kinds.index("table")]["html"].startswith("<table>")
    assert sorted(D.versions(P2)) == [1, 2] and D.papers() == sorted([P1, P2, P4, P5, JD])
    D.close()


def test_delta_holds_only_what_the_latest_version_adds(env):
    from compilescholar.documents.build import Documents
    D = Documents()
    d = D.delta(P2)
    assert d["v1_key"] == f"{P2}@v1" and d["latest_key"] == f"{P2}@v2"
    assert len(d["sentences"]) == 1 and "convergence bound" in d["sentences"][0]["text"]
    assert len(d["revised"]) == 1 and "three days" in d["revised"][0]["text"] \
        and "two days" in d["revised"][0]["v1_text"]
    assert list(d["entries"]) == ["b1"]
    assert d["cites"] and all(c[1] == "b1" for c in d["cites"])         # the new citation rides with the delta
    assert D.get(f"{P2}@v2")["text_date"] == "2021-06-01"               # the date delta content carries (v2.5)
    assert D.delta(P1) is None                                          # single version: no delta
    D.close()


def test_rebuild_without_change_does_no_work(env, monkeypatch):
    from compilescholar.documents import assemble as A
    from compilescholar.documents import build as DB

    def no_rematerialise(*a, **k):
        raise AssertionError("assemble.get called on an unchanged item")

    monkeypatch.setattr(A, "get", no_rematerialise)
    before = _work(env)
    DB.build(workers=2, log=lambda *a: None)
    assert _work(env) == before
    assert env.read_manifest("documents")["complete"]


def test_reparse_and_date_fix_reopen_exactly_one_item_each(env):
    from compilescholar.documents import build as DB
    from compilescholar.documents.build import Documents
    before = _work(env)
    con = _reg()                                                        # a re-parse: the parses row moves on
    con.execute("UPDATE parses SET created_at='2026-10-07T01:00:00' WHERE sha256=?", (SHA["p1v1"],))
    con.commit()
    con.close()
    DB.build(workers=2, log=lambda *a: None)
    changed = {i for i, (k, *_ ) in _work(env).items() if before[i][0] != k}
    assert changed == {f"{P1}@v1"}

    before = _work(env)
    con = _reg()                                                        # a corrected version date
    con.execute("UPDATE dates SET lo='2020-02-10', hi='2020-02-10' WHERE paper_id=? AND version=1", (P5,))
    con.commit()
    con.close()
    DB.build(workers=2, log=lambda *a: None)
    changed = {i for i, (k, *_ ) in _work(env).items() if before[i][0] != k}
    assert changed == {f"{P5}@v1"}
    D = Documents()
    assert D.get(f"{P5}@v1")["text_date"] == "2020-02-10"
    D.close()


def test_merged_away_paper_is_swept_and_the_survivor_gains_the_version(env):
    from compilescholar.documents import build as DB
    from compilescholar.documents.build import Documents
    con = _reg()                                                        # what library.identity.merge does: assets
    con.execute("UPDATE assets SET paper_id=?, version=2 WHERE paper_id=?", (P1, P4))   # and dates move over
    con.execute("UPDATE dates SET paper_id=?, version=2 WHERE paper_id=?", (P1, P4))
    con.execute("DELETE FROM papers WHERE paper_id=?", (P4,))
    con.commit()
    con.close()
    DB.build(workers=2, log=lambda *a: None)
    D = Documents()
    assert D.get(f"{P4}@v1") is None and f"{P4}@v1" not in _work(env)   # swept with its work row
    assert D.get(f"{P1}@v2") is not None and D.get(f"{P1}@v2")["text_date"] == "2020-06-01"
    d = D.delta(P1)                                                     # the survivor became multi-version
    assert d and len(d["sentences"]) == 1 and "appendix" in d["sentences"][0]["text"].lower()
    D.close()


def test_broken_product_fails_retries_gives_up_and_recovers(env):
    from compilescholar.core import paths
    from compilescholar.documents import build as DB
    from compilescholar.documents.build import Documents
    gz = paths.library() / "parsed" / "grobid" / SHA["p5v1"][:2] / f"{SHA['p5v1']}.tei.xml.gz"
    stamp = [10]

    def bump():                                                         # a re-parse attempt changes the fingerprint
        stamp[0] += 1
        con = _reg()
        con.execute("UPDATE parses SET created_at=? WHERE sha256=?",
                    (f"2026-10-07T00:{stamp[0]}:00", SHA["p5v1"]))
        con.commit()
        con.close()

    gz.write_bytes(b"not gzip at all")
    bump()
    DB.build(workers=2, log=lambda *a: None)
    m = env.read_manifest("documents")
    assert not m["complete"] and m["work"].get("failed") == 1
    DB.build(workers=2, log=lambda *a: None)                            # retried on the next build
    assert _work(env)[f"{P5}@v1"][1:] == ("failed", 2)
    DB.build(workers=2, log=lambda *a: None)                            # ... and given up after MAX_ATTEMPTS
    assert _work(env)[f"{P5}@v1"][1:] == ("gave_up", 3)
    DB.build(workers=2, log=lambda *a: None)                            # a given-up item is not retried under its key
    assert _work(env)[f"{P5}@v1"][1:] == ("gave_up", 3)
    gz.write_bytes(gzip.compress(TEIS["p5v1"], 6))                      # repaired product + a new parse stamp
    bump()
    DB.build(workers=2, log=lambda *a: None)
    assert _work(env)[f"{P5}@v1"][1:] == ("ok", 0)
    assert env.read_manifest("documents")["complete"]
    D = Documents()
    assert D.get(f"{P5}@v1")["fast"]["sentences"]
    D.close()


def test_stale_documents_blocks_downstream(env):
    env.write_manifest("citations", {}, {})                             # a built downstream stage
    assert not env.status("citations")["stale"]
    env.write_manifest("documents", {"v": 999}, {})                     # documents reconfigured
    s = env.status("citations")
    assert s["stale"] and any("reconfigured" in w or "stale" in w for w in s["why"])
    with pytest.raises(RuntimeError):
        env.require_fresh("citations")
    with pytest.raises(RuntimeError):                                   # extract refuses: upstream citations is stale
        with env.Run("extract", {}):
            pass


# ---------------------------------------------------------------- citations (C②)

def test_resolution_cascade_and_self_citation(env):
    cit = _cit(env)
    m = dict(cit.execute("SELECT citing, method FROM entries WHERE key='b0'"))
    assert m == {P1: "registry_title", P2: "registry_title", P4: "entry_arxiv", JD: "entry_doi"}
    rows = cit.execute("SELECT citing, cited, self_cite, date FROM cites").fetchall()
    assert (P1, P0, 0, "2020-01-10") in rows
    assert (P2, P0, 1, "2021-01-12") in rows                            # Gilmer (J.) cites Gilmer (Justin)
    assert (P4, P0, 0, "2020-06-01") in rows
    assert (JD, P0, 0, "2021-05-01") in rows
    stub = [r for r in rows if r[1].startswith("stub:")]
    assert len(stub) == 1 and stub[0][0] == P2 and stub[0][3] == "2021-06-01"   # the delta cite, v2-dated
    s = sorted(cit.execute("SELECT citing_key, in_delta, date FROM sentences WHERE citing=?", (P2,)))
    assert s == [(f"{P2}@v1", 0, "2021-01-12"), (f"{P2}@v2", 1, "2021-06-01")]
    # the delta's entry is stored under the latest version, keyed apart from v1's b0
    e = cit.execute("SELECT version, cited FROM entries WHERE citing=? AND key='b1'", (P2,)).fetchone()
    assert e[0] == 2 and e[1].startswith("stub:")
    cit.close()


def test_no_citation_visible_before_its_text_version(env):
    """Phase B leak gate, now at the citations level: at a T between P2's v1 and v2, only what v1 already cited
    exists; the v2-only reference is dated 2021-06-01 and invisible."""
    cit = _cit(env)
    T = "2021-03-01"
    rows = cit.execute("SELECT cited FROM cites WHERE citing=? AND date<=?", (P2, T)).fetchall()
    assert rows == [(P0,)]
    assert cit.execute("SELECT count(*) FROM cites WHERE date<=? AND cited LIKE 'stub:%'", (T,)).fetchone()[0] == 0
    cit.close()


def test_resolver_cascade_edges(env):
    from compilescholar.citations.build import Resolver
    from compilescholar.core import paths
    reg = env.read_only(paths.library() / "registry.sqlite")
    R = Resolver(reg)
    # stage 1 beats everything else
    assert R.resolve({"title": "", "year": "", "raw": "x", "doi": "10.1000/nmp"}) == (P0, "entry_doi")
    assert R.resolve({"title": "", "year": "", "raw": "x", "arxiv": "1704.01212"}) == (P0, "entry_arxiv")
    # two live papers sharing a title_key: a stub until the merge queue decides
    assert R.resolve({"title": "Ambiguous title test", "year": "2020", "raw": ""})[1] == "stub"
    # year window: |Δ|<=1 plain, <=4 with a surname in the raw entry, beyond is a stub
    assert R.resolve({"title": "Year window test", "year": "2021",
                      "raw": "Someone. Year window test. 2021."}) == (PC, "registry_title")
    assert R.resolve({"title": "Year window test", "year": "2023",
                      "raw": "W. Zhang. Year window test. 2023."}) == (PC, "registry_title")
    assert R.resolve({"title": "Year window test", "year": "2026",
                      "raw": "W. Zhang. Year window test. 2026."})[1] == "stub"
    # self-citation: surname + compatible initial; a differing initial is a different person
    assert R.self_cite(PX, PD) == 0 and R.self_cite(PD, PD) == 1
    assert R.self_cite(P1, "stub:x") == 0
    reg.close()


def test_citations_rebuild_without_change_does_no_work(env):
    from compilescholar.citations import build as CB
    before = _work(env, "citations")
    CB.build(workers=2, log=lambda *a: None)
    assert _work(env, "citations") == before
    assert env.read_manifest("citations")["complete"]


def test_registry_growth_reresolves_stubs(env):
    """A registry import changes the generation digest: every citing paper re-resolves, and a former stub finds
    its paper (the pre-C stage had the same semantics via the papers-stage fingerprint)."""
    from compilescholar.citations import build as CB
    cit = _cit(env)
    assert cit.execute("SELECT count(*) FROM cites WHERE cited LIKE 'stub:%'").fetchone()[0] == 1
    cit.close()
    before = _work(env, "citations")
    con = _reg()
    con.execute("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?)",
                (PNEW, "active", None, "Convergence of attention", "2021-03-01", "2021-03-01", "day", "test",
                 "arxiv_v", "2026-10-07T00:00:00"))
    con.execute("INSERT INTO identifiers VALUES (?,?,?,?,?)", ("arxiv", "2104.77777", PNEW, "test", "primary"))
    con.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)",
                (PNEW, "arxiv", 1, "2021-03-01", "Convergence of attention",
                 ids.title_key("Convergence of attention"), None, "cs.LG", None))
    con.commit()
    con.close()
    CB.build(workers=2, log=lambda *a: None)
    after = _work(env, "citations")
    assert all(after[i][0] != before[i][0] for i in before)             # generation moved: everything re-opened
    cit = _cit(env)
    assert cit.execute("SELECT count(*) FROM cites WHERE cited LIKE 'stub:%'").fetchone()[0] == 0
    assert cit.execute("SELECT cited FROM cites WHERE citing=? AND date='2021-06-01'", (P2,)).fetchone()[0] == PNEW
    cit.close()


def test_citations_sweep_after_merge(env):
    from compilescholar.citations import build as CB
    from compilescholar.documents import build as DB
    con = _reg()                                                        # P4 merges into P1 as its v2
    con.execute("UPDATE assets SET paper_id=?, version=2 WHERE paper_id=?", (P1, P4))
    con.execute("UPDATE dates SET paper_id=?, version=2 WHERE paper_id=?", (P1, P4))
    con.execute("DELETE FROM papers WHERE paper_id=?", (P4,))
    con.commit()
    con.close()
    DB.build(workers=2, log=lambda *a: None)
    CB.build(workers=2, log=lambda *a: None)
    cit = _cit(env)
    for t in ("cites", "sentences", "entries", "docs"):
        col = "paper_id" if t == "docs" else "citing"
        assert cit.execute(f"SELECT count(*) FROM {t} WHERE {col}=?", (P4,)).fetchone()[0] == 0
    # P1 keeps its own v1 cite; the moved version adds no new citation pair (its citing sentence was unchanged)
    assert cit.execute("SELECT cited, date FROM cites WHERE citing=?", (P1,)).fetchall() == [(P0, "2020-01-10")]
    cit.close()


# ---------------------------------------------------------------- Sciverse content tier (v2.5, ②′)

SV_MD = """# 1 Introduction

We propose FastGF, an efficient graph transformer. Training takes three days on one GPU.

We also prove a convergence bound for the attention layer.

<table><tr><td>0.91</td></tr></table>

# References

[1] A first reference here. 2020.
[2] A second reference here. 2021.
[3] A third reference sits here. 2019.
"""


def _texts(tmp_path, pid, text, fetched_at="2026-10-07T01:00:00", doc_id="DOC1"):
    """Write one cached Sciverse text row (the fetch marathon's output) into the env's derived dir."""
    import zlib
    from compilescholar.documents import sciverse as SV
    tp = tmp_path / "data" / "derived" / "sciverse_text.sqlite"
    tp.parent.mkdir(parents=True, exist_ok=True)
    con = SV._connect(tp)
    con.execute("INSERT OR REPLACE INTO texts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (pid, doc_id, "t", None, "ok", "", len(text), 1, 1, fetched_at, zlib.compress(text.encode())))
    con.commit()
    con.close()
    return tp


def test_sv_pass_dates_sentences_by_version(env, tmp_path):
    from compilescholar.documents import build as DB
    from compilescholar.documents.build import Documents
    _texts(tmp_path, P2, SV_MD)
    counts = DB.build(workers=2, log=lambda *a: None)
    assert counts["sv"] == 1 and counts["sv_held"] == {"latest": 1}
    assert _work(env, "sv")[P2][1] == "ok"
    D = Documents()
    sv = D.sv(P2)
    assert sv["held"] == "latest" and sv["doc_id"] == "DOC1"
    d = {s["text"]: s["date"] for s in sv["sentences"]}
    assert d["We propose FastGF, an efficient graph transformer."] == "2021-01-12"        # in v1 -> v1's date
    assert d["Training takes three days on one GPU."] == "2021-06-01"                     # revised -> latest
    assert d["We also prove a convergence bound for the attention layer."] == "2021-06-01"  # latest-only
    assert not any("0.91" in s["text"] for s in sv["sentences"])                          # tables are not prose
    D.close()


def test_sv_pass_reopens_on_new_fetch_and_sweeps(env, tmp_path):
    from compilescholar.documents import build as DB
    from compilescholar.documents.build import Documents
    t1 = ("GraphFormer applies attention to graphs. Experiments show consistent gains over strong baselines.")
    tp = _texts(tmp_path, P1, t1)
    DB.build(workers=2, log=lambda *a: None)
    D = Documents()
    assert D.sv(P1)["held"] == "single"                                                   # one version: nothing to decide
    assert {s["date"] for s in D.sv(P1)["sentences"]} == {"2020-01-10"}                   # all dated v1
    n1 = len(D.sv(P1)["sentences"])
    D.close()
    _texts(tmp_path, P1, t1 + " A later addition sentence comes here.", fetched_at="2026-10-07T02:00:00")
    DB.build(workers=2, log=lambda *a: None)                                              # fetched_at moved -> reopened
    D = Documents()
    assert len(D.sv(P1)["sentences"]) == n1 + 1
    D.close()
    con = sqlite3.connect(tp)
    con.execute("UPDATE texts SET status='absent' WHERE paper_id=?", (P1,))
    con.commit()
    con.close()
    counts = DB.build(workers=2, log=lambda *a: None)                                     # text gone -> row swept
    assert counts["sv"] == 0
    D = Documents()
    assert D.sv(P1) is None
    D.close()


# ---------------------------------------------------------------- extract (C④)

SV_MD_EXT = """# 1 Introduction

We propose FastGF, an efficient graph transformer. It trains quickly.

Table 1: main results for the benchmark.

<table><tr><td>Method</td><td>Score</td></tr><tr><td>FastGF</td><td>91.2</td></tr><tr><td>Base</td><td>88.0</td></tr></table>

We also prove a convergence bound for the attention layer.

# References

[1] A first reference here. 2020.
[2] A second reference here. 2021.
[3] A third reference sits here. 2019.
"""

OTHER_RE = __import__("re").compile(r"^\[(\d+)\]|cites \"([^\"]+)\"")


def _stub_llm(log):
    """Prompt-routed LLM stub: answers each pass's JSON contract generically; the other pass answers exactly the
    (sentence, key) pairs the prompt lists."""
    def chat(prompt, **kw):
        log.append(prompt[:40])
        if "numbered abstract sentences" in prompt:
            return json.dumps({"items": [{"n": 1, "facet": "contribution", "role": "proposes",
                                          "epistemic": "stated", "condition": "",
                                          "text": "proposes a graph attention model"}],
                               "names": [{"name": "GraphFormer", "aliases": [], "artefact": "method",
                                          "relation": "proposes", "n": 1},
                                         {"name": "FastGF", "aliases": [], "artefact": "method",
                                          "relation": "proposes", "n": 1}]})
        if "chunk of a research paper" in prompt:
            return json.dumps({"items": [{"n": 1, "facet": "method", "role": "describes", "epistemic": "stated",
                                          "condition": "", "text": "uses an efficient attention mechanism",
                                          "mentions": []}],
                               "config": [], "own_methods": ["FastGF"]})
        if "repaired into a grid" in prompt:
            return json.dumps({"usable": True, "object_axis": "row", "object_index": 0, "header_rows": [0],
                               "conditions": [], "skip_rows": [],
                               "measures": [{"index": 1, "metric": "score", "unit": "", "direction": "higher"}]})
        if "citation sentences from ONE" in prompt:
            pairs, s = [], None
            for line in prompt.splitlines():
                m = OTHER_RE.match(line.strip())
                if m and m.group(1):
                    s = int(m.group(1))
                elif m and m.group(2) and s is not None:
                    pairs.append({"s": s, "key": m.group(2), "function": "basis", "role": "uses", "about": None,
                                  "facet": "contribution", "epistemic": "stated", "category": None,
                                  "limitation": None, "name": None, "outcome": None, "builds_on": None,
                                  "builds_on_relation": None})
            return json.dumps({"pairs": pairs})
        raise AssertionError(f"unrouted prompt: {prompt[:80]}")
    return chat


def test_extract_stage_endtoend(env, tmp_path, monkeypatch):
    from compilescholar.citations import build as CB
    from compilescholar.documents import build as DB
    from compilescholar.extract import build as EB
    from compilescholar.extract import passes as PS
    # scope: P0 (the cited target), P1 and P2 as benchmark members with arXiv records
    con = _reg()
    for i, (pid, t) in enumerate([(P0, "Neural message passing for quantum chemistry"),
                                  (P1, "GraphFormer: attention for graphs"), (P2, "Faster graph transformers")]):
        con.execute("INSERT OR REPLACE INTO members VALUES (?,?,?)", ("testbench", f"k{i}", pid))
        if pid != P0:
            con.execute("INSERT OR REPLACE INTO records(paper_id, source, version, date_hi, title, title_key, "
                        "abstract, categories, venue) VALUES (?,?,?,?,?,?,?,?,?)",
                        (pid, "arxiv", 1, None, t, ids.title_key(t), None, "cs.LG", None))
    con.commit()
    con.close()
    _texts(tmp_path, P2, SV_MD_EXT)                       # the sv tier for P2 (table + prose)
    DB.build(workers=2, log=lambda *a: None)
    CB.build(workers=2, log=lambda *a: None)              # documents grew (sv): citations re-validates, no-op
    # the other pass targets the DEEP set (ring-2 reception records; the 10-10 mega-run separates deep from
    # scope). The fixture's cites all point at P0, which has no full text -> not deep. Add two rows citing P1
    # so the pass stays covered end to end, preserving the v1-date and self-citation chain checks.
    cw = env.connect("citations")
    for citing, self_cite in ((P2, 1), (JD, 0)):
        r = cw.execute("SELECT sentence_id, date, version, key, n_group FROM cites WHERE citing=? AND cited=?",
                       (citing, P0)).fetchone()
        cw.execute("INSERT INTO cites VALUES (?,?,?,?,?,?,?,?)", (r[0], citing, r[1], r[2], r[3], P1, r[4], self_cite))
    cw.commit()
    cw.close()
    calls = []
    monkeypatch.setattr(PS, "call_local", _stub_llm(calls))
    counts = EB.build(benchmark="testbench", n_deep=5, workers=2, log=lambda *a: None)
    assert counts["scope"] == 3 and counts["deep"] == 2   # P0 has no full text -> not deep
    bp = counts["by_pass"]
    assert bp.get("t1", 0) >= 4                           # P1: item + GraphFormer; P2: item + FastGF
    assert bp.get("t2", 0) >= 2 and bp.get("results") == 2
    assert bp.get("other") == 2                           # the two citing papers of the DEEP paper P1
    con = env.connect("extract", readonly=True)
    row = con.execute("SELECT date, epistemic, loc, schema_version, run_id, model, prompt_sha, meta FROM statements "
                      "WHERE pass='other' AND speaker=?", (P2,)).fetchone()
    assert row[0] == "2021-01-12"                         # the v1 citing sentence keeps its v1 date
    assert row[3] == 2 and row[4] and row[5] and row[6]
    assert json.loads(row[7])["self_cite"] == 1           # Gilmer cites Gilmer (C② -> C④ chain)
    res = con.execute("SELECT text, epistemic, quote, meta FROM statements WHERE pass='results' ORDER BY text"
                      ).fetchall()
    assert any("FastGF" in r[0] and "91.2" in r[0] and r[1] == "demonstrated" for r in res)
    assert any("Base" in r[0] and "88.0" in r[0] and r[1] == "cited" for r in res)
    assert all("91.2" in r[2] or "88.0" in r[2] for r in res)     # quotes carry the cell values verbatim
    cc = con.execute("SELECT month, n FROM cite_counts WHERE cited=?", (P0,)).fetchall()
    assert cc and sum(n for _, n in cc) >= 2
    n_first = con.execute("SELECT count(*) FROM statements").fetchone()[0]
    con.close()
    # rebuild with nothing changed: no work, no new LLM calls, no new rows
    calls.clear()
    counts2 = EB.build(benchmark="testbench", n_deep=5, workers=2, log=lambda *a: None)
    assert calls == [] and counts2["statements"] == n_first


# ---------------------------------------------------------------- cognition -> index -> tools (phase D full chain)

def _fake_embed(texts):
    """Deterministic bag-of-words vectors (dim 16) for the index stage — no embedding service in tests."""
    import re as _re
    import numpy as _np
    out = []
    for t in texts:
        v = _np.zeros(16, dtype="float32")
        for w in _re.findall(r"[a-z]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 16] += 1.0
        n = float(_np.linalg.norm(v))
        out.append((v / n).tolist() if n else v.tolist())
    return out


class _CountingEmbed:
    def __init__(self):
        self.n = 0

    def __call__(self, texts):
        self.n += len(texts)
        return _fake_embed(texts)


def test_full_chain_cognition_index_tools(env, tmp_path, monkeypatch):
    """The whole derived pipeline on one library: documents -> citations -> extract -> cognition -> index ->
    tools, with the leak invariants at every layer (a paper invisible before its first public date; a later
    version's sentence invisible before that version) and the index's incremental discipline."""
    from compilescholar.citations import build as CB
    from compilescholar.cognition import build as COG
    from compilescholar.documents import build as DB
    from compilescholar.extract import build as EB
    from compilescholar.extract import passes as PS
    from compilescholar.index import build as IB
    from compilescholar.index.search import Index
    from compilescholar.tools import api

    con = _reg()
    for i, (pid, t, ab) in enumerate([
            (P0, "Neural message passing for quantum chemistry", None),
            (P1, "GraphFormer: attention for graphs", "We propose GraphFormer, an attention model for graphs."),
            (P2, "Faster graph transformers", "We propose FastGF, an efficient graph transformer.")]):
        con.execute("INSERT OR REPLACE INTO members VALUES (?,?,?)", ("testbench", f"k{i}", pid))
        if pid != P0:
            con.execute("INSERT OR REPLACE INTO records(paper_id, source, version, date_hi, title, title_key, "
                        "abstract, categories, venue) VALUES (?,?,?,?,?,?,?,?,?)",
                        (pid, "arxiv", 1, None, t, ids.title_key(t), ab, "cs.LG", None))
    con.commit()
    con.close()
    _texts(tmp_path, P2, SV_MD_EXT)
    DB.build(workers=2, log=lambda *a: None)
    CB.build(workers=2, log=lambda *a: None)
    monkeypatch.setattr(PS, "call_local", _stub_llm([]))
    EB.build(benchmark="testbench", n_deep=5, workers=2, log=lambda *a: None)

    # cognition on this fixture is fully deterministic: unambiguous method names, no co-citation pairs,
    # no shift candidate reaches MIN_N votes, no families -> zero LLM adjudications
    def no_chat(prompt, **kw):
        raise AssertionError(f"cognition wanted the LLM on a deterministic fixture: {prompt[:80]}")
    COG.build(workers=2, log=lambda *a: None, chat=no_chat)

    emb = _CountingEmbed()
    counts = IB.build(dense=("papers", "statements"), embed=emb, model_label="fake", log=lambda *a: None)
    assert counts["papers"] >= 3 and counts["statements"] >= 8 and counts["passages"] >= 4
    assert counts["vec_papers"] >= 2 and counts["vec_statements"] >= 2
    for s in env.STAGES:                                    # the whole chain is fresh end to end
        st = env.status(s)
        assert st["built"] and not st["stale"] and not st["behind"], (s, st["why"])

    idx = Index(embed=_fake_embed, model_label="fake")
    api.reset_caches()
    api.set_index(idx)
    api.configure(api.ToolConfig(arm="full", external=False))
    try:
        late = "2022-01-01"
        got = api.search_papers("graph attention transformer", late, 5)
        assert {x["id"] for x in got} >= {P1, P2}
        card = api.paper_card(P2, late)
        assert any(u["value"] == "91.2" and u["own"] for u in card["result_units"])
        ev = api.find_evidence("efficient attention mechanism", late, 8)
        assert ev and all(e.get("quote") for e in ev)

        # ---- leak invariants
        early = "2020-12-31"                                  # before P2's first public date (2021-01-12)
        assert api.paper_card(P2, early)["error"] == "not visible at as_of"
        assert all(x["id"] != P2 for x in api.search_papers("faster graph transformers", early, 10))
        assert api.read(P2, early, k=10) == []
        mid = "2021-03-01"                                    # v1 visible, v2 (2021-06-01) not
        mid_hits = api.read(P2, mid, k=50)
        assert mid_hits and all(h["date"] <= mid for h in mid_hits)
        assert not any("convergence bound" in h["quote"] for h in mid_hits)
        late_hits = api.read(P2, late, k=50)
        assert any("convergence bound" in h["quote"] and h["date"] == "2021-06-01" for h in late_hits)
        q_mid = api.read(P2, mid, query="convergence bound attention", k=5)
        assert not any("convergence bound" in h["quote"] for h in q_mid)   # the passages index agrees

        # ---- the index is incremental: an unchanged chain re-embeds and re-indexes nothing
        # (a reader must close its memmaps before a rebuild replaces the vector files — Windows locks
        # mapped files; Index.close's docstring is the operational contract)
        idx.close()
        api.set_index(None)
        emb.n = 0
        counts2 = IB.build(dense=("papers", "statements"), embed=emb, model_label="fake", log=lambda *a: None)
        assert emb.n == 0 and counts2 == counts
    finally:
        idx.close()
        api.set_index(None)
        api.reset_caches()
        api.configure(api.ToolConfig())
