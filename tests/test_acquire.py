# -*- coding: utf-8 -*-
"""acquire: identity check on real-shaped PDFs (made with PyMuPDF), version plan, assets / attempts / quarantine."""
import pymupdf
import pytest

from compilescholar.acquire import channels as CH
from compilescholar.acquire import run as R
from compilescholar.acquire import verify as V
from compilescholar.library import import_arxiv as A
from compilescholar.library import identity as I
from compilescholar.library import store


def pdf(*lines) -> bytes:
    d = pymupdf.open()
    p = d.new_page()
    y = 72
    for ln in lines:
        p.insert_text((72, y), ln, fontsize=10, fontname="china-s")      # a builtin font with Greek glyphs
        y += 14
    filler = " ".join(["The method is evaluated on several benchmarks and compared with baselines."] * 4)
    for i in range(6):
        p.insert_text((72, y + 14 * i), filler[:90], fontsize=9)
    return d.tobytes()


TITLE = "Experience Replay for Continual Learning"


def test_own_arxiv_stamp_with_title_is_ok():
    b = pdf("arXiv:1811.11682v1 [cs.LG] 28 Nov 2018", TITLE, "David Rolnick, Arun Ahuja")
    assert V.check(b, TITLE, ["Rolnick", "Ahuja"], "1811.11682", None)["verdict"] == "ok"


def test_title_and_surname_is_ok_unicode():
    t = "α-Synuclein aggregation in neurons"
    b = pdf("α-Synuclein Aggregation in Neurons", "Jürgen Müller and Ana Pérez")
    assert V.check(b, t, ["Müller"], None, None)["verdict"] == "ok"


def test_wrong_paper_is_mismatch_and_beta_is_not_alpha():
    b = pdf("Protein folding with deep networks", "A. Other")
    assert V.check(b, "Deep learning for image segmentation", ["Smith"], None, None)["verdict"] == "mismatch"
    b = pdf("β-Synuclein aggregation in neurons", "Jürgen Müller")
    r = V.check(b, "α-Synuclein aggregation in neurons", ["Müller"], None, None)
    assert r["verdict"] != "ok"


def test_no_text_layer_is_unsure_not_ok():
    d = pymupdf.open()
    d.new_page()
    assert V.check(d.tobytes(), TITLE, ["Rolnick"], None, None)["verdict"] == "unsure"


def test_surname_with_affiliation_marks():
    b = pdf("Visual Acoustic Matching", "Changan Chen1,4  Ruohan Gao2  Kristen Grauman1,4")
    r = V.check(b, "Visual Acoustic Matching", ["Chen", "Gao"], None, None)
    assert r["surname"] is True and r["verdict"] == "ok"


def test_unreadable_is_mismatch():
    assert V.check(b"not a pdf", TITLE, [], None, None)["verdict"] == "mismatch"


def test_gcs_url_new_and_old_style():
    assert CH.gcs_url("1706.03762", 1).endswith("/arxiv/pdf/1706/1706.03762v1.pdf")
    assert CH.gcs_url("cs/0408007", 1).endswith("/arxiv/cs/pdf/0408/0408007v1.pdf")
    assert CH.gcs_url("math.GT/0309136", 2).endswith("/arxiv/math/pdf/0309/0309136v2.pdf")


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setenv("CS_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(R.paths, "library", lambda: tmp_path / "data" / "library")
    db = tmp_path / "data" / "library" / "registry.sqlite"
    con = store.connect(db)
    w = I.Writer(con, "arxiv")
    w.add(A._incoming("1811.11682", {"v1": "2018-11-28", "v2": "2018-12-05", "v3": "2019-01-10"}, TITLE, "abs",
                      [{"name": "David Rolnick", "surname": "Rolnick"}], ["cs.LG"], None))
    w.add(A._incoming("1902.00001", {"v1": "2019-02-01"}, "A Paper Nobody Mirrored", "abs",
                      [{"name": "Ann Lee", "surname": "Lee"}], ["cs.LG"], None))
    w.close()
    con.close()
    return db


def test_plan_wants_v1_and_latest(lib):
    con = store.connect(lib)
    assert [(w.arxiv, w.version) for w, _ in R.plan(con, "arxiv:1811.11682")] == [("1811.11682", 1),
                                                                                  ("1811.11682", 3)]
    assert [w.version for w, _ in R.plan(con, "arxiv:1902.00001")] == [1]


def test_acquire_records_assets_attempts_and_quarantine(lib, monkeypatch):
    good = pdf("arXiv:1811.11682v1 [cs.LG]", TITLE, "David Rolnick")
    wrong = pdf("Some Other Paper Entirely About Graphs", "Bob Stone")

    def nas(w):
        if w.arxiv == "1811.11682" and w.version == 1:
            return CH.Fetched("arxiv_nas", 1, good, {"channel": "arxiv_nas", "path": "x"})
        return None

    def gcs(w):
        if w.arxiv == "1902.00001":
            return CH.Fetched("arxiv_gcs", 1, wrong, {"channel": "arxiv_gcs", "url": "u"})
        if w.version == 3:
            return CH.Fetched("arxiv_gcs", 3, good, {"channel": "arxiv_gcs", "url": "u3"})
        return None
    monkeypatch.setitem(CH.CHANNELS, "arxiv_nas", nas)
    monkeypatch.setitem(CH.CHANNELS, "arxiv_gcs", gcs)
    n = R.acquire(["arxiv:1811.11682", "arxiv:1902.00001"], workers=2, llm=False, path=lib)
    assert n["assets"] == 2 and n["mismatch"] == 1
    con = store.connect(lib)
    rows = con.execute("SELECT paper_id, version, channel, identity FROM assets ORDER BY version").fetchall()
    assert rows == [("arxiv:1811.11682", 1, "arxiv_nas", "ok"), ("arxiv:1811.11682", 3, "arxiv_gcs", "ok")]
    sha_wrong = CH.sha256(wrong)
    assert R.quarantine_path(sha_wrong).exists()
    assert con.execute("SELECT status FROM attempts WHERE paper_id='arxiv:1902.00001' AND channel='arxiv_gcs'"
                       ).fetchone()[0] == "mismatch"
    assert R.pdf_path(CH.sha256(good)).exists()                     # the GCS download is stored, NAS is a pointer
    # a second run skips what is already acquired
    assert R.acquire(["arxiv:1811.11682"], workers=1, llm=False, path=lib)["assets"] == 0
