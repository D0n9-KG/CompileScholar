# -*- coding: utf-8 -*-
"""Schema v2 (INTEGRATED-SYSTEM-1005 §7.1): the closed vocabularies, the new first-class fields (epistemic,
condition, loc, schema_version), bare paper_id objects, config-statement shape, and the DDL's compatibility with
v1-style explicit-column inserts (cognition reads SELECT * and inserts by column name until phase D)."""
from __future__ import annotations

import json
import sqlite3

from compilescholar.extract.schema import DDL, EPISTEMIC, FACETS, SCHEMA_VERSION, Statement

GOOD = dict(speaker="arxiv:1706.06083", date="2017-06-12", kind="other", about="arxiv:1234.5678",
            role="uses", facet="method", text="uses a graph encoder", quote="We use a graph encoder.",
            loc={"unit_id": "arxiv:1706.06083@v1#para3", "sent_id": "arxiv:1706.06083@v1#s7"})


def _st(**over):
    return Statement(**{**GOOD, **over})


def test_good_statement_validates():
    assert _st().validate() == []
    assert _st().schema_version == SCHEMA_VERSION == 2


def test_vocabularies_are_closed():
    assert _st(role="inspires").validate()                     # not a relation
    assert _st(facet="vibes").validate()
    assert _st(epistemic="guessed").validate()
    assert _st(function="inspiration").validate()
    assert set(EPISTEMIC) == {"demonstrated", "stated", "hypothesized", "cited"}
    for f in ("config", "absence", "definition"):              # §7.1's new facets
        assert f in FACETS


def test_self_only_roles_and_other_target():
    assert _st(kind="other", role="proposes").validate()       # proposes is self-only
    assert _st(kind="other", role="describes").validate()
    # kind=other may carry a target: a third party's claim "A extends B" (§7.1)
    assert _st(kind="other", role="extends", target="arxiv:9999.8888").validate() == []
    assert _st(kind="other", role="extends", target="paper:9999.8888").validate()   # old prefix abolished


def test_objects_are_bare_paper_ids_or_stubs():
    for about in ("arxiv:1234.5678", "doi:10.1/x", "title:some work|2020", "stub:some work",
                  "method:FastGF@arxiv:2101.00002"):
        assert _st(about=about).validate() == [], about
    assert _st(about="paper:1234.5678").validate()             # the pre-C prefix is gone
    assert _st(about="1234.5678").validate()                   # a bare arXiv id without its scheme is wrong too


def test_config_statement_shape():
    cfg = {"item": "learning rate", "value": "3e-4", "unit": "", "applies_to": "all experiments"}
    assert _st(facet="config", meta={"config": cfg}).validate() == []
    assert _st(facet="config", meta={}).validate()             # config without meta.config
    assert _st(facet="config", meta={"config": {"item": "lr"}}).validate()          # no value
    assert _st(facet="config", meta={"config": {"item": "lr", "value": 3}}).validate()  # missing unit/applies_to


def test_loc_anchor_and_dates_required():
    assert _st(loc={}).validate()                              # no anchor at all
    assert _st(loc={"sent_id": "x#s1"}).validate() == []       # one anchor is enough (chars come at the final check)
    assert _st(date="2017").validate()                         # a statement must carry a full text-version date
    assert _st(text="").validate() and _st(quote="").validate()


def test_row_and_ddl_roundtrip():
    st = _st(group=("arxiv:1", "stub:x"), pass_name="t2", item="arxiv:1706.06083", run_id="r1",
             model="m", prompt_sha="abc")
    r = st.row()
    assert r["pass"] == "t2" and "pass_name" not in r and r["group"] == ["arxiv:1", "stub:x"]
    con = sqlite3.connect(":memory:")
    con.executescript(DDL)
    con.execute("INSERT INTO statements(speaker,date,kind,about,role,facet,text,quote,epistemic,condition,loc,"
                "target,grp,function,meta,schema_version,pass,item,run_id,model,prompt_sha) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r["speaker"], r["date"], r["kind"], r["about"], r["role"], r["facet"], r["text"], r["quote"],
                 r["epistemic"], r["condition"], json.dumps(r["loc"]), r["target"], json.dumps(r["group"]),
                 r["function"], json.dumps(r["meta"]), r["schema_version"], r["pass"], r["item"], r["run_id"],
                 r["model"], r["prompt_sha"]))
    # the natural key: the same (pass, item, statement) twice is an error, not a silent copy
    try:
        con.execute("INSERT INTO statements(pass,item,speaker,kind,about,role,facet,text,quote) "
                    "VALUES ('t2','arxiv:1706.06083','arxiv:1706.06083','other','arxiv:1234.5678','uses','method',"
                    "'uses a graph encoder','We use a graph encoder.')")
        raise AssertionError("duplicate natural key accepted")
    except sqlite3.IntegrityError:
        pass
    con.close()


def test_ddl_accepts_v1_style_inserts():
    """cognition (until phase D) inserts by the v1 column list and reads SELECT * — the v2 DDL must keep every
    v1 column and leave the new ones nullable."""
    con = sqlite3.connect(":memory:")
    con.executescript(DDL)
    con.execute("INSERT INTO statements(speaker,date,kind,about,role,facet,text,quote,target,grp,function,meta,"
                "model,prompt_sha) VALUES ('2302.1','2023-02-23','other','paper:1706.06083','uses','method','x','y',"
                "NULL,'[]',NULL,'{}','m','sha')")
    assert con.execute("SELECT count(*) FROM statements").fetchone()[0] == 1
    con.close()
