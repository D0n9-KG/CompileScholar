# -*- coding: utf-8 -*-
"""Stage `extract` (L2): tiered extraction over a question-blind scope, written to data/dfc/extract.sqlite.

Tiers (DESIGN-LITERATURE-LAYER §3.5; the old system's coarse-for-breadth / deep-for-important split, kept):
  T1  self pass on title + abstract          every paper in scope (papers stage)
  T2  self pass on abstract + introduction    deep papers (selected below) that have a full text (documents stage),
      + deep call on method/experiment text   same papers
      + result pass (deterministic tables)    same papers
  R   other pass on citation sentences        every cited object in scope, <= N_OTHER pairs each, stratified over
                                              citing months (round-robin, oldest first) and distinct citing papers
Scope and deep selection are question-blind: they use only corpus statistics (categories, citation counts from the
citations stage), never benchmark annotations:
  scope = papers whose primary category is in `categories` and v1_date >= since
  deep  = scope papers with a full text, ranked by in-corpus citation count, top `n_deep`
Resumable: done_self / done_deep / done_pairs tables. Every statement passes schema validation and its quote check."""
from __future__ import annotations

import concurrent.futures as cf
import json
import threading
from collections import defaultdict

from ..corpus.papers import Papers
from ..dfc import store
from ..documents import units as U
from ..documents.build import Documents
from . import other_pass as O
from . import result_pass as R
from . import self_pass as S
from .schema import DDL

N_OTHER = 30
EXTRA_DDL = """CREATE TABLE IF NOT EXISTS done_self(arxiv_id TEXT PRIMARY KEY, tier TEXT, stats TEXT);
CREATE TABLE IF NOT EXISTS done_pairs(sentence_id INT, cited TEXT, PRIMARY KEY(sentence_id, cited));
CREATE TABLE IF NOT EXISTS scope(arxiv_id TEXT PRIMARY KEY, deep INT);"""


def select_scope(categories: tuple[str, ...], since: str, n_deep: int) -> tuple[list[str], set[str]]:
    """Question-blind scope + deep set (corpus statistics only)."""
    pap = store.connect("papers", readonly=True)
    cit = store.connect("citations", readonly=True)
    docs = set(Documents().ids())
    q = f"SELECT arxiv_id FROM papers WHERE primary_cat IN ({','.join('?' * len(categories))}) AND v1_date >= ?"
    scope = [r[0] for r in pap.execute(q, (*categories, since))]
    counts = dict(cit.execute("SELECT substr(cited, 7), count(DISTINCT citing) FROM cites "
                              "WHERE cited LIKE 'paper:%' GROUP BY cited").fetchall())
    ranked = sorted((a for a in scope if a in docs), key=lambda a: (-counts.get(a, 0), a))
    return scope, set(ranked[:n_deep])


def select_pairs(cit, targets, n: int = N_OTHER) -> dict[str, list[tuple]]:
    """{target: [(sentence_id, citing, date, n_group), ...]} capped at n, spread over citing months (round-robin,
    oldest first) and, within a month, over distinct citing papers first."""
    out = {}
    q = "SELECT sentence_id, citing, date, n_group FROM cites WHERE cited = ? ORDER BY date, citing, sentence_id"
    for t in targets:
        months: dict[str, list] = defaultdict(list)
        extra: dict[str, list] = defaultdict(list)
        seen = set()
        for sid, citing, date, ng in cit.execute(q, (t,)):
            (extra if (citing, date[:7]) in seen else months)[date[:7]].append((sid, citing, date, ng))
            seen.add((citing, date[:7]))
        picked = []
        for pool in (months, extra):
            queues = [list(pool[m]) for m in sorted(pool)]
            while len(picked) < n and any(queues):
                for qq in queues:
                    if qq and len(picked) < n:
                        picked.append(qq.pop(0))
        if picked:
            out[t] = picked
    return out


def build(categories: tuple[str, ...] = ("cs.LG",), since: str = "2018-01-01", n_deep: int = 500,
          workers: int = 48, log=print) -> dict:
    store.require_fresh("papers", "documents", "citations")
    papers, docs = Papers(), Documents()
    cit = store.connect("citations", readonly=True)
    con = store.connect("extract")
    con.executescript(DDL + EXTRA_DDL)
    scope, deep = select_scope(categories, since, n_deep)
    con.executemany("INSERT OR REPLACE INTO scope VALUES (?,?)", [(a, int(a in deep)) for a in scope])
    con.commit()
    log(f"[extract] scope {len(scope)} papers ({categories}, since {since}); deep {len(deep)}")
    lock = threading.Lock()
    tot = defaultdict(int)

    def write(stmts, prompt_sha):
        rows = []
        for s in stmts:
            if s.validate():
                tot["schema_rejected"] += 1
                continue
            r = s.row()
            rows.append((r["speaker"], r["date"], r["kind"], r["about"], r["role"], r["facet"], r["text"], r["quote"],
                         r["target"], json.dumps(r["group"]), r["function"], json.dumps(r["meta"], ensure_ascii=False),
                         S.MODEL, prompt_sha))
        con.executemany("INSERT INTO statements(speaker,date,kind,about,role,facet,text,quote,target,grp,function,"
                        "meta,model,prompt_sha) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    # ---- T1 / T2 self pass (+ deep call + result pass for T2)
    done_self = {r[0] for r in con.execute("SELECT arxiv_id FROM done_self")}

    def do_self(aid):
        p = papers.get(aid)
        if not p or not p["v1_date"]:
            return
        body = method = None
        res_stmts, res_st = [], {}
        if aid in deep:
            d = docs.get(aid)
            body, method = U.abstract_and_intro(d["units"]), U.method_and_experiments(d["units"])
            res_stmts, res_st = R.run(aid, p["v1_date"], d["raw"], d["source"])
        stmts, st = S.run(aid, p["v1_date"], p["title"], p["abstract"], body, method_text=method)
        with lock:
            write(stmts, S.PROMPT_SHA)
            write(res_stmts, "results-deterministic")
            con.execute("INSERT OR REPLACE INTO done_self VALUES (?,?,?)",
                        (aid, "T2" if aid in deep else "T1", json.dumps({**st, **res_st})))
            con.commit()
            for k, v in {**st, **res_st}.items():
                tot["self_" + k] += v

    todo = [a for a in scope if a not in done_self]
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(do_self, todo)):
            if (i + 1) % 500 == 0:
                log(f"[extract] self {i + 1}/{len(todo)} {dict(tot)}")

    # ---- R: other pass over citation sentences about scope papers
    pairs = select_pairs(cit, [f"paper:{a}" for a in scope])
    done = {(a, b) for a, b in con.execute("SELECT sentence_id, cited FROM done_pairs")}
    by_citing: dict[str, list[dict]] = defaultdict(list)
    for t, lst in pairs.items():
        for sid, citing, date, ng in lst:
            if (sid, t) in done:
                continue
            sent = cit.execute("SELECT sentence FROM sentences WHERE id=?", (sid,)).fetchone()[0]
            ent = cit.execute("SELECT raw, title FROM entries WHERE citing=? AND cited=? LIMIT 1", (citing, t)).fetchone()
            grp = [r[0] for r in cit.execute("SELECT cited FROM cites WHERE sentence_id=?", (sid,))]
            by_citing[citing].append({"sentence_id": sid, "sentence": sent, "cited": t, "date": date,
                                      "title": ent[1] if ent else "", "raw": ent[0] if ent else "", "group": grp})
    batches = [(c, items[i:i + O.BATCH]) for c, items in by_citing.items() for i in range(0, len(items), O.BATCH)]
    log(f"[extract] other pass: {sum(len(v) for v in pairs.values())} pairs for {len(pairs)} cited papers, "
        f"{len(batches)} batches")

    def do_other(b):
        citing, items = b
        stmts, st = O.run_batch(citing, items[0]["date"], items)
        with lock:
            write(stmts, O.PROMPT_SHA)
            con.executemany("INSERT OR IGNORE INTO done_pairs VALUES (?,?)",
                            [(it["sentence_id"], it["cited"]) for it in items])
            con.commit()
            for k, v in st.items():
                tot["other_" + k] += v

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(do_other, batches)):
            if (i + 1) % 200 == 0:
                log(f"[extract] other {i + 1}/{len(batches)} {dict(tot)}")

    counts = {"statements": con.execute("SELECT count(*) FROM statements").fetchone()[0],
              "self": con.execute("SELECT count(*) FROM statements WHERE kind='self'").fetchone()[0],
              "other": con.execute("SELECT count(*) FROM statements WHERE kind='other'").fetchone()[0],
              "results": con.execute("SELECT count(*) FROM statements WHERE facet='result' AND kind='self'").fetchone()[0],
              "scope": len(scope), "deep": len(deep), **tot}
    con.close()
    store.write_manifest("extract", {"categories": list(categories), "since": since, "n_deep": n_deep,
                                     "n_other": N_OTHER, "self_prompt": S.PROMPT_SHA, "deep_prompt": S.DEEP_SHA,
                                     "other_prompt": O.PROMPT_SHA, "model": S.MODEL}, counts)
    return counts
