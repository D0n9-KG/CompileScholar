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
Incremental and exact through dfc.store item bookkeeping (passes `self` and `other`); every statement row carries the
(pass, item) that produced it. Every statement passes schema validation and its quote check."""
from __future__ import annotations

import json
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
EXTRA_DDL = """CREATE TABLE IF NOT EXISTS tiers(arxiv_id TEXT PRIMARY KEY, tier TEXT, stats TEXT);
CREATE TABLE IF NOT EXISTS scope(arxiv_id TEXT PRIMARY KEY, deep INT);"""
INSERT = ("INSERT OR IGNORE INTO statements(speaker,date,kind,about,role,facet,text,quote,target,grp,function,meta,"
          "model,prompt_sha,pass,item) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")


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
          workers: int = 48, rebuild: bool = False, log=print) -> dict:
    """Two passes, each with its own item bookkeeping (dfc.store.Work):
      self   item = paper; key = (self prompts, model, the stage's code) + (paper record, tier, document key)
      other  item = "<citing>|<cited>" (the sentences one citing paper wrote about one cited object); key = (other
             prompt, model, code) + (the citations item of the citing paper, the selected sentence ids)
    A changed prompt re-opens its own pass only; a failed LLM call is retried on the next build and never counted as
    done; items that left the scope are removed with their statements."""
    params = {"categories": list(categories), "since": since, "n_deep": n_deep, "n_other": N_OTHER}
    with store.Run("extract", params, rebuild=rebuild) as run:
        papers, docs = Papers(), Documents()
        cit = store.connect("citations", readonly=True)
        con = run.con
        con.executescript(DDL + EXTRA_DDL)
        scope, deep = select_scope(categories, since, n_deep)
        con.execute("DELETE FROM scope")
        con.executemany("INSERT INTO scope VALUES (?,?)", [(a, int(a in deep)) for a in scope])
        con.commit()
        log(f"[extract] scope {len(scope)} papers ({categories}, since {since}); deep {len(deep)}")
        lock = run.lock
        tot = defaultdict(int)

        def rows(stmts, prompt_sha, pass_name, item):
            out = []
            for s_ in stmts:
                if s_.validate():
                    tot["schema_rejected"] += 1
                    continue
                r = s_.row()
                out.append((r["speaker"], r["date"], r["kind"], r["about"], r["role"], r["facet"], r["text"],
                            r["quote"], r["target"], json.dumps(r["group"]), r["function"],
                            json.dumps(r["meta"], ensure_ascii=False), S.MODEL, prompt_sha, pass_name, item))
            return out

        def replace_item(pass_name, item, new_rows):
            con.execute("DELETE FROM statements WHERE pass=? AND item=?", (pass_name, item))
            con.executemany(INSERT, new_rows)

        # ---- T1 / T2 self pass (+ deep call + result pass for T2)
        doc_keys = store.item_keys("documents", "documents")
        w_self = run.work("self", store.sha(run.digest, S.PROMPT_SHA, S.DEEP_SHA, S.MODEL))

        def self_fp(aid):
            p = papers.get(aid) or {}
            return store.sha(p.get("v1_date"), p.get("title"), p.get("abstract"), aid in deep,
                             doc_keys.get(aid) if aid in deep else None)

        todo = w_self.todo((a, self_fp(a)) for a in scope)

        def drop_self(aid):
            replace_item("self", aid, [])
            con.execute("DELETE FROM tiers WHERE arxiv_id=?", (aid,))

        def do_self(aid):
            p = papers.get(aid)
            if not p or not p["v1_date"]:
                with lock:
                    replace_item("self", aid, [])
                    w_self.ok(aid)
                return
            body = method = None
            res_stmts, res_st = [], {}
            if aid in deep:
                d = docs.get(aid)
                body, method = U.abstract_and_intro(d["units"]), U.method_and_experiments(d["units"])
                res_stmts, res_st = R.run(aid, p["v1_date"], d["raw"], d["source"])
            stmts, st = S.run(aid, p["v1_date"], p["title"], p["abstract"], body, method_text=method)
            if st.get("parse_failed") or st.get("deep_parse_failed"):
                with lock:      # the item has no valid output under its current key: same as a clean build
                    drop_self(aid)
                    w_self.fail(aid, "self pass: no parseable answer" if st.get("parse_failed")
                                else "deep call: no parseable answer")
                return
            with lock:
                replace_item("self", aid, rows(stmts, S.PROMPT_SHA, "self", aid) +
                             rows(res_stmts, "results-deterministic", "self", aid))
                con.execute("INSERT OR REPLACE INTO tiers VALUES (?,?,?)",
                            (aid, "T2" if aid in deep else "T1", json.dumps({**st, **res_st})))
                w_self.ok(aid)
                con.commit()
                for k, v in {**st, **res_st}.items():
                    tot["self_" + k] += v

        store.parallel(do_self, todo, workers, log, 500, "extract self")
        tot["self_removed"] = w_self.sweep(scope, drop_self)

        # ---- R: other pass over citation sentences about scope papers
        cit_keys = store.item_keys("citations", "citations")
        pairs = select_pairs(cit, [f"paper:{a}" for a in scope])
        groups: dict[str, list[dict]] = defaultdict(list)
        for t, lst in pairs.items():
            for sid, citing, date, ng in lst:
                groups[f"{citing}|{t}"].append({"sentence_id": sid, "citing": citing, "cited": t, "date": date})
        w_other = run.work("other", store.sha(run.digest, O.PROMPT_SHA, O.MODEL))
        todo = set(w_other.todo((g, store.sha(cit_keys.get(g.split("|")[0]), [it["sentence_id"] for it in items]))
                                for g, items in groups.items()))
        by_citing: dict[str, list[list[dict]]] = defaultdict(list)
        for g in sorted(todo):
            grp_items = []
            for it in groups[g]:
                sent = cit.execute("SELECT sentence FROM sentences WHERE id=?", (it["sentence_id"],)).fetchone()[0]
                ent = cit.execute("SELECT raw, title FROM entries WHERE citing=? AND cited=? LIMIT 1",
                                  (it["citing"], it["cited"])).fetchone()
                grp = [r[0] for r in cit.execute("SELECT cited FROM cites WHERE sentence_id=?", (it["sentence_id"],))]
                grp_items.append({**it, "group_id": g, "sentence": sent, "title": ent[1] if ent else "",
                                  "raw": ent[0] if ent else "", "group": grp})
            by_citing[groups[g][0]["citing"]].append(grp_items)
        # whole groups per batch (an item's rows are replaced as a unit, so a group never spans two calls)
        batches = []
        for c, glist in by_citing.items():
            cur: list[dict] = []
            for gi in glist:
                if cur and len(cur) + len(gi) > O.BATCH:
                    batches.append((c, cur))
                    cur = []
                cur = cur + gi
            if cur:
                batches.append((c, cur))
        log(f"[extract] other pass: {sum(len(v) for v in groups.values())} pairs in {len(groups)} groups, "
            f"{len(todo)} to do, {len(batches)} batches")

        def do_other(b):
            citing, items = b
            gids = list(dict.fromkeys(it["group_id"] for it in items))
            stmts, st = O.run_batch(citing, items[0]["date"], items)
            if st.get("failed"):
                with lock:
                    for g in gids:
                        replace_item("other", g, [])
                    w_other.fail_many(gids, "other pass: no parseable answer")
                return
            by_g: dict[str, list] = defaultdict(list)
            for stmt in stmts:
                by_g[f"{citing}|{stmt.about}"].append(stmt)
            with lock:
                for g in gids:
                    replace_item("other", g, rows(by_g.get(g, []), O.PROMPT_SHA, "other", g))
                w_other.ok_many(gids)
                con.commit()
                for k, v in st.items():
                    tot["other_" + k] += v

        store.parallel(do_other, batches, workers, log, 200, "extract other")
        tot["other_removed"] = w_other.sweep(groups, lambda g: replace_item("other", g, []))

        counts = {"statements": con.execute("SELECT count(*) FROM statements").fetchone()[0],
                  "self": con.execute("SELECT count(*) FROM statements WHERE kind='self'").fetchone()[0],
                  "other": con.execute("SELECT count(*) FROM statements WHERE kind='other'").fetchone()[0],
                  "results": con.execute("SELECT count(*) FROM statements WHERE facet='result' AND kind='self'"
                                         ).fetchone()[0],
                  "scope": len(scope), "deep": len(deep), **tot}
        run.finish(counts)
    return counts

