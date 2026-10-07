# -*- coding: utf-8 -*-
"""Stage `extract` (phase C④): the four passes of INTEGRATED-SYSTEM-1005 §7.3 over a question-blind scope, on
registry keys. The per-paper logic lives in extract/passes.py (shared verbatim with the runtime deep_read —
v2.4 item 1); this file is the stage wiring: scope, work items, persistence, sweeps.

Work passes (every statement row carries pass+item; redoing an item replaces exactly its rows):
  t1       item = a scope paper: title + numbered v1-abstract sentences (registry abstract as marked fallback;
           a paper with neither is a valid empty item)
  t2       item = a deep paper with a full text: chunks <= 8k chars over the tier priority sciverse -> mineru
           -> grobid, sentences dated by their earliest version; seeded with t1's proposed names, returns
           own_methods for the results pass
  results  item = a deep paper with MinerU-family tables (sv/careful tiers): LLM axis roles, structural gate,
           every value from a cell
  other    item = a citing paper: its sampled citation sentences — per cited paper a time-stratified quota
           (§7.3 他述抽样: year buckets, sqrt-of-count allocation, distinct citing papers first within a year)
  figures  item = a paper with a careful parse: deterministic figure units (number, caption, page, bbox — the
           image itself is never stored, tools crop it from the PDF on demand)
Deterministic tail (no work items): cite_counts(cited, month, n) — FULL monthly cited counts (v2.4: the count
is no longer truncated by the sampled subset).

Scope and deep selection are question-blind (the 10-03 user ruling; §13.1): scope = the benchmark's members
(--benchmark) or the registry papers whose primary category is in `categories` with first_hi >= `since`; deep =
scope papers with a materialised full text, ranked by in-corpus cited count, top n_deep. Only corpus statistics
enter — never questions, never gold.

Fingerprints: (prompt sha, model, the documents/citations work keys of everything the pass reads) — a re-parsed
PDF, a new Sciverse fetch, a re-resolved citation or a changed prompt re-opens exactly the items that read it.
A pass whose LLM answer did not parse fails the item (dfc retries, breaker at 3) — never an empty success."""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict

from ..core import paths
from ..dfc import store
from ..documents.build import Documents
from . import passes as PS
from . import prompts as PR
from . import reading as RD
from .schema import DDL

N_OTHER = 30
BATCH_PAIRS = 24
EXTRA_DDL = """CREATE TABLE IF NOT EXISTS scope(paper_id TEXT PRIMARY KEY, deep INT);
CREATE TABLE IF NOT EXISTS pass_stats(paper_id TEXT NOT NULL, pass TEXT NOT NULL, stats TEXT,
  PRIMARY KEY(paper_id, pass));
CREATE TABLE IF NOT EXISTS figure_units(paper_id TEXT NOT NULL, uid TEXT NOT NULL, fig_no INT, caption TEXT,
  page INT, bbox TEXT, PRIMARY KEY(paper_id, uid));
CREATE TABLE IF NOT EXISTS cite_counts(cited TEXT NOT NULL, month TEXT NOT NULL, n INT, PRIMARY KEY(cited, month));
CREATE INDEX IF NOT EXISTS ix_cc_month ON cite_counts(month);"""
INSERT = ("INSERT OR IGNORE INTO statements(speaker,date,kind,about,role,facet,text,quote,epistemic,condition,loc,target,"
          "grp,function,meta,schema_version,pass,item,run_id,model,prompt_sha) "
          "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
_FIG_NO = re.compile(r"(?i)^\s*(?:figure|fig\.?)\s*(\d+)")


def _rows(stmts, run_id: str, tot: dict) -> list[tuple]:
    out = []
    for s in stmts:
        errs = s.validate()
        if errs:
            tot["schema_rejected"] += 1
            continue
        r = s.row()
        out.append((r["speaker"], r["date"], r["kind"], r["about"], r["role"], r["facet"], r["text"], r["quote"],
                    r["epistemic"], r["condition"], json.dumps(r["loc"], ensure_ascii=False), r["target"],
                    json.dumps(r["group"]), r["function"], json.dumps(r["meta"], ensure_ascii=False),
                    r["schema_version"], r["pass"], r["item"], run_id, r["model"], r["prompt_sha"]))
    return out


def select_pairs(cit, targets, cap: int = N_OTHER) -> dict:
    """{cited paper_id: [(sentence_id, citing, date, version, key, self_cite), ...]} — per cited paper a
    time-stratified quota over its citation lifetime (§7.3): year buckets, allocation proportional to
    sqrt(bucket size) with at least one per non-empty year while the cap allows, within a year distinct citing
    papers first and earlier dates first. Deterministic given the store."""
    out = {}
    q = ("SELECT sentence_id, citing, date, version, key, self_cite FROM cites WHERE cited = ? "
         "ORDER BY date, citing, sentence_id")
    for t in targets:
        rows = cit.execute(q, (t,)).fetchall()
        if not rows:
            continue
        years: dict[str, list] = defaultdict(list)
        for r in rows:
            years[(r[2] or "")[:4]].append(r)
        ks = sorted(years)
        if len(rows) <= cap:
            picked = rows
        else:
            weights = {k: len(years[k]) ** 0.5 for k in ks}
            tot_w = sum(weights.values())
            quota = {k: max(1, int(round(cap * weights[k] / tot_w))) for k in ks}
            while sum(quota.values()) > cap:                     # trim from the largest quota
                k = max(quota, key=lambda x: (quota[x], x))
                if quota[k] <= 1:
                    break
                quota[k] -= 1
            picked = []
            for k in ks:
                by_citing: dict[str, list] = defaultdict(list)
                for r in years[k]:
                    by_citing[r[1]].append(r)
                queues = [v for _, v in sorted(by_citing.items())]
                got = 0
                while got < quota[k]:
                    progressed = False
                    for qq in queues:      # distinct citing papers first; each queue is date-ordered (the query)
                        if qq and got < quota[k]:
                            picked.append(qq.pop(0))
                            got += 1
                            progressed = True
                    if not progressed:
                        break
        out[t] = picked[:cap]
    return out


def build(benchmark: str | None = None, categories: tuple[str, ...] = ("cs.LG",), since: str = "2018-01-01",
          n_deep: int = 500, workers: int | None = None, rebuild: bool = False, log=print) -> dict:
    workers = workers or 48                        # LLM lanes (the client paces per backend)
    params = {"benchmark": benchmark, "categories": list(categories), "since": since,
              "n_deep": n_deep, "n_other": N_OTHER}
    run_id = f"extract-{time.strftime('%Y%m%dT%H%M%S')}"
    with store.Run("extract", params, rebuild=rebuild) as run:
        con = run.con
        con.executescript(DDL + EXTRA_DDL)
        if "schema_version" not in {r[1] for r in con.execute("PRAGMA table_info(statements)")}:
            raise RuntimeError("extract.sqlite has the pre-C (v1) schema; build it with --rebuild")
        reg = store.read_only(paths.library() / "registry.sqlite")
        cit_path = store.db_path("citations")
        cit = store.read_only(cit_path) if cit_path.exists() else None
        D = Documents()
        rconn = store.read_only(run.path)          # per-thread reads of this run's file (rebuild writes a .new)
        try:
            # ---- scope + deep (question-blind)
            if benchmark:
                scope = [p for (p,) in reg.execute(
                    "SELECT DISTINCT paper_id FROM members WHERE benchmark=? ORDER BY paper_id", (benchmark,))]
            else:
                likes = " OR ".join("categories LIKE ?" for _ in categories)
                scope = [p for (p,) in reg.execute(
                    "SELECT DISTINCT paper_id FROM records r JOIN papers p ON p.paper_id = r.paper_id "
                    f"WHERE p.status='active' AND p.first_hi >= ? AND r.source='arxiv' AND ({likes}) "
                    "ORDER BY paper_id", (since, *[f"{c}%" for c in categories]))]
            recs: dict[str, tuple] = {}
            scope_set = set(scope)
            for p, t, ab in reg.execute("SELECT paper_id, title, abstract FROM records "
                                        "ORDER BY paper_id, source = 'arxiv'"):
                if p in scope_set:
                    recs[p] = (t or "", ab or "")      # the arxiv row comes last and wins
            cited_counts = dict(cit.execute(
                "SELECT cited, count(DISTINCT citing) FROM cites WHERE cited NOT LIKE 'stub:%' "
                "GROUP BY cited")) if cit is not None else {}
            with_text = set(D.papers())
            deep = set(sorted((p for p in scope if p in with_text),
                              key=lambda p: (-cited_counts.get(p, 0), p))[:n_deep])
            con.execute("DELETE FROM scope")
            con.executemany("INSERT INTO scope VALUES (?,?)", [(p, int(p in deep)) for p in scope])
            con.commit()
            log(f"[extract] scope {len(scope):,} papers; deep {len(deep):,}")

            doc_keys = store.item_keys("documents", "docs")
            delta_keys = store.item_keys("documents", "delta")
            sv_keys = store.item_keys("documents", "sv")
            vers: dict[str, list] = {}
            for p_, v_ in D.con.execute("SELECT paper_id, version FROM docs"):
                vers.setdefault(p_, []).append(v_)

            def base_latest(pid):
                vs = vers.get(pid) or []
                if not vs:
                    return None, None
                bv = 1 if 1 in vs else (0 if 0 in vs else max(vs))
                lv = max(vs)
                return f"{pid}@v{bv}", (f"{pid}@v{lv}" if lv != bv else None)

            lock = run.lock
            tot: defaultdict = defaultdict(int)

            def replace(pass_name, item, rows):
                con.execute("DELETE FROM statements WHERE pass=? AND item=?", (pass_name, item))
                con.executemany(INSERT, rows)

            def stats_row(pid, pass_name, st):
                con.execute("INSERT OR REPLACE INTO pass_stats VALUES (?,?,?)",
                            (pid, pass_name, json.dumps(st, ensure_ascii=False, default=str)))

            counter = {"n": 0}

            def commit_soon():
                counter["n"] += 1
                if counter["n"] % 200 == 0:
                    con.commit()

            # ---- t1: every scope paper
            w_t1 = run.work("t1")
            t1_fp = {}
            for pid in scope:
                bk, _ = base_latest(pid)
                title, abstract = recs.get(pid, ("", ""))
                t1_fp[pid] = store.sha(PR.T1_SHA, PS.MODEL, title,
                                       doc_keys.get(bk) if bk else store.sha(abstract)[:12])
            todo1 = set(w_t1.todo([(p, t1_fp[p]) for p in scope]))
            log(f"[extract] t1: {len(todo1):,} of {len(scope):,}")

            def one_t1(pid):
                if pid not in todo1:
                    return
                try:
                    inp = RD.t1_input(D, reg, pid)
                    if inp is None:                       # no title/abstract/date anywhere: a valid empty item
                        with lock:
                            replace("t1", pid, [])
                            w_t1.ok(pid)
                            commit_soon()
                        return
                    stmts, names, st = PS.t1(inp)
                    if stmts is None:
                        w_t1.fail(pid, "t1: no parseable answer")
                        return
                    with lock:
                        replace("t1", pid, _rows(stmts, run_id, tot))
                        stats_row(pid, "t1", st)
                        w_t1.ok(pid)
                        commit_soon()
                except Exception as e:
                    w_t1.fail(pid, f"{type(e).__name__}: {e}")

            store.parallel(one_t1, scope, workers, log=log, every=5000, label="extract:t1")
            with lock:
                con.commit()
            w_t1.sweep(scope, lambda p: replace("t1", p, []))

            # ---- t2 + results: deep papers with a full text
            w_t2 = run.work("t2")
            w_res = run.work("results")
            t2_items = []
            for pid in sorted(deep):
                bk, lk = base_latest(pid)
                t2_items.append((pid, bk, lk))
            t2_fp = {pid: store.sha(PR.T2_SHA, PS.MODEL, sv_keys.get(pid), doc_keys.get(bk),
                                    (doc_keys.get(lk), delta_keys.get(pid)) if lk else None)
                     for pid, bk, lk in t2_items}
            todo2 = set(w_t2.todo([(p, t2_fp[p]) for p, _, _ in t2_items]))
            log(f"[extract] t2: {len(todo2):,} of {len(t2_items):,}")

            def _t1_names(pid):
                names = []
                for (meta,) in rconn.execute("SELECT meta FROM statements WHERE pass='t1' AND item=? "
                                             "AND role='proposes'", (pid,)):
                    try:
                        m = json.loads(meta)
                    except (TypeError, ValueError):
                        continue
                    if m.get("name"):
                        names.append(m["name"])
                return names

            def one_t2(triple):
                pid = triple[0]
                if pid not in todo2:
                    return
                try:
                    full = RD.full_text(D, pid)
                    if full is None:                      # deep-selected but the text vanished: valid empty
                        with lock:
                            replace("t2", pid, [])
                            w_t2.ok(pid)
                            commit_soon()
                        return
                    title = recs.get(pid, ("", ""))[0]
                    stmts, own, st = PS.t2(pid, title, full, seed_own=_t1_names(pid))
                    if stmts is None:
                        w_t2.fail(pid, f"t2: chunk {st.get('failed_chunk')} gave no parseable answer")
                        return
                    st["own_methods"] = own
                    with lock:
                        replace("t2", pid, _rows(stmts, run_id, tot))
                        stats_row(pid, "t2", st)
                        w_t2.ok(pid)
                        commit_soon()
                except Exception as e:
                    w_t2.fail(pid, f"{type(e).__name__}: {e}")

            store.parallel(one_t2, [t for t in t2_items if t[0] in todo2], workers, log=log, every=2000,
                           label="extract:t2")
            with lock:
                con.commit()
            w_t2.sweep([p for p, _, _ in t2_items], lambda p: replace("t2", p, []))

            res_fp = {pid: store.sha(PR.RESULTS_SHA, PS.MODEL, w_t2.key(pid), t2_fp[pid])
                      for pid, _, _ in t2_items}
            todo_r = set(w_res.todo([(p, res_fp[p]) for p, _, _ in t2_items]))
            log(f"[extract] results: {len(todo_r):,} of {len(t2_items):,}")

            def one_results(pid):
                if pid not in todo_r:
                    return
                try:
                    full = RD.full_text(D, pid)
                    tbs = RD.tables_of(D, pid, full) if full else []
                    if not tbs:
                        with lock:
                            replace("results", pid, [])
                            w_res.ok(pid)
                            commit_soon()
                        return
                    own = list(_t1_names(pid))
                    r = rconn.execute("SELECT stats FROM pass_stats WHERE paper_id=? AND pass='t2'", (pid,)).fetchone()
                    if r:
                        try:
                            own += json.loads(r[0]).get("own_methods") or []
                        except (TypeError, ValueError):
                            pass
                    default_date = max((s["date"] for s in full["sentences"] if s.get("date")), default=None)
                    stmts, st = PS.results(pid, tbs, own_methods=own, default_date=default_date)
                    with lock:
                        replace("results", pid, _rows(stmts, run_id, tot))
                        stats_row(pid, "results", st)
                        w_res.ok(pid)
                        commit_soon()
                except Exception as e:
                    w_res.fail(pid, f"{type(e).__name__}: {e}")

            store.parallel(one_results, [p for p, _, _ in t2_items if p in todo_r], workers, log=log, every=2000,
                           label="extract:results")
            with lock:
                con.commit()
            w_res.sweep([p for p, _, _ in t2_items], lambda p: replace("results", p, []))

            # ---- other: sampled citation sentences, item = citing paper
            w_other = run.work("other")
            if cit is not None:
                pairs = select_pairs(cit, scope, N_OTHER)
                by_citing: dict[str, list] = defaultdict(list)
                for cited_, lst in pairs.items():
                    for sent_id, citing, date, version, key, self_cite in lst:
                        by_citing[citing].append((sent_id, cited_, version, key, date, self_cite))
                cit_keys = store.item_keys("citations", "citations")
                other_items = sorted(by_citing)
                other_fp = {c: store.sha(PR.OTHER_SHA, PS.MODEL, cit_keys.get(c),
                                         sorted((s, k) for s, _, _, k, _, _ in by_citing[c]))
                            for c in other_items}
                todo_o = set(w_other.todo([(c, other_fp[c]) for c in other_items]))
                n_pairs = sum(len(v) for v in by_citing.values())
                log(f"[extract] other: {n_pairs:,} sampled pairs over {len(other_items):,} citing papers; "
                    f"{len(todo_o):,} to do")

                def one_other(citing):
                    if citing not in todo_o:
                        return
                    try:
                        per_sent: dict[int, dict] = {}
                        for sent_id, cited_, version, key, date, self_cite in by_citing[citing]:
                            it = per_sent.get(sent_id)
                            if it is None:
                                r = cit.execute("SELECT citing_key, sid, sentence, in_delta FROM sentences "
                                                "WHERE id=?", (sent_id,)).fetchone()
                                if r is None:
                                    continue
                                it = per_sent[sent_id] = {"s": len(per_sent) + 1, "sentence": r[2], "date": date,
                                                          "citing_key": r[0], "sid": r[1], "in_delta": r[3],
                                                          "cites": []}
                            e = cit.execute("SELECT title, raw FROM entries WHERE citing=? AND version=? AND key=?",
                                            (citing, version, key)).fetchone()
                            it["cites"].append({"key": key, "cited": cited_, "self_cite": self_cite,
                                                "title": (e[0] if e else "") or "", "raw": (e[1] if e else "") or ""})
                        items = sorted(per_sent.values(), key=lambda x: x["s"])
                        batches, cur, n = [], [], 0
                        for it in items:                   # whole sentences per batch (an item's rows replace as one)
                            if cur and n + len(it["cites"]) > BATCH_PAIRS:
                                batches.append(cur)
                                cur, n = [], 0
                            cur.append(it)
                            n += len(it["cites"])
                        if cur:
                            batches.append(cur)
                        out, merged = [], defaultdict(int)
                        for b in batches:
                            stmts, st = PS.other_batch(citing, b)
                            if stmts is None:
                                w_other.fail(citing, "other: a batch gave no parseable answer")
                                return
                            out += stmts
                            for k, v in st.items():
                                merged[k] += v
                        with lock:
                            replace("other", citing, _rows(out, run_id, tot))
                            stats_row(citing, "other", dict(merged))
                            w_other.ok(citing)
                            commit_soon()
                    except Exception as e:
                        w_other.fail(citing, f"{type(e).__name__}: {e}")

                store.parallel(one_other, [c for c in other_items if c in todo_o], workers, log=log, every=2000,
                               label="extract:other")
                with lock:
                    con.commit()
                w_other.sweep(other_items, lambda c: replace("other", c, []))
                tot["other_pairs_sampled"] = n_pairs

            # ---- figures: deterministic, careful-tier papers (the docs table is the documents stage's — via D)
            w_fig = run.work("figures")
            fig_items = [(p, k) for k, p in D.con.execute(
                "SELECT key, paper_id FROM docs WHERE careful_z IS NOT NULL")]
            todo_f = set(w_fig.todo([(p, doc_keys.get(k, k)) for p, k in fig_items]))

            def one_figures(pair):
                pid, key = pair
                if pid not in todo_f:
                    return
                try:
                    d = D.get(key)
                    rows = []
                    for u in (d or {}).get("careful", {}).get("units", []) if d and d.get("careful") else []:
                        if u.get("kind") != "figure":
                            continue
                        m = _FIG_NO.match(u.get("text") or "")
                        rows.append((pid, u["uid"], int(m.group(1)) if m else None, u.get("text") or "",
                                     u.get("page"), json.dumps(u.get("bbox"))))
                    with lock:
                        con.execute("DELETE FROM figure_units WHERE paper_id=?", (pid,))
                        con.executemany("INSERT OR REPLACE INTO figure_units VALUES (?,?,?,?,?,?)", rows)
                        w_fig.ok(pid)
                        commit_soon()
                except Exception as e:
                    w_fig.fail(pid, f"{type(e).__name__}: {e}")

            store.parallel(one_figures, [t for t in fig_items if t[0] in todo_f], 8, log=log, every=2000,
                           label="extract:figures")
            with lock:
                con.commit()
            w_fig.sweep([p for p, _ in fig_items],
                        lambda p: con.execute("DELETE FROM figure_units WHERE paper_id=?", (p,)))

            # ---- deterministic tail: full monthly cited counts
            if cit is not None:
                con.execute("DELETE FROM cite_counts")
                cur = cit.execute("SELECT cited, substr(date, 1, 7) AS month, count(*) FROM cites "
                                  "GROUP BY cited, month")
                while True:
                    chunk = cur.fetchmany(50000)
                    if not chunk:
                        break
                    con.executemany("INSERT OR REPLACE INTO cite_counts VALUES (?,?,?)", chunk)
                con.commit()
                tot["cite_count_rows"] = con.execute("SELECT count(*) FROM cite_counts").fetchone()[0]

            q = lambda s: con.execute(s).fetchone()[0] or 0        # noqa: E731
            counts = {"scope": len(scope), "deep": len(deep),
                      "statements": q("SELECT count(*) FROM statements"),
                      "by_pass": dict(con.execute("SELECT pass, count(*) FROM statements GROUP BY pass")),
                      "by_facet": dict(con.execute("SELECT facet, count(*) FROM statements GROUP BY facet")),
                      "by_epistemic": dict(con.execute("SELECT epistemic, count(*) FROM statements GROUP BY 1")),
                      "figures": q("SELECT count(*) FROM figure_units"),
                      "cited_papers_counted": q("SELECT count(DISTINCT cited) FROM cite_counts"),
                      **tot}
            run.finish(counts)
        finally:
            reg.close()
            if cit is not None:
                cit.close()
            D.close()
            rconn.close()
    return counts
