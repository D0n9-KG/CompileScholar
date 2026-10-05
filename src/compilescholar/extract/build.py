# -*- coding: utf-8 -*-
"""Stage `extract` (L2): self pass over target papers + other pass over citation sentences about target papers,
written to data/dfc/extract.sqlite (table `statements`, schema.DDL).

Targets = the set of papers whose field cognition we compile in this build (e.g. a benchmark's candidate pool, a
field). Budget control (DESIGN §5): for each target, at most N_OTHER citation pairs, stratified over citing months so
every period is represented (round-robin over months, oldest first); self pass only for targets (and, optionally,
for the citing papers). Resumable: a (citing, sentence_id, cited) pair or a self-pass paper already done is skipped.
Every statement passes schema validation and the quote check before it is written."""
from __future__ import annotations

import concurrent.futures as cf
import json
import threading
from collections import defaultdict

import pyarrow.parquet as pq

from ..citations.build import ideaforecast_dir
from ..corpus.papers import Papers
from ..dfc import store
from . import other_pass as O
from . import self_pass as S
from .schema import DDL

N_OTHER = 30
EXTRA_DDL = """CREATE TABLE IF NOT EXISTS done_self(arxiv_id TEXT PRIMARY KEY, stats TEXT);
CREATE TABLE IF NOT EXISTS done_pairs(sentence_id INT, cited TEXT, PRIMARY KEY(sentence_id, cited));
CREATE TABLE IF NOT EXISTS targets(about TEXT PRIMARY KEY);"""


def select_pairs(cit, targets: set[str], n: int = N_OTHER) -> dict[str, list[tuple]]:
    """targets: 'paper:<id>' strings -> {target: [(sentence_id, citing, date, n_group), ...]} capped at n, spread
    over citing months (round-robin, oldest month first) and, within a month, over distinct citing papers."""
    by_t: dict[str, dict[str, list[tuple]]] = defaultdict(lambda: defaultdict(list))
    q = "SELECT sentence_id, citing, date, cited, n_group FROM cites WHERE cited = ?"
    for t in targets:
        seen_citing = set()
        for sid, citing, date, cited, ng in cit.execute(q, (t,)):
            key = (citing, date[:7])
            if key in seen_citing:  # one sentence per citing paper per month first; extras only if budget remains
                by_t[t]["~" + date[:7]].append((sid, citing, date, ng))
            else:
                seen_citing.add(key)
                by_t[t][date[:7]].append((sid, citing, date, ng))
    out = {}
    for t, months in by_t.items():
        firsts = sorted(m for m in months if not m.startswith("~"))
        rest = sorted(m for m in months if m.startswith("~"))
        picked = []
        for order in (firsts, rest):
            queues = [list(months[m]) for m in order]
            while len(picked) < n and any(queues):
                for qq in queues:
                    if qq and len(picked) < n:
                        picked.append(qq.pop(0))
        out[t] = picked
    return out


def _bodies(ids: set[str]) -> dict[str, str]:
    out = {}
    for f in sorted(ideaforecast_dir().glob("*.parquet")):
        for r in pq.read_table(f, columns=["arxiv_id", "text"]).to_pylist():
            if r["arxiv_id"] in ids:
                out[r["arxiv_id"]] = r["text"]
    return out


def build(targets: set[str], workers: int = 48, self_for_citing: bool = False, log=print) -> dict:
    store.require_fresh("papers", "citations")
    papers = Papers()
    cit = store.connect("citations", readonly=True)
    con = store.connect("extract")
    con.executescript(DDL + EXTRA_DDL)
    con.executemany("INSERT OR IGNORE INTO targets VALUES (?)", [(t,) for t in targets])
    con.commit()
    lock = threading.Lock()
    model = S.MODEL

    def write(stmts, prompt_sha):
        rows = []
        for s in stmts:
            if s.validate():
                continue
            r = s.row()
            rows.append((r["speaker"], r["date"], r["kind"], r["about"], r["role"], r["facet"], r["text"], r["quote"],
                         r["target"], json.dumps(r["group"]), r["function"], json.dumps(r["meta"], ensure_ascii=False),
                         model, prompt_sha))
        con.executemany("INSERT INTO statements(speaker,date,kind,about,role,facet,text,quote,target,grp,function,"
                        "meta,model,prompt_sha) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    # ---- other pass
    pairs = select_pairs(cit, targets)
    done = {(a, b) for a, b in con.execute("SELECT sentence_id, cited FROM done_pairs")}
    by_citing: dict[str, list[dict]] = defaultdict(list)
    for t, lst in pairs.items():
        for sid, citing, date, ng in lst:
            if (sid, t) in done:
                continue
            sent, keys = cit.execute("SELECT sentence, keys FROM sentences WHERE id=?", (sid,)).fetchone()
            ent = cit.execute("SELECT raw, title FROM entries WHERE citing=? AND cited=? LIMIT 1", (citing, t)).fetchone()
            grp = [r[0] for r in cit.execute("SELECT cited FROM cites WHERE sentence_id=?", (sid,))]
            by_citing[citing].append({"sentence_id": sid, "sentence": sent, "cited": t, "date": date,
                                      "title": ent[1] if ent else "", "raw": ent[0] if ent else "", "group": grp})
    batches = []
    for citing, items in by_citing.items():
        for i in range(0, len(items), O.BATCH):
            batches.append((citing, items[i:i + O.BATCH]))
    tot = defaultdict(int)
    log(f"[extract] other pass: {sum(len(v) for v in pairs.values())} pairs for {len(pairs)} targets, "
        f"{len(batches)} batches to run")

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
        return st

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(do_other, batches)):
            if (i + 1) % 100 == 0:
                log(f"[extract] other {i + 1}/{len(batches)} {dict(tot)}")

    # ---- self pass
    selfs = {t[6:] for t in targets if t.startswith("paper:")}
    if self_for_citing:
        selfs |= set(by_citing)
    done_self = {r[0] for r in con.execute("SELECT arxiv_id FROM done_self")}
    todo = sorted(selfs - done_self)
    bodies = _bodies(set(todo))
    log(f"[extract] self pass: {len(todo)} papers ({len(bodies)} with full text)")

    def do_self(aid):
        p = papers.get(aid)
        if not p or not p["v1_date"]:
            return {"missing": 1}
        stmts, st = S.run(aid, p["v1_date"], p["title"], p["abstract"], bodies.get(aid))
        with lock:
            write(stmts, S.PROMPT_SHA)
            con.execute("INSERT OR REPLACE INTO done_self VALUES (?,?)", (aid, json.dumps(st)))
            con.commit()
            for k, v in st.items():
                tot["self_" + k] += v
        return st

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(do_self, todo)):
            if (i + 1) % 200 == 0:
                log(f"[extract] self {i + 1}/{len(todo)} {dict(tot)}")
    counts = {"statements": con.execute("SELECT count(*) FROM statements").fetchone()[0],
              "self": con.execute("SELECT count(*) FROM statements WHERE kind='self'").fetchone()[0],
              "other": con.execute("SELECT count(*) FROM statements WHERE kind='other'").fetchone()[0],
              "targets": con.execute("SELECT count(*) FROM targets").fetchone()[0], **tot}
    con.close()
    store.write_manifest("extract", {"n_other": N_OTHER, "self_for_citing": self_for_citing,
                                     "targets_n": len(targets), "other_prompt": O.PROMPT_SHA,
                                     "self_prompt": S.PROMPT_SHA, "model": model}, counts)
    return counts
