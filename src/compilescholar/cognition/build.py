# -*- coding: utf-8 -*-
"""Stage `cognition` (phase D①): the materialised tables of INTEGRATED-SYSTEM-1005 §8 — deterministic ones first.

Passes in this first cut (all deterministic, zero LLM; the five LLM adjudications land as their own passes, each
piloted on 200 items with the dual-model agreement gate — §12 D②):
  cocite        (a, b, day, n): pairs co-cited in one citation sentence, aggregated per day — the as-of strength
                is SUM(n) WHERE day <= T (bitemporal: no snapshot needed)
  reception     reception_daily(cited, day, n): full daily citation counts — the count is never truncated by the
                other pass's sampled subset (v2.4 item 2)
  authors       author_link(author_key, paper_id): surname+initial keys over the registry's active papers, for
                the independence checks (facts) and author-aware tools
  comparisons   comparison_edge from the other pass's outcome meta (citing_better / cited_better / mixed), dated
                by the citing sentence's text-version date

Tables for the later passes (mention_link, lineage_edge/hyper, category_canon/daily, fact_member,
fact_status_event, shift_event, family_snapshot) are created up front so their schemas are stable; they stay
empty until their passes land.

Work items: one item ("all") per pass, fingerprinted by the upstream manifests' data fingerprints — any growth
in citations/extract/registry re-opens the affected aggregate (they are cheap full recomputes by SQL)."""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict

import ahocorasick

from ..compile.skeleton.proposes import GENERIC
from ..core import paths
from ..dfc import store
from ..extract.schema import LINEAGE
from ..llm.client import call_local
from ..llm.jsonparse import parse_json_response
from . import prompts as PR

DDL = """CREATE TABLE IF NOT EXISTS cocite(a TEXT NOT NULL, b TEXT NOT NULL, day TEXT NOT NULL, n INT NOT NULL,
  PRIMARY KEY(a, b, day));
CREATE INDEX IF NOT EXISTS ix_cocite_a ON cocite(a, day);
CREATE INDEX IF NOT EXISTS ix_cocite_b ON cocite(b, day);
CREATE TABLE IF NOT EXISTS reception_daily(cited TEXT NOT NULL, day TEXT NOT NULL, n INT NOT NULL,
  PRIMARY KEY(cited, day));
CREATE INDEX IF NOT EXISTS ix_reception_day ON reception_daily(day);
CREATE TABLE IF NOT EXISTS author_link(author_key TEXT NOT NULL, paper_id TEXT NOT NULL,
  PRIMARY KEY(author_key, paper_id));
CREATE INDEX IF NOT EXISTS ix_author_paper ON author_link(paper_id);
CREATE TABLE IF NOT EXISTS comparison_edge(stmt_id INT PRIMARY KEY, a TEXT, b TEXT, outcome TEXT, date TEXT);
CREATE INDEX IF NOT EXISTS ix_comp_b ON comparison_edge(b, date);
CREATE TABLE IF NOT EXISTS mention_link(id INTEGER PRIMARY KEY, name TEXT, sid TEXT, citing TEXT, date TEXT,
  candidates TEXT, paper_id TEXT, status TEXT);
CREATE INDEX IF NOT EXISTS ix_mention_name ON mention_link(name);
CREATE TABLE IF NOT EXISTS method_identity(name TEXT PRIMARY KEY, paper_id TEXT, status TEXT, decided_by TEXT,
  decided_at TEXT);
CREATE TABLE IF NOT EXISTS lineage_edge(stmt_id INT, child TEXT, parent TEXT, relation TEXT, speaker TEXT,
  date TEXT, valid_from TEXT, kind TEXT);
CREATE TABLE IF NOT EXISTS lineage_hyper(stmt_id INT, child TEXT, member TEXT, relation TEXT, speaker TEXT,
  date TEXT, valid_from TEXT);
CREATE TABLE IF NOT EXISTS category_canon(phrase TEXT PRIMARY KEY, canonical TEXT, umbrella INT, decided_by TEXT);
CREATE TABLE IF NOT EXISTS category_daily(category TEXT NOT NULL, day TEXT NOT NULL, n INT,
  PRIMARY KEY(category, day));
CREATE TABLE IF NOT EXISTS fact_member(fact_id TEXT, statement_id INT, role TEXT, PRIMARY KEY(fact_id, statement_id));
CREATE TABLE IF NOT EXISTS fact_status_event(fact_id TEXT, date TEXT, status TEXT, evidence TEXT);
CREATE TABLE IF NOT EXISTS shift_event(id INTEGER PRIMARY KEY, subject TEXT, facet TEXT, window_start TEXT,
  window_end TEXT, direction TEXT, evidence TEXT, decided_by TEXT);
CREATE TABLE IF NOT EXISTS family_snapshot(snapshot TEXT, family_id TEXT, name TEXT, members TEXT, named_by TEXT,
  PRIMARY KEY(snapshot, family_id));"""


def _manifest_fp(stage: str) -> str | None:
    m = store.read_manifest(stage) or {}
    return m.get("fingerprint")


# ---------------------------------------------------------------- method-name vocabulary (identity evidence)
_MODIFIER = re.compile(r"\b(the|a|an|model|method|framework|approach|algorithm|network)s?\b")


def norm_name(s: str) -> str:
    """Surface name -> comparison key (the old cognition.identity rule, kept: dash folds, parentheticals out,
    genre modifiers out)."""
    s = re.sub(r"[‐-―]", "-", (s or "").lower())
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = re.sub(r"[^a-z0-9+\-. ]+", " ", s)
    s = _MODIFIER.sub(" ", s)
    return " ".join(s.split()).strip(" .-")


def name_vocab(ext) -> dict:
    """norm_name -> {"papers": Counter(paper_id -> evidence weight), "surface": str}. Evidence weights keep the
    old Identity discipline: a self-reported proposal (3) outweighs its aliases (2) outweigh a third party's
    naming (1); generic names are never in the vocabulary."""
    vocab: dict[str, dict] = defaultdict(lambda: {"papers": Counter(), "surface": ""})

    def note(name, pid, w):
        n = norm_name(name)
        if not n or n in GENERIC or len(n) < 3:
            return
        v = vocab[n]
        v["papers"][pid] += w
        if len(name) > len(v["surface"]):
            v["surface"] = name

    for meta, about in ext.execute("SELECT meta, about FROM statements WHERE kind='self' AND role='proposes'"):
        try:
            m = json.loads(meta or "{}")
        except (TypeError, ValueError):
            continue
        if m.get("name") and not m.get("generic"):
            note(m["name"], about, 3)
            for a in m.get("aliases") or []:
                note(a, about, 2)
    for meta, about in ext.execute("SELECT meta, about FROM statements WHERE kind='other'"):
        try:
            m = json.loads(meta or "{}")
        except (TypeError, ValueError):
            continue
        if m.get("name") and not about.startswith("stub:"):
            note(m["name"], about, 1)
    return dict(vocab)


def _sent_text(D, pid: str, sid: str) -> str:
    """The verbatim sentence text behind a mention's sid (sv tier or a version's fast tier); '' when the
    document or sentence is gone (a merge moved it) — the adjudication then runs on the remaining contexts."""
    try:
        if "@sv#" in sid:
            d = D.sv(pid)
            sents = (d or {}).get("sentences") or []
        else:
            m = re.match(r"(.+)@v(\d+)#", sid or "")
            if not m:
                return ""
            d = D.get(f"{pid}@v{m.group(2)}")
            sents = ((d or {}).get("fast") or {}).get("sentences") or []
        return next((s["text"] for s in sents if s.get("sid") == sid), "")
    except Exception:
        return ""


def resolve_name(cand: Counter) -> tuple[str | None, str]:
    """Deterministic identity: one candidate resolves; several resolve only on clear evidence dominance
    (>= 2 and >= 2x the runner-up — the old rule); otherwise 'ambiguous', which is the LLM adjudication's queue
    (D② method identity)."""
    if not cand:
        return None, "empty"
    if len(cand) == 1:
        return next(iter(cand)), "ok"
    (best, w), *rest = cand.most_common()
    if w >= 2 and w >= 2 * rest[0][1]:
        return best, "ok"
    return None, "ambiguous"


def _author_key(surname: str, name: str) -> str | None:
    sur = (surname or "").strip().lower()
    if not sur:
        return None
    toks = (name or "").split()
    ini = toks[0][0].lower() if toks and toks[0][:1].isalpha() else ""
    return f"{sur}|{ini}"


def build(workers: int | None = None, rebuild: bool = False, log=print, cit=None, ext=None, reg=None,
          chat=None) -> dict:
    chat = chat or call_local
    params = {"v": 1}
    with store.Run("cognition", params, rebuild=rebuild) as run:
        con = run.con
        con.executescript(DDL)
        mine = []
        if cit is None:
            cit = store.connect("citations", readonly=True)
            mine.append(cit)
        if ext is None:
            ext = store.connect("extract", readonly=True)
            mine.append(ext)
        if reg is None:
            reg = store.read_only(paths.library() / "registry.sqlite")
            mine.append(reg)
        try:
            cit_fp, ext_fp = _manifest_fp("citations"), _manifest_fp("extract")
            reg_fp = store.sha(reg.execute("SELECT count(*) FROM authors").fetchone()[0],
                               reg.execute("SELECT count(*) FROM papers WHERE status='active'").fetchone()[0])

            # ---- cocite: pairs co-cited in one sentence, aggregated per day
            w = run.work("cocite")
            if w.todo([("all", store.sha(cit_fp))]):
                con.execute("DELETE FROM cocite")
                cur = cit.execute("SELECT a.cited, b.cited, a.date, count(*) FROM cites a JOIN cites b "
                                  "ON a.sentence_id = b.sentence_id AND a.cited < b.cited "
                                  "WHERE a.date IS NOT NULL GROUP BY a.cited, b.cited, a.date")
                n = 0
                while True:
                    rows = cur.fetchmany(50000)
                    if not rows:
                        break
                    con.executemany("INSERT OR REPLACE INTO cocite VALUES (?,?,?,?)", rows)
                    n += len(rows)
                con.commit()
                w.ok("all")
                log(f"[cognition] cocite: {n:,} (pair, day) rows")

            # ---- reception: full daily citation counts
            w = run.work("reception")
            if w.todo([("all", store.sha(cit_fp))]):
                con.execute("DELETE FROM reception_daily")
                cur = cit.execute("SELECT cited, date, count(*) FROM cites WHERE date IS NOT NULL "
                                  "GROUP BY cited, date")
                n = 0
                while True:
                    rows = cur.fetchmany(50000)
                    if not rows:
                        break
                    con.executemany("INSERT OR REPLACE INTO reception_daily VALUES (?,?,?)", rows)
                    n += len(rows)
                con.commit()
                w.ok("all")
                log(f"[cognition] reception: {n:,} (paper, day) rows")

            # ---- authors: surname+initial keys of every active paper
            w = run.work("authors")
            if w.todo([("all", store.sha(reg_fp))]):
                con.execute("DELETE FROM author_link")
                active = {p for (p,) in reg.execute("SELECT paper_id FROM papers WHERE status='active'")}
                rows, n = [], 0
                for pid, names in reg.execute("SELECT paper_id, names FROM authors"):
                    if pid not in active:
                        continue
                    try:
                        lst = json.loads(names)
                    except (TypeError, ValueError):
                        continue
                    for a in lst:
                        k = _author_key(a.get("surname"), a.get("name"))
                        if k:
                            rows.append((k, pid))
                    if len(rows) >= 50000:
                        con.executemany("INSERT OR IGNORE INTO author_link VALUES (?,?)", rows)
                        n += len(rows)
                        rows = []
                if rows:
                    con.executemany("INSERT OR IGNORE INTO author_link VALUES (?,?)", rows)
                    n += len(rows)
                con.commit()
                w.ok("all")
                log(f"[cognition] author_link: {n:,} rows")

            # ---- comparisons: the other pass's qualitative outcomes
            w = run.work("comparisons")
            if w.todo([("all", store.sha(ext_fp))]):
                con.execute("DELETE FROM comparison_edge")
                cur = ext.execute("SELECT id, speaker, about, json_extract(meta, '$.outcome'), date "
                                  "FROM statements WHERE kind='other' AND date IS NOT NULL "
                                  "AND json_extract(meta, '$.outcome') IS NOT NULL")
                n = 0
                while True:
                    rows = cur.fetchmany(50000)
                    if not rows:
                        break
                    con.executemany("INSERT OR REPLACE INTO comparison_edge VALUES (?,?,?,?,?)", rows)
                    n += len(rows)
                con.commit()
                w.ok("all")
                log(f"[cognition] comparison_edge: {n:,} rows")

            # ---- method-name vocabulary (shared by the mentions and lineage passes)
            vocab = name_vocab(ext)
            resolved = {n: resolve_name(v["papers"]) for n, v in vocab.items()}
            vocab_fp = store.sha(sorted((k, sorted(v["papers"].items())) for k, v in vocab.items()))

            # ---- mentions: every known method name in every sentence, including ones without a citation mark
            from ..documents.build import Documents
            from ..extract import reading as RD
            D = Documents()
            try:
                w = run.work("mentions")
                doc_keys = store.item_keys("documents", "docs")
                delta_keys = store.item_keys("documents", "delta")
                sv_keys = store.item_keys("documents", "sv")
                vers: dict[str, list] = {}
                for p_, v_ in D.con.execute("SELECT paper_id, version FROM docs"):
                    vers.setdefault(p_, []).append(v_)
                m_items = []
                for pid, vs in sorted(vers.items()):
                    bv = 1 if 1 in vs else (0 if 0 in vs else max(vs))
                    lv = max(vs)
                    bk, lk = f"{pid}@v{bv}", (f"{pid}@v{lv}" if lv != bv else None)
                    m_items.append((pid, bk, lk))
                auto = ahocorasick.Automaton()
                for n in vocab:
                    auto.add_word(n, n)
                if vocab:
                    auto.make_automaton()
                todo_m = set(w.todo([(pid, store.sha(vocab_fp, doc_keys.get(bk),
                                                      (doc_keys.get(lk), delta_keys.get(pid)) if lk else None,
                                                      sv_keys.get(pid))) for pid, bk, lk in m_items]))

                def one_mentions(triple):
                    pid = triple[0]
                    if pid not in todo_m:
                        return
                    try:
                        full = RD.full_text(D, pid)
                        rows = []
                        if full and vocab:
                            for s in full["sentences"]:
                                t = norm_name(s["text"])       # the vocabulary's normalisation on both sides
                                if len(t) < 3:
                                    continue
                                for end, name in auto.iter(t):
                                    start = end - len(name) + 1
                                    if (start > 0 and t[start - 1] != " ") or (end + 1 < len(t) and t[end + 1] != " "):
                                        continue                      # inside a longer token: not a mention
                                    p_res, status = resolved.get(name, (None, "empty"))
                                    rows.append((name, s["sid"], pid, s.get("date"),
                                                 json.dumps(sorted(vocab[name]["papers"].items())), p_res, status))
                        with run.lock:
                            con.execute("DELETE FROM mention_link WHERE citing=?", (pid,))
                            con.executemany("INSERT INTO mention_link(name, sid, citing, date, candidates, "
                                            "paper_id, status) VALUES (?,?,?,?,?,?,?)", rows)
                            mcounter["n"] += 1
                            if mcounter["n"] % 200 == 0:
                                con.commit()
                            w.ok(pid)
                    except Exception as e:
                        w.fail(pid, f"{type(e).__name__}: {e}")

                mcounter = {"n": 0}
                store.parallel(one_mentions, [t for t in m_items if t[0] in todo_m], workers or 8, log=log,
                               every=5000, label="cognition:mentions")
                with run.lock:
                    con.commit()
                w.sweep([p for p, _, _ in m_items],
                        lambda p: con.execute("DELETE FROM mention_link WHERE citing=?", (p,)))

                # ---- method identity: the LLM adjudication of ambiguous names (§8: 18.3% of named papers share
                #      a name with someone else; §12 D②: pilot 200 with the dual-model gate before full runs)
                rconn = store.read_only(run.path)          # per-thread reads of this run's file
                w = run.work("identities")
                amb = sorted(n for n, r in resolved.items() if r[1] == "ambiguous")
                now = time.strftime("%Y-%m-%dT%H:%M:%S")
                con.execute("DELETE FROM method_identity WHERE status IN ('ok', 'ambiguous')")
                for n, (p_, st_) in sorted(resolved.items()):
                    if st_ == "ok":
                        con.execute("INSERT OR REPLACE INTO method_identity VALUES (?,?,?,?,?)",
                                    (n, p_, "ok", "deterministic", now))
                for n in amb:                              # pending rows survive; adjudicated rows are never reset
                    con.execute("INSERT OR IGNORE INTO method_identity VALUES (?,?,?,?,?)",
                                (n, None, "ambiguous", None, None))
                con.commit()
                todo_i = set(w.todo([(n, store.sha(PR.METHOD_ID_SHA, PR.MODEL, sorted(vocab[n]["papers"].items())))
                                     for n in amb]))
                if todo_i:
                    log(f"[cognition] identities: {len(todo_i):,} of {len(amb):,} ambiguous names to adjudicate")

                def one_identity(name):
                    if name not in todo_i:
                        return
                    try:
                        lines, cand_pids = [], []
                        for i, (pid, _wt) in enumerate(vocab[name]["papers"].most_common()[:8], 1):
                            r = reg.execute(
                                "SELECT first_hi, (SELECT title FROM records r WHERE r.paper_id=p.paper_id "
                                "ORDER BY r.source='arxiv' DESC LIMIT 1), (SELECT abstract FROM records r "
                                "WHERE r.paper_id=p.paper_id AND abstract IS NOT NULL AND abstract!='' "
                                "ORDER BY r.source='arxiv' DESC LIMIT 1) FROM papers p WHERE p.paper_id=?",
                                (pid,)).fetchone()
                            title = (r[1] if r else "") or pid
                            year = ((r[0] or "")[:4]) if r else ""
                            lines.append(f"{i}. {title} ({year}) — {((r[2] if r else '') or '')[:220]}")
                            cand_pids.append(pid)
                        ctx = []
                        for sid, citing in rconn.execute("SELECT sid, citing FROM mention_link "
                                                         "WHERE name=? AND status='ambiguous' LIMIT 6", (name,)):
                            t = _sent_text(D, citing, sid)
                            if t:
                                ctx.append(f"[{citing}] {t[:220]}")
                        raw = chat(PR.METHOD_ID.format(name=vocab[name]["surface"] or name,
                                                       candidates="\n".join(lines),
                                                       contexts="\n".join(ctx) or "(none captured)"),
                                   model=PR.MODEL, max_tokens=200, temperature=0.0, enable_thinking=False,
                                   item=f"identity:{name}")
                        obj = parse_json_response(raw or "")
                        if not isinstance(obj, dict) or not isinstance(obj.get("paper"), int):
                            w.fail(name, "identity: no parseable answer")
                            return
                        pick = obj["paper"]
                        pid_res = cand_pids[pick - 1] if 1 <= pick <= len(cand_pids) else None
                        with run.lock:
                            con.execute("INSERT OR REPLACE INTO method_identity VALUES (?,?,?,?,?)",
                                        (name, pid_res, "adjudicated" if pid_res else "unresolved", PR.MODEL, now))
                            if pid_res:
                                con.execute("UPDATE mention_link SET paper_id=?, status='adjudicated' "
                                            "WHERE name=? AND status='ambiguous'", (pid_res, name))
                            icounter["n"] += 1
                            if icounter["n"] % 100 == 0:
                                con.commit()
                            w.ok(name)
                    except Exception as e:
                        w.fail(name, f"{type(e).__name__}: {e}")

                icounter = {"n": 0}
                store.parallel(one_identity, [n for n in amb if n in todo_i], workers or 8, log=log, every=1000,
                               label="cognition:identities")
                with run.lock:
                    con.commit()
                id_map = dict(rconn.execute("SELECT name, paper_id FROM method_identity "
                                            "WHERE status='adjudicated' AND paper_id IS NOT NULL"))
                id_fp = store.sha(sorted((k, v) for k, v in rconn.execute(
                    "SELECT name, paper_id FROM method_identity WHERE status='adjudicated'")))
                rconn.close()
            finally:
                D.close()

            def owner(n):
                """The paper a method name belongs to: the deterministic resolution first, then the adjudicated
                identity (the ambiguity is exactly what the LLM decided)."""
                p_, st_ = resolved.get(n, (None, "empty"))
                return p_ if st_ == "ok" else id_map.get(n)

            # ---- lineage: paper -> paper edges with the §2.4 effective date (deterministic assembly; parent
            #      identities come from the resolution/adjudication above)
            w = run.work("lineage")
            if w.todo([("all", store.sha(ext_fp, vocab_fp, id_fp))]):
                first_hi = dict(reg.execute("SELECT paper_id, first_hi FROM papers "
                                            "WHERE status='active' AND first_hi IS NOT NULL"))
                edges, hyper, dropped = [], [], {"time": 0, "stub": 0, "unknown": 0}

                def add_edge(sid, child, parent, rel, kind, speaker, date):
                    if child == parent:
                        return
                    if child.startswith("stub:") or parent.startswith("stub:"):
                        dropped["stub"] += 1
                        return
                    fc, fp2 = first_hi.get(child), first_hi.get(parent)
                    if not fc or not fp2:
                        dropped["unknown"] += 1
                        return
                    if fc < fp2:                       # cannot extend a work that did not exist yet
                        dropped["time"] += 1
                        return
                    edges.append((sid, child, parent, rel, speaker, date, max(date, fc, fp2), kind))

                def add_hyper(sid, child, members, rel, speaker, date):
                    members = [m for m in dict.fromkeys(members) if m != child and not m.startswith("stub:")
                               and m in first_hi]
                    if len(members) < 2 or child.startswith("stub:") or child not in first_hi:
                        return
                    vf = max([date, first_hi[child]] + [first_hi[m] for m in members])
                    for m in members:
                        hyper.append((sid, child, m, rel, speaker, date, vf))

                lin = ",".join("?" * len(LINEAGE))
                for sid, speaker, about, role, date, meta, kind, grp in ext.execute(
                        f"SELECT id, speaker, about, role, date, meta, kind, grp FROM statements "
                        f"WHERE date IS NOT NULL AND (role IN ({lin}) "
                        f"OR json_extract(meta, '$.builds_on') IS NOT NULL)", tuple(LINEAGE)):
                    try:
                        m = json.loads(meta or "{}")
                    except (TypeError, ValueError):
                        m = {}
                    group = json.loads(grp or "[]")
                    if kind == "self" and role in LINEAGE:
                        parents = []
                        for mn in m.get("mentions") or []:
                            rel = mn.get("relation") if mn.get("relation") in LINEAGE else role
                            p = owner(norm_name(mn.get("name") or ""))
                            if p:
                                add_edge(sid, speaker, p, rel, "self", speaker, date)
                                parents.append(p)
                        if role == "combines" and len(parents) >= 2:
                            add_hyper(sid, speaker, parents, "combines", speaker, date)
                    elif kind == "other":
                        if role in LINEAGE:
                            add_edge(sid, speaker, about, role, "self", speaker, date)
                            if role == "combines":
                                add_hyper(sid, speaker, group, "combines", speaker, date)
                        bo, bor = m.get("builds_on"), m.get("builds_on_relation")
                        if bo and bor in LINEAGE:
                            bn = norm_name(bo)
                            p = owner(bn)
                            if p is None and bn in vocab:
                                # a globally ambiguous name is disambiguated by the sentence: a co-cited work
                                # that claims the name is the parent (the old Identity fallback, kept)
                                cand = vocab[bn]["papers"]
                                p = next((g for g in group if g != about and g in cand), None)
                            if p:
                                add_edge(sid, about, p, bor, "third", speaker, date)
                con.execute("DELETE FROM lineage_edge")
                con.executemany("INSERT INTO lineage_edge VALUES (?,?,?,?,?,?,?,?)", edges)
                con.execute("DELETE FROM lineage_hyper")
                con.executemany("INSERT INTO lineage_hyper VALUES (?,?,?,?,?,?,?)", hyper)
                con.commit()
                w.ok("all")
                log(f"[cognition] lineage: {len(edges):,} edges, {len(hyper):,} hyper rows, dropped {dropped}")

            q = lambda s: con.execute(s).fetchone()[0] or 0        # noqa: E731
            counts = {"cocite": q("SELECT count(*) FROM cocite"),
                      "reception": q("SELECT count(*) FROM reception_daily"),
                      "author_link": q("SELECT count(*) FROM author_link"),
                      "comparison_edge": q("SELECT count(*) FROM comparison_edge"),
                      "mention_link": q("SELECT count(*) FROM mention_link"),
                      "mentions_ambiguous": q("SELECT count(*) FROM mention_link WHERE status='ambiguous'"),
                      "mentions_adjudicated": q("SELECT count(*) FROM mention_link WHERE status='adjudicated'"),
                      "names": len(vocab),
                      "identities": dict(con.execute("SELECT status, count(*) FROM method_identity GROUP BY status")),
                      "lineage_edge": q("SELECT count(*) FROM lineage_edge"),
                      "lineage_hyper": q("SELECT count(*) FROM lineage_hyper")}
            run.finish(counts)
        finally:
            for c in mine:
                c.close()
    return counts
