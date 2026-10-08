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
CREATE TABLE IF NOT EXISTS fact_pair(stmt_a INT NOT NULL, stmt_b INT NOT NULL, relation TEXT, decided_by TEXT,
  PRIMARY KEY(stmt_a, stmt_b));
CREATE TABLE IF NOT EXISTS fact_member(fact_id TEXT, statement_id INT, role TEXT, PRIMARY KEY(fact_id, statement_id));
CREATE INDEX IF NOT EXISTS ix_fact_member_stmt ON fact_member(statement_id);
CREATE TABLE IF NOT EXISTS fact_status_event(fact_id TEXT, date TEXT, status TEXT, evidence TEXT);
CREATE INDEX IF NOT EXISTS ix_fact_event ON fact_status_event(fact_id, date);
CREATE TABLE IF NOT EXISTS shift_event(id INTEGER PRIMARY KEY, subject TEXT, facet TEXT, window_start TEXT,
  window_end TEXT, direction TEXT, evidence TEXT, decided_by TEXT, status TEXT,
  UNIQUE(subject, facet, window_start, window_end, direction));
CREATE INDEX IF NOT EXISTS ix_shift_subject ON shift_event(subject, status);
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


SNAPSHOT_SEED = 20261007
MIN_FAMILY = 2

# Reception-shift screening (pre-registered constants, carried from the old shifts module; the arbitrary
# half-split verdicts are replaced by Fisher + BH — the old rule-based shifts measured ~75% false positives)
MIN_N = 8
COMP, COMP_GAIN = 0.5, 0.2
BASE, BASE_GAIN = 0.4, 0.2
COMPONENT_ROLES = {"uses"}
COMPONENT_FN = {"tool", "data", "metric"}
BASELINE_FN = {"baseline", "contrast"}
FDR_Q = 0.05


def fisher_p(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test of [[a, b], [c, d]] — pure Python (no scipy dependency)."""
    from math import comb
    n, r1, c1 = a + b + c + d, a + b, a + c
    if n == 0 or r1 == 0 or r1 == n or c1 == 0 or c1 == n:
        return 1.0
    denom = comb(n, c1)

    def hg(k):
        return comb(r1, k) * comb(n - r1, c1 - k) / denom

    p0 = hg(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return min(1.0, sum(hg(k) for k in range(lo, hi + 1) if hg(k) <= p0 * (1 + 1e-9)))


def bh_select(ps: list[float], q: float = FDR_Q) -> set[int]:
    """Benjamini-Hochberg: indices of the p-values that pass at FDR level q."""
    m = len(ps)
    if not m:
        return set()
    order = sorted(range(m), key=lambda i: ps[i])
    k_max = -1
    for rank, i in enumerate(order, 1):
        if ps[i] <= q * rank / m:
            k_max = rank
    return set(order[:k_max]) if k_max >= 0 else set()


def screen_shifts(obj_votes: dict, author_of: dict, replaces: dict) -> list[dict]:
    """Deterministic candidate screen for reception shifts. obj_votes: subject -> date-sorted vote dicts
    (one per (speaker, sentence): date, role, function, facet, category, speaker, quote); author_of: paper ->
    author keys (independence); replaces: parent subject -> earliest 'replaces' claim date.
    Returns candidate dicts (facet, windows, direction, evidence, p) after the effect floors, Fisher and BH."""
    cands = []
    for obj, votes in obj_votes.items():
        if len(votes) < MIN_N:
            continue
        mid = len(votes) // 2
        early, late = votes[:mid], votes[mid:]
        w0, w1 = early[0]["date"], late[-1]["date"]

        def share(rows, pred):
            return sum(1 for r in rows if pred(r)) / len(rows)

        def is_comp(r):
            return r.get("role") in COMPONENT_ROLES or r.get("function") in COMPONENT_FN

        def is_base(r):
            return r.get("function") in BASELINE_FN or r.get("role") == "compares"

        for facet, pred, floor, gain in (("became_component", is_comp, COMP, COMP_GAIN),
                                         ("became_baseline", is_base, BASE, BASE_GAIN)):
            ce, cl = share(early, pred), share(late, pred)
            if cl >= floor and cl - ce >= gain:
                a = sum(1 for r in late if pred(r))
                b0 = sum(1 for r in early if pred(r))
                p = fisher_p(a, len(late) - a, b0, len(early) - b0)
                cands.append({"subject": obj, "facet": facet, "window_start": w0, "window_end": w1,
                              "direction": json.dumps({"early": round(ce, 3), "late": round(cl, 3),
                                                       "p": round(p, 6)}),
                              "evidence": json.dumps({"early": [r["quote"][:200] for r in early if pred(r)][:2],
                                                      "late": [r["quote"][:200] for r in late if pred(r)][:3]},
                                                     ensure_ascii=False),
                              "p": p})
        ec = Counter(r.get("category") or "" for r in early if r.get("category"))
        lc = Counter(r.get("category") or "" for r in late if r.get("category"))
        if ec and lc:
            (e1, en), (l1, ln) = ec.most_common(1)[0], lc.most_common(1)[0]
            if e1 != l1 and en >= 2 and ln >= 2:
                le = ec.get(l1, 0)                    # the late top category's count in the EARLY window
                p = fisher_p(ln, len(late) - ln, le, len(early) - le)
                cands.append({"subject": obj, "facet": "recategorized", "window_start": w0, "window_end": w1,
                              "direction": json.dumps({"from": e1, "to": l1, "p": round(p, 6)},
                                                      ensure_ascii=False),
                              "evidence": json.dumps({"early": [r["quote"][:200] for r in early
                                                                if (r.get("category") or "") == e1][:2],
                                                      "late": [r["quote"][:200] for r in late
                                                               if (r.get("category") or "") == l1][:3]},
                                                     ensure_ascii=False),
                              "p": p})
        seen: dict[str, tuple] = {}                 # speaker -> (date, quote): one quote per independent speaker
        for r in sorted(votes, key=lambda x: x["date"]):
            if r.get("facet") == "limitation":
                sp = r["speaker"]
                if all(not (author_of.get(sp, {sp}) & author_of.get(o, {o})) for o in seen):
                    seen[sp] = (r["date"], r["quote"][:200])
                if len(seen) >= 2:
                    cands.append({"subject": obj, "facet": "limitation_exposed",
                                  "window_start": min(d for d, _ in seen.values()), "window_end": r["date"],
                                  "direction": json.dumps({"speakers": sorted(seen)}),
                                  "evidence": json.dumps([q for _, q in seen.values()], ensure_ascii=False),
                                  "p": 0.0})
                    break
        rep = replaces.get(obj)
        if rep:
            after = [r for r in votes if r["date"] >= rep]
            if len(after) >= 4 and share(after, is_base) > share(
                    after, lambda r: r.get("role") in ("extends", "improves", "adapts")):
                cands.append({"subject": obj, "facet": "superseded", "window_start": rep,
                              "window_end": after[-1]["date"], "direction": json.dumps({"since": rep}),
                              "evidence": json.dumps({"baseline": [r["quote"][:200] for r in after
                                                                   if is_base(r)][:3],
                                                     "extends": [r["quote"][:200] for r in after
                                                                 if r.get("role") in
                                                                 ("extends", "improves", "adapts")][:2]},
                                                     ensure_ascii=False),
                              "p": 0.0})
    passed = bh_select([c["p"] for c in cands])
    return [c for i, c in enumerate(cands) if i in passed or c["p"] == 0.0]


def _shift_claim(facet: str, dirj: dict, ev) -> tuple[str, list]:
    """Facet-aware claim sentence + labelled evidence blocks for SHIFT_VERIFY. Structural formatting only —
    the judgement stays with the LLM. (10-09 fix: single-window facets used to render an empty 'early' block
    and the judge rejected 88/91 candidates with 'no early sentences exist' — a rendering artifact, not a
    verdict on the evidence.)"""
    if facet in ("became_component", "became_baseline"):
        what = "a building-block component" if facet == "became_component" else "a comparison baseline"
        claim = (f"between the early window (share {dirj.get('early', '—')}) and the late window "
                 f"(share {dirj.get('late', '—')}, Fisher p={dirj.get('p', 'n/a')}), citing papers "
                 f"increasingly treat the work as {what}.")
        d = ev if isinstance(ev, dict) else {"late": ev}
        return claim, [("Early-window sentences (from citing papers)", d.get("early", [])),
                       ("Late-window sentences", d.get("late", []))]
    if facet == "recategorized":
        claim = (f"the dominant category citing papers use for the work changed from "
                 f"'{dirj.get('from', '?')}' to '{dirj.get('to', '?')}' (Fisher p={dirj.get('p', 'n/a')}).")
        d = ev if isinstance(ev, dict) else {"late": ev}
        return claim, [("Early-window sentences (old dominant category)", d.get("early", [])),
                       ("Late-window sentences (new dominant category)", d.get("late", []))]
    if facet == "limitation_exposed":
        claim = ("at least two author-independent citing papers explicitly expose a limitation of this work "
                 "(single-window claim; no early/late contrast).")
        return claim, [("Limitation-exposing sentences (from independent citing papers)",
                        ev if isinstance(ev, list) else [])]
    if facet == "superseded":
        claim = (f"since {dirj.get('since', '?')} (an explicit 'replaces' claim), citing papers treat the "
                 "work mainly as a comparison baseline rather than extending it.")
        d = ev if isinstance(ev, dict) else {"baseline": ev}
        return claim, [("Baseline-style sentences since then", d.get("baseline", [])),
                       ("Extension-style sentences since then (should be rarer)", d.get("extends", []))]
    return (f"{facet} — {json.dumps(dirj, ensure_ascii=False)[:200]}",
            [("Evidence sentences", ev if isinstance(ev, list) else [])])


def snapshot_grid(earliest: str, latest: str, years: int = 15, extra=()) -> list[str]:
    """The §8 snapshot grid: monthly over the last 3 years of the range, quarterly before that, plus explicit
    extra dates (benchmark cut-offs). Capped at `years` back from the latest date (the design's ~80 snapshots:
    36 monthly + ~48 quarterly); queries before the grid clamp to the earliest snapshot and say so."""
    import datetime as dt
    hi = dt.date.fromisoformat(latest[:10])
    lo = max(dt.date.fromisoformat(earliest[:10]), hi - dt.timedelta(days=years * 365))
    monthly_from = hi - dt.timedelta(days=3 * 365)
    out = set()
    d = dt.date(lo.year, lo.month, 1)
    while d <= hi:
        if d >= monthly_from or d.month in (1, 4, 7, 10):
            out.add(d.isoformat())
        d = dt.date(d.year + (d.month == 12), (d.month % 12) + 1, 1)
    out.update(x[:10] for x in extra)
    return sorted(out)


# Family-level facts (§8): deterministic recall + LLM relation adjudication (the old Jaccard-only clustering
# with a negation regex is what the design replaces — "not slow" and "slow" landed in one class)
FACT_FACETS = ("limitation", "method", "result", "categorization", "contribution")
FACT_RECALL_SIM = 0.3
FACT_MAX_FAMILY_STMTS = 900
_FACT_STOP = set("a an the of in on for to and or with by from as is are was were be been this that these those "
                 "it its their they we our which such via using use used based into than also can may method "
                 "methods model models approach approaches work works paper papers all".split())


def _content_words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in _FACT_STOP and len(w) > 2}


def _independent(speakers, author_of: dict) -> int:
    """Greedy merge of speakers sharing >= half of the smaller author-key set (one group counts once) — the old
    facts.independent rule, on author_link keys."""
    groups: list[set] = []
    for s in sorted(speakers):
        a = author_of.get(s) or {s}
        for g in groups:
            if a and g and len(a & g) * 2 >= min(len(a), len(g)):
                g |= a
                break
        else:
            groups.append(set(a))
    return len(groups)


def build(workers: int | None = None, rebuild: bool = False, log=print, cit=None, ext=None, reg=None,
          chat=None, snapshot_extra=()) -> dict:
    chat = chat or call_local
    params = {"v": 1, "snapshot_extra": list(snapshot_extra)}
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

            # ---- category canon: normalise the other pass's category phrases (LLM, item = the phrase — stable
            #      under corpus growth; the prompt offers the current canonical names for reuse)
            w = run.work("categories")
            phrases = sorted({c.strip() for (c,) in ext.execute(
                "SELECT DISTINCT json_extract(meta, '$.category') FROM statements "
                "WHERE kind='other' AND json_extract(meta, '$.category') IS NOT NULL")
                if c and len(c.strip()) >= 3})
            existing = [r[0] for r in con.execute("SELECT canonical FROM category_canon "
                                                  "GROUP BY canonical ORDER BY count(*) DESC LIMIT 40")]
            todo_c = set(w.todo([(p, store.sha(PR.CATEGORY_CANON_SHA, PR.MODEL)) for p in phrases]))
            if todo_c:
                log(f"[cognition] categories: {len(todo_c):,} of {len(phrases):,} phrases to canonicalise")

            def one_category(phrase):
                if phrase not in todo_c:
                    return
                try:
                    raw = chat(PR.CATEGORY_CANON.format(
                        existing="\n".join(f"- {x}" for x in existing) or "(none yet)", phrase=phrase[:200]),
                        model=PR.MODEL, max_tokens=120, temperature=0.0, enable_thinking=False,
                        item=f"category:{phrase[:60]}")
                    obj = parse_json_response(raw or "")
                    canon = (obj or {}).get("canonical") if isinstance(obj, dict) else None
                    if not isinstance(canon, str) or not canon.strip():
                        w.fail(phrase, "category: no parseable answer")
                        return
                    with run.lock:
                        con.execute("INSERT OR REPLACE INTO category_canon VALUES (?,?,?,?)",
                                    (phrase, canon.strip().lower()[:120],
                                     int(bool((obj or {}).get("umbrella"))), PR.MODEL))
                        ccounter["n"] += 1
                        if ccounter["n"] % 200 == 0:
                            con.commit()
                        w.ok(phrase)
                except Exception as e:
                    w.fail(phrase, f"{type(e).__name__}: {e}")

            ccounter = {"n": 0}
            store.parallel(one_category, [p for p in phrases if p in todo_c], workers or 8, log=log, every=2000,
                           label="cognition:categories")
            with run.lock:
                con.commit()
            w.sweep(phrases, lambda p: con.execute("DELETE FROM category_canon WHERE phrase=?", (p,)))

            # ---- category_daily: statement counts per canonical category per day (deterministic; phrases with
            #      no adjudicated canonical yet count under their own lowercased surface)
            w = run.work("category_daily")
            canon_fp = store.sha(list(con.execute("SELECT phrase, canonical FROM category_canon ORDER BY phrase")))
            if w.todo([("all", store.sha(ext_fp, canon_fp))]):
                canon_map = dict(con.execute("SELECT phrase, canonical FROM category_canon"))
                agg: Counter = Counter()
                for phr, day, n in ext.execute(
                        "SELECT json_extract(meta, '$.category'), date, count(*) FROM statements "
                        "WHERE kind='other' AND date IS NOT NULL AND json_extract(meta, '$.category') IS NOT NULL "
                        "GROUP BY 1, 2"):
                    phr = (phr or "").strip()
                    agg[(canon_map.get(phr, phr.lower()), day)] += n
                con.execute("DELETE FROM category_daily")
                con.executemany("INSERT OR REPLACE INTO category_daily VALUES (?,?,?)",
                                [(c, d, n) for (c, d), n in sorted(agg.items()) if c])
                con.commit()
                w.ok("all")

            # ---- families: time-sliced Leiden snapshots (§8: the only grid-snapshot object; fixed seed, weighted
            #      modularity — its configuration null model is the degree-based hub de-weighting)
            import igraph as ig
            import leidenalg
            w = run.work("families")
            lo_row = reg.execute("SELECT min(first_hi) FROM papers WHERE status='active' "
                                 "AND first_hi IS NOT NULL").fetchone()
            hi_cands = [x for x in (reg.execute("SELECT max(first_hi) FROM papers WHERE status='active'").fetchone()[0],
                                    con.execute("SELECT max(day) FROM cocite").fetchone()[0],
                                    con.execute("SELECT max(valid_from) FROM lineage_edge").fetchone()[0],
                                    ext.execute("SELECT max(date) FROM statements").fetchone()[0]) if x]
            if lo_row[0] and hi_cands:
                data_hi = max(hi_cands)
                snaps = snapshot_grid(lo_row[0], data_hi, extra=snapshot_extra)
                fam_fp = store.sha(cit_fp, ext_fp, SNAPSHOT_SEED, ig.__version__, leidenalg.version)
                todo_f = set(w.todo([(s, store.sha(fam_fp, s)) for s in snaps]))
                if todo_f:
                    log(f"[cognition] families: {len(todo_f):,} of {len(snaps):,} snapshots to cluster")
                first_hi = dict(reg.execute("SELECT paper_id, first_hi FROM papers "
                                            "WHERE status='active' AND first_hi IS NOT NULL"))
                coc = sorted(con.execute("SELECT a, b, day, n FROM cocite WHERE day IS NOT NULL"),
                             key=lambda r: r[2])
                lin = sorted(con.execute("SELECT child, parent, valid_from FROM lineage_edge"), key=lambda r: r[2])
                canon_map = dict(con.execute("SELECT phrase, canonical FROM category_canon"))
                umbrella = {p for p, u in con.execute("SELECT phrase, umbrella FROM category_canon") if u}
                cat_first: dict[tuple, str] = {}
                for about, phr, date in ext.execute(
                        "SELECT about, json_extract(meta, '$.category'), date FROM statements "
                        "WHERE kind='other' AND date IS NOT NULL "
                        "AND json_extract(meta, '$.category') IS NOT NULL"):
                    raw = (phr or "").strip()
                    c = canon_map.get(raw, raw.lower())
                    if c and raw not in umbrella:
                        k = (c, about)
                        if k not in cat_first or date < cat_first[k]:
                            cat_first[k] = date
                by_cat: dict[str, dict] = defaultdict(dict)
                for (c, p_), d_ in cat_first.items():
                    by_cat[c][p_] = d_
                cat_edges = []
                for c, papers_ in by_cat.items():
                    if len(papers_) > 50:                  # umbrella-scale categories carry no family signal
                        continue
                    ps = sorted(papers_)
                    for i, a_ in enumerate(ps):
                        for b_ in ps[i + 1:]:
                            cat_edges.append((a_, b_, max(papers_[a_], papers_[b_])))
                cat_edges.sort(key=lambda r: r[2])
                weights: Counter = Counter()
                ci = li = ki = 0
                for snap in snaps:                         # cumulative: every snapshot advances the pointers
                    while ci < len(coc) and coc[ci][2] <= snap:
                        weights[(coc[ci][0], coc[ci][1])] += coc[ci][3]
                        ci += 1
                    while li < len(lin) and lin[li][2] <= snap:
                        weights[tuple(sorted(lin[li][:2]))] += 2      # a direct lineage claim weighs 2 co-cites
                        li += 1
                    while ki < len(cat_edges) and cat_edges[ki][2] <= snap:
                        weights[tuple(sorted(cat_edges[ki][:2]))] += 1
                        ki += 1
                    if snap not in todo_f:
                        continue
                    edges = [(a_, b_) for (a_, b_), wt in weights.items()
                             if wt > 0 and first_hi.get(a_, "9999") <= snap and first_hi.get(b_, "9999") <= snap]
                    con.execute("DELETE FROM family_snapshot WHERE snapshot=?", (snap,))
                    if edges:
                        nodes = sorted({x for e in edges for x in e})
                        idx = {p_: i for i, p_ in enumerate(nodes)}
                        g = ig.Graph(n=len(nodes), edges=[(idx[a_], idx[b_]) for a_, b_ in edges], directed=False)
                        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                                        weights=[weights[e] for e in edges],
                                                        seed=SNAPSHOT_SEED, n_iterations=3)
                        rows = []
                        comms = sorted(part, key=lambda c: (-len(c), min(c)))
                        for k, comm in enumerate(comms):
                            members = sorted(nodes[i] for i in comm)
                            if len(members) < MIN_FAMILY:
                                continue
                            rows.append((snap, f"{snap}#f{k}", None, json.dumps(members), None))
                        con.executemany("INSERT OR REPLACE INTO family_snapshot VALUES (?,?,?,?,?)", rows)
                    con.commit()
                    w.ok(snap)
                w.sweep(snaps, lambda s: con.execute("DELETE FROM family_snapshot WHERE snapshot=?", (s,)))

            # ---- reception shifts: deterministic screen (effect floors + Fisher + BH) -> LLM verification
            w = run.work("shift_screen")
            if w.todo([("all", store.sha(ext_fp))]):
                votes: dict[str, dict] = defaultdict(dict)     # subject -> (speaker, sent_id) -> one vote
                for about, speaker, date, role, function, facet, cat, quote, sid in ext.execute(
                        "SELECT about, speaker, date, role, function, facet, json_extract(meta, '$.category'), "
                        "quote, json_extract(loc, '$.sent_id') FROM statements "
                        "WHERE kind='other' AND date IS NOT NULL ORDER BY date"):
                    votes[about][(speaker, sid)] = {"date": date, "role": role, "function": function,
                                                    "facet": facet, "category": cat, "speaker": speaker,
                                                    "quote": quote or ""}
                author_of: dict[str, set] = {}
                for k, p_ in con.execute("SELECT author_key, paper_id FROM author_link"):
                    author_of.setdefault(p_, set()).add(k)
                replaces = dict(con.execute("SELECT parent, min(valid_from) FROM lineage_edge "
                                            "WHERE relation='replaces' GROUP BY parent"))
                cands = screen_shifts({o: sorted(v.values(), key=lambda r: r["date"])
                                       for o, v in votes.items()}, author_of, replaces)
                # approved rows survive; rejected rows are re-screened and re-adjudicated every build (the
                # evidence format and the verify prompt evolve — 10-09 fix: the old single-window evidence
                # rendering made 88/91 rejections structural, and keeping them would freeze that bug in)
                con.execute("DELETE FROM shift_event WHERE status IN ('candidate','rejected')")
                con.executemany("INSERT OR IGNORE INTO shift_event(subject, facet, window_start, window_end, "
                                "direction, evidence, status) VALUES (?,?,?,?,?,?,?)",
                                [(c["subject"], c["facet"], c["window_start"], c["window_end"], c["direction"],
                                  c["evidence"], "candidate") for c in cands])
                con.commit()
                w.ok("all")
                log(f"[cognition] shift screen: {len(cands):,} candidates")

            w = run.work("shifts")
            sconn = store.read_only(run.path)
            try:
                s_items = [(str(i), store.sha(PR.SHIFT_VERIFY_SHA, PR.MODEL, ev)) for i, ev in
                           sconn.execute("SELECT id, evidence FROM shift_event WHERE status='candidate'")]
                todo_s = set(w.todo(s_items))
                if todo_s:
                    log(f"[cognition] shifts: {len(todo_s):,} candidates to verify")

                def one_shift(id_str):
                    if id_str not in todo_s:
                        return
                    try:
                        r = sconn.execute("SELECT subject, facet, direction, evidence FROM shift_event "
                                          "WHERE id=? AND status='candidate'", (int(id_str),)).fetchone()
                        if r is None:
                            w.ok(id_str)
                            return
                        t = reg.execute("SELECT title FROM papers WHERE paper_id=?", (r[0],)).fetchone()
                        dirj = json.loads(r[2] or "{}")
                        ev = json.loads(r[3] or "[]")
                        claim, blocks = _shift_claim(r[1], dirj, ev)
                        raw = chat(PR.SHIFT_VERIFY.format(
                            subject=f"{r[0]}" + (f" — {t[0][:120]}" if t and t[0] else ""),
                            claim=claim,
                            evidence="\n\n".join(
                                f"{title}:\n" + ("\n".join(f"- {x[:200]}" for x in rows[:3]) or "(none)")
                                for title, rows in blocks)),
                            model=PR.MODEL, max_tokens=150, temperature=0.0, enable_thinking=False,
                            item=f"shift:{id_str}")
                        obj = parse_json_response(raw or "")
                        if not isinstance(obj, dict) or not isinstance(obj.get("holds"), bool):
                            w.fail(id_str, "shift: no parseable verdict")
                            return
                        with run.lock:
                            con.execute("UPDATE shift_event SET status=?, decided_by=? WHERE id=?",
                                        ("approved" if obj["holds"] else "rejected", PR.MODEL, int(id_str)))
                            con.commit()
                            w.ok(id_str)
                    except Exception as e:
                        w.fail(id_str, f"{type(e).__name__}: {e}")

                store.parallel(one_shift, [i for i, _ in s_items if i in todo_s], workers or 8, log=log,
                               every=1000, label="cognition:shifts")
                with run.lock:
                    con.commit()
                w.sweep([i for i, _ in s_items], lambda i: None)
            finally:
                sconn.close()

            # ---- family names: the latest snapshot (+ benchmark cut-offs) gets LLM names and boundary-member
            #      verdicts; earlier snapshots inherit by member overlap — naming all ~80 snapshots x thousands
            #      of families would be an LLM explosion for near-identical member sets
            w = run.work("family_names")
            nconn = store.read_only(run.path)
            try:
                latest_snap = con.execute("SELECT max(snapshot) FROM family_snapshot").fetchone()[0]
                naming_snaps = ({latest_snap} if latest_snap else set()) | {x[:10] for x in snapshot_extra}
                fam_rows = [(s, f, json.loads(m), n) for s, f, m, n in
                            con.execute("SELECT snapshot, family_id, members, name FROM family_snapshot")]
                to_name = [(s, f, mm) for s, f, mm, n in fam_rows if s in naming_snaps and not n]
                todo_n = set(w.todo([(f, store.sha(PR.FAMILY_NAME_SHA, PR.MODEL, store.sha(sorted(mm))))
                                     for _, f, mm in to_name]))

                def one_name(fid):
                    if fid not in todo_n:
                        return
                    try:
                        r = nconn.execute("SELECT members FROM family_snapshot WHERE family_id=?", (fid,)).fetchone()
                        members = json.loads(r[0])
                        head = members[:15]
                        deg: Counter = Counter()
                        if len(head) > 1:
                            ph = ",".join("?" * len(head))
                            for a_, b_ in nconn.execute(f"SELECT a, b FROM cocite WHERE a IN ({ph}) AND b IN ({ph})",
                                                        (*head, *head)):
                                deg[a_] += 1
                                deg[b_] += 1
                            for a_, b_ in nconn.execute(
                                    f"SELECT DISTINCT child, parent FROM lineage_edge WHERE child IN ({ph}) "
                                    f"AND parent IN ({ph})", (*head, *head)):
                                deg[a_] += 1
                                deg[b_] += 1
                        titles = {}
                        for pid in head:
                            t = reg.execute("SELECT title, first_hi FROM papers WHERE paper_id=?", (pid,)).fetchone()
                            titles[pid] = ((t[0] if t else "") or pid)[:120] + (f" ({t[1][:4]})" if t and t[1] else "")
                        lines = [f"{i}. {titles.get(p_, p_)}" + ("" if deg.get(p_, 0) > 1 else " (weak link)")
                                 for i, p_ in enumerate(head, 1)]
                        _q = (f"SELECT json_extract(meta, '$.category') FROM statements WHERE kind='other' "
                              f"AND about IN ({','.join('?' * len(head))}) "
                              "AND json_extract(meta, '$.category') IS NOT NULL LIMIT 200")
                        cats = [c for (c,) in ext.execute(_q, tuple(head))] if head else []
                        top_cats = ", ".join(c for c, _ in Counter(x.lower() for x in cats if x).most_common(5)) \
                            or "(none)"
                        raw = chat(PR.FAMILY_NAME.format(members="\n".join(lines), categories=top_cats),
                                   model=PR.MODEL, max_tokens=300, temperature=0.0, enable_thinking=False,
                                   item=f"family:{fid}")
                        obj = parse_json_response(raw or "")
                        name = (obj or {}).get("name") if isinstance(obj, dict) else None
                        if not isinstance(name, str) or not name.strip():
                            w.fail(fid, "family: no parseable name")
                            return
                        # weak links only mean something in a family big enough to have an interior
                        weak = [i for i, p_ in enumerate(head, 1) if deg.get(p_, 0) <= 1] if len(members) > 4 else []
                        keep = {int(x) for x in (obj or {}).get("keep") or [] if str(x).lstrip("-").isdigit()}
                        drop = {i for i in weak if i not in keep} if weak else set()
                        kept = sorted(set([p_ for i, p_ in enumerate(members, 1) if i not in drop]))
                        with run.lock:
                            con.execute("UPDATE family_snapshot SET name=?, members=?, named_by=? "
                                        "WHERE family_id=?",
                                        (name.strip().lower()[:80], json.dumps(kept), PR.MODEL, fid))
                            con.commit()
                            w.ok(fid)
                    except Exception as e:
                        w.fail(fid, f"{type(e).__name__}: {e}")

                store.parallel(one_name, [f for _, f, _ in to_name if f in todo_n], workers or 8, log=log,
                               every=500, label="cognition:family_names")
                with run.lock:
                    con.commit()
                named_fams = [(n, set(json.loads(m))) for _s, f, m, n in
                              con.execute("SELECT snapshot, family_id, members, name FROM family_snapshot "
                                          "WHERE name IS NOT NULL")]
                inherited = 0
                for s, f, mm, n in fam_rows:
                    if n or s in naming_snaps:
                        continue
                    ms = set(mm)
                    best, bj = None, 0.0
                    for n2, ms2 in named_fams:
                        j = len(ms & ms2) / len(ms | ms2) if (ms | ms2) else 0.0
                        if j > bj:
                            best, bj = n2, j
                    if best is not None and bj >= 0.3:
                        con.execute("UPDATE family_snapshot SET name=?, named_by='inherit' WHERE family_id=?",
                                    (best, f))
                        inherited += 1
                con.commit()
                if inherited:
                    log(f"[cognition] family names: {inherited:,} older-snapshot families inherited a name")
            finally:
                nconn.close()

            # ---- facts: family-level fact groups (§8). Recall is deterministic (same-facet content-word
            #      Jaccard inside one family), the relation verdict is the LLM's, the assembly and the status
            #      timeline are deterministic again.
            latest_snap2 = con.execute("SELECT max(snapshot) FROM family_snapshot").fetchone()[0]
            fam_members = {f: json.loads(m) for f, m in con.execute(
                "SELECT family_id, members FROM family_snapshot WHERE snapshot=?",
                (latest_snap2,))} if latest_snap2 else {}
            stmt_rows: dict[int, dict] = {}
            by_about: dict[str, list] = defaultdict(list)
            for i_, about, facet, text, quote, date, speaker, kind in ext.execute(
                    f"SELECT id, about, facet, text, quote, date, speaker, kind FROM statements "
                    f"WHERE facet IN ({','.join('?' * len(FACT_FACETS))}) AND date IS NOT NULL "
                    f"AND text IS NOT NULL", FACT_FACETS):
                stmt_rows[i_] = {"about": about, "facet": facet, "text": text, "quote": quote or "",
                                 "date": date, "speaker": speaker, "kind": kind}
                by_about[about].append(i_)
            cand_pairs: set = set()
            too_large = 0
            for f_, members in fam_members.items():
                ids_ = [i_ for m_ in members for i_ in by_about.get(m_, [])]
                if len(ids_) > FACT_MAX_FAMILY_STMTS:
                    too_large += 1
                    continue
                per_facet: dict[str, list] = defaultdict(list)
                for i_ in ids_:
                    per_facet[stmt_rows[i_]["facet"]].append(i_)
                for fct, fids in per_facet.items():
                    words = {i_: _content_words(stmt_rows[i_]["text"]) for i_ in fids}
                    for xi, a_ in enumerate(fids):
                        for b_ in fids[xi + 1:]:
                            wa, wb = words[a_], words[b_]
                            if wa and wb and len(wa & wb) / len(wa | wb) >= FACT_RECALL_SIM:
                                cand_pairs.add((min(a_, b_), max(a_, b_)))

            w = run.work("fact_pairs")
            todo_p = set(w.todo([(f"{a_}:{b_}", store.sha(PR.FACT_REL_SHA, PR.MODEL,
                                                          stmt_rows[a_]["text"], stmt_rows[b_]["text"]))
                                 for a_, b_ in sorted(cand_pairs) if a_ in stmt_rows and b_ in stmt_rows]))
            if todo_p:
                log(f"[cognition] facts: {len(todo_p):,} of {len(cand_pairs):,} candidate pairs to adjudicate")

            def one_pair(item):
                if item not in todo_p:
                    return
                a_, b_ = (int(x) for x in item.split(":"))
                sa, sb = stmt_rows.get(a_), stmt_rows.get(b_)
                if sa is None or sb is None:
                    w.ok(item)
                    return
                try:
                    raw = chat(PR.FACT_REL.format(about_a=sa["about"], date_a=sa["date"], text_a=sa["text"][:300],
                                                  about_b=sb["about"], date_b=sb["date"], text_b=sb["text"][:300]),
                               model=PR.MODEL, max_tokens=80, temperature=0.0, enable_thinking=False,
                               item=f"factpair:{item}")
                    obj = parse_json_response(raw or "")
                    rel = (obj or {}).get("relation") if isinstance(obj, dict) else None
                    if rel not in ("same", "opposite", "narrower", "broader", "unrelated"):
                        w.fail(item, "fact pair: no usable relation")
                        return
                    with run.lock:
                        con.execute("INSERT OR REPLACE INTO fact_pair VALUES (?,?,?,?)", (a_, b_, rel, PR.MODEL))
                        pcounter["n"] += 1
                        if pcounter["n"] % 200 == 0:
                            con.commit()
                        w.ok(item)
                except Exception as e:
                    w.fail(item, f"{type(e).__name__}: {e}")

            pcounter = {"n": 0}
            store.parallel(one_pair, sorted(todo_p), workers or 8, log=log, every=1000,
                           label="cognition:fact_pairs")
            with run.lock:
                con.commit()

            w = run.work("facts")
            pair_digest = store.sha(list(con.execute("SELECT stmt_a, stmt_b, relation FROM fact_pair "
                                                     "ORDER BY stmt_a, stmt_b")))
            if w.todo([("all", store.sha(ext_fp, pair_digest, store.sha(sorted(fam_members))))]):
                parent: dict[int, int] = {}

                def find(x):
                    while parent.get(x, x) != x:
                        parent[x] = parent.get(parent[x], parent[x])
                        x = parent[x]
                    return x

                def union(x, y):
                    rx, ry = find(x), find(y)
                    if rx != ry:
                        parent[max(rx, ry)] = min(rx, ry)

                for a_, b_ in con.execute("SELECT stmt_a, stmt_b FROM fact_pair WHERE relation='same'"):
                    if a_ in stmt_rows and b_ in stmt_rows:
                        union(a_, b_)
                group_of: dict[int, int] = {}
                for f_, members in fam_members.items():
                    for m_ in members:
                        for i_ in by_about.get(m_, []):
                            if i_ in stmt_rows:
                                group_of[i_] = find(i_)
                groups: dict[int, list] = defaultdict(list)
                for i_, root in group_of.items():
                    groups[root].append(i_)
                fid_of = {root: "fact:" + store.sha(sorted(ids_))[:12] for root, ids_ in groups.items()}
                author_of: dict[str, set] = {}
                for k_, p_ in con.execute("SELECT author_key, paper_id FROM author_link"):
                    author_of.setdefault(p_, set()).add(k_)
                member_rows, event_rows = [], []
                for root, ids_ in groups.items():
                    ids_ = sorted(ids_)
                    fid = fid_of[root]
                    rep = max(ids_, key=lambda i_: (stmt_rows[i_]["kind"] == "other", len(stmt_rows[i_]["text"])))
                    for i_ in ids_:
                        member_rows.append((fid, i_, "representative" if i_ == rep else "support"))
                    seen_sp: set = set()
                    seen_ab: set = set()
                    status = None
                    for i_ in sorted(ids_, key=lambda x: stmt_rows[x]["date"]):
                        r_ = stmt_rows[i_]
                        seen_sp.add(r_["speaker"])
                        seen_ab.add(r_["about"])
                        n_ind = _independent(seen_sp, author_of)
                        new = ("consensus" if n_ind >= 3 and len(seen_ab) >= 2
                               else "established" if n_ind >= 2 else "single-source")
                        if new != status:
                            event_rows.append((fid, r_["date"], new,
                                               json.dumps({"n_independent": n_ind, "n_subjects": len(seen_ab)})))
                            status = new
                for a_, b_ in con.execute("SELECT stmt_a, stmt_b FROM fact_pair WHERE relation='opposite'"):
                    fa = fid_of.get(group_of.get(a_)) if a_ in group_of else None
                    fb = fid_of.get(group_of.get(b_)) if b_ in group_of else None
                    if not fa and not fb:
                        continue
                    d_ = max(stmt_rows[x]["date"] for x in (a_, b_) if x in stmt_rows)
                    ev = json.dumps([stmt_rows[x]["quote"][:200] for x in (a_, b_) if x in stmt_rows],
                                    ensure_ascii=False)
                    for fid in {fa, fb} - {None}:
                        event_rows.append((fid, d_, "contested", ev))
                con.execute("DELETE FROM fact_member")
                con.execute("DELETE FROM fact_status_event")
                con.executemany("INSERT OR REPLACE INTO fact_member VALUES (?,?,?)", member_rows)
                con.executemany("INSERT INTO fact_status_event VALUES (?,?,?,?)", sorted(event_rows))
                con.commit()
                w.ok("all")
                log(f"[cognition] facts: {len(groups):,} facts over {len(member_rows):,} statements "
                    f"({too_large} families skipped as too large)")

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
                      "lineage_hyper": q("SELECT count(*) FROM lineage_hyper"),
                      "category_canon": q("SELECT count(*) FROM category_canon"),
                      "category_daily": q("SELECT count(*) FROM category_daily"),
                      "family_snapshots": q("SELECT count(DISTINCT snapshot) FROM family_snapshot"),
                      "families": q("SELECT count(*) FROM family_snapshot"),
                      "families_named": q("SELECT count(*) FROM family_snapshot WHERE name IS NOT NULL"),
                      "facts": q("SELECT count(DISTINCT fact_id) FROM fact_member"),
                      "facts_contested": q("SELECT count(DISTINCT fact_id) FROM fact_status_event "
                                           "WHERE status='contested'"),
                      "shifts": dict(con.execute("SELECT status, count(*) FROM shift_event GROUP BY status"))}
            run.finish(counts)
        finally:
            for c in mine:
                c.close()
    return counts
