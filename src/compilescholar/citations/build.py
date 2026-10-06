# -*- coding: utf-8 -*-
"""Stage `citations` (phase C②): citing sentences + bibliography entries -> paper_id, built from the documents stage
(INTEGRATED-SYSTEM-1005 §7.2). Zero LLM.

Per citing paper: the v1 document (GROBID fast tier: sentences, entries, citation pairs) and, when the paper is
multi-version, the stored delta — the citation pairs only the latest version has. Every pair becomes a cites row
dated by the version whose text it was read from: v1 cites carry the v1 date, delta cites the latest version's date
(phase B gate: references newer than v1 land only in the delta, so nothing is visible before its time).

Entry -> paper resolution, deterministic cascade (offline; the network stage is the `citations.external` batch):
  1. the entry's own DOI or arXiv id -> registry identifiers (point lookup);
  2. registry ref_resolutions: raw entry text -> paper_id, validated and remembered by citations.external
     (Crossref query.bibliographic / OpenAlex). That pass also creates the metadata-only papers for DOI-only
     entries (which stage 1 then finds by the entry's own DOI) and for titled stubs (which stage 3 finds by
     title_key) — method = "external_ref";
  3. title_key exact match in the registry records, year window |Δyear| <= 1 (<= 4 when a surname of the candidate
     appears in the raw entry), candidates restricted to status='active', unique hit only — two papers sharing a
     title_key stay a stub until the merge queue decides they are one work (same philosophy as the registry);
  4. stub:<title_key> (or the first 80 chars of the normalised raw entry) — never empty.

cited values are bare paper_ids ("arxiv:…", "doi:…") or "stub:…" — the pre-C "paper:<arxiv_id>" prefix and its
[6:] slicing convention are gone; extract/index consumers move to this in C③–C④.
Self-citation: citing and cited share an author surname, and given-name initials agree where both sides have one
(registry authors.names). Ambiguity is possible with big groups; it is a sampling signal downstream, not a verdict.

Tables (citations.sqlite):
  docs(key PK, paper_id, version, date, n_entries, n_sentences, n_cites, n_resolved)
  entries(citing, version, key, raw, title, year, doi, arxiv, cited, method, PK(citing, version, key))
  sentences(id INTEGER PK, citing, citing_key, date, sid, sentence, keys, in_delta)
  cites(sentence_id, citing, date, version, key, cited, n_group, self_cite)

Work items: item = citing paper_id, fingerprint = (registry generation, the documents work keys of the v1/latest
docs and the delta) — a re-materialised document re-opens its paper; a changed registry (imports, merges) re-opens
every paper, because any entry may resolve better now (the pre-C stage had the same semantics via the papers-stage
fingerprint). Items that left (a merge moved the paper's assets) are swept."""
from __future__ import annotations

import json
import re

from ..compile.skeleton.resolve import entry_title_year
from ..core import ids, paths
from ..dfc import store
from ..documents.build import Documents
from .external import raw_sha

DDL = """CREATE TABLE IF NOT EXISTS docs(key TEXT PRIMARY KEY, paper_id TEXT, version INT, date TEXT,
  n_entries INT, n_sentences INT, n_cites INT, n_resolved INT);
CREATE TABLE IF NOT EXISTS entries(citing TEXT, version INT, key TEXT, raw TEXT, title TEXT, year INT, doi TEXT,
  arxiv TEXT, cited TEXT, method TEXT, PRIMARY KEY(citing, version, key));
CREATE TABLE IF NOT EXISTS sentences(id INTEGER PRIMARY KEY, citing TEXT, citing_key TEXT, date TEXT, sid TEXT,
  sentence TEXT, keys TEXT, in_delta INT);
CREATE TABLE IF NOT EXISTS cites(sentence_id INT, citing TEXT, date TEXT, version INT, key TEXT, cited TEXT,
  n_group INT, self_cite INT);
CREATE INDEX IF NOT EXISTS ix_cites_cited ON cites(cited, date);
CREATE INDEX IF NOT EXISTS ix_cites_citing ON cites(citing);
CREATE INDEX IF NOT EXISTS ix_cites_sid ON cites(sentence_id);
CREATE INDEX IF NOT EXISTS ix_sent_citing ON sentences(citing);"""


def _initial(name: str | None) -> str | None:
    toks = (name or "").split()
    return toks[0][0].lower() if toks and toks[0][:1].isalpha() else None


class Resolver:
    """The deterministic cascade, over a read-only registry connection (thread-safe ReadConn; the caches are plain
    dicts — concurrent fills compute the same values, so racing is harmless)."""

    def __init__(self, reg):
        self.reg = reg
        self._cands: dict[str, list] = {}
        self._auth: dict[str, set] = {}
        # registries created before stage 2 was wired have no ref_resolutions table (store.connect adds it)
        self._has_ref = reg.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ref_resolutions'").fetchone() is not None

    def surnames(self, pid: str) -> set:
        s = self._auth.get(pid)
        if s is None:
            s = set()
            for (names,) in self.reg.execute("SELECT names FROM authors WHERE paper_id=?", (pid,)):
                try:
                    rows = json.loads(names)
                except (TypeError, ValueError):
                    continue
                for a in rows:
                    sur = (a.get("surname") or "").strip().lower()
                    if sur:
                        s.add((sur, _initial(a.get("name"))))
            self._auth[pid] = s
        return s

    def self_cite(self, citing: str, cited: str) -> int:
        a = self.surnames(citing)
        if not a or cited.startswith("stub:"):
            return 0
        for sur, ia in a:
            for sur2, ib in self.surnames(cited):
                if sur == sur2 and (ia is None or ib is None or ia == ib):
                    return 1
        return 0

    def _candidates(self, tk: str) -> list:
        c = self._cands.get(tk)
        if c is None:
            c = [(pid, (hi or "")[:4]) for pid, hi in self.reg.execute(
                "SELECT DISTINCT r.paper_id, p.first_hi FROM records r JOIN papers p ON p.paper_id = r.paper_id "
                "AND p.status = 'active' WHERE r.title_key = ?", (tk,))]
            self._cands[tk] = c
        return c

    def resolve(self, entry: dict) -> tuple[str, str]:
        """-> (cited, method); cited is a paper_id or 'stub:…'."""
        doi, ax = entry.get("doi"), entry.get("arxiv")
        if doi:
            r = self.reg.execute("SELECT paper_id FROM identifiers WHERE scheme='doi' AND value=?", (doi,)).fetchone()
            if r:
                return r[0], "entry_doi"
        if ax:
            r = self.reg.execute("SELECT paper_id FROM identifiers WHERE scheme='arxiv' AND value=?", (ax,)).fetchone()
            if r:
                return r[0], "entry_arxiv"
        raw = entry.get("raw") or ""
        if raw and self._has_ref:
            r = self.reg.execute("SELECT paper_id FROM ref_resolutions WHERE raw_sha=?", (raw_sha(raw),)).fetchone()
            if r:
                return r[0], "external_ref"
        title, year = entry.get("title") or "", str(entry.get("year") or "")
        if not title:
            t2, y2 = entry_title_year(raw)
            title, year = t2, year or (str(y2) if y2 else "")
        tk = ids.title_key(title)
        if tk:
            cands = self._candidates(tk)
            if cands:
                raw = (entry.get("raw") or "").lower()
                y = int(year[:4]) if year[:4].isdigit() else None

                def within(cy: str, backed: bool) -> bool:
                    if y is None or not (cy or "").isdigit():
                        return True
                    return abs(y - int(cy)) <= (4 if backed else 1)

                def backed(cand) -> bool:
                    return any(len(sur) >= 3 and re.search(rf"\b{re.escape(sur)}\b", raw)
                               for sur, _ in self.surnames(cand[0]))

                keep = [c for c in cands if backed(c) and within(c[1], True)]
                if not keep:
                    keep = [c for c in cands if within(c[1], False)]
                if len(keep) == 1:
                    return keep[0][0], "registry_title"
        return f"stub:{tk or ids.norm_title(entry.get('raw') or '')[:80]}", "stub"


def _generation(reg) -> str:
    """One digest of the registry's resolution-relevant state: any import, merge or external resolution changes
    it."""
    has_ref = reg.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ref_resolutions'").fetchone()
    return store.sha(sorted(reg.execute("SELECT status, count(*) FROM papers GROUP BY status").fetchall()),
                     reg.execute("SELECT count(*) FROM identifiers").fetchone()[0],
                     reg.execute("SELECT count(*) FROM records").fetchone()[0],
                     reg.execute("SELECT count(*) FROM authors").fetchone()[0],
                     reg.execute("SELECT count(*) FROM ref_resolutions").fetchone()[0] if has_ref else 0)


def _items(D: Documents) -> list[tuple]:
    """[(citing paper_id, base docs key, latest docs key or None)] — base = v1 when held, else the version of
    record; latest from the stored delta. Reads the documents store, not this stage's."""
    vers: dict[str, list] = {}
    for pid, v in D.con.execute("SELECT paper_id, version FROM docs"):
        vers.setdefault(pid, []).append(v)
    latest = dict(D.con.execute("SELECT paper_id, latest_key FROM deltas"))
    out = []
    for pid, vs in sorted(vers.items()):
        base = 1 if 1 in vs else (0 if 0 in vs else max(vs))
        out.append((pid, f"{pid}@v{base}", latest.get(pid)))
    return out


def _delete(con, pid: str) -> None:
    for t in ("entries", "sentences", "cites"):
        con.execute(f"DELETE FROM {t} WHERE citing=?", (pid,))
    con.execute("DELETE FROM docs WHERE paper_id=?", (pid,))


def build(workers: int | None = None, rebuild: bool = False, log=print) -> dict:
    workers = workers or 8
    params = {"v": 2}
    with store.Run("citations", params, rebuild=rebuild) as run:
        con = run.con
        con.executescript(DDL)
        if "version" not in {r[1] for r in con.execute("PRAGMA table_info(entries)")}:
            raise RuntimeError("citations.sqlite has the pre-C (arXiv-id) schema; build it with --rebuild")
        D = Documents()
        reg = store.read_only(paths.library() / "registry.sqlite")
        try:
            gen = _generation(reg)
            resolver = Resolver(reg)
            doc_keys = store.item_keys("documents", "docs")
            delta_keys = store.item_keys("documents", "delta")
            items = _items(D)
            w = run.work("citations")
            todo = set(w.todo([(pid, store.sha(gen, doc_keys.get(bk, bk),
                                              (doc_keys.get(lk, lk), delta_keys.get(pid)) if lk else None))
                               for pid, bk, lk in items]))
            log(f"[citations] {len(todo):,} of {len(items):,} citing papers to resolve")
            methods: dict[str, int] = {}
            counter = {"items": 0}

            def one(triple):
                pid, bk, lk = triple
                if pid not in todo:
                    return
                try:
                    item_methods = _process(D, con, run.lock, resolver, pid, bk, lk, counter)
                    with run.lock:
                        for k, v in item_methods.items():
                            methods[k] = methods.get(k, 0) + v
                    w.ok(pid)
                except Exception as e:      # a corrupt document is this paper's failure, not the stage's
                    with run.lock:
                        _delete(con, pid)
                    w.fail(pid, f"{type(e).__name__}: {e}")

            store.parallel(one, items, workers, log=log, every=5000, label="citations")
            with run.lock:
                con.commit()
            w.sweep([p for p, _, _ in items], lambda p: _delete(con, p))
            q = lambda s: con.execute(s).fetchone()[0] or 0        # noqa: E731
            counts = {"docs": q("SELECT count(*) FROM docs"), "entries": q("SELECT count(*) FROM entries"),
                      "sentences": q("SELECT count(*) FROM sentences"), "cites": q("SELECT count(*) FROM cites"),
                      "resolved": q("SELECT count(*) FROM cites WHERE cited NOT LIKE 'stub:%'"),
                      "self_cites": q("SELECT count(*) FROM cites WHERE self_cite=1"),
                      "delta_sentences": q("SELECT count(*) FROM sentences WHERE in_delta=1"),
                      "methods": dict(sorted(methods.items()))}
            run.finish(counts)
        finally:
            reg.close()
            D.close()
    return counts


def _process(D: Documents, con, lock, resolver: Resolver, pid: str, base_key: str, latest_key: str | None,
             counter: dict) -> dict:
    """Resolve and store everything one citing paper says. Base doc first (v1 or the version of record), then the
    delta side (the latest version's own citation pairs). Returns this item's method counts."""
    base = D.get(base_key)
    if base is None or not base["fast"]:
        with lock:                      # no fast tier: no citation layer for this paper (valid outcome)
            _delete(con, pid)
        return {}
    delta = D.delta(pid)
    latest = D.get(latest_key) if latest_key else None
    if delta and (latest is None or not latest["fast"]):
        delta = latest = None

    methods: dict[str, int] = {}
    entry_rows, sent_rows, cite_rows, doc_rows = [], [], [], []

    def collect(doc, cites, entries, in_delta):
        version, date = doc["version"], doc["text_date"]
        sents = {s["sid"]: s["text"] for s in doc["fast"]["sentences"]}
        resolved = {}
        for gk, e in entries.items():
            cited, method = resolver.resolve(e)
            resolved[gk] = cited
            methods[method] = methods.get(method, 0) + 1
            entry_rows.append((pid, version, gk, e.get("raw"), e.get("title") or None,
                               int(e["year"]) if str(e.get("year") or "")[:4].isdigit() else None,
                               e.get("doi"), e.get("arxiv"), cited, method))
        per_sent: dict[str, list] = {}
        for sid, gk, ngrp in cites:
            per_sent.setdefault(sid, []).append((gk, ngrp))
        n_res = 0
        for sid, gks in per_sent.items():
            text = sents.get(sid)
            if text is None:
                continue
            sent_rows.append((doc["key"], sid, date, text,
                              json.dumps([f"v{version}:{g}" for g, _ in gks], ensure_ascii=False), in_delta))
            for gk, ngrp in gks:
                cited = resolved[gk]
                cite_rows.append((len(sent_rows) - 1, version, gk, cited, ngrp, resolver.self_cite(pid, cited), date))
                n_res += not cited.startswith("stub:")
        doc_rows.append((doc["key"], pid, version, date, len(entries), len(per_sent),
                         sum(len(v) for v in per_sent.values()), n_res))

    collect(base, [tuple(c) for c in base["fast"]["cites"]], base["fast"]["entries"], 0)
    if delta:
        lfast = latest["fast"]
        used = {c[1] for c in delta["cites"]} | set(delta["entries"])
        collect(latest, [tuple(c) for c in delta["cites"]],
                {k: e for k, e in lfast["entries"].items() if k in used}, 1)

    with lock:
        _delete(con, pid)
        con.executemany("INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?)", entry_rows)
        sent_ids = []
        for citing_key, sid, date, text, keys, in_delta in sent_rows:
            cur = con.execute("INSERT INTO sentences(citing, citing_key, date, sid, sentence, keys, in_delta) "
                              "VALUES (?,?,?,?,?,?,?)", (pid, citing_key, date, sid, text, keys, in_delta))
            sent_ids.append(cur.lastrowid)
        con.executemany("INSERT INTO cites VALUES (?,?,?,?,?,?,?,?)",
                        [(sent_ids[i], pid, date, version, gk, cited, ngrp, sc)
                         for i, version, gk, cited, ngrp, sc, date in cite_rows])
        con.executemany("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?,?)", doc_rows)
        counter["items"] += 1
        if counter["items"] % 200 == 0:
            con.commit()
    return methods
