# -*- coding: utf-8 -*-
"""The Sciverse content layer (INTEGRATED-SYSTEM-1005 v2.5 §2.3): Sciverse's pre-parsed full text (MinerU 2.5
markdown) is the preferred body/table/formula source; GROBID stays the citation layer, MinerU-runs-here only cover
papers Sciverse does not hold.

Two halves with different disciplines:

  fetch()   the network marathon (account limit 30 req/min, paced by the shared token bucket in sources.sciverse).
            Per paper: exact-title semantic search (Sciverse has no id lookup), accepted only on title_key
            equality; then the full text through /content. Cached in data/derived/sciverse_text.sqlite (its own
            file, NOT the dfc stage store — a code change must not re-download 10^5 texts): status ok | absent |
            error | gave_up, attempts <= 3, idempotent skip on re-run. Measured coverage on a 56-paper stratified
            SC sample: 71% (2013–2023 ~75%, 2024 ~50%, 2025 ~0%).

  materialize()   the CPU half, a documents-stage work pass (build.py): adapter (markdown -> units + sentences +
            display formulas, reference block cut — citations are GROBID's job), version determination and
            per-sentence dates, stored in documents.sqlite `sv` as the sv tier.

Version rule (v2.5): Sciverse's own date field is NOT trusted (measured: 4/18 said v1 while the text was the
latest). We determine the held version from each version's own sentences: sentences only v1 has and sentences
only the latest version has (versions.delta both ways) are looked up in the Sciverse text by word-bigram
coverage >= 0.8; > 50% of one side's sentences present and of the other's absent decides it (14/18 decidable in
the measurement). Sentence dates: found in the v1 GROBID text -> the v1 date; otherwise the latest known
version's date — and an undecidable text dates everything not in v1 by the latest version, which is the
leak-safe direction (later, never earlier).
"""
from __future__ import annotations

import json
import queue
import re
import sqlite3
import threading
import time
import zlib

from rapidfuzz import fuzz, process

from ..core import ids, paths
from ..sources import sciverse as S
from . import versions as VV

TEXT_DDL = """CREATE TABLE IF NOT EXISTS texts(paper_id TEXT PRIMARY KEY, doc_id TEXT, title TEXT, sv_date TEXT,
  status TEXT NOT NULL, detail TEXT, chars INT NOT NULL DEFAULT 0, has_refs INT NOT NULL DEFAULT 0,
  attempts INT NOT NULL DEFAULT 0, fetched_at TEXT, text_z BLOB);"""

MAX_ATTEMPTS = 3
_DONE = ("ok", "absent", "gave_up")


def text_store(path=None) -> str:
    return str(path or (paths.derived() / "sciverse_text.sqlite"))


def _connect(path=None) -> sqlite3.Connection:
    p = text_store(path)
    con = sqlite3.connect(p, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.executescript(TEXT_DDL)
    return con


# ---------------------------------------------------------------- fetch (network marathon)
_TOK_LOCK = threading.Lock()
_TOK_I = [0]


def _next_token() -> str | None:
    """Round-robin over every configured Sciverse key — the 30 req/min limit is per account, so three keys
    triple the marathon's throughput; sources.sciverse keeps one token bucket per key."""
    toks = S.all_tokens()
    if not toks:
        return None
    with _TOK_LOCK:
        t = toks[_TOK_I[0] % len(toks)]
        _TOK_I[0] += 1
    return t


def _search(title: str) -> dict | None:
    """The hit whose title_key equals the queried title's (the coverage experiment's rule)."""
    r = S.request_json("POST", "/agentic-search", payload={"query": title, "page_size": 10}, timeout_seconds=60,
                       token=_next_token())
    key = ids.title_key(title)
    return next((h for h in (r or {}).get("hits") or []
                 if ids.title_key(re.sub(r"^title:", "", h.get("title") or "")) == key), None)


def _content(doc_id: str) -> str:
    """Full text; one big read (the versions experiment pulled up to 400 kB in one request), paging as fallback."""
    out, off = "", 0
    for _ in range(40):
        c = S.request_json("GET", "/content", query={"doc_id": doc_id, "offset": off, "limit": 200000},
                           timeout_seconds=120, token=_next_token())
        out += (c or {}).get("text") or ""
        if str((c or {}).get("more")) != "True":
            return out
        off = int((c or {}).get("next_offset") or off + 200000)
    return out


def fetch(paper_ids, workers: int = 4, log=print, path=None, limit: int | None = None) -> dict:
    """Fetch + cache the Sciverse text of every paper that does not have a definitive row yet. Single writer
    thread (acquire's discipline); the token bucket paces every process on the account."""
    from ..dfc.store import parallel
    con = _connect(path)
    todo = []
    for pid in paper_ids:
        r = con.execute("SELECT status, attempts FROM texts WHERE paper_id=?", (pid,)).fetchone()
        if r and (r[0] in _DONE or r[1] >= MAX_ATTEMPTS):
            continue
        todo.append(pid)
    if limit:
        todo = todo[:limit]
    log(f"[sciverse] {len(todo):,} papers to fetch")
    if not todo:
        con.close()
        return {"todo": 0}

    reg = sqlite3.connect(f"file:{(paths.library() / 'registry.sqlite').as_posix()}?mode=ro", uri=True)
    titles: dict[str, str] = {}
    for i in range(0, len(todo), 900):                 # SQLite binds at most 999/32766 params — chunk
        part = todo[i:i + 900]
        for p, t in reg.execute(f"SELECT paper_id, title FROM records WHERE paper_id IN "
                                f"({','.join('?' * len(part))}) ORDER BY paper_id, source = 'arxiv'", part):
            titles[p] = t                              # arxiv rows come last, so they win the dict
    reg.close()

    out_q: queue.Queue = queue.Queue()
    n = {"ok": 0, "absent": 0, "error": 0, "gave_up": 0}
    stop = threading.Event()

    def one(pid):
        title = (titles.get(pid) or "").strip()
        if not title:
            out_q.put((pid, None, title, None, "absent", "no title", "", 0))
            return
        try:
            hit = _search(title)
            if hit is None:
                out_q.put((pid, None, title, None, "absent", "no title_key hit", "", 0))
                return
            text = _content(str(hit.get("doc_id") or ""))
            if len(text) < 2000:                      # a stub page is not a full text
                out_q.put((pid, hit.get("doc_id"), title, hit.get("publication_published_date"),
                           "error", f"content too short ({len(text)})", "", 0))
                return
            out_q.put((pid, hit.get("doc_id"), title, hit.get("publication_published_date"), "ok", "",
                       text, len(text)))
        except Exception as e:                        # one broken paper must not kill the marathon
            out_q.put((pid, None, title, None, "error", f"{type(e).__name__}: {e}"[:300], "", 0))

    def writer():
        while not (stop.is_set() and out_q.empty()):
            try:
                pid, doc_id, title, sv_date, status, detail, text, chars = out_q.get(timeout=0.5)
            except queue.Empty:
                continue
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            att = con.execute("SELECT attempts FROM texts WHERE paper_id=?", (pid,)).fetchone()
            att = (att[0] if att else 0) + 1
            if status == "error" and att >= MAX_ATTEMPTS:
                status = "gave_up"
            con.execute("INSERT OR REPLACE INTO texts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (pid, doc_id, title, sv_date, status, detail, chars,
                         int(bool(re.search(r"(?im)^#*\s*(references|bibliography)\s*$", text))) if text else 0,
                         att, now, zlib.compress(text.encode("utf-8"), 6) if text else None))
            n[status if status in n else "error"] += 1
            if sum(n.values()) % 200 == 0:
                con.commit()
        con.commit()

    wt = threading.Thread(target=writer)
    wt.start()
    try:
        parallel(one, todo, workers, log=log, every=2000, label="sciverse")
    finally:
        stop.set()
        wt.join()
        con.close()
    log(f"[sciverse] fetch: {n}")
    return n


# ---------------------------------------------------------------- adapter (markdown -> units + sentences)
_HEAD = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
_TABLE = re.compile(r"<table.*?</table>", re.S)
_DISPLAY_MATH = re.compile(r"\$\$.+?\$\$", re.S)
_CAPTION = re.compile(r"(?mi)^\s*((?:table|figure|fig\.)\s*\d+[.:]\s*.+)$")
_IMAGE_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MATH_HOLD = re.compile(r"\$[^$\n]+\$|\\\(.+?\\\)")
_BOUND = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\\$0-9])")
_ABBR = {"e.g", "i.e", "et al", "cf", "vs", "etc", "fig", "figs", "eq", "eqs", "sec", "secs", "tab", "no", "vol",
         "pp", "ch", "al", "approx", "resp", "refs", "ref", "eds", "ed", "trans", "dept", "univ", "inc", "ltd"}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", _IMAGE_MD.sub(" ", s)).strip()      # base64/linked figures are noise for prose


def split_sentences(text: str) -> list[str]:
    """Conservative splitter for cleaned paragraph text: a boundary needs [.!?] + whitespace + a capital / digit /
    opening bracket / math start, the token before it must not be an abbreviation or a single initial, and inline
    math is held out so dots and '$' inside formulas never cut. Sentences are exact substrings of `text`
    (whitespace already collapsed) — the final-check quote discipline relies on it. When in doubt: do not split."""
    spans: list[str] = []

    def hold(m):
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    t = _MATH_HOLD.sub(hold, text)
    cuts = [0]
    for m in _BOUND.finditer(t):
        prev = t[:m.start()].rstrip().rsplit(" ", 1)[-1].strip(".").lower()
        if prev in _ABBR or len(prev) <= 1:
            continue
        cuts.append(m.end())
    cuts.append(len(t))
    out = []
    for a, b in zip(cuts, cuts[1:]):
        s = t[a:b].strip()
        if len(s) < 2:
            continue
        for i, sp in enumerate(spans):
            if f"\x00{i}\x00" in s:
                s = s.replace(f"\x00{i}\x00", sp)
        out.append(s)
    return out


_TRAILING_ENTRIES = re.compile(r"(?ms)(?:^\s*\[\d+\][^\n]*\n?){3,}\s*$")


def _cut_refs(text: str) -> tuple[str, bool]:
    """(body, has_refs): the reference block is cut by heading (citations.markdown.find_refs); MinerU texts with
    an UNTITLED reference list (the chemistry/medicine sentinel form) are cut at the trailing [n]-entry block."""
    from ..citations.markdown import find_refs
    parts = find_refs(text)
    if parts:
        return parts[0], True
    m = _TRAILING_ENTRIES.search(text)
    if m:
        return text[:m.start()], True
    return text, False


def doc(pid: str, text: str) -> dict:
    """Sciverse markdown -> the sv tier: {"units", "sentences", "has_refs"}. Unit kinds: section | para | table |
    caption | formula ($$display math$$; inline math stays inside its paragraph). Sentences come from para and
    caption units with stable ids '<pid>@sv#s<n>' — table and formula content is not prose (tei.py's rule).
    Page/bbox do not exist in Sciverse text; units needing coordinates come from the local MinerU tier."""
    body, has_refs = _cut_refs(text)
    units, sentences = [], []
    path: list[str] = []
    n = {"u": 0, "s": 0}

    def add_unit(kind: str, text_: str, section: str) -> str:
        n["u"] += 1
        uid = f"{pid}@sv#{kind}{n['u']}"
        units.append({"uid": uid, "kind": kind, "section": section, "text": text_})
        return uid

    def add_sentences(kind: str, raw: str, section: str) -> None:
        t = _clean(raw)
        if len(t) < 20:
            return
        uid = add_unit("caption" if _CAPTION.match(t) else kind, t, section)
        for s in split_sentences(t):
            n["s"] += 1
            sentences.append({"sid": f"{pid}@sv#s{n['s']}", "unit": uid, "text": s})

    # tables and display formulas first, so their content is not re-split into paragraphs
    spans = [(m.start(), m.end(), "table") for m in _TABLE.finditer(body)]
    spans += [(m.start(), m.end(), "formula") for m in _DISPLAY_MATH.finditer(body)]
    spans = sorted(spans)
    kept = []                                          # drop overlapping spans (a formula inside a table)
    end = -1
    for s, e, k in spans:
        if s >= end:
            kept.append((s, e, k))
            end = e
    pos = 0
    chunks = []
    for s, e, k in kept:
        if s > pos:
            chunks.append((pos, s, "text"))
        chunks.append((s, e, k))
        pos = e
    chunks.append((pos, len(body), "text"))
    for s, e, kind in chunks:
        seg = body[s:e]
        if kind == "table":
            add_unit("table", seg.strip(), " > ".join(path))
            continue
        if kind == "formula":
            add_unit("formula", _clean(seg), " > ".join(path))
            continue
        cursor = 0
        for m in _HEAD.finditer(seg):
            for para in re.findall(r"(?:[^\n]+\n?)+", seg[cursor:m.start()]):
                add_sentences("para", para, " > ".join(path))
            level, title = len(m.group(1)), _clean(m.group(2))
            num = re.match(r"^(\d+(?:\.\d+)*)\.?\s", title)
            if num:                                    # MinerU renders every heading '#'; the number carries depth
                level = num.group(1).count(".") + 1
            path[:] = path[:max(0, level - 1)] + [title]
            add_unit("section", title, " > ".join(path))
            cursor = m.end()
        for para in re.findall(r"(?:[^\n]+\n?)+", seg[cursor:]):
            add_sentences("para", para, " > ".join(path))
    return {"units": units, "sentences": sentences, "has_refs": has_refs}


# ---------------------------------------------------------------- version determination + sentence dates
def bigrams(s: str) -> set:
    w = re.findall(r"[a-z0-9]+", s.lower())
    return set(zip(w, w[1:]))


def _found(sent_texts: list[str], bg: set) -> tuple[int, int]:
    """(found, countable) — a sentence counts when it has >= 6 bigrams, and is found when >= 0.8 of them are in
    the other text's bigram set (sciverse_versions.py's measured rule)."""
    hit = n = 0
    for t in sent_texts:
        b = bigrams(t)
        if len(b) < 6:
            continue
        n += 1
        hit += len(b & bg) / len(b) >= 0.8
    return hit, n


def _bg_of(fast: dict) -> set:
    return bigrams(" ".join(s["text"] for s in fast["sentences"]))


def held_version(sv_sentences: list[str], v1_fast: dict | None, latest_fast: dict | None) -> tuple[str, dict]:
    """'v1' | 'latest' | 'ambiguous' | 'single' — by each version's own sentences (v2.5: Sciverse's date field is
    not trusted). 'single' when the library holds one version only (nothing to decide; dating still runs)."""
    if not v1_fast or not latest_fast:
        return "single", {}
    only_latest = VV.delta(v1_fast, latest_fast)["sentences"]
    only_v1 = VV.delta(latest_fast, v1_fast)["sentences"]
    bg = bigrams(" ".join(sv_sentences))
    h1, n1 = _found([s["text"] for s in only_v1], bg)
    hl, nl = _found([s["text"] for s in only_latest], bg)
    d = {"v1_only": [h1, n1], "latest_only": [hl, nl]}
    r1 = h1 / n1 if n1 else 0.0
    rl = hl / nl if nl else 0.0
    if r1 > 0.5 and rl < 0.5:
        return "v1", d
    if rl > 0.5 and r1 < 0.5:
        return "latest", d
    return "ambiguous", d


def date_sentences(sentences: list[dict], v1_fast: dict | None, v1_date: str | None, latest_date: str | None,
                   delta: dict | None = None) -> list[dict]:
    """Per-sentence dates (v2.5). With a delta (the calibrated GROBID-vs-GROBID revision screen) each sentence
    runs a contest: `slat` = best token_set against the delta's new + revised sentences (the latest version's own
    forms), `sv1` = best token_set against the v1 sentences. slat >= NEW_BELOW and slat >= sv1 -> the latest
    date; otherwise the v1 date when the sentence is in the v1 text — word-bigram coverage >= 0.8, or sv1 >=
    NEW_BELOW. Everything else -> the latest known date (later, never earlier: leak-safe; ties go latest too).
    The fuzzy legs matter because MinerU and GROBID render the same sentence differently: measured on the smoke
    rows, a bigram-only rule misdated 46% of the v1 sentences of multi-version papers to the latest version,
    almost all of them math-heavy ('$\\zeta_{k}$' vs 'ζ k').
    Without a delta the fuzzy legs stay off — an unscreened revised sentence must not fuzzy-match its own v1
    original into an earlier date."""
    new_n = [ids.norm_title(s["text"]) for s in (delta.get("sentences") or [])] if delta else []
    rev_n = [ids.norm_title(s["text"]) for s in (delta.get("revised") or [])] if delta else []
    v1n = [ids.norm_title(s["text"]) for s in v1_fast["sentences"]] if v1_fast and delta else []
    bg = _bg_of(v1_fast) if v1_fast and v1_fast.get("sentences") else set()

    def best(t, choices):
        m = process.extractOne(t, choices, scorer=fuzz.token_set_ratio) if choices and t else None
        return m[1] if m is not None else 0.0

    out = []
    for s in sentences:
        t = ids.norm_title(s["text"])
        b = bigrams(s["text"])
        in_v1 = bool(bg) and len(b) >= 6 and len(b & bg) / len(b) >= 0.8
        if v1n:
            sv1 = best(t, v1n)
            slat = max(best(t, new_n), best(t, rev_n))
            if slat >= VV.NEW_BELOW and slat >= sv1:
                in_v1 = False
            elif not in_v1:
                in_v1 = sv1 >= VV.NEW_BELOW
        out.append({**s, "date": (v1_date or latest_date) if in_v1 else (latest_date or v1_date)})
    return out


def materialize(scon, pid: str, v1_fast: dict | None, v1_date: str | None, latest_fast: dict | None,
                latest_date: str | None, delta: dict | None = None):
    """One paper's sv row for the documents stage: (row, payload-stats) or None when no cached text. `delta` is
    the stored versions.delta(v1, latest) dict — the revision screen date_sentences needs for the fuzzy leg."""
    r = scon.execute("SELECT doc_id, chars, text_z FROM texts WHERE paper_id=? AND status='ok'", (pid,)).fetchone()
    if not r or not r[2]:
        return None
    doc_id, chars, text = r[0], r[1], zlib.decompress(r[2]).decode("utf-8")
    d = doc(pid, text)
    held, detail = held_version([s["text"] for s in d["sentences"]], v1_fast, latest_fast)
    dated = date_sentences(d["sentences"], v1_fast, v1_date, latest_date, delta=delta)
    n_v1 = sum(1 for s in dated if s["date"] == v1_date and v1_date != latest_date)
    payload = {"doc_id": doc_id, "held": held, "detail": detail, "has_refs": d["has_refs"],
               "units": d["units"], "sentences": dated}
    return (pid, doc_id, held, chars, len(d["units"]), len(dated), n_v1,
            zlib.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"), 6))
