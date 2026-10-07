# -*- coding: utf-8 -*-
"""The stage contract of the derived store (INTEGRATED-SYSTEM-1005 v2 §10.1).

Layout: <root>/<stage>.sqlite, one file per stage, <root>/manifests/<stage>.json, <root>/locks/<stage>.lock.

Two levels of change, two mechanisms:
  config  the stage's params, the source of every module it imports (import closure, computed from the AST — no
          hand-kept file list), or an upstream stage's config. Changes the stage `digest`; the stage is `stale`.
  data    an upstream stage gained or changed items without a config change. Changes that stage's `fingerprint`;
          downstream is `behind` and only the affected items are recomputed.
Item-level bookkeeping makes both incremental and exact: every unit of work (a paper, a citation pair, ...) is a row
in the stage's `_work` table keyed by (pass, item) and carrying `key` = sha(pass digest, item input fingerprint). An
item is done only when its status is ok under the current key; a config change of one pass re-opens that pass only;
a failed item is retried (MAX_ATTEMPTS) and never counted as done; items that left the current set are swept
(their outputs deleted) so the tables always equal what a clean build would produce.

    with store.Run("citations", params) as run:          # lock, upstream must be fresh, config digest
        w = run.work("citations", pass_digest)            # item bookkeeping for one pass
        for item in w.todo(items_with_fingerprints): ... w.ok(item) / w.fail(item, err)
        w.sweep(current_items, delete_fn)
        run.finish(counts)                                # atomic manifest; never written if the block raised

A stage whose block raises writes no manifest; consecutive failures past BREAKER abort the build (StageAborted)."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from filelock import FileLock, Timeout

from ..core import paths

STAGES = ("documents", "citations", "extract", "cognition", "index")
UPSTREAM = {"documents": (), "citations": ("documents",), "extract": ("documents", "citations"),
            "cognition": ("documents", "citations", "extract"),
            "index": ("documents", "citations", "extract", "cognition")}
ENTRY = {"documents": ("documents/build.py",),
         "citations": ("citations/build.py",), "extract": ("extract/build.py",),
         "cognition": ("cognition/build.py",), "index": ("index/build.py",)}
SRC = Path(__file__).resolve().parents[1]
_EXCLUDE = {SRC / "__init__.py", SRC / "dfc" / "store.py", SRC / "dfc" / "__init__.py"}
MAX_ATTEMPTS = 3
BREAKER = 50
BUSY_MS = 30000


class StageLocked(RuntimeError):
    pass


class StageAborted(RuntimeError):
    pass


def root() -> Path:
    return paths.derived()


def db_path(stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(stage)
    return root() / f"{stage}.sqlite"


def _check_local(p: Path) -> None:
    if str(p).startswith(("\\\\", "//")):
        raise RuntimeError(f"derived store must be on a local disk (SQLite WAL), got {p}")


class ReadConn:
    """A read-only connection that is safe to share across threads: each thread gets its own sqlite3 connection on first
    use (one sqlite3 connection used by several threads at once returns wrong rows and raises — measured 10-06:
    16 threads x 3,000 point reads on one shared connection gave 2,659 spurious misses and 1,758 errors). Exposes the
    sqlite3.Connection read API used here: execute, executemany-free, close."""

    def __init__(self, uri: str):
        self.uri = uri
        self._tl = threading.local()
        self._all: list[sqlite3.Connection] = []
        self._lock = threading.Lock()

    def _con(self) -> sqlite3.Connection:
        c = getattr(self._tl, "con", None)
        if c is None:
            c = sqlite3.connect(self.uri, uri=True, check_same_thread=False)
            c.execute(f"PRAGMA busy_timeout={BUSY_MS}")
            self._tl.con = c
            with self._lock:
                self._all.append(c)
        return c

    def execute(self, *a, **k):
        return self._con().execute(*a, **k)

    def close(self) -> None:
        with self._lock:
            for c in self._all:
                c.close()
            self._all.clear()
        self._tl = threading.local()


def read_only(p: Path) -> ReadConn:
    return ReadConn(f"file:{p}?mode=ro")


def connect(stage: str, readonly: bool = False, path: Path | None = None):
    p = path or db_path(stage)
    if readonly:
        return read_only(p)
    else:
        _check_local(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        # writers are shared by worker threads that serialize writes with their own lock
        con = sqlite3.connect(p, check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
    con.execute(f"PRAGMA busy_timeout={BUSY_MS}")
    return con


# ---- code identity: the import closure of a module set inside the package

def _module_files(base: Path) -> list[Path]:
    return [c for c in (base.with_suffix(".py"), base / "__init__.py") if c.is_file()]


_IMPORTS: dict[tuple, list[Path]] = {}


def _imports(f: Path) -> list[Path]:
    st = f.stat()
    k = (str(f), st.st_mtime_ns, st.st_size)
    if k in _IMPORTS:
        return _IMPORTS[k]
    out: list[Path] = []
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if not (node.module or "").startswith("compilescholar"):
                    continue
                base = SRC.parent / Path(*node.module.split("."))
            else:
                base = f.parent
                for _ in range(node.level - 1):
                    base = base.parent
                if node.module:
                    base = base / Path(*node.module.split("."))
            out += _module_files(base)
            for a in node.names:
                out += _module_files(base / a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("compilescholar"):
                    out += _module_files(SRC.parent / Path(*a.name.split(".")))
    _IMPORTS[k] = out
    return out


def closure(entries: tuple[str, ...]) -> list[Path]:
    """Every package source file reachable by imports (any depth, including imports inside functions) from the entry
    files, plus the __init__.py of each package on the way. Paths relative to SRC (e.g. 'llm/client.py')."""
    seen: set[Path] = set()
    todo = [SRC / e for e in entries]
    while todo:
        f = todo.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        d = f.parent
        while d != SRC and SRC in d.parents:
            todo += _module_files(d)
            d = d.parent
        todo += _imports(f)
    return sorted(seen - _EXCLUDE)


def _file_sha(f: Path) -> str:
    return hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def code_hashes(entries: tuple[str, ...]) -> dict[str, str]:
    return {str(f.relative_to(SRC)).replace("\\", "/"): _file_sha(f) for f in closure(entries)}


def code_digest(*entries: str) -> str:
    """One hash for the import closure of `entries` (used by stages to build per-pass digests)."""
    return _sha(code_hashes(tuple(entries)))


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(*parts) -> str:
    """Stable digest of JSON-serialisable parts (for pass digests and item fingerprints)."""
    return _sha(list(parts))


def item_keys(stage: str, pass_name: str) -> dict[str, str]:
    """{item: key} of the ok items of an upstream stage's pass (a downstream item fingerprint)."""
    p = db_path(stage)
    if not p.exists():
        return {}
    con = connect(stage, readonly=True)
    try:
        return dict(con.execute("SELECT item, key FROM _work WHERE pass=? AND status='ok'", (pass_name,)).fetchall())
    except sqlite3.OperationalError:
        return {}
    finally:
        con.close()


def parallel(fn, items, workers: int, log=None, every: int = 0, label: str = "") -> None:
    """Run fn over items on a thread pool; the first exception cancels everything not yet started and is re-raised
    (a breaker trip must stop the build, not drain the queue)."""
    import concurrent.futures as cf
    items = list(items)
    ex = cf.ThreadPoolExecutor(max(1, workers))
    futs = [ex.submit(fn, it) for it in items]
    try:
        for i, f in enumerate(cf.as_completed(futs)):
            f.result()
            if log and every and (i + 1) % every == 0:
                log(f"[{label}] {i + 1}/{len(items)}")
    except BaseException:
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    ex.shutdown(wait=True)


# ---- manifests (atomic)

def _manifest_path(stage: str) -> Path:
    return root() / "manifests" / f"{stage}.json"


def read_manifest(stage: str) -> dict | None:
    p = _manifest_path(stage)
    for attempt in range(3):
        if not p.exists():
            return None
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, PermissionError):
            if attempt == 2:
                raise
            time.sleep(0.05)
    return None


def _write_json_atomic(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    for attempt in range(20):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:     # Windows: a reader has the target open for a moment
            time.sleep(0.05)
    os.replace(tmp, p)


def _upstream_state(stage: str) -> dict:
    out = {}
    for u in UPSTREAM[stage]:
        m = read_manifest(u)
        if m is None:
            raise RuntimeError(f"stage {stage!r} needs {u!r}, which has not been built")
        out[u] = {"digest": m["digest"], "fingerprint": m.get("fingerprint", m["digest"])}
    return out


def config_digest(stage: str, params: dict) -> str:
    up = {u: s["digest"] for u, s in _upstream_state(stage).items()}
    return _sha({"stage": stage, "params": params, "upstream": up, "code": code_hashes(ENTRY[stage])})


def write_manifest(stage: str, params: dict, counts: dict, fingerprint: str | None = None,
                   work: dict | None = None) -> dict:
    """Low-level: record a finished build (Run.finish calls this; tests use it to stand up upstream stages)."""
    up = _upstream_state(stage)
    code = code_hashes(ENTRY[stage])
    digest = _sha({"stage": stage, "params": params, "upstream": {u: s["digest"] for u, s in up.items()}, "code": code})
    work = work or {}
    m = {"stage": stage, "params": params, "upstream": up, "code": code, "counts": counts, "work": work,
         "complete": not (work.get("failed") or work.get("gave_up")),
         "digest": digest, "fingerprint": fingerprint or _sha({"digest": digest, "counts": counts}),
         "build_id": f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}",
         "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _write_json_atomic(_manifest_path(stage), m)
    return m


def status(stage: str, _memo: dict | None = None) -> dict:
    """{'built', 'stale', 'behind', 'complete', 'why'}.
    stale  = own code changed, an upstream stage reconfigured, or an upstream stage is stale (config level);
    behind = an upstream stage has new data since this build, or is itself behind (data level);
    complete = no failed / given-up items in the last build."""
    memo = {} if _memo is None else _memo
    if stage in memo:
        return memo[stage]
    m = read_manifest(stage)
    if m is None:
        memo[stage] = {"built": False, "stale": True, "behind": False, "complete": False, "why": ["not built"]}
        return memo[stage]
    why, stale, behind = [], False, False
    for u in UPSTREAM[stage]:
        um = read_manifest(u)
        rec = (m.get("upstream") or {}).get(u) or {}
        if isinstance(rec, str):            # manifest written before data fingerprints existed
            rec = {"digest": rec, "fingerprint": None}
        if um is None:
            why.append(f"upstream {u} missing")
            stale = True
            continue
        if um["digest"] != rec.get("digest"):
            why.append(f"upstream {u} reconfigured")
            stale = True
        elif um.get("fingerprint", um["digest"]) != rec.get("fingerprint"):
            why.append(f"upstream {u} has new data")
            behind = True
        us = status(u, memo)
        if us["stale"]:
            why.append(f"upstream {u} stale")
            stale = True
        elif us["behind"]:
            why.append(f"upstream {u} behind")
            behind = True
    code = code_hashes(ENTRY[stage])
    changed = sorted(k for k in set(code) | set(m["code"]) if code.get(k) != m["code"].get(k))
    if changed:
        why.append("code changed: " + ", ".join(changed))
        stale = True
    complete = bool(m.get("complete", True))
    if not complete:
        why.append(f"incomplete: {m.get('work')}")
    memo[stage] = {"built": True, "stale": stale, "behind": behind, "complete": complete, "why": why}
    return memo[stage]


def require_fresh(*stages: str) -> None:
    """Raise unless every given stage is built, not stale and not behind (incomplete stages are usable; their
    failure counts are in the manifest and in status())."""
    memo: dict = {}
    bad = {}
    for s in stages:
        st = status(s, memo)
        if not st["built"] or st["stale"] or st["behind"]:
            bad[s] = st["why"]
    if bad:
        raise RuntimeError(f"upstream not fresh: {bad}")


# ---- item-level work

WORK_DDL = """CREATE TABLE IF NOT EXISTS _work(pass TEXT, item TEXT, key TEXT, status TEXT, attempts INT,
  last_error TEXT, updated_at TEXT, PRIMARY KEY(pass, item));"""


class Work:
    """Bookkeeping for one pass of one stage. Thread-safe; writes go through the run's connection."""

    def __init__(self, run: "Run", name: str, digest: str, breaker: int = BREAKER):
        self.run, self.name, self.digest, self.breaker = run, name, digest, breaker
        self.con = run.con
        self.lock = run.lock
        self.consecutive = 0
        self._keys: dict[str, str] = {}

    def key(self, item: str) -> str:
        return self._keys.get(item) or self.digest

    def todo(self, items) -> list[str]:
        """items: iterable of str or (str, input_fingerprint). Returns those not ok under the current key and not given
        up (a given-up item comes back when its key changes). May be called repeatedly (keys accumulate)."""
        pairs = [(i, None) if isinstance(i, str) else (i[0], i[1]) for i in items]
        self._keys.update({i: (_sha([self.digest, fp]) if fp is not None else self.digest) for i, fp in pairs})
        have = {}
        for item, key, st, att in self.con.execute("SELECT item, key, status, attempts FROM _work WHERE pass=?",
                                                     (self.name,)):
            have[item] = (key, st, att)
        out = []
        for i, _ in pairs:
            k = self._keys[i]
            h = have.get(i)
            if h is None or h[0] != k:
                out.append(i)
            elif h[1] == "ok" or h[1] == "gave_up":
                continue
            else:
                out.append(i)
        return out

    def ok(self, item: str) -> None:
        self.ok_many([item])

    def ok_many(self, items) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.lock:
            self.con.executemany("INSERT OR REPLACE INTO _work VALUES (?,?,?,?,?,?,?)",
                                 [(self.name, i, self.key(i), "ok", 0, None, now) for i in items])
            self.consecutive = 0

    def fail(self, item: str, err: str) -> None:
        self.fail_many([item], err)

    def fail_many(self, items, err: str) -> None:
        """Record a failure (one breaker tick per call: a failed batch is one failure)."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.lock:
            for i in items:
                r = self.con.execute("SELECT key, attempts FROM _work WHERE pass=? AND item=?", (self.name, i)).fetchone()
                att = (r[1] if r and r[0] == self.key(i) else 0) + 1
                st = "gave_up" if att >= MAX_ATTEMPTS else "failed"
                self.con.execute("INSERT OR REPLACE INTO _work VALUES (?,?,?,?,?,?,?)",
                                 (self.name, i, self.key(i), st, att, str(err)[:500], now))
            self.consecutive += 1
            if self.consecutive >= self.breaker:
                self.con.commit()
                raise StageAborted(f"{self.run.stage}/{self.name}: {self.consecutive} consecutive failures, last: {err}")

    def sweep(self, current, delete) -> int:
        """Delete the outputs (via `delete(item)`) and the work rows of items no longer in `current`."""
        cur = set(current)
        gone = [r[0] for r in self.con.execute("SELECT item FROM _work WHERE pass=?", (self.name,)) if r[0] not in cur]
        with self.lock:
            for i in gone:
                delete(i)
                self.con.execute("DELETE FROM _work WHERE pass=? AND item=?", (self.name, i))
        return len(gone)

    def summary(self) -> dict:
        rows = self.con.execute("SELECT status, count(*) FROM _work WHERE pass=? GROUP BY status", (self.name,))
        return {k: v for k, v in rows}


class Run:
    """One build of one stage: holds the stage lock, checks upstream freshness, owns the write connection.
    rebuild=True writes a new file and swaps it in on finish (clean slate); otherwise the build is incremental."""

    def __init__(self, stage: str, params: dict, rebuild: bool = False):
        if stage not in STAGES:
            raise ValueError(stage)
        self.stage, self.params, self.rebuild = stage, params, rebuild
        self.digest = None
        self.con = None
        self.passes: list[Work] = []
        self.lock = threading.RLock()        # guards every write on `con` (stage code takes it around its own writes)
        self._lock = FileLock(str(root() / "locks" / f"{stage}.lock"))
        self._finished = False

    def __enter__(self) -> "Run":
        (root() / "locks").mkdir(parents=True, exist_ok=True)
        try:
            self._lock.acquire(timeout=0)
        except Timeout:
            raise StageLocked(f"stage {self.stage!r} is being built by another process") from None
        try:
            require_fresh(*UPSTREAM[self.stage])
            self.upstream = _upstream_state(self.stage)
            self.digest = config_digest(self.stage, self.params)
            self.path = db_path(self.stage)
            if self.rebuild:
                self.path = self.path.with_name(self.path.name + ".new")
                for suf in ("", "-wal", "-shm"):
                    Path(str(self.path) + suf).unlink(missing_ok=True)
            self.con = connect(self.stage, path=self.path)
            tables = {r[0] for r in self.con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables and "_work" not in tables:
                raise RuntimeError(f"{self.path} was built before item bookkeeping existed; build it with --rebuild")
            self.con.executescript(WORK_DDL)
        except BaseException:
            if self.con is not None:
                self.con.close()
                self.con = None
            self._lock.release()
            raise
        return self

    def fingerprint(self, name: str) -> str:
        """Data fingerprint of an upstream stage (for per-item keys that must change when upstream data grows)."""
        return self.upstream[name]["fingerprint"]

    def work(self, name: str, digest: str | None = None, breaker: int = BREAKER) -> Work:
        w = Work(self, name, digest or self.digest, breaker)
        self.passes.append(w)
        return w

    def finish(self, counts: dict, fingerprint: str | None = None) -> dict:
        self.con.commit()
        work = {}
        for w in self.passes:
            for k, v in w.summary().items():
                work[k] = work.get(k, 0) + v
        if fingerprint is None and self.passes:
            rows = self.con.execute("SELECT pass, item, key FROM _work WHERE status='ok' ORDER BY pass, item")
            h = hashlib.sha256()
            for r in rows:
                h.update("\x1f".join(r).encode())
                h.update(b"\x1e")
            fingerprint = h.hexdigest()
        if self.rebuild:
            self.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.con.close()
            final = db_path(self.stage)
            for suf in ("-wal", "-shm"):
                Path(str(final) + suf).unlink(missing_ok=True)
            os.replace(self.path, final)
            self.con = None
        m = write_manifest(self.stage, self.params, counts, fingerprint=fingerprint, work=work)
        self._finished = True
        return m

    def __exit__(self, et, ev, tb):
        try:
            if self.con is not None:
                if et is None:
                    self.con.commit()
                self.con.close()
            if self.rebuild and not self._finished:
                for suf in ("", "-wal", "-shm"):
                    Path(str(self.path) + suf).unlink(missing_ok=True)
        finally:
            self._lock.release()
        return False
