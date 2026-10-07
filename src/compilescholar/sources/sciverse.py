# -*- coding: utf-8 -*-
"""Sciverse semantic search (/agentic-search) — the external retrieval channel shared by our pipeline and the
harness MCP server. Moved from retrieval.sources (SciverseClient.semantic_search, sciverse_request_json, the
cross-process token bucket); request payloads, retries and pacing are unchanged.

Changes from the original:
  - the knowledge cutoff comes from compilescholar.core.cutoff directly (the original looked up a module named
    "cutoff" in sys.modules, i.e. it depended on the caller having imported the experiment-folder module first);
  - the token comes from compilescholar.core.secrets.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..core import cutoff as _cutoff
from ..core import secrets

USER_AGENT = "sci-evo-extract/0.1"


class SourceAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceCandidate:
    source_name: str
    query_kind: str
    status: str
    source_record_id: str | None = None
    normalized_doi: str | None = None
    title: str | None = None
    year: int | None = None
    venue: str | None = None
    candidate_score: float | None = None
    pdf_candidates: list[dict[str, Any]] = field(default_factory=list)
    license: str | None = None
    open_access_status: str | None = None
    source_url: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    error_summary: str | None = None


def _safe_int(value: object) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _semantic_candidate(payload: dict[str, Any], score: float) -> SourceCandidate:
    return SourceCandidate(
        source_name="sciverse-semantic", query_kind="semantic", status="ready",
        source_record_id=str(payload.get("doc_id") or "") or None,
        title=str(payload.get("title")) if payload.get("title") else None,
        year=_safe_int(payload.get("publication_published_year")),
        venue=str(payload.get("publication_venue_name_unified")) if payload.get("publication_venue_name_unified") else None,
        candidate_score=score,
        raw={"doc_id": payload.get("doc_id"), "chunk_id": payload.get("chunk_id"), "chunk": payload.get("chunk"),
             "abstract": payload.get("abstract"), "citation_count": payload.get("citation_count"),
             "primary_topic": payload.get("primary_topic"), "endpoint": "/agentic-search"})


def _failed(source_name: str, query_kind: str, error_summary: str) -> SourceCandidate:
    return SourceCandidate(source_name=source_name, query_kind=query_kind, status="failed", error_summary=error_summary)


def _blocked(source_name: str, query_kind: str, error_summary: str) -> SourceCandidate:
    return SourceCandidate(source_name=source_name, query_kind=query_kind, status="blocked", error_summary=error_summary)


# ---------------------------------------------------------------- pacing (account limit 30 req/min)
class _TokenBucket:
    """In-process token bucket: waits for a token instead of firing and getting a 429."""

    def __init__(self, rate_per_min: float, capacity: int):
        self.rate = rate_per_min / 60.0
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, max_wait_s: float = 10.0) -> bool:
        deadline = time.monotonic() + max_wait_s
        while True:
            with self._lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
                need_s = (1.0 - self.tokens) / self.rate
            remaining = deadline - time.monotonic()
            if need_s >= remaining:
                return False
            time.sleep(min(need_s, remaining))


class _FileTokenBucket:
    """Cross-process token bucket (the 30 req/min limit is per account): state in a JSON file guarded by a filelock,
    shared by every process on the machine. Enabled by SCIVERSE_SHARED_BUCKET=<path>."""

    def __init__(self, path: str, rate_per_min: float, capacity: int):
        from filelock import FileLock
        self.path, self.rate, self.capacity = path, rate_per_min / 60.0, float(capacity)
        self._lock = FileLock(path + ".lock")

    def acquire(self, max_wait_s: float = 10.0) -> bool:
        deadline = time.time() + max_wait_s
        while True:
            with self._lock:
                try:
                    st = json.load(open(self.path, encoding="utf-8"))
                except Exception:
                    st = {"tokens": self.capacity, "last": time.time()}
                now = time.time()
                tokens = min(self.capacity, st["tokens"] + (now - st["last"]) * self.rate)
                if tokens >= 1.0:
                    json.dump({"tokens": tokens - 1.0, "last": now}, open(self.path, "w", encoding="utf-8"))
                    return True
                json.dump({"tokens": tokens, "last": now}, open(self.path, "w", encoding="utf-8"))
                need_s = (1.0 - tokens) / self.rate
            remaining = deadline - time.time()
            if need_s >= remaining:
                return False
            time.sleep(min(need_s, remaining) + 0.05)


_BUCKETS: dict = {}
_BUCKET_LOCK = threading.Lock()

# the account limit (30 req/min) is per key; with several keys the caller rotates them and each key keeps its
# own bucket (and its own shared-bucket state file, so cross-process pacing stays per account)
TOKEN_ENVS = ("SCIVERSE_API_TOKEN", "SCIVERSE_KRY_2", "SCIVERSE_KRY_3")


def all_tokens() -> list[str]:
    out = []
    for k in TOKEN_ENVS:
        v = secrets.get(k)
        if v and v not in out:
            out.append(v)
    return out


def _bucket(token: str):
    import hashlib
    key = hashlib.sha1(token.encode()).hexdigest()[:10]
    with _BUCKET_LOCK:
        b = _BUCKETS.get(key)
        if b is None:
            rate = float(os.environ.get("SCIVERSE_RATE_PER_MIN", "30"))
            shared = os.environ.get("SCIVERSE_SHARED_BUCKET")
            b = _BUCKETS[key] = (_FileTokenBucket(f"{shared}.{key}", rate, max(1, int(rate))) if shared
                                 else _TokenBucket(rate, max(1, int(rate))))
        return b


def request_json(method: str, path: str, *, payload: object | None = None, query: dict[str, object] | None = None,
                 timeout_seconds: int, token: str | None = None, base_url: str = "https://api.sciverse.space") -> object:
    token = token or secrets.get("SCIVERSE_API_TOKEN")
    if not token:
        raise SourceAdapterError("SCIVERSE_API_TOKEN missing")
    # Wait for a token before firing (SCIVERSE_MAX_WAIT_S; the answer path uses 600 — rate limiting is a queueing
    # problem, not a source failure).
    _bucket(token).acquire(max_wait_s=float(os.environ.get("SCIVERSE_MAX_WAIT_S", "10")))
    url = base_url.rstrip("/") + path
    if query:
        url += "?" + urlencode(query)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
    if body is not None:
        headers["Content-Type"] = "application/json"
    for attempt in range(5):
        req = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(req, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 429 and attempt < 4:
                time.sleep(3.0 * (attempt + 1))
                _bucket().acquire(max_wait_s=120.0)
                continue
            raise SourceAdapterError(f"HTTP {exc.code} {exc.reason} for {method} {url}") from exc
        except URLError as exc:
            raise SourceAdapterError(f"Network error for {method} {url}: {exc.reason}") from exc
        except Exception as exc:  # noqa: BLE001
            raise SourceAdapterError(str(exc)) from exc
    raise SourceAdapterError(f"exhausted 429 retries for {method} {url}")


class SciverseClient:
    def __init__(self, *, token: str | None = None, request_json_fn: Callable[..., object] | None = None,
                 base_url: str = "https://api.sciverse.space", timeout_seconds: int = 30):
        self.token = token if token is not None else secrets.get("SCIVERSE_API_TOKEN")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._request = request_json_fn or self._bound

    def _bound(self, method, path, *, payload=None, query=None, timeout_seconds=30):
        return request_json(method, path, payload=payload, query=query, timeout_seconds=timeout_seconds,
                            token=self.token, base_url=self.base_url)

    def semantic_search(self, query: str, *, limit: int = 10, year_lte: int | None = None) -> list[SourceCandidate]:
        """Semantic full-text search, one candidate per document (consecutive chunks of one paper collapse to the
        first hit). The cutoff is pushed to the server as publication_published_year <= year_lte. Default: the
        env cutoff's year minus one (year granularity cannot decide same-year visibility, so the cutoff year is
        excluded). A caller that re-checks every hit's exact date itself (the D④ search_external tool) may pass
        year_lte = the cutoff year. Never raises: errors become a failed candidate."""
        q = (query or "").strip()
        if not q:
            return []
        if not self.token:
            return [_blocked("sciverse-semantic", "semantic", "SCIVERSE_API_TOKEN missing")]
        payload: dict[str, Any] = {"query": q, "page_size": limit}
        if year_lte is None:
            cut = _cutoff.raw_cutoff()
            if cut[:4].isdigit():
                year_lte = int(cut[:4]) - 1
        if year_lte is not None:
            payload["filters"] = {"publication_published_year": {"lte": int(year_lte)}}
        try:
            response = self._request("POST", "/agentic-search", payload=payload, query=None,
                                     timeout_seconds=self.timeout_seconds)
            if not isinstance(response, dict):
                raise SourceAdapterError("Sciverse agentic-search response is not a JSON object")
            hits = response.get("hits") or []
            if not isinstance(hits, list):
                return []
            seen: set[str] = set()
            out: list[SourceCandidate] = []
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                doc = str(hit.get("doc_id") or "")
                if doc and doc in seen:
                    continue
                if doc:
                    seen.add(doc)
                score = hit.get("score")
                out.append(_semantic_candidate(hit, float(score) if isinstance(score, (int, float)) else 0.5))
                if len(out) >= limit:
                    break
            return out
        except SourceAdapterError as exc:
            return [_failed("sciverse-semantic", "semantic", str(exc))]

    def read_content(self, *, doc_id: str, chunk_id: str | None = None, offset: int | None = None,
                     limit: int | None = None) -> dict[str, Any]:
        if not self.token:
            raise SourceAdapterError("SCIVERSE_API_TOKEN missing")
        if not doc_id:
            raise SourceAdapterError("Sciverse content read needs doc_id")
        query: dict[str, object] = {"doc_id": doc_id}
        if chunk_id:
            query["chunk_id"] = chunk_id
        else:
            query["offset"] = offset or 0
            if limit is not None:
                query["limit"] = limit
        response = self._request("GET", "/content", payload=None, query=query, timeout_seconds=self.timeout_seconds)
        if not isinstance(response, dict):
            raise SourceAdapterError("Sciverse content response is not a JSON object")
        return response
