"""Embedding rerank + two-level cache (P4-A A4, 2026-09-24).

The gap this closes (RETRIEVAL-MODULE-DESIGN.md §一.4/二): discovery
candidates arrive ordered by each source's own relevance heuristic, and
title-string similarity is the only local signal — a question phrased in
different vocabulary than the paper title ranks poorly. This layer:

  - reconstructs per-candidate text (title + abstract when the source
    payload carries one — OpenAlex abstract_inverted_index, arXiv Atom
    summary, S2 abstract field)
  - embeds query + candidate texts with ONE model (env-driven local
    endpoint; caller may inject CompileScholar's kb_infra embed_local for
    ledger/semaphore integration) and reorders by cosine
  - caches: query->result LRU (in-process, bounded) and DOI->candidate
    (file-backed, cross-process) so repeat/related queries skip work

Mixed-dim cosine silently corrupts — query and candidates MUST embed
through the same provider/model (same discipline as CompileScholar
kb_infra).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from retrieval.sources import SourceCandidate


def candidate_text(c: SourceCandidate) -> str:
    """Title + reconstructable abstract, used as the embedding input."""
    parts = [c.title or ""]
    abstract = ""
    raw = c.raw or {}
    inv = raw.get("abstract_inverted_index")
    if isinstance(inv, dict) and inv:
        # OpenAlex: word -> [positions]; rebuild in position order
        slots: dict[int, str] = {}
        for word, positions in inv.items():
            for p in positions if isinstance(positions, list) else []:
                slots[int(p)] = str(word)
        abstract = " ".join(slots[k] for k in sorted(slots))
    elif isinstance(raw.get("summary"), str):
        abstract = raw["summary"]           # arXiv Atom summary
    elif isinstance(raw.get("abstract"), str) and raw["abstract"]:
        abstract = raw["abstract"]          # S2 abstract field
    if abstract:
        parts.append(abstract[:1500])
    if c.venue:
        parts.append(str(c.venue))
    return " ".join(p for p in parts if p).strip()


def _default_embed(texts: list[str]) -> list[list[float]]:
    """OpenAI-compatible /embeddings against the local GPUStack box."""
    import urllib.request

    base = (os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")
            .rstrip("/"))
    key = os.environ.get("LOCAL_API_KEY", "local")
    model = (os.environ.get("LOCAL_EMBED_MODEL")
             or os.environ.get("EMBEDDING_MODEL", "qwen3-embedding-8b-local"))
    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        base + "/embeddings", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())
    data = sorted(payload.get("data", []), key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


@dataclass(frozen=True)
class RerankResult:
    candidates: list[SourceCandidate]
    scores: list[float]
    embedded: bool   # False = passthrough on embed failure (original order)


def rerank_candidates(
    query: str,
    candidates: list[SourceCandidate],
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> RerankResult:
    """Reorder discovery candidates by cosine(query, candidate_text).
    Deterministic passthrough on any embedding failure — rerank is an
    enhancement, never a blocker."""
    if not candidates:
        return RerankResult([], [], embedded=False)
    fn = embed or _default_embed
    try:
        texts = [query] + [candidate_text(c) for c in candidates]
        vecs = fn(texts)
        if len(vecs) != len(texts):
            raise ValueError(f"embed returned {len(vecs)}/{len(texts)}")
        qv = vecs[0]
        scored = sorted(
            zip(candidates, [_cosine(qv, v) for v in vecs[1:]]),
            key=lambda kv: -kv[1],
        )
        return RerankResult([c for c, _ in scored], [s for _, s in scored],
                            embedded=True)
    except Exception:
        return RerankResult(candidates, [0.0] * len(candidates),
                            embedded=False)


class QueryResultCache:
    """Bounded in-process LRU: query string -> cached value (the caller
    stores whatever the fast path needs — e.g. the ranked candidate ids)."""

    def __init__(self, max_entries: int = 256, ttl_s: float = 3600.0):
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            self.misses += 1
            return None
        ts, value = entry
        if time.time() - ts > self.ttl_s:
            del self._data[key]
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        self._data[key] = (time.time(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "entries": len(self._data)}


class DoiMetaCache:
    """File-backed DOI -> candidate memo (JSON lines; cross-process).
    Discovery hits on a DOI we've already seen skip re-fetching metadata
    in the fusion layer. Best-effort: IO errors never propagate."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mem: dict[str, dict] = {}
        try:
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._mem[rec["doi"]] = rec["candidate"]
        except Exception:
            self._mem = {}
        self._dirty = False

    def get(self, doi: str | None) -> dict | None:
        if not doi:
            return None
        return self._mem.get(doi)

    def put(self, c: SourceCandidate) -> None:
        if not c.normalized_doi:
            return
        row = {
            "source_name": c.source_name, "status": c.status,
            "normalized_doi": c.normalized_doi, "title": c.title,
            "year": c.year, "venue": c.venue, "source_url": c.source_url,
            "open_access_status": c.open_access_status,
        }
        if self._mem.get(c.normalized_doi) != row:
            self._mem[c.normalized_doi] = row
            self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for doi, cand in self._mem.items():
                    f.write(json.dumps({"doi": doi, "candidate": cand},
                                       ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)
            self._dirty = False
        except Exception:
            pass  # cache write failures are never fatal
