# -*- coding: utf-8 -*-
"""Local OpenAI-compatible /embeddings over the unified transport (llm.client: httpx, lanes, ledger, breaker).

Never falls back to another model (chunks and queries must be embedded by the same model); raises RuntimeError when a
batch cannot be embedded. `embed()` returns the vectors with the model label and dimension so stores can record them;
`embed_local()` keeps the old list-of-vectors signature for existing callers."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from ..core import secrets
from . import client as C

LOCAL_EMBED_MODEL_DEFAULT = "qwen3-embedding-8b-local"


def model_label() -> str:
    return secrets.get("EMBEDDING_MODEL") or LOCAL_EMBED_MODEL_DEFAULT


def embed(texts: list[str], batch_size: int = 64, workers: int = 8) -> tuple[list[list[float]], str, int]:
    """(vectors, model_label, dim). Same model for every batch; a dimension mismatch between batches raises."""
    p = C.settings().providers["local"]
    base = (secrets.get("LOCAL_BASE_URL") or "").rstrip("/")
    if not base:
        raise RuntimeError("LOCAL_BASE_URL not configured")
    key = secrets.get("LOCAL_API_KEY", "local")
    model = model_label()
    if not texts:
        return [], model, 0
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    lane = C._lane(p, False)

    def _one(batch: list[str]) -> list[list[float]]:
        last = None
        for attempt in range(p.attempts):
            t0 = time.monotonic()
            try:
                with lane:
                    r = C._client(p).post(base + "/embeddings", json={"model": model, "input": batch},
                                          headers={"Authorization": f"Bearer {key}"}, timeout=min(p.wall, 300))
                if r.status_code != 200:
                    raise RuntimeError(f"http {r.status_code}: {r.text[:200]}")
                data = sorted(r.json().get("data", []), key=lambda x: x.get("index", 0))
                out = [d["embedding"] for d in data]
                if len(out) != len(batch):
                    raise RuntimeError(f"count mismatch {len(out)}/{len(batch)}")
                C._log({"provider": "local-embed", "model": model, "ok": True, "attempt": attempt, "n_items": len(batch),
                        "latency_ms": round((time.monotonic() - t0) * 1000)})
                return out
            except (httpx.HTTPError, RuntimeError, ValueError) as e:
                last = e
                C._log({"provider": "local-embed", "model": model, "ok": False, "attempt": attempt,
                        "n_items": len(batch), "error": str(e)[:300],
                        "latency_ms": round((time.monotonic() - t0) * 1000)})
                if attempt < p.attempts - 1:
                    C._backoff(attempt, None)
        raise RuntimeError(f"local embed batch failed: {last}")

    if workers <= 1 or len(batches) == 1:
        results = [_one(b) for b in batches]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as ex:
            results = list(ex.map(_one, batches))
    vecs = [e for r in results for e in r]
    dims = {len(v) for v in vecs}
    if len(dims) != 1:
        raise RuntimeError(f"embedding dimensions differ across batches: {sorted(dims)}")
    return vecs, model, dims.pop()


def embed_local(texts: list[str], batch_size: int = 64, workers: int = 8) -> list[list[float]]:
    return embed(texts, batch_size=batch_size, workers=workers)[0]
