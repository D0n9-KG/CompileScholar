# -*- coding: utf-8 -*-
"""Local OpenAI-compatible /embeddings (moved from kb_infra.embedding.embed_local; same batching, retries, wall and
ledger). Raises RuntimeError on failure — callers decide degradation; never mixes providers silently (chunks and
queries must be embedded by the same model)."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ..core import secrets
from .client import _log_call, walled_open

LOCAL_EMBED_MODEL_DEFAULT = "qwen3-embedding-8b-local"


def embed_local(texts: list[str], batch_size: int = 64, workers: int = 2) -> list[list[float]]:
    base = (secrets.get("LOCAL_BASE_URL") or "").rstrip("/")
    if not base:
        raise RuntimeError("LOCAL_BASE_URL not configured")
    key = secrets.get("LOCAL_API_KEY", "local")
    model = secrets.get("EMBEDDING_MODEL") or LOCAL_EMBED_MODEL_DEFAULT
    if not texts:
        return []
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    wall = float(os.environ.get("LLM_WALL_TIMEOUT", "240"))

    def _one(batch: list[str]) -> list[list[float]]:
        body = json.dumps({"model": model, "input": batch}).encode()
        t_start = time.time()
        last_err = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(base + "/embeddings", data=body,
                                             headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                r = walled_open(req, sock_timeout=60, wall_s=wall)
                t0 = time.time()
                chunks = []
                while True:
                    if time.time() - t0 > wall:
                        raise TimeoutError(f"local-embed wall {wall}s exceeded")
                    b = r.read(65536)
                    if not b:
                        break
                    chunks.append(b)
                data = json.loads(b"".join(chunks)).get("data", [])
                data.sort(key=lambda x: x.get("index", 0))
                out = [d["embedding"] for d in data]
                if len(out) == len(batch):
                    _log_call("local-embed", model, True, (time.time() - t_start) * 1000, None, attempt,
                              payload={"model": model, "input": batch}, extra={"n_items": len(batch)})
                    return out
                last_err = RuntimeError(f"count mismatch {len(out)}/{len(batch)}")
            except Exception as e:
                last_err = e
                time.sleep(3 * (attempt + 1))
        _log_call("local-embed", model, False, (time.time() - t_start) * 1000, None, 3,
                  payload={"model": model, "input": batch}, extra={"n_items": len(batch)})
        raise RuntimeError(f"local embed batch failed: {last_err}")

    if workers <= 1 or len(batches) == 1:
        results = [_one(b) for b in batches]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as ex:
            results = list(ex.map(_one, batches))
    return [e for r in results for e in r]
