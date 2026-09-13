"""Embedding infra (Stage B).

Consolidates three proven implementations:
- granular_agent.llm_client.embed_batch      (Paratera GLM-Embedding-2, dim 1024,
                                              server-side per-item ~3s — slow tier)
- granular_agent.hypergraph_evolution._embed_texts_robust
                                              (two-tier fallback, honest None)
- stageA ext_bench paperscope_run.embed_cst  (CST qwen3-embedding:8b TRUE batch,
                                              dim 4096, measured 64 texts / ~10s)

DISCIPLINE (measured failure, silent): chunks and queries MUST be embedded by
the SAME model / SAME dim (GLM 1024 vs CST 4096 — mixing corrupts cosine with
no error raised). Therefore embed_texts_robust here returns (embs, provider)
instead of bare embs: callers must store the provider tag alongside any
persisted embedding and refuse to compare across tags.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .llm import ENV, _CTX, _walled_open, embed_batch  # noqa: F401  (embed_batch re-exported)

CST_EMBED_MODEL = "qwen3-embedding:8b"
CST_EMBED_DIM = 4096
GLM_EMBED_DIM = 1024
CST_EMBED_BATCH = 64  # measured true-batch size (64 texts / ~10s)


def embed_cst(texts: list[str], batch_size: int = CST_EMBED_BATCH,
              workers: int = 4) -> list[list[float]]:
    """CST qwen3-embedding:8b, true batched (dim 4096). Raises RuntimeError
    if any batch still fails after retries — caller decides degradation."""
    key = ENV.get("CST_API_KEY", "")
    base = (ENV.get("CST_BASE_URL", "") or "").rstrip("/")
    if not key or not base:
        raise RuntimeError("CST embedding not configured")
    if not texts:
        return []
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    wall = float(os.environ.get("LLM_WALL_TIMEOUT", "240"))

    def _one(batch: list[str]) -> list[list[float]]:
        body = json.dumps({"model": CST_EMBED_MODEL, "input": batch}).encode()
        last_err = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    base + "/embeddings", data=body,
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"})
                # wall-clock watchdog: same slow-drip hole as chat endpoints
                # (2026-08-30 third blind spot was exactly a CST embed tier)
                r = _walled_open(req, sock_timeout=60, wall_s=wall)
                t0 = time.time()
                chunks = []
                while True:
                    if time.time() - t0 > wall:
                        raise TimeoutError(f"cst-embed wall {wall}s exceeded")
                    b = r.read(65536)
                    if not b:
                        break
                    chunks.append(b)
                data = json.loads(b"".join(chunks)).get("data", [])
                data.sort(key=lambda x: x.get("index", 0))
                out = [d["embedding"] for d in data]
                if len(out) == len(batch):
                    return out
                last_err = RuntimeError(f"count mismatch {len(out)}/{len(batch)}")
            except Exception as e:
                last_err = e
                time.sleep(6 * (attempt + 1))
        raise RuntimeError(f"CST embed batch failed: {last_err}")

    if workers <= 1 or len(batches) == 1:
        results = [_one(b) for b in batches]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as ex:
            results = list(ex.map(_one, batches))
    return [e for r in results for e in r]


def embed_texts_robust(texts: list[str]) -> tuple[list[list[float]] | None, str | None]:
    """Two-tier embedding with provider fallback (ported from the frozen
    stack's _embed_texts_robust, provider tag added — see module docstring).

    tier 1: Paratera GLM-Embedding-2 (dim 1024; per-item server-side, slow)
    tier 2: CST qwen3-embedding:8b  (dim 4096; true batch, fast)

    Returns (embs, provider_tag) where provider_tag ∈ {"paratera-glm",
    "cst-qwen3"}; (None, None) only if BOTH tiers fail — caller then
    degrades honestly and must report the degradation.
    """
    if not texts:
        return [], None
    try:
        embs = embed_batch(texts)
        if embs and len(embs) == len(texts):
            return embs, "paratera-glm"
    except Exception:
        pass  # 429 / SSL / wall-timeout -> fall through to CST
    try:
        embs = embed_cst(texts)
        if embs and len(embs) == len(texts):
            return embs, "cst-qwen3"
    except Exception:
        pass
    return None, None
