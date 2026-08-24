"""LLM client: wraps DeepSeek + Paratera APIs for extraction and embedding."""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.request
from typing import Any

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def load_env(path: str = "C:/Users/D0n9/Desktop/LogicKG/.env") -> dict:
    env = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


ENV = load_env()


def _chat_once(url, key, body, timeout):
    """Single HTTP POST to an OpenAI-compat chat endpoint. Raises on any failure."""
    req = urllib.request.Request(
        url, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    raw = urllib.request.urlopen(req, context=_CTX, timeout=timeout).read()
    return json.loads(raw)["choices"][0]["message"]["content"]


def call_llm(prompt: str, model: str = "deepseek-chat", max_tokens: int = 4000,
             temperature: float = 0.0) -> str | None:
    """Call DeepSeek chat API with exponential-backoff retry + Paratera fallback.

    deepseek intermittently hangs at socket level (TCP connected, server never
    replies) — a short per-call timeout + 3 backoff retries on the primary,
    then one shot at the Paratera fallback model (GLM-5-Turbo) so a single
    hung section can't stall the whole extraction."""
    key = ENV.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("no DEEPSEEK_API_KEY")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    url = "https://api.deepseek.com/v1/chat/completions"
    last_err = None
    # primary: 3 attempts with exponential backoff (2s, 6s)
    for attempt in range(3):
        try:
            return _chat_once(url, key, body, timeout=120)
        except Exception as e:
            last_err = e
            if attempt < 2:
                import time
                time.sleep(2 * (attempt + 1))
    # fallback: Paratera GLM-5-Turbo (same prompt, one shot)
    fb = call_paratera(prompt, model="GLM-5-Turbo", max_tokens=max_tokens,
                       temperature=temperature)
    if fb is not None:
        return fb
    # both failed
    print(f"  [llm] all retries + fallback failed: {last_err}", flush=True)
    return None


def call_paratera(prompt: str, model: str = "Kimi-K2.6", max_tokens: int = 4000,
                  temperature: float = 0.0, enable_thinking: bool = None) -> str | None:
    """Call Paratera API (Kimi/GLM/Qwen/DeepSeek).

    enable_thinking: for reasoning models (DeepSeek-V4-Flash etc), set False
    to suppress reasoning_content (which eats max_tokens budget + slows). None
    = don't send the param (model default)."""
    key = ENV.get("PARATERA_API_KEY")
    base = ENV.get("PARATERA_BASE_URL", "").rstrip("/")
    if not key:
        return None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(2):
        try:
            raw = urllib.request.urlopen(req, context=_CTX, timeout=300).read()
            return json.loads(raw)["choices"][0]["message"]["content"]
        except Exception:
            if attempt == 1:
                return None
    return None


def embed_batch(texts: list[str], model: str = "GLM-Embedding-2") -> list[list[float]]:
    """Call Paratera embedding API. Returns one embedding per input text; texts
    that 400 (empty/oversized/odd chars) get a zero vector so callers keep
    index alignment rather than crashing the whole batch."""
    key = ENV.get("PARATERA_API_KEY")
    base = ENV.get("PARATERA_BASE_URL", "").rstrip("/")
    if not key or not texts:
        return []
    dim = 1024  # GLM-Embedding-2 dim; used for zero-vector fallback
    out: list[list[float]] = [[0.0] * dim for _ in texts]
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        body = json.dumps({"model": model, "input": chunk}).encode()
        req = urllib.request.Request(
            base + "/embeddings",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            raw = urllib.request.urlopen(req, context=_CTX, timeout=60).read()
            data = json.loads(raw).get("data", [])
            data.sort(key=lambda x: x.get("index", 0))
            for d in data:
                out[i + d.get("index", 0)] = d["embedding"]
        except Exception as e:
            # whole batch failed (likely one bad text); retry each individually
            for j, t in enumerate(chunk):
                if any(out[i + j]):
                    continue
                b = json.dumps({"model": model, "input": [t[:8000]]}).encode()
                rq = urllib.request.Request(
                    base + "/embeddings", data=b,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                try:
                    r = urllib.request.urlopen(rq, context=_CTX, timeout=30).read()
                    d = json.loads(r).get("data", [])
                    if d:
                        out[i + j] = d[0]["embedding"]
                except Exception:
                    pass  # leave zero vector
    return out


def cosine_sim(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---- CST (CSTCloud) provider — OpenAI-compatible API ----
_CST_MODELS = {"qwen3.5", "gpt-oss-120b", "deepseek-v4-flash", "minimax-m27",
               "S1-Base-Lite", "S1-Base-Pro", "S1-Base-Ultra"}


def _is_cst_model(model: str) -> bool:
    return model.lower() in _CST_MODELS


def call_cst(prompt: str, model: str = "qwen3.5", max_tokens: int = 4000,
             temperature: float = 0.0) -> str | None:
    """Call CSTCloud API (qwen3.5 / gpt-oss-120b / deepseek-v4-flash etc).
    OpenAI-compatible. enable_thinking not sent (CST models ignore it)."""
    key = ENV.get("CST_API_KEY")
    base = ENV.get("CST_BASE_URL", "").rstrip("/")
    if not key:
        return None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(2):
        try:
            raw = urllib.request.urlopen(req, context=_CTX, timeout=120).read()
            return json.loads(raw)["choices"][0]["message"]["content"]
        except Exception:
            if attempt == 1:
                return None
    return None


def parse_json_response(text: str | None) -> Any:
    """Parse JSON from LLM response, handling markdown fences and extra text."""
    if not text:
        return None
    text = re.sub(r"```json|```", "", text, flags=re.S)
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
