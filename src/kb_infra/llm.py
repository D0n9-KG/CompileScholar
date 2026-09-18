"""LLM client: wraps DeepSeek + Paratera APIs for extraction and embedding.

Provenance (port): faithful copy of src/granular_agent/llm_client.py (frozen
legacy stack) at Stage B kickoff 2026-09-05 — same env, same wall-timeout
watchdogs, same call log, same balanced-JSON parser. The legacy file stays
untouched; this port is the new stack's only LLM entry point. Embedding lives
in kb_infra/embedding.py (embed_batch below is the Paratera GLM tier).

Provenance (D3 fix): every call is recorded in CALL_LOG (module-level list)
and, if env LLM_CALL_LOG gives a path, appended as one JSONL line per call:
{ts, run_id, provider, model, ok, latency_ms, prompt_tokens, completion_tokens,
 attempt, fallback_for}. The DeepSeek->GLM-5-Turbo silent fallback is now
VISIBLE in the log (fallback_for="deepseek-chat") and can be disabled entirely
with env LLM_ALLOW_FALLBACK=0 (scientific runs want a single model).
"""

from __future__ import annotations

import json
import os
import random
import re
import socket
import ssl
import threading
import time
import urllib.request
from datetime import datetime, timezone
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

# ---- call provenance (D3) ----
RUN_ID = os.environ.get("LLM_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
CALL_LOG: list[dict] = []


def _env_seed() -> int | None:
    """Default seed from env LLM_SEED (set by run_kernel --seed). Explicit
    per-call seed argument takes precedence. NOTE: whether the provider
    actually honors `seed` is unverified — two same-seed runs must be
    compared before claiming reproducibility."""
    v = os.environ.get("LLM_SEED")
    try:
        return int(v) if v not in (None, "") else None
    except ValueError:
        return None


def _log_call(provider: str, model: str, ok: bool, latency_ms: float,
              usage: dict | None = None, attempt: int = 0,
              fallback_for: str = "") -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "provider": provider,
        "model": model,
        "ok": ok,
        "latency_ms": round(latency_ms),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "attempt": attempt,
        "fallback_for": fallback_for,
    }
    CALL_LOG.append(rec)
    path = os.environ.get("LLM_CALL_LOG")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass  # logging must never kill the call


def call_log_summary() -> dict:
    """Aggregate the in-memory CALL_LOG: per (provider, model) call count,
    tokens, and fallback count. For run-end provenance reports."""
    agg: dict[tuple, dict] = {}
    for r in CALL_LOG:
        k = (r["provider"], r["model"])
        a = agg.setdefault(k, {"calls": 0, "ok": 0, "prompt_tokens": 0,
                               "completion_tokens": 0, "fallback": 0})
        a["calls"] += 1
        a["ok"] += 1 if r["ok"] else 0
        a["prompt_tokens"] += r["prompt_tokens"] or 0
        a["completion_tokens"] += r["completion_tokens"] or 0
        if r["fallback_for"]:
            a["fallback"] += 1
    return {"run_id": RUN_ID, "total_calls": len(CALL_LOG),
            "by_model": {f"{p}/{m}": v for (p, m), v in sorted(agg.items())}}



def _walled_open(req, sock_timeout: float = 60, wall_s: float | None = None):
    """(v3) thin alias: opens via _timed_open (daemon-thread wall) then the
    caller reads via _walled_read. Kept as the stable entry point name."""
    return _timed_open(req, sock_timeout=sock_timeout, wall_s=wall_s)


def _timed_open(req, sock_timeout: float = 60, wall_s: float | None = None):
    """THE FINAL WALL (GOAL 2026-08-30 v3): urlopen executed in a DAEMON
    thread; main flow joins with a hard timeout. Whatever the SSL/socket
    stack does on Windows (socket-timeout silently unapplied to blocked
    reads — measured twice today), the MAIN thread proceeds at wall_s and
    the hung reader is abandoned to its daemon grave. The only construct
    that cannot be bypassed by any network-stack behavior."""
    import threading
    if wall_s is None:
        wall_s = float(os.environ.get("LLM_WALL_TIMEOUT", "240"))
    box = {}
    def _run():
        try:
            box["r"] = urllib.request.urlopen(req, context=_CTX,
                                              timeout=sock_timeout)
        except Exception as e:
            box["e"] = e
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(wall_s)
    if "r" in box:
        return box["r"]
    if "e" in box:
        raise box["e"]
    raise TimeoutError(f"open wall {wall_s}s (thread-abandon)")


def _walled_read(r, t0: float, wall_s: float | None = None) -> bytes:
    if wall_s is None:
        wall_s = float(os.environ.get("LLM_WALL_TIMEOUT", "240"))
    chunks = []
    while True:
        if time.time() - t0 > wall_s:
            raise TimeoutError(f"wall-clock {wall_s}s exceeded (slow-drip)")
        b = r.read(65536)
        if not b:
            break
        chunks.append(b)
    return b"".join(chunks)

def _chat_once(url, key, body, timeout):
    """Single HTTP POST to an OpenAI-compat chat endpoint. Returns
    (content, usage). Raises on any failure."""
    req = urllib.request.Request(
        url, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    _t0 = time.time()
    r = _walled_open(req, sock_timeout=min(timeout, 60))
    raw = _walled_read(r, _t0)
    resp = json.loads(raw)
    return resp["choices"][0]["message"]["content"], resp.get("usage") or {}


def call_llm(prompt: str, model: str = "deepseek-chat", max_tokens: int = 4000,
             temperature: float = 0.0, seed: int | None = None) -> str | None:
    """Call DeepSeek chat API with exponential-backoff retry + Paratera fallback.

    deepseek intermittently hangs at socket level (TCP connected, server never
    replies) — a short per-call timeout + 3 backoff retries on the primary,
    then one shot at the Paratera fallback model (GLM-5-Turbo) so a single
    hung section can't stall the whole extraction. The fallback is logged
    (fallback_for) and disabled when env LLM_ALLOW_FALLBACK=0."""
    key = ENV.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("no DEEPSEEK_API_KEY")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is None:
        seed = _env_seed()
    if seed is not None:
        payload["seed"] = seed
    body = json.dumps(payload).encode()
    url = "https://api.deepseek.com/v1/chat/completions"
    last_err = None
    # primary: 3 attempts with exponential backoff (2s, 4s)
    for attempt in range(3):
        t0 = time.time()
        try:
            content, usage = _chat_once(url, key, body, timeout=120)
            _log_call("deepseek", model, True, (time.time() - t0) * 1000, usage, attempt)
            return content
        except Exception as e:
            last_err = e
            _log_call("deepseek", model, False, (time.time() - t0) * 1000,
                      None, attempt)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    # fallback: Paratera GLM-5-Turbo (same prompt, one shot) — logged + switchable
    if os.environ.get("LLM_ALLOW_FALLBACK", "1") != "0":
        fb = call_paratera(prompt, model="GLM-5-Turbo", max_tokens=max_tokens,
                           temperature=temperature, fallback_for=model)
        if fb is not None:
            return fb
    # both failed
    print(f"  [llm] all retries + fallback failed: {last_err}", flush=True)
    return None


def call_paratera(prompt: str, model: str = "Kimi-K2.6", max_tokens: int = 4000,
                  temperature: float = 0.0, enable_thinking: bool = None,
                  seed: int | None = None, fallback_for: str = "") -> str | None:
    """Call Paratera API (Kimi/GLM/Qwen/DeepSeek).

    enable_thinking: for reasoning models (DeepSeek-V4-Flash etc), set False
    to suppress reasoning_content via the CORRECT DeepSeek API param:
    {"thinking": {"type": "disabled"}} (verified: reasoning_tokens=0, 1s
    response. The old 'enable_thinking' param name did NOT work — it left
    reasoning on, eating max_tokens + slowing 67x).

    fallback_for: set when this call serves as another provider's fallback
    (provenance marker, logged per call)."""
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
    if seed is None:
        seed = _env_seed()
    if seed is not None:
        payload["seed"] = seed
    if enable_thinking is False:
        # Two verified param forms (both measured, do not merge blindly):
        # - DeepSeek-family: {"thinking": {"type": "disabled"}} per official
        #   docs (verified reasoning_tokens=0; the old 'enable_thinking' param
        #   name did NOT work — left reasoning on, eating max_tokens, 67x slow)
        # - Qwen-family (Qwen3.8-Max): chat_template_kwargs={"enable_thinking":
        #   False} (verified in Stage A; the thinking param is ignored)
        if model.lower().startswith("qwen"):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            payload["thinking"] = {"type": "disabled"}
    body = json.dumps(payload).encode()

    # WALL-CLOCK WATCHDOG (GOAL 2026-08-30): urllib's timeout only bounds
    # connect + each socket read — a server that drips bytes (measured
    # twice today: boost2/boost3 hung 30+ min inside ONE call while the
    # ledger went silent) never trips it. Hard total-deadline via a socket
    # poll: read in chunks, abort the moment total elapsed exceeds the cap.
    _WALL = float(os.environ.get("LLM_WALL_TIMEOUT", "240"))

    def _open_with_wall(req, wall_s: float, sock_timeout: float = 60):
        r = urllib.request.urlopen(req, context=_CTX, timeout=sock_timeout)
        return r

    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    # LLM_SOCK_TIMEOUT override (2026-09-06): dense-table chunks make Qwen take
    # ~120s to first byte (measured: curl#c21 probe succeeded in 123s with
    # timeout=300 after failing 3 rounds at the hardcoded 60s). Default stays
    # 60 for normal traffic; retry rounds set 200 + LLM_WALL_TIMEOUT=420.
    _sock_to = float(os.environ.get("LLM_SOCK_TIMEOUT", "60"))
    for attempt in range(2):
        t0 = time.time()
        try:
            r = _walled_open(req, sock_timeout=_sock_to)
            chunks = []
            while True:
                if time.time() - t0 > _WALL:
                    raise TimeoutError(f"wall-clock {_WALL}s exceeded (slow-drip server)")
                try:
                    b = r.read(65536)
                except socket.timeout:
                    raise
                if not b:
                    break
                chunks.append(b)
            raw = b"".join(chunks)
            resp = json.loads(raw)
            _log_call("paratera", model, True, (time.time() - t0) * 1000,
                      resp.get("usage") or {}, attempt, fallback_for)
            return resp["choices"][0]["message"]["content"]
        except Exception:
            _log_call("paratera", model, False, (time.time() - t0) * 1000,
                      None, attempt, fallback_for)
            if attempt == 1:
                return None
    return None


def call_paratera_vision(prompt: str, image_path: str,
                         model: str = "GLM-4V-Flash", max_tokens: int = 4000,
                         temperature: float = 0.0) -> str | None:
    """Call a Paratera vision model (GLM-4.5V / GLM-4.6V / GLM-4V-Flash / ...)
    with one local image. Multimodal OpenAI-compatible content array.

    figure_channel v2 (2026-09-18 batch-1 parallel line): the VLM is a
    PROPOSER only — every value it reads is gated deterministically
    downstream (FIGURE-FIND-REDESIGN-SPEC G1-G5). Same wall-clock watchdog
    and ledger discipline as call_paratera; vision input is NOT token-counted
    by the ledger (image tokens vary by model), chat-side tokens are.
    """
    import base64
    import mimetypes
    key = ENV.get("PARATERA_API_KEY")
    base = ENV.get("PARATERA_BASE_URL", "").rstrip("/")
    if not key:
        return None
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode()
    _WALL = float(os.environ.get("LLM_WALL_TIMEOUT", "240"))

    req = urllib.request.Request(
        base + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    _sock_to = float(os.environ.get("LLM_SOCK_TIMEOUT", "60"))
    for attempt in range(2):
        t0 = time.time()
        try:
            r = _walled_open(req, sock_timeout=_sock_to)
            chunks = []
            while True:
                if time.time() - t0 > _WALL:
                    raise TimeoutError(f"wall-clock {_WALL}s exceeded (slow-drip server)")
                b = r.read(65536)
                if not b:
                    break
                chunks.append(b)
            raw = b"".join(chunks)
            resp = json.loads(raw)
            _log_call("paratera-vision", model, True, (time.time() - t0) * 1000,
                      resp.get("usage") or {}, attempt)
            return resp["choices"][0]["message"]["content"]
        except Exception:
            _log_call("paratera-vision", model, False, (time.time() - t0) * 1000,
                      None, attempt)
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
            # wall-clock watchdog (GOAL 2026-08-30): same slow-drip hole as
            # chat — boost3's second hang was HERE (embedding endpoint,
            # verified via netstat to Paratera IP while ledger silent).
            r = _walled_open(req, sock_timeout=60)
            _t0 = time.time()
            _wall = float(os.environ.get("LLM_WALL_TIMEOUT", "240"))
            chunks = []
            while True:
                if time.time() - _t0 > _wall:
                    raise TimeoutError(f"embed wall-clock {_wall}s exceeded")
                b = r.read(65536)
                if not b:
                    break
                chunks.append(b)
            raw = b"".join(chunks)
            data = json.loads(raw).get("data", [])
            data.sort(key=lambda x: x.get("index", 0))
            for d in data:
                out[i + d.get("index", 0)] = d["embedding"]
        except Exception as e:
            if isinstance(e, TimeoutError):
                # wall timeout = provider slow-drip-dead (not a bad text):
                # per-item retry would multiply the wall 16x per batch
                # (measured avalanche 2026-08-31: embed_batch joined >400s on
                # a dead Paratera while walls fired correctly beneath).
                # Fail fast so _embed_texts_robust falls to the CST tier.
                raise
            # whole batch failed (likely one bad text); retry each individually
            for j, t in enumerate(chunk):
                if any(out[i + j]):
                    continue
                b = json.dumps({"model": model, "input": [t[:8000]]}).encode()
                rq = urllib.request.Request(
                    base + "/embeddings", data=b,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                try:
                    _t2 = time.time()
                    _rr2 = _walled_open(rq, sock_timeout=30)
                    r = _walled_read(_rr2, _t2)
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


# ---- CST hardening (D8, 2026-09-10) ----
# Probe evidence (precheck_2026-09-10/cst_chat_probe.md): CST qwen3.5 is ALIVE
# but slow-first-byte — measured 79.3s TTFB for a one-word reply (free-tier
# queue), while the old hardcoded sock_timeout=60 killed every call. deepseek-
# v4-flash answered in 0.9s, minimax-m27 in 7.7s on the same channel.
# D8 spec: same-model retries only (exponential backoff + jitter + Retry-After),
# concurrency semaphore (free tier), NO cross-channel fallback, arm purity
# asserted post-hoc via check_arm_purity().
_CST_SEM = threading.Semaphore(int(os.environ.get("CST_MAX_CONCURRENT", "4")))
_CST_RETRY_STATUS = {429, 500, 502, 503, 504}


def _cst_backoff_sleep(attempt: int, retry_after: str | None = None) -> None:
    """Exponential backoff (2,4,8,16,32s cap 60) + jitter; Retry-After wins
    when the server sends a sane value."""
    if retry_after:
        try:
            ra = float(retry_after)
            if 0 < ra <= 300:
                time.sleep(ra + random.uniform(0, 1.0))
                return
        except ValueError:
            pass
    time.sleep(min(60.0, 2.0 * (2 ** attempt)) + random.uniform(0, 1.5))


def call_cst(prompt: str, model: str = "qwen3.5", max_tokens: int = 4000,
             temperature: float = 0.0, seed: int | None = None,
             enable_thinking: bool | None = None) -> str | None:
    """Call CSTCloud API (qwen3.5 / gpt-oss-120b / deepseek-v4-flash etc).
    OpenAI-compatible.

    enable_thinking (2026-09-10 probe cst_thinking_probe.md — overturns the
    old "CST models ignore it" note): CST qwen3.5 DOES think by default
    (reasoning 350-380 tokens, TTFB 77-176s for a one-word reply) and DOES
    honor the Qwen-family form chat_template_kwargs={"enable_thinking":
    false} (reasoning=0, TTFB 1.3s — same form documented by CST for
    qwen3:235b). The DeepSeek-family form {"thinking": {"type": "disabled"}}
    is rejected with HTTP 422 on CST. Non-qwen CST models (deepseek-v4-flash
    measured reasoning=0) default to thinking-off: send nothing for them.
    NOTE: CST intermittently 422s VALID payloads (1/3 observed) — 422 stays
    terminal here (surfaces real payload errors); call_json's outer retries
    absorb the flake.

    Hardened per D8 (2026-09-10): CST_SOCK_TIMEOUT default 150s (measured
    79.3s TTFB on qwen3.5 free tier), CST_MAX_ATTEMPTS default 5 same-model
    retries with exponential backoff + jitter + Retry-After, module-level
    concurrency semaphore CST_MAX_CONCURRENT (default 4). No cross-channel
    fallback: a persistently dead channel returns None and the batch runner
    stops-and-resumes rather than switching models (arm purity)."""
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
    if seed is None:
        seed = _env_seed()
    if seed is not None:
        payload["seed"] = seed
    if enable_thinking is False and model.lower().startswith("qwen"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    attempts = int(os.environ.get("CST_MAX_ATTEMPTS", "5"))
    sock_to = float(os.environ.get("CST_SOCK_TIMEOUT", "150"))
    for attempt in range(attempts):
        t0 = time.time()
        retry_after = None
        try:
            req = urllib.request.Request(
                base + "/chat/completions",
                data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
            )
            with _CST_SEM:
                _r3 = _walled_open(req, sock_timeout=sock_to)
                raw = _walled_read(_r3, t0)
            resp = json.loads(raw)
            _log_call("cst", model, True, (time.time() - t0) * 1000,
                      resp.get("usage") or {}, attempt)
            return resp["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            try:
                retry_after = e.headers.get("Retry-After") if e.headers else None
            except Exception:
                retry_after = None
            _log_call("cst", model, False, (time.time() - t0) * 1000,
                      None, attempt)
            if e.code in _CST_RETRY_STATUS and attempt < attempts - 1:
                _cst_backoff_sleep(attempt, retry_after)
                continue
            return None
        except Exception:
            _log_call("cst", model, False, (time.time() - t0) * 1000,
                      None, attempt)
            if attempt < attempts - 1:
                _cst_backoff_sleep(attempt)
                continue
            return None
    return None


# ---- LOCAL provider (self-hosted vLLM, OpenAI-compatible; added 2026-09-16
#      for the local Qwen3.8-27B deployment) ----
_LOCAL_SEM = threading.Semaphore(int(os.environ.get("LOCAL_MAX_CONCURRENT", "4")))


def call_local(prompt: str, model: str = "qwen3.8-27b-local", max_tokens: int = 4000,
               temperature: float = 0.0, seed: int | None = None,
               enable_thinking: bool | None = None) -> str | None:
    """Call a local vLLM OpenAI-compatible server.

    Config: LOCAL_BASE_URL (default http://127.0.0.1:8000/v1) and optional
    LOCAL_API_KEY, read from .env first then os.environ (deployment can set
    either). Served model name must start with 'qwen' for the thinking-off
    path below (same convention as call_cst); thinking is disabled via the
    Qwen-family form chat_template_kwargs={"enable_thinking": False} — the
    only form verified to work on our stack (Paratera probe + CST probe;
    the bare 'enable_thinking' param silently leaves reasoning ON).
    FIRST-USE CANARY: verify usage shows reasoning_tokens=0 in the ledger
    (GLM-4.6V lesson: thinking models mis-read table values; GLM-5.3
    lesson: thinking eats the output budget).

    Retries absorb server warm-up / transient 5xx with short backoff.
    No cross-channel fallback: a dead local server returns None and the
    batch runner stops-and-resumes (arm purity)."""
    base = (ENV.get("LOCAL_BASE_URL")
            or os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
    key = ENV.get("LOCAL_API_KEY") or os.environ.get("LOCAL_API_KEY", "local")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is None:
        seed = _env_seed()
    if seed is not None:
        payload["seed"] = seed
    if enable_thinking is False and model.lower().startswith("qwen"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    attempts = int(os.environ.get("LOCAL_MAX_ATTEMPTS", "4"))
    sock_to = float(os.environ.get("LOCAL_SOCK_TIMEOUT", "300"))
    for attempt in range(attempts):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                base + "/chat/completions",
                data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
            )
            with _LOCAL_SEM:
                _rl = _walled_open(req, sock_timeout=sock_to)
                raw = _walled_read(_rl, t0)
            resp = json.loads(raw)
            _log_call("local", model, True, (time.time() - t0) * 1000,
                      resp.get("usage") or {}, attempt)
            return resp["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            _log_call("local", model, False, (time.time() - t0) * 1000,
                      None, attempt)
            if e.code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except Exception:
            _log_call("local", model, False, (time.time() - t0) * 1000,
                      None, attempt)
            if attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return None
    return None


def check_arm_purity(log_path: str | None = None,
                     allowed_pairs: list[tuple[str, str]] | None = None) -> dict:
    """D8 arm-purity assertion. Reads a JSONL call log (default: env
    LLM_CALL_LOG) and reports the distinct (provider, model) pairs with
    ok=True. Purity rule: if allowed_pairs is given, any ok pair outside it
    is a violation; otherwise the log must contain exactly ONE distinct pair.
    Runner calls this at batch end; violations => arm void (discipline D8).
    Judge/scoring calls must go to a SEPARATE log file, else pass them in
    allowed_pairs explicitly."""
    path = log_path or os.environ.get("LLM_CALL_LOG")
    if not path or not os.path.exists(path):
        return {"error": f"log not found: {path}", "pure": False}
    counts: dict[tuple, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("ok"):
                k = (rec.get("provider"), rec.get("model"))
                counts[k] = counts.get(k, 0) + 1
    if allowed_pairs:
        allowed = {(p, m) for p, m in allowed_pairs}
        violations = sorted(k for k in counts if k not in allowed)
    else:
        violations = sorted(counts, key=lambda k: -counts[k])[1:]
    return {
        "pairs": {f"{p}/{m}": c for (p, m), c in sorted(counts.items())},
        "violations": [f"{p}/{m}" for p, m in violations],
        "pure": not violations,
    }


def parse_json_response(text: str | None) -> Any:
    r"""Parse JSON from LLM response, handling markdown fences and extra text.

    Long-chunk fix: the old greedy `(\[.*\]|\{.*\})` matched the FIRST `{` to
    the LAST `}` — on long LLM outputs (which drift into prose with stray
    braces) it grabbed a huge span full of non-JSON → json.loads failed → 0
    nodes/edges parsed (a whole chunk lost). This now extracts the FIRST
    BALANCED JSON object/array (counting brace depth, respecting strings), so
    even if the LLM wraps JSON in prose, the first real JSON object is found."""
    if not text:
        return None
    text = re.sub(r"```json|```", "", text, flags=re.S)
    # find the first balanced { ... } or [ ... ] (string-aware, depth-counted)
    obj = _extract_first_json(text)
    if obj is not None:
        try:
            return json.loads(obj)
        except Exception:
            pass
    # fallback: old greedy regex (last resort, may fail on long prose)
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _extract_first_json(text: str) -> str | None:
    """Extract the first balanced JSON object/array from text (string-aware,
    depth-counted). Returns the substring or None. Handles LLM output that
    drifts into prose with stray braces — grabs only the first complete {…} or
    […] block, not a greedy first-{ to last-} span."""
    start = -1
    depth = 0
    in_str = False
    esc = False
    open_ch = ""
    close_ch = ""
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            if depth == 0:
                start = i
                open_ch = ch
                close_ch = "}" if ch == "{" else "]"
            depth += 1
        elif ch in "}]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = text[start:i + 1]
                    # only return if the closing matches the opening type
                    if ch == close_ch:
                        return candidate
                    # mismatched (e.g. opened { closed ]) — reset, keep scanning
                    start = -1
                    open_ch = ""
                    close_ch = ""
    return None
