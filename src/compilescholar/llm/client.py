# -*- coding: utf-8 -*-
"""LLM gateway for the answer path and the judges: local OpenAI-compatible server (GPUStack/vLLM) and Paratera.

Moved from kb_infra.llm (call_local / call_paratera) with the same request payloads, retries, wall-clock watchdogs,
concurrency lanes and JSONL call ledger. Differences from the original, all intentional:
  - TLS certificate verification is ON (the original disabled it process-wide; set CS_TLS_INSECURE=1 to restore that
    for a self-signed endpoint). The local endpoints are plain http, so the answer path is unaffected.
  - credentials come from compilescholar.core.secrets (process environment first, then .env).
  - concurrency semaphores are created on first use (not at import), so the value of LOCAL_MAX_CONCURRENT in effect
    when the first call is made decides the lane width — callers no longer have to set it before importing.
"""
from __future__ import annotations

import hashlib
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from ..core import secrets

RUN_ID = os.environ.get("LLM_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
CALL_LOG: list[dict] = []


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if os.environ.get("CS_TLS_INSECURE") == "1":
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


_CTX = _ssl_ctx()


def _env_seed() -> int | None:
    v = os.environ.get("LLM_SEED")
    try:
        return int(v) if v not in (None, "") else None
    except ValueError:
        return None


def _phash(payload) -> str:
    if not payload:
        return ""
    try:
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    except Exception:
        return ""


def _log_call(provider: str, model: str, ok: bool, latency_ms: float, usage: dict | None = None, attempt: int = 0,
              fallback_for: str = "", payload=None, extra: dict | None = None) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run_id": RUN_ID,
           "caller": os.environ.get("LLM_CALLER", ""), "provider": provider, "model": model, "ok": ok,
           "latency_ms": round(latency_ms), "prompt_tokens": (usage or {}).get("prompt_tokens"),
           "completion_tokens": (usage or {}).get("completion_tokens"), "attempt": attempt, "fallback_for": fallback_for}
    if payload is not None:
        rec["prompt_hash"] = _phash(payload)
    if extra:
        rec.update(extra)
    CALL_LOG.append(rec)
    path = os.environ.get("LLM_CALL_LOG")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass  # logging must never kill the call


def _provider_allowed(prov: str) -> bool:
    """LLM_PROVIDER_ALLOWLIST gate: unset = everything allowed."""
    allow = os.environ.get("LLM_PROVIDER_ALLOWLIST", "")
    if not allow:
        return True
    return prov in {p.strip() for p in allow.split(",") if p.strip()}


def _gate_block(prov: str, model: str):
    import sys
    print(f"[llm-gate] BLOCKED {prov}:{model} — LLM_PROVIDER_ALLOWLIST={os.environ.get('LLM_PROVIDER_ALLOWLIST')!r}",
          file=sys.stderr, flush=True)
    _log_call(prov, model, False, 0.0, None, 0)


def _wall() -> float:
    return float(os.environ.get("LLM_WALL_TIMEOUT", "240"))


def walled_open(req, sock_timeout: float = 60, wall_s: float | None = None):
    """urlopen in a daemon thread joined with a hard wall: a hung SSL/socket read on Windows cannot stall the caller."""
    wall_s = _wall() if wall_s is None else wall_s
    box: dict = {}

    def _run():
        try:
            box["r"] = urllib.request.urlopen(req, context=_CTX, timeout=sock_timeout)
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


def walled_read(r, t0: float, wall_s: float | None = None) -> bytes:
    """Read the whole body in a daemon thread; give up when the wall since t0 is exceeded."""
    wall_s = _wall() if wall_s is None else wall_s
    chunks: list[bytes] = []
    box: dict = {}

    def _reader():
        try:
            while True:
                b = r.read(65536)
                if not b:
                    break
                chunks.append(b)
            box["done"] = True
        except Exception as e:
            box["e"] = e
    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(max(1.0, wall_s - (time.time() - t0)))
    if "done" in box:
        return b"".join(chunks)
    if "e" in box:
        raise box["e"]
    raise TimeoutError(f"read wall {wall_s}s exceeded (thread-abandon)")


# ---------------------------------------------------------------- local server
_SEM_LOCK = threading.Lock()
_SEMS: dict[str, threading.Semaphore] = {}


def _lane(large: bool) -> threading.Semaphore:
    """Two concurrency lanes: large-output calls (max_tokens >= 8000) get their own narrow lane so they cannot starve
    short calls. Widths: LOCAL_MAX_CONCURRENT (default 4) and LOCAL_LARGE_MAX_CONCURRENT (default max(2, shared // 8))."""
    name = "large" if large else "shared"
    with _SEM_LOCK:
        if name not in _SEMS:
            shared = int(os.environ.get("LOCAL_MAX_CONCURRENT", "4"))
            n = int(os.environ.get("LOCAL_LARGE_MAX_CONCURRENT", str(max(2, shared // 8)))) if large else shared
            _SEMS[name] = threading.Semaphore(n)
        return _SEMS[name]


def call_local(prompt: str, model: str = "Qwen3.8-27B", max_tokens: int = 4000, temperature: float = 0.0,
               seed: int | None = None, enable_thinking: bool | None = None) -> str | None:
    """Local OpenAI-compatible chat. Thinking is disabled for Qwen-family models via
    chat_template_kwargs={"enable_thinking": False} (the only verified form). Retries 429/5xx and transport errors with
    short backoff; no cross-channel fallback — a dead server returns None."""
    if not _provider_allowed("local"):
        _gate_block("local", model)
        return None
    base = (secrets.get("LOCAL_BASE_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
    key = secrets.get("LOCAL_API_KEY", "local")
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
               "max_tokens": max_tokens}
    seed = _env_seed() if seed is None else seed
    if seed is not None:
        payload["seed"] = seed
    if enable_thinking is False and model.lower().startswith("qwen"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    attempts = int(os.environ.get("LOCAL_MAX_ATTEMPTS", "4"))
    sock_to = float(os.environ.get("LOCAL_SOCK_TIMEOUT", "900"))
    for attempt in range(attempts):
        t0 = time.time()
        try:
            req = urllib.request.Request(base + "/chat/completions", data=body,
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with _lane(max_tokens >= 8000):
                raw = walled_read(walled_open(req, sock_timeout=sock_to), t0)
            resp = json.loads(raw)
            _log_call("local", model, True, (time.time() - t0) * 1000, resp.get("usage") or {}, attempt, payload=payload)
            return resp["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            _log_call("local", model, False, (time.time() - t0) * 1000, None, attempt, payload=payload)
            if e.code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except Exception:
            _log_call("local", model, False, (time.time() - t0) * 1000, None, attempt, payload=payload)
            if attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return None
    return None


# ---------------------------------------------------------------- Paratera (field-eval judge and baselines)
def call_paratera(prompt: str, model: str = "Kimi-K2.6", max_tokens: int = 4000, temperature: float = 0.0,
                  enable_thinking: bool | None = None, seed: int | None = None, fallback_for: str = "") -> str | None:
    """Paratera chat (2 attempts). Thinking off: Qwen family via chat_template_kwargs, others via
    {"thinking": {"type": "disabled"}}."""
    if not _provider_allowed("paratera"):
        _gate_block("paratera", model)
        return None
    key = secrets.get("PARATERA_API_KEY")
    base = (secrets.get("PARATERA_BASE_URL") or "").rstrip("/")
    if not key:
        return None
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
               "max_tokens": max_tokens}
    seed = _env_seed() if seed is None else seed
    if seed is not None:
        payload["seed"] = seed
    if enable_thinking is False:
        if model.lower().startswith("qwen"):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            payload["thinking"] = {"type": "disabled"}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    sock_to = float(os.environ.get("LLM_SOCK_TIMEOUT", "60"))
    for attempt in range(2):
        t0 = time.time()
        try:
            raw = walled_read(walled_open(req, sock_timeout=sock_to), t0)
            resp = json.loads(raw)
            _log_call("paratera", model, True, (time.time() - t0) * 1000, resp.get("usage") or {}, attempt,
                      fallback_for, payload=payload)
            return resp["choices"][0]["message"]["content"]
        except Exception:
            _log_call("paratera", model, False, (time.time() - t0) * 1000, None, attempt, fallback_for, payload=payload)
            if attempt == 1:
                return None
    return None
