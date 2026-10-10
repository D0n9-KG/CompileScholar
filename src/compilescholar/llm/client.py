# -*- coding: utf-8 -*-
"""The one LLM client of the system (INTEGRATED-SYSTEM-1005 v2 §10.2): local OpenAI-compatible server (GPUStack/vLLM),
Paratera and CSTCloud; chat, vision and JSON calls. Replaces kb_infra.llm and retrieval._see_llm.

Transport  httpx, streamed. A wall-clock deadline closes the connection, which makes the server stop generating
           (measured 10-06 on the local server: 24 long streams cut after 3 s, no residual load), and the lane is held
           until the connection is really closed — the old thread-abandon wall freed the lane while the server kept
           generating, and retries stacked up to 4 copies of the same prompt on the server.
Lanes      per provider: a normal lane and a narrow lane for large outputs (max_tokens >= large_at); widths from the
           config (`llm.providers.<p>.lanes`), the same for `build` and `answer`.
Rate       per provider token bucket (requests/min), cross-process when `shared_bucket` is set.
Retries    transport errors, timeouts, 429 and 5xx: exponential backoff with jitter, Retry-After honoured; 4xx other
           than 429 are terminal. Breaker: `breaker` consecutive transport failures on a provider raise ChannelDead.
Ledger     always on: one JSON line per attempt in runs/<run_id>/llm_calls.jsonl (or `ledger_path`), with template id,
           finish_reason, served model, http status and error class. Only aggregates are kept in memory.
Cache      temperature-0 responses are cached in cache/llm/responses.sqlite keyed by (provider, model, messages,
           sampling); a rebuild replays them byte for byte (temperature 0 is not reproducible on the local server —
           measured). Sampled calls are not cached by default (repeated runs must stay independent draws). Truncated
           replies (finish_reason=length) are never cached.
Purity     an allowlist of (provider[, model]) blocks other calls before they are sent; check_arm_purity() audits a
           ledger afterwards.

Configuration comes from configure(cfg) (core.config RunConfig.llm); every value has a default, so library code and
tests can call without configuring. Credentials come from core.secrets. Environment variables are not read here except
the legacy ones listed in _LEGACY_ENV, honoured only when configure() was never called (old scripts)."""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import random
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ..core import paths, secrets
from .jsonparse import parse_json_response, salvage_json_records


class ChannelDead(RuntimeError):
    """A provider failed `breaker` times in a row: stop the stage instead of marking items done."""


# ---------------------------------------------------------------- configuration

@dataclass
class Provider:
    name: str
    base_url_key: str
    api_key_key: str | None
    default_model: str
    lanes: int = 4
    large_lanes: int = 2
    large_at: int = 8000
    rate_per_min: float = 0.0             # 0 = no rate limit
    shared_bucket: bool = False
    connect_timeout: float = 15.0
    read_timeout: float = 300.0           # gap between two streamed chunks
    wall: float = 900.0                   # whole call
    attempts: int = 4
    trust_env: bool = False               # local endpoints must bypass the registry proxy (GPUStack lesson)
    thinking_off: str = "qwen"            # qwen = chat_template_kwargs for Qwen models; deepseek = also "thinking"
    breaker: int = 20


DEFAULT_PROVIDERS = {
    "local": Provider("local", base_url_key="LOCAL_BASE_URL", api_key_key="LOCAL_API_KEY",
                      default_model="Qwen3.8-27B", lanes=48, large_lanes=48, read_timeout=600, wall=1800, attempts=4),
    "paratera": Provider("paratera", base_url_key="PARATERA_BASE_URL", api_key_key="PARATERA_API_KEY",
                         default_model="DeepSeek-V4-Flash", lanes=16, large_lanes=8, rate_per_min=0, read_timeout=120,
                         wall=600, attempts=5, trust_env=True, thinking_off="deepseek"),
    "cst": Provider("cst", base_url_key="CST_BASE_URL", api_key_key="CST_API_KEY", default_model="qwen3.5", lanes=4,
                    large_lanes=2, read_timeout=200, wall=900, attempts=5, trust_env=True, thinking_off="qwen"),
}


@dataclass
class LLMSettings:
    providers: dict = field(default_factory=lambda: {k: Provider(**vars(v)) for k, v in DEFAULT_PROVIDERS.items()})
    allow: tuple = ()                     # () = everything; else "provider" or "provider:model" entries
    seed: int | None = None
    cache: bool = True
    cache_path: str | None = None
    ledger_path: str | None = None
    run_id: str = ""
    caller: str = ""
    mirror: dict = field(default_factory=dict)
    # call_local overflow routing: {"provider": "paratera", "model": "Qwen3.8-27B",
    #                                "for_model": "Qwen3.8-27B", "fraction": 0.3}
    # a deterministic share of FRESH call_local traffic (hashed on the work item, stable across restarts)
    # goes to a second provider serving the same model; the ledger records the true provider per call.


_SETTINGS = LLMSettings()
_CONFIGURED = False
_LEGACY_ENV = ("LOCAL_MAX_CONCURRENT", "LOCAL_LARGE_MAX_CONCURRENT", "LLM_WALL_TIMEOUT", "LOCAL_MAX_ATTEMPTS",
               "LLM_SEED", "LLM_PROVIDER_ALLOWLIST", "LLM_CALL_LOG", "LLM_RUN_ID", "LLM_CALLER")


def configure(llm: dict | None = None, run_id: str | None = None, ledger_path: str | Path | None = None,
              caller: str | None = None) -> LLMSettings:
    """Apply the `llm:` section of a resolved config. Resets lanes, buckets and the breaker."""
    global _SETTINGS, _CONFIGURED
    llm = llm or {}
    s = LLMSettings()
    for name, over in (llm.get("providers") or {}).items():
        base = s.providers.get(name) or Provider(name, f"{name.upper()}_BASE_URL", f"{name.upper()}_API_KEY", "")
        s.providers[name] = Provider(**{**vars(base), **(over or {})})
    s.allow = tuple(llm.get("allow") or ())
    s.mirror = dict(llm.get("mirror") or {})
    s.seed = llm.get("seed")
    s.cache = bool(llm.get("cache", True))
    s.cache_path = llm.get("cache_path")
    s.ledger_path = str(ledger_path) if ledger_path else llm.get("ledger_path")
    s.run_id = run_id or llm.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    s.caller = caller or llm.get("caller") or ""
    with _STATE_LOCK:
        _SETTINGS = s
        _CONFIGURED = True
        _LANES.clear()
        _BUCKETS.clear()
        _FAILS.clear()
    return s


def settings() -> LLMSettings:
    if not _CONFIGURED:
        _apply_legacy_env()
    return _SETTINGS


def _apply_legacy_env() -> None:
    """Old scripts set LOCAL_MAX_CONCURRENT etc. before the first call; honour them until they call configure()."""
    global _CONFIGURED
    env = {k: os.environ.get(k) for k in _LEGACY_ENV if os.environ.get(k) not in (None, "")}
    over: dict = {"providers": {"local": {}}}
    if "LOCAL_MAX_CONCURRENT" in env:
        over["providers"]["local"]["lanes"] = int(env["LOCAL_MAX_CONCURRENT"])
        over["providers"]["local"]["large_lanes"] = int(env.get("LOCAL_LARGE_MAX_CONCURRENT")
                                                        or max(2, int(env["LOCAL_MAX_CONCURRENT"]) // 8))
    if "LLM_WALL_TIMEOUT" in env:
        for p in ("local", "paratera", "cst"):
            over["providers"].setdefault(p, {})["wall"] = float(env["LLM_WALL_TIMEOUT"])
    if "LOCAL_MAX_ATTEMPTS" in env:
        over["providers"]["local"]["attempts"] = int(env["LOCAL_MAX_ATTEMPTS"])
    if "LLM_SEED" in env:
        over["seed"] = int(env["LLM_SEED"])
    if "LLM_PROVIDER_ALLOWLIST" in env:
        over["allow"] = [p.strip() for p in env["LLM_PROVIDER_ALLOWLIST"].split(",") if p.strip()]
    configure(over, run_id=env.get("LLM_RUN_ID"), ledger_path=env.get("LLM_CALL_LOG"), caller=env.get("LLM_CALLER"))
    _CONFIGURED = False          # keep tracking the environment until a real configure()


# ---------------------------------------------------------------- shared state

_STATE_LOCK = threading.Lock()
_LANES: dict[tuple, threading.BoundedSemaphore] = {}
_BUCKETS: dict[str, object] = {}
_FAILS: dict[str, int] = {}
_STATS: dict[tuple, dict] = {}
_LEDGER_LOCK = threading.Lock()


def _lane(p: Provider, large: bool) -> threading.BoundedSemaphore:
    k = (p.name, large)
    with _STATE_LOCK:
        if k not in _LANES:
            _LANES[k] = threading.BoundedSemaphore(max(1, p.large_lanes if large else p.lanes))
        return _LANES[k]


def _bucket(p: Provider):
    if p.rate_per_min <= 0:
        return None
    with _STATE_LOCK:
        if p.name not in _BUCKETS:
            from ..sources.sciverse import _FileTokenBucket, _TokenBucket
            cap = max(1, int(p.rate_per_min // 6))
            _BUCKETS[p.name] = (_FileTokenBucket(str(paths.cache() / "pace" / f"llm_{p.name}.json"), p.rate_per_min, cap)
                                if p.shared_bucket else _TokenBucket(p.rate_per_min / 60.0, cap))
            if p.shared_bucket:
                (paths.cache() / "pace").mkdir(parents=True, exist_ok=True)
        return _BUCKETS[p.name]


# ---------------------------------------------------------------- ledger

def _ledger_file() -> Path:
    s = settings()
    if s.ledger_path:
        return Path(s.ledger_path)
    return paths.runs() / (s.run_id or "adhoc") / "llm_calls.jsonl"


def _log(rec: dict) -> None:
    s = settings()
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run_id": s.run_id,
           "caller": rec.pop("caller", None) or s.caller, **rec}
    k = (rec.get("provider"), rec.get("model"))
    with _LEDGER_LOCK:
        a = _STATS.setdefault(k, {"calls": 0, "ok": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0})
        a["calls"] += 1
        a["ok"] += bool(rec.get("ok"))
        a["cached"] += bool(rec.get("cached"))
        a["prompt_tokens"] += rec.get("prompt_tokens") or 0
        a["completion_tokens"] += rec.get("completion_tokens") or 0
        try:
            f = _ledger_file()
            f.parent.mkdir(parents=True, exist_ok=True)
            with open(f, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass                 # the ledger must never kill a call


def call_log_summary() -> dict:
    with _LEDGER_LOCK:
        return {"run_id": settings().run_id, "ledger": str(_ledger_file()),
                "by_model": {f"{p}/{m}": dict(v) for (p, m), v in sorted(_STATS.items(), key=lambda x: str(x[0]))}}


def check_arm_purity(log_path: str | Path | None = None, allowed_pairs: list[tuple[str, str]] | None = None,
                     min_calls: int = 0) -> dict:
    """Distinct (provider, model) pairs with ok=True in a ledger. With allowed_pairs, any other ok pair is a violation;
    otherwise the ledger must hold exactly one pair. min_calls > 0 fails a near-empty ledger (the PaperQA case)."""
    path = Path(log_path) if log_path else _ledger_file()
    if not path.exists():
        return {"error": f"log not found: {path}", "pure": False}
    counts: dict[tuple, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("ok"):
            k = (rec.get("provider"), rec.get("model"))
            counts[k] = counts.get(k, 0) + 1
    if allowed_pairs:
        allowed = {(p, m) for p, m in allowed_pairs}
        violations = sorted(k for k in counts if k not in allowed)
    else:
        violations = sorted(counts, key=lambda k: -counts[k])[1:]
    total = sum(counts.values())
    insufficient = min_calls > 0 and total < min_calls
    return {"pairs": {f"{p}/{m}": c for (p, m), c in sorted(counts.items())},
            "violations": [f"{p}/{m}" for p, m in violations] + ([f"<insufficient_calls: {total}<{min_calls}>"]
                                                                  if insufficient else []),
            "total_ok": total, "pure": not violations and not insufficient}


# ---------------------------------------------------------------- response cache

_CACHE_LOCK = threading.Lock()
_CACHE_CON: dict[str, sqlite3.Connection] = {}


def _cache_con() -> sqlite3.Connection | None:
    s = settings()
    if not s.cache:
        return None
    p = Path(s.cache_path) if s.cache_path else paths.cache() / "llm" / "responses.sqlite"
    with _CACHE_LOCK:
        con = _CACHE_CON.get(str(p))
        if con is None:
            p.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(p, check_same_thread=False)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA busy_timeout=30000")
            con.execute("CREATE TABLE IF NOT EXISTS responses(key TEXT PRIMARY KEY, provider TEXT, model TEXT, "
                        "text TEXT, finish_reason TEXT, usage TEXT, created_at TEXT)")
            _CACHE_CON[str(p)] = con
        return con


def _cache_key(provider: str, payload: dict) -> str:
    keep = {k: payload.get(k) for k in ("model", "messages", "temperature", "max_tokens", "seed", "top_p",
                                       "chat_template_kwargs", "thinking", "response_format")}
    return hashlib.sha256(json.dumps([provider, keep], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _cache_get(key: str):
    con = _cache_con()
    if con is None:
        return None
    with _CACHE_LOCK:
        return con.execute("SELECT text, finish_reason, usage FROM responses WHERE key=?", (key,)).fetchone()


def _cache_put(key: str, provider: str, model: str, text: str, finish: str | None, usage: dict) -> None:
    con = _cache_con()
    if con is None:
        return
    with _CACHE_LOCK:
        con.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?)",
                    (key, provider, model, text, finish, json.dumps(usage or {}),
                     datetime.now(timezone.utc).isoformat(timespec="seconds")))
        con.commit()


# ---------------------------------------------------------------- transport

class _Retryable(Exception):
    def __init__(self, msg: str, status: int | None = None, retry_after: str | None = None):
        super().__init__(msg)
        self.status, self.retry_after = status, retry_after


class _Terminal(Exception):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


_CLIENTS: dict[tuple, httpx.Client] = {}


def _client(p: Provider) -> httpx.Client:
    k = (p.name, p.trust_env, p.connect_timeout, p.read_timeout)
    with _STATE_LOCK:
        c = _CLIENTS.get(k)
        if c is None:
            c = httpx.Client(trust_env=p.trust_env, timeout=httpx.Timeout(p.connect_timeout, read=p.read_timeout,
                                                                          write=60.0, pool=None),
                             limits=httpx.Limits(max_connections=max(p.lanes, p.large_lanes) + 8,
                                                 max_keepalive_connections=max(p.lanes, p.large_lanes)))
            _CLIENTS[k] = c
        return c


def _stream(p: Provider, url: str, key: str | None, payload: dict, deadline: float) -> tuple[str, str | None, dict,
                                                                                            str | None, int]:
    """One streamed request. Returns (text, finish_reason, usage, served_model, http_status). Closing the response
    (leaving the `with`) on a deadline aborts the server-side generation."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    parts: list[str] = []
    finish = served = None
    usage: dict = {}
    try:
        with _client(p).stream("POST", url, json=body, headers=headers) as r:
            if r.status_code != 200:
                txt = r.read().decode("utf-8", "replace")[:300]
                if r.status_code == 429 or r.status_code >= 500:
                    raise _Retryable(f"http {r.status_code}: {txt}", r.status_code, r.headers.get("retry-after"))
                raise _Terminal(f"http {r.status_code}: {txt}", r.status_code)
            for line in r.iter_lines():
                if time.monotonic() > deadline:
                    raise _Retryable(f"wall {p.wall:.0f}s exceeded", 200)
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    o = json.loads(d)
                except ValueError:
                    continue
                served = o.get("model") or served
                if o.get("usage"):
                    usage = o["usage"]
                for ch in o.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        parts.append(delta["content"])
                    finish = ch.get("finish_reason") or finish
            return "".join(parts), finish, usage, served, r.status_code
    except (httpx.TransportError, httpx.TimeoutException) as e:
        raise _Retryable(f"{type(e).__name__}: {e}") from None


def _backoff(attempt: int, retry_after: str | None) -> None:
    if retry_after:
        try:
            ra = float(retry_after)
            if 0 < ra <= 300:
                time.sleep(ra + random.uniform(0, 1.0))
                return
        except ValueError:
            pass
    time.sleep(min(60.0, 2.0 * (2 ** attempt)) + random.uniform(0, 1.0))


def _allowed(provider: str, model: str) -> bool:
    allow = settings().allow
    return not allow or provider in allow or f"{provider}:{model}" in allow


def chat(provider: str, prompt: str | list, model: str | None = None, max_tokens: int = 4000,
         temperature: float = 0.0, seed: int | None = None, enable_thinking: bool | None = None,
         template: str = "", item: str = "", cache: bool | None = None) -> str | None:
    """One chat completion. `prompt` is a user string or a full messages list. Returns the text, or None when every
    attempt failed (the failure is in the ledger). Raises ChannelDead when the provider's breaker trips."""
    s = settings()
    p = s.providers.get(provider)
    if p is None:
        raise ValueError(f"unknown provider {provider!r}")
    model = model or p.default_model
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    payload: dict = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    seed = s.seed if seed is None else seed
    if seed is not None:
        payload["seed"] = seed
    if enable_thinking is False:
        if model.lower().startswith("qwen"):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif p.thinking_off == "deepseek":
            payload["thinking"] = {"type": "disabled"}
    rec = {"provider": provider, "model": model, "template": template, "item": item, "max_tokens": max_tokens,
           "temperature": temperature, "seed": seed, "thinking": enable_thinking}
    if not _allowed(provider, model):
        print(f"[llm-gate] BLOCKED {provider}:{model} (allow={s.allow})", file=sys.stderr, flush=True)
        _log({**rec, "ok": False, "error_class": "blocked", "attempt": 0, "latency_ms": 0})
        return None
    key = _cache_key(provider, payload)
    # sampled calls (temperature > 0) are not cached unless asked: repeated runs (r1/r2) must stay independent draws
    use_cache = (s.cache and temperature == 0) if cache is None else cache
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            _log({**rec, "ok": True, "cached": True, "finish_reason": hit[1], "attempt": 0, "latency_ms": 0,
                  "payload_sha": key[:16]})
            return hit[0]
    base = (secrets.get(p.base_url_key) or ("http://127.0.0.1:8000/v1" if provider == "local" else "")).rstrip("/")
    api_key = secrets.get(p.api_key_key, "local" if provider == "local" else None) if p.api_key_key else None
    if not base or (provider != "local" and not api_key):
        _log({**rec, "ok": False, "error_class": "not_configured", "attempt": 0, "latency_ms": 0})
        return None
    url = base + "/chat/completions"
    lane = _lane(p, max_tokens >= p.large_at)
    bucket = _bucket(p)
    for attempt in range(p.attempts):
        if bucket is not None and not bucket.acquire(max_wait_s=600):
            _log({**rec, "ok": False, "error_class": "rate_wait_exceeded", "attempt": attempt, "latency_ms": 0})
            return None
        t0 = time.monotonic()
        try:
            with lane:
                text, finish, usage, served, status = _stream(p, url, api_key, payload, t0 + p.wall)
        except _Retryable as e:
            _log({**rec, "ok": False, "error_class": "retryable", "http_status": e.status, "error": str(e)[:300],
                  "attempt": attempt, "latency_ms": round((time.monotonic() - t0) * 1000), "payload_sha": key[:16]})
            if _trip(p, str(e)):
                raise ChannelDead(f"{provider}: {p.breaker} consecutive failures, last: {e}") from None
            if attempt < p.attempts - 1:
                _backoff(attempt, e.retry_after)
            continue
        except _Terminal as e:
            _log({**rec, "ok": False, "error_class": "terminal", "http_status": e.status, "error": str(e)[:300],
                  "attempt": attempt, "latency_ms": round((time.monotonic() - t0) * 1000), "payload_sha": key[:16]})
            _reset(p)
            return None
        _reset(p)
        _log({**rec, "ok": True, "cached": False, "http_status": status, "finish_reason": finish,
              "served_model": served, "prompt_tokens": usage.get("prompt_tokens"),
              "completion_tokens": usage.get("completion_tokens"),
              "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
              "attempt": attempt, "latency_ms": round((time.monotonic() - t0) * 1000), "payload_sha": key[:16]})
        if use_cache and text and finish != "length":
            _cache_put(key, provider, model, text, finish, usage)
        return text
    return None


def _trip(p: Provider, err: str) -> bool:
    with _STATE_LOCK:
        _FAILS[p.name] = _FAILS.get(p.name, 0) + 1
        return _FAILS[p.name] >= p.breaker


def _reset(p: Provider) -> None:
    with _STATE_LOCK:
        _FAILS[p.name] = 0


# ---------------------------------------------------------------- the old entry points (same signatures)

def _mirror_route(model: str, key: str) -> tuple[str, str]:
    """Deterministic overflow routing for call_local (llm.mirror config): a fixed fraction of traffic for
    `for_model` goes to a second provider serving the same model. The route is a hash of the work-item key,
    so an item always lands on the same provider — response-cache replay stays coherent across restarts,
    and the ledger records the true provider per call."""
    m = settings().mirror or {}
    frac = float(m.get("fraction") or 0.0)
    if frac <= 0.0 or model != (m.get("for_model") or "Qwen3.8-27B"):
        return "local", model
    h = int(hashlib.sha256(key.encode("utf-8", "replace")).hexdigest()[:8], 16)
    if h % 10000 < int(frac * 10000):
        return m.get("provider") or "paratera", m.get("model") or model
    return "local", model


def call_local(prompt: str, model: str = "Qwen3.8-27B", max_tokens: int = 4000, temperature: float = 0.0,
               seed: int | None = None, enable_thinking: bool | None = None, template: str = "",
               item: str = "") -> str | None:
    provider, mdl = _mirror_route(model, item or template or prompt[:256])
    return chat(provider, prompt, model=mdl, max_tokens=max_tokens, temperature=temperature, seed=seed,
                enable_thinking=enable_thinking, template=template, item=item)


def call_paratera(prompt: str, model: str = "Kimi-K2.6", max_tokens: int = 4000, temperature: float = 0.0,
                  enable_thinking: bool | None = None, seed: int | None = None, fallback_for: str = "",
                  template: str = "", item: str = "") -> str | None:
    return chat("paratera", prompt, model=model, max_tokens=max_tokens, temperature=temperature, seed=seed,
                enable_thinking=enable_thinking, template=template or fallback_for, item=item)


def call_cst(prompt: str, model: str = "qwen3.5", max_tokens: int = 4000, temperature: float = 0.0,
             seed: int | None = None, enable_thinking: bool | None = None, template: str = "",
             item: str = "") -> str | None:
    """CSTCloud: Qwen models honour chat_template_kwargs; the DeepSeek-style "thinking" field is rejected (422), so
    the provider's thinking_off is "qwen" and nothing is sent for non-Qwen models."""
    return chat("cst", prompt, model=model, max_tokens=max_tokens, temperature=temperature, seed=seed,
                enable_thinking=enable_thinking, template=template, item=item)


def vision(prompt: str, images: list[str | Path], provider: str = "paratera", model: str = "GLM-4.6V",
           max_tokens: int = 4000, temperature: float = 0.0, template: str = "", item: str = "") -> str | None:
    """Multimodal call (OpenAI content array, images inlined as data URLs). Same transport, lanes, ledger, cache."""
    content: list = []
    for img in images:
        mime = mimetypes.guess_type(str(img))[0] or "image/png"
        content.append({"type": "image_url", "image_url": {
            "url": f"data:{mime};base64,{base64.b64encode(Path(img).read_bytes()).decode()}"}})
    content.append({"type": "text", "text": prompt})
    return chat(provider, [{"role": "user", "content": content}], model=model, max_tokens=max_tokens,
                temperature=temperature, template=template, item=item)


def call_json(prompt: str, provider: str = "local", model: str | None = None, max_tokens: int = 4000,
              retries: int = 2, validate=None, salvage: tuple[str, str] | None = None, template: str = "",
              item: str = "", enable_thinking: bool | None = False):
    """Chat -> parsed JSON. A reply that does not parse, or that `validate(obj)` rejects (returns False), is retried
    (`retries` extra calls, cache bypassed so a bad cached answer is not replayed). salvage=(item_key, wrapper_key)
    recovers complete objects from a truncated reply. Returns None when nothing usable came back."""
    raw = None
    for att in range(retries + 1):
        raw = chat(provider, prompt, model=model, max_tokens=max_tokens, temperature=0.0,
                   enable_thinking=enable_thinking, template=template, item=item, cache=None if att == 0 else False)
        obj = parse_json_response(raw)
        if obj is not None and (validate is None or validate(obj) is not False):
            return obj
        if salvage and raw:
            obj = salvage_json_records(raw, *salvage)
            if obj is not None and (validate is None or validate(obj) is not False):
                return obj
    return None
