# -*- coding: utf-8 -*-
"""One-shot: harden call_intern (batch workhorse per user directive 09-21)
and wire the intern provider into records/common.py routing."""
import io

LLM = r"C:/Users/D0n9/Desktop/CompileScholar/src/kb_infra/llm.py"
COMMON = r"C:/Users/D0n9/Desktop/CompileScholar/src/kb_compiler/records/common.py"

s = io.open(LLM, encoding="utf-8").read()

OLD_SIG = '''def call_intern(prompt: str, model: str = "qwen3.8-27b", max_tokens: int = 4000,
                temperature: float = 0.0, seed: int | None = None) -> str | None:'''
NEW_HEAD = '''_INTERN_SEM = threading.Semaphore(int(os.environ.get("INTERN_MAX_CONCURRENT", "8")))
_INTERN_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


def call_intern(prompt: str, model: str = "qwen3.8-27b", max_tokens: int = 4000,
                temperature: float = 0.0, seed: int | None = None,
                enable_thinking: bool | None = None) -> str | None:'''
assert OLD_SIG in s, "intern signature not found"
s = s.replace(OLD_SIG, NEW_HEAD, 1)

OLD_DOC = '''    Thinking-disable is INTERN-SPECIFIC (measured 2026-09-18): ONLY
    {"thinking": {"type": "disabled"}} works — chat_template_kwargs and the
    enable_thinking param are both ignored. Same wall-clock watchdog + ledger
    discipline as the other providers. Use for: arbitration initial rulings,
    smoke tests, low-stakes batch work; formal runs stay on billed channels.
    """'''
NEW_DOC = '''    Thinking-disable is INTERN-SPECIFIC (measured 2026-09-18, re-probed
    2026-09-21): ONLY {"thinking": {"type": "disabled"}} works —
    chat_template_kwargs and the top-level enable_thinking param are both
    IGNORED (reasoning tokens stay billed). The enable_thinking kwarg is
    accepted for call-interface parity and changes nothing.

    Promoted to formal batch workhorse (user directive 2026-09-21: free
    quota first, switch back to local when exhausted; the build ledger
    records provider+model per call, and check_arm_purity takes the
    (intern, local) allowed-pair list for mixed-provider builds).
    Hardened for batch use: INTERN_MAX_CONCURRENT semaphore (default 8),
    5 attempts with exponential backoff + jitter + Retry-After (reuses the
    CST backoff), fresh Request per attempt. No cross-channel fallback
    (arm purity): a dead channel returns None and the batch runner
    stops-and-resumes.
    """'''
assert OLD_DOC in s, "intern docstring not found"
s = s.replace(OLD_DOC, NEW_DOC, 1)

OLD_LOOP = '''    body = json.dumps(payload).encode()
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
            resp = json.loads(b"".join(chunks))
            _log_call("intern", model, True, (time.time() - t0) * 1000,
                      resp.get("usage") or {}, attempt)
            return resp["choices"][0]["message"]["content"]
        except Exception:
            _log_call("intern", model, False, (time.time() - t0) * 1000,
                      None, attempt)
            if attempt == 1:
                return None
    return None'''
NEW_LOOP = '''    body = json.dumps(payload).encode()
    attempts = int(os.environ.get("INTERN_MAX_ATTEMPTS", "5"))
    _WALL = float(os.environ.get("LLM_WALL_TIMEOUT", "240"))
    _sock_to = float(os.environ.get("LLM_SOCK_TIMEOUT", "60"))
    for attempt in range(attempts):
        t0 = time.time()
        retry_after = None
        try:
            req = urllib.request.Request(
                base + "/chat/completions", data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            with _INTERN_SEM:
                r = _walled_open(req, sock_timeout=_sock_to)
                chunks = []
                while True:
                    if time.time() - t0 > _WALL:
                        raise TimeoutError(f"wall-clock {_WALL}s exceeded (slow-drip server)")
                    b = r.read(65536)
                    if not b:
                        break
                    chunks.append(b)
            resp = json.loads(b"".join(chunks))
            _log_call("intern", model, True, (time.time() - t0) * 1000,
                      resp.get("usage") or {}, attempt)
            return resp["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            try:
                retry_after = e.headers.get("Retry-After") if e.headers else None
            except Exception:
                retry_after = None
            _log_call("intern", model, False, (time.time() - t0) * 1000,
                      None, attempt)
            if e.code in _INTERN_RETRY_STATUS and attempt < attempts - 1:
                _cst_backoff_sleep(attempt, retry_after)
                continue
            return None
        except Exception:
            _log_call("intern", model, False, (time.time() - t0) * 1000,
                      None, attempt)
            if attempt < attempts - 1:
                _cst_backoff_sleep(attempt)
                continue
            return None
    return None'''
assert OLD_LOOP in s, "intern retry loop not found"
s = s.replace(OLD_LOOP, NEW_LOOP, 1)
io.open(LLM, "w", encoding="utf-8").write(s)
print("llm.py patched")

s2 = io.open(COMMON, encoding="utf-8").read()
OLD_IMP = "from kb_infra.llm import call_paratera, call_cst, call_local, parse_json_response  # noqa: E402"
NEW_IMP = ("from kb_infra.llm import (call_paratera, call_cst, call_local,  # noqa: E402\n"
           "                          call_intern, parse_json_response)")
assert OLD_IMP in s2
s2 = s2.replace(OLD_IMP, NEW_IMP, 1)

OLD_DISP = '''    prov, model = route_model(model_spec)
    fn = {"cst": call_cst, "local": call_local}.get(prov, call_paratera)'''
NEW_DISP = '''    prov, model = route_model(model_spec)
    fn = {"cst": call_cst, "local": call_local,
          "intern": call_intern}.get(prov, call_paratera)'''
assert OLD_DISP in s2
s2 = s2.replace(OLD_DISP, NEW_DISP, 1)

OLD_DOC2 = '''Model routing convention: "provider:model" — provider ∈ {paratera, cst,
local}; bare model name defaults to paratera.'''
NEW_DOC2 = '''Model routing convention: "provider:model" — provider ∈ {paratera, cst,
local, intern}; bare model name defaults to paratera. "intern" = INTERN
free channel (intern:qwen3.8-27b; its thinking-disabled form is provider-
specific and handled inside call_intern).'''
assert OLD_DOC2 in s2
s2 = s2.replace(OLD_DOC2, NEW_DOC2, 1)
io.open(COMMON, "w", encoding="utf-8").write(s2)
print("common.py patched")
