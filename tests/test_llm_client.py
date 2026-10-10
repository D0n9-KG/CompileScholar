# -*- coding: utf-8 -*-
"""llm.client against a local fake OpenAI-compatible server (no network): streaming, retries (429 + Retry-After, 5xx,
transport), terminal 4xx, wall deadline that frees the lane only when the connection is closed, ledger fields, response
cache, allowlist, breaker, call_json validate/salvage, purity check."""
import importlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class Script:
    """Per-request behaviour, consumed in order; the last entry repeats."""

    def __init__(self):
        self.plan, self.seen, self.lock = [], [], threading.Lock()
        self.open_streams = 0
        self.max_open = 0

    def next(self):
        with self.lock:
            return self.plan.pop(0) if len(self.plan) > 1 else self.plan[0]


def _handler(script):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(n))
            with script.lock:
                script.seen.append(body)
            kind, arg = script.next()
            if kind == "status":
                code, extra = arg
                self.send_response(code)
                for k, v in extra.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(b'{"error": "x"}')
                return
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            with script.lock:
                script.open_streams += 1
                script.max_open = max(script.max_open, script.open_streams)
            try:
                if kind == "hang":
                    for _ in range(int(arg * 20)):
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        time.sleep(0.05)
                    return
                text, finish = arg
                for ch in [text[i:i + 3] for i in range(0, len(text), 3)]:
                    self.wfile.write(b"data: " + json.dumps({"model": body["model"], "choices": [
                        {"delta": {"content": ch}, "finish_reason": None}]}).encode() + b"\n\n")
                self.wfile.write(b"data: " + json.dumps({"model": body["model"], "choices": [
                    {"delta": {}, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": len(text)}}).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            finally:
                with script.lock:
                    script.open_streams -= 1
    return H


@pytest.fixture()
def llm(tmp_path, monkeypatch):
    script = Script()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(script))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("LOCAL_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1")
    monkeypatch.setenv("CS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("CS_RUNS", str(tmp_path / "runs"))
    from compilescholar.core import paths
    importlib.reload(paths)
    from compilescholar.llm import client as C
    importlib.reload(C)
    monkeypatch.setattr(C, "_backoff", lambda attempt, ra: time.sleep(0.01 if not ra else float(ra) / 100))
    C.configure({"providers": {"local": {"lanes": 2, "large_lanes": 1, "attempts": 3, "wall": 1.0,
                                         "read_timeout": 5, "breaker": 4}}},
                run_id="t", ledger_path=tmp_path / "ledger.jsonl")
    yield C, script, tmp_path / "ledger.jsonl"
    srv.shutdown()


def _ledger(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def test_stream_ok_and_ledger_fields(llm):
    C, s, led = llm
    s.plan = [("ok", ('{"a": 1}', "stop"))]
    assert C.call_local("hi", template="t1", item="p1") == '{"a": 1}'
    r = _ledger(led)[-1]
    assert r["ok"] and r["finish_reason"] == "stop" and r["served_model"] == "Qwen3.8-27B"
    assert r["template"] == "t1" and r["item"] == "p1" and r["completion_tokens"] == 8 and r["http_status"] == 200
    assert s.seen[-1]["stream"] is True


def test_retry_after_429_then_ok(llm):
    C, s, led = llm
    s.plan = [("status", (429, {"Retry-After": "1"})), ("status", (503, {})), ("ok", ("fine", "stop"))]
    assert C.call_local("x") == "fine"
    rows = _ledger(led)
    assert [r.get("http_status") for r in rows] == [429, 503, 200]
    assert [r["error_class"] for r in rows[:2]] == ["retryable", "retryable"]


def test_terminal_4xx_not_retried(llm):
    C, s, led = llm
    s.plan = [("status", (400, {})), ("ok", ("never", "stop"))]
    assert C.call_local("x") is None
    assert len(s.seen) == 1 and _ledger(led)[-1]["error_class"] == "terminal"


def test_wall_closes_connection_and_frees_lane(llm):
    C, s, led = llm
    s.plan = [("hang", 3.0)]
    t0 = time.monotonic()
    assert C.call_local("x", item="slow") is None
    took = time.monotonic() - t0
    assert took < 6                                   # 3 attempts x 1 s wall (+ backoff), not 3 x 3 s
    time.sleep(0.3)
    assert s.open_streams == 0                        # the server saw the disconnect: nothing keeps generating
    assert all(r["error_class"] == "retryable" and "wall" in r["error"] for r in _ledger(led))


def test_lane_bounds_concurrency(llm):
    C, s, led = llm
    s.plan = [("hang", 0.4)]
    C.configure({"providers": {"local": {"lanes": 2, "attempts": 1, "wall": 0.3, "breaker": 100}}},
                run_id="t", ledger_path=led)
    inner, live, peak, lk = C._stream, [0], [0], threading.Lock()

    def counted(*a, **k):                              # a connection is open exactly while _stream runs
        with lk:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        try:
            return inner(*a, **k)
        finally:
            with lk:
                live[0] -= 1

    C._stream = counted
    try:
        ts = [threading.Thread(target=C.call_local, args=("x",)) for _ in range(6)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    finally:
        C._stream = inner
    assert peak[0] == 2


def test_cache_replays_temperature_zero_only(llm):
    C, s, led = llm
    s.plan = [("ok", ("one", "stop")), ("ok", ("two", "stop"))]
    assert C.call_local("same") == "one"
    assert C.call_local("same") == "one"              # replayed, server not called
    assert len(s.seen) == 1 and _ledger(led)[-1]["cached"] is True
    assert C.call_local("same", temperature=0.7) == "two"   # sampled calls are fresh draws
    assert len(s.seen) == 2


def test_truncated_reply_not_cached(llm):
    C, s, led = llm
    s.plan = [("ok", ('{"a": [1, 2', "length")), ("ok", ('{"a": [1, 2]}', "stop"))]
    assert C.call_local("t") == '{"a": [1, 2'
    assert C.call_local("t") == '{"a": [1, 2]}'


def test_allowlist_blocks_before_sending(llm):
    C, s, led = llm
    C.configure({"allow": ["paratera"]}, run_id="t", ledger_path=led)
    s.plan = [("ok", ("x", "stop"))]
    assert C.call_local("x") is None and not s.seen
    assert _ledger(led)[-1]["error_class"] == "blocked"


def test_breaker_raises_channel_dead(llm):
    C, s, led = llm
    s.plan = [("status", (503, {}))]
    with pytest.raises(C.ChannelDead):
        for _ in range(5):
            C.call_local("x")


def test_call_json_validate_and_salvage(llm):
    C, s, led = llm
    s.plan = [("ok", ('{"pairs": "not a list"}', "stop")), ("ok", ('{"pairs": [1]}', "stop"))]
    obj = C.call_json("j", validate=lambda o: isinstance(o.get("pairs"), list))
    assert obj == {"pairs": [1]}                      # rejected first answer retried without the cache
    s.plan = [("ok", ('{"records": [{"kind": "a"}, {"kind": "b", "x": [1,', "length"))]
    obj = C.call_json("k", retries=0, salvage=("kind", "records"))
    assert obj == {"records": [{"kind": "a"}]}


def test_check_arm_purity(llm):
    C, s, led = llm
    s.plan = [("ok", ("x", "stop"))]
    C.call_local("a")
    C.call_local("b", model="Other-Model")
    r = C.check_arm_purity(led)
    assert not r["pure"] and r["violations"] == ["local/Other-Model"]
    r = C.check_arm_purity(led, allowed_pairs=[("local", "Qwen3.8-27B"), ("local", "Other-Model")], min_calls=3)
    assert not r["pure"] and "insufficient_calls" in r["violations"][0]


def test_thinking_off_payloads(llm):
    C, s, led = llm
    s.plan = [("ok", ("x", "stop"))]
    C.call_local("q", enable_thinking=False)
    assert s.seen[-1]["chat_template_kwargs"] == {"enable_thinking": False} and "thinking" not in s.seen[-1]


def test_mirror_route_deterministic_and_gated():
    """10-10 mega-run: call_local overflow routing — deterministic per item, model-gated, fraction-bound."""
    from compilescholar.llm import client as LC
    LC.configure({"mirror": {"provider": "paratera", "model": "Qwen3.8-27B",
                             "for_model": "Qwen3.8-27B", "fraction": 0.3}})
    try:
        routes = [LC._mirror_route("Qwen3.8-27B", f"item:{i}") for i in range(400)]
        assert routes == [LC._mirror_route("Qwen3.8-27B", f"item:{i}") for i in range(400)]   # deterministic
        n_mirror = sum(1 for p, _ in routes if p == "paratera")
        assert 80 < n_mirror < 160, n_mirror                                                  # ~30% +- band
        assert all(m == "Qwen3.8-27B" for p, m in routes if p == "paratera")
        assert LC._mirror_route("some-other-model", "x") == ("local", "some-other-model")     # model-gated
        LC.configure({"mirror": {"provider": "paratera", "fraction": 0.0}})
        assert all(LC._mirror_route("Qwen3.8-27B", f"i{i}") == ("local", "Qwen3.8-27B") for i in range(50))
        LC.configure({})
        assert LC._mirror_route("Qwen3.8-27B", "x") == ("local", "Qwen3.8-27B")               # no mirror config
    finally:
        LC.configure({})
