"""P2-4 regression: ledger rows carry caller + prompt_hash (+ embed rows)."""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kb_infra import llm  # noqa: E402


def test_log_call_fields(tmp_path):
    log = tmp_path / "ledger.jsonl"
    old = (os.environ.get("LLM_CALL_LOG"), os.environ.get("LLM_CALLER"))
    try:
        os.environ["LLM_CALL_LOG"] = str(log)
        os.environ["LLM_CALLER"] = "unit-test-stage"
        llm._log_call("local", "Qwen3.8-27B", True, 123.4, {"prompt_tokens": 10},
                      payload={"model": "Qwen3.8-27B",
                               "messages": [{"role": "user", "content": "hi"}]},
                      extra={"n_items": 1})
        rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        r = rows[0]
        assert r["caller"] == "unit-test-stage"
        assert r["prompt_tokens"] == 10
        # stable 12-char hash, identical payload -> identical hash
        h1 = r["prompt_hash"]
        assert len(h1) == 12
        assert llm._phash({"model": "Qwen3.8-27B",
                           "messages": [{"role": "user", "content": "hi"}]}) == h1
        # no payload -> no hash key (old rows stay shape-compatible)
        llm._log_call("local", "Qwen3.8-27B", False, 5.0)
        rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
        assert "prompt_hash" not in rows[1]
        assert rows[1]["caller"] == "unit-test-stage"
    finally:
        os.environ["LLM_CALL_LOG"] = old[0] or ""
        if old[0] is None:
            os.environ.pop("LLM_CALL_LOG", None)
        os.environ["LLM_CALLER"] = old[1] or ""
        if old[1] is None:
            os.environ.pop("LLM_CALLER", None)


def test_phash_deterministic_and_scoped():
    assert llm._phash(None) == ""
    assert llm._phash({}) == ""
    a = llm._phash({"input": ["x", "y"]})
    b = llm._phash({"input": ["x", "y"]})
    c = llm._phash({"input": ["y", "x"]})
    assert a == b and a != c and len(a) == 12
