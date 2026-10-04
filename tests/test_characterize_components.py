# -*- coding: utf-8 -*-
"""New Sciverse client, harness proxy + MCP server, field-state compiler and GPTR adapter vs goldens generated from
the old code before it moved to legacy/ (tests/fixtures/characterize/make_goldens_components.py)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import _characterize_impl as C

GOLD = json.load(open(C.FIX / "goldens_components.json", encoding="utf-8"))


@pytest.fixture(scope="module")
def new():
    code = (f"import sys, json; sys.path.insert(0, {str(C.REPO / 'tests')!r}); sys.path.insert(0, {str(C.FIX)!r}); "
            "import make_goldens_components as M; "
            "print(json.dumps(M.compute('new'), ensure_ascii=False, sort_keys=True, default=str))")
    env = {**os.environ, "PYTHONUTF8": "1", "LLM_PROVIDER_ALLOWLIST": "none"}
    p = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, encoding="utf-8",
                       env=env, timeout=600, cwd=str(C.REPO))
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("key", sorted(GOLD))
def test_component_matches_golden(new, key):
    assert C.canon(new[key]) == C.canon(GOLD[key])


def test_goldens_exercise_the_interesting_paths():
    fs = GOLD["field_state"]["batched_merge"]["prompts"]
    assert any(p.startswith("Below are method families proposed separately") for p in fs)
    assert any(c["cands"] and c["cands"][0][2] == "failed" for k, c in GOLD["sciverse"].items() if k.endswith("error"))
