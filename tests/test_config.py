# -*- coding: utf-8 -*-
"""configs/bench/cs2_test_v9b.yaml must resolve to exactly the switches recorded for the frozen v9b test run."""
from __future__ import annotations

import json

import pytest

from compilescholar.core import config, paths

V9B_RUN_CONFIG = paths.legacy_bench() / "cs2" / "arm_vnext" / "config_test100_r1.json"
FREEZE = paths.legacy_bench() / "cs2" / "FREEZE_CS2_TEST_1003_v9b.json"


def test_v9b_config_matches_recorded_run():
    c = config.load(paths.REPO / "configs" / "bench" / "cs2_test_v9b.yaml")
    rec = json.load(open(V9B_RUN_CONFIG, encoding="utf-8"))
    mine = {"split": c.bench.split, "offset": c.bench.offset, "limit": c.bench.limit, "kb": c.answer.kb,
            "ext": c.answer.ext, "cite": c.answer.cite, "screen": c.answer.screen, "state": c.answer.state,
            "probe": c.answer.probe, "word_budget": c.answer.word_budget, "model": c.answer.model,
            "cutoff": c.answer.cutoff}
    assert mine == rec
    assert c.runtime.workers == 4
    assert c.kb_path() == paths.legacy_bench() / "cs2" / "base_kb_v2"


def test_v9b_config_matches_freeze_record():
    c = config.load(paths.REPO / "configs" / "bench" / "cs2_test_v9b.yaml")
    fz = json.load(open(FREEZE, encoding="utf-8"))["config"]
    for k, v in {"word_budget": c.answer.word_budget, "cutoff": c.answer.cutoff, "model": c.answer.model}.items():
        if k in fz:
            assert fz[k] == v, k


def test_overrides_and_unknown_keys():
    c = config.load(paths.REPO / "configs" / "bench" / "cs2_test_v9b.yaml", ["answer.cite=false", "bench.limit=5"])
    assert c.answer.cite is False and c.bench.limit == 5
    with pytest.raises(ValueError):
        config.load(None, ["answer.nonexistent=1"])
    assert config.load().sha256() == config.load().sha256()
