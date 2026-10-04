# -*- coding: utf-8 -*-
"""Prompt templates as files (byte-identical to the strings in the original answer_pipeline.py). Loaded once; the
sha256 of each template goes into every run manifest."""
from __future__ import annotations

import hashlib
from importlib import resources


def _load(name: str) -> str:
    return resources.files(__package__).joinpath(f"{name}.txt").read_bytes().decode("utf-8")


PLAN = _load("plan")
PROBE_BLOCK = _load("probe_block")
WRITE = _load("write")
SCREEN = _load("screen")


def hashes() -> dict[str, str]:
    return {n: hashlib.sha256(s.encode("utf-8")).hexdigest() for n, s in
            (("plan", PLAN), ("probe_block", PROBE_BLOCK), ("write", WRITE), ("screen", SCREEN))}
