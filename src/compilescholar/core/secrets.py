# -*- coding: utf-8 -*-
"""The single reader of the repository .env file.

Precedence: process environment first, then .env (the reverse of the old kb_infra.llm behaviour, so a command line
can override a key for one run). Values are never logged.
"""
from __future__ import annotations

import os
from functools import lru_cache

from . import paths


@lru_cache(maxsize=1)
def _dotenv() -> dict[str, str]:
    p = paths.env_file()
    out: dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def get(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v not in (None, ""):
        return v
    v = _dotenv().get(name)
    return v if v not in (None, "") else default
