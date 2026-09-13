# -*- coding: utf-8 -*-
"""Pipeline common: corpus loading, model routing, JSON call wrapper, caching.

Model routing convention: "provider:model" — provider ∈ {paratera, cst};
bare model name defaults to paratera. enable_thinking=False is ALWAYS sent
on paratera (kb_infra.llm dispatches the correct param form per model family).
"""
from __future__ import annotations

import json
import os
import sys

_REPO_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from kb_infra.llm import call_paratera, call_cst, parse_json_response  # noqa: E402

MAX_PAPER_CHARS = 110_000  # stage-A proven cap for single-call full-text passes


def route_model(spec: str):
    """'provider:model' or bare model -> (provider, model)."""
    if ":" in spec:
        prov, model = spec.split(":", 1)
        return prov, model
    return "paratera", spec


def salvage_json_records(raw: str):
    """Truncation salvage (deterministic): extract complete brace-balanced
    objects containing "kind" from an unparseable/truncated response.
    Measured need 2026-09-05: DSF memory-blurt on a famous paper produced
    37k chars, truncated mid-JSON, whole chunk lost without salvage."""
    if not raw:
        return None
    objs, depth, in_str, esc = [], 0, False, False
    stack = []  # (start_idx, open_depth)
    for i, ch in enumerate(raw):
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
        elif ch == "{":
            stack.append((i, depth))
            depth += 1
        elif ch == "}":
            if depth > 0 and stack:
                start, open_depth = stack.pop()
                depth -= 1
                # candidates: records nested in the wrapper (open_depth==1)
                # or bare-array/top-level records (open_depth==0). A complete
                # wrapper object parses but has no "kind" -> filtered below.
                if open_depth > 1:
                    continue
                blob = raw[start:i + 1]
                if '"kind"' not in blob:
                    continue
                try:
                    o = json.loads(blob)
                except Exception:
                    continue
                if isinstance(o, dict) and o.get("kind"):
                    objs.append(o)
    return {"records": objs} if objs else None


def call_json(prompt: str, model_spec: str, max_tokens: int = 8000,
              retries: int = 3, salvage: bool = False):
    """LLM call -> parsed JSON (balanced-extract parser) or None. Retries on
    empty/unparseable. salvage=True adds the truncation-salvage tier.
    Never raises."""
    prov, model = route_model(model_spec)
    fn = call_cst if prov == "cst" else call_paratera
    for _ in range(retries):
        raw = fn(prompt, model=model, max_tokens=max_tokens,
                 temperature=0.0, enable_thinking=False)
        obj = parse_json_response(raw)
        if obj is not None:
            return obj
        if salvage:
            obj = salvage_json_records(raw)
            if obj:
                return obj
    return None


def load_corpus(texts_dir: str) -> list[tuple[str, str]]:
    """[(paper_id, text)] sorted; .md/.txt only (logs etc. excluded)."""
    out = []
    for fn in sorted(os.listdir(texts_dir)):
        if fn.endswith((".md", ".txt")):
            pid = fn.rsplit(".", 1)[0]
            with open(os.path.join(texts_dir, fn), encoding="utf-8",
                      errors="replace") as f:
                out.append((pid, f.read()))
    return out


def load_manifest(path: str) -> dict[str, dict]:
    recs = json.load(open(path, encoding="utf-8"))
    return {r["paper_id"]: r for r in recs}


def load_json(path: str, default=None):
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return default


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
