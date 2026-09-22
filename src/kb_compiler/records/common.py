# -*- coding: utf-8 -*-
"""Pipeline common: corpus loading, model routing, JSON call wrapper, caching.

Model routing convention: "provider:model" — provider ∈ {paratera, cst,
local, intern}; bare model name defaults to paratera. "intern" = INTERN
free channel (intern:qwen3.8-27b; its thinking-disabled form is provider-
specific and handled inside call_intern). "local" = self-hosted vLLM
(LOCAL_BASE_URL, e.g. local:qwen3.8-27b-local). enable_thinking=False is
ALWAYS sent on paratera (kb_infra.llm dispatches the correct param form
per model family).
"""
from __future__ import annotations

import json
import os
import sys

_REPO_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from kb_infra.llm import (call_paratera, call_cst, call_local,  # noqa: E402
                          call_intern, parse_json_response)

MAX_PAPER_CHARS = 110_000  # stage-A proven cap for single-call full-text passes

# Chunk-based extraction cap (slot deep-extraction). Was MAX_PAPER_CHARS —
# an API-billing-era cost bound that silently truncated 47/430 Multi corpus
# texts (GPT-3 33%, Llama-3 Herd 27%, LSST book 6% covered). User-approved
# 2026-09-21: free local/intern channels remove the cost rationale; 500k
# covers every corpus text except the LSST book (disclosed separately).
# Single-call passes (skeleton card, absence) KEEP MAX_PAPER_CHARS — that is
# their physical context budget, not a cost knob.
SLOT_MAX_CHARS = int(os.environ.get("SLOT_MAX_CHARS", "500000"))


def route_model(spec: str):
    """'provider:model' or bare model -> (provider, model)."""
    if ":" in spec:
        prov, model = spec.split(":", 1)
        return prov, model
    return "paratera", spec


def salvage_json_records(raw: str, item_key: str = "kind",
                         wrapper_key: str = "records"):
    """Truncation salvage (deterministic): extract complete brace-balanced
    objects containing item_key from an unparseable/truncated response.
    Measured need 2026-09-05: DSF memory-blurt on a famous paper produced
    37k chars, truncated mid-JSON, whole chunk lost without salvage.
    item_key/wrapper_key generalize the original records-shape ("kind" ->
    "records") to other contracts, e.g. round2 mapping assignments
    ("i" -> "assignments") — 2026-09-23 run5 forensics: salvage silently
    returned nothing for assignment-shaped output because it only matched
    "kind"."""
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
                if f'"{item_key}"' not in blob:
                    continue
                try:
                    o = json.loads(blob)
                except Exception:
                    continue
                if isinstance(o, dict) and o.get(item_key) is not None:
                    objs.append(o)
    return {wrapper_key: objs} if objs else None


def _dump_raw(path: str, attempt: int, raw: str | None, cap: int = 60000):
    """Forensic capture (append-only): full raw output of a response that
    failed direct parse — for post-hoc contract analysis. 2026-09-23 lesson:
    the run5 debug capture truncated at 2000 chars and the visible head looked
    like clean JSON, hiding the real tail defect. Per-dump cap bounds growth;
    hard file cap 5MB stops unbounded append on salvage-recovered runs."""
    try:
        if os.path.exists(path) and os.path.getsize(path) > 5 * 1024 * 1024:
            return
        import time as _time
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"=== attempt {attempt} ts={_time.strftime('%m-%d %H:%M:%S')} "
                    f"len={len(raw or '')} ===\n")
            f.write((raw or "")[:cap] + "\n")
    except OSError:
        pass


def call_json(prompt: str, model_spec: str, max_tokens: int = 8000,
              retries: int = 3, salvage: bool = False,
              salvage_key: str = "kind", salvage_wrapper: str = "records",
              fail_dump: str | None = None):
    """LLM call -> parsed JSON (balanced-extract parser) or None. Retries on
    empty/unparseable. salvage=True adds the truncation-salvage tier
    (item_key/wrapper_key select the contract shape). fail_dump path gets
    every raw that failed direct parse (full capture, even when salvage
    recovers — diagnosis without blocking progress). Never raises."""
    prov, model = route_model(model_spec)
    fn = {"cst": call_cst, "local": call_local,
          "intern": call_intern}.get(prov, call_paratera)
    for _att in range(retries):
        raw = fn(prompt, model=model, max_tokens=max_tokens,
                 temperature=0.0, enable_thinking=False)
        obj = parse_json_response(raw)
        if obj is not None:
            return obj
        if fail_dump:
            _dump_raw(fail_dump, _att, raw)
        if salvage:
            obj = salvage_json_records(raw, salvage_key, salvage_wrapper)
            if obj:
                return obj
    return None


BLOCK_WORKERS = int(os.environ.get("REGISTRY_BLOCK_WORKERS", "6"))


def par_map(fn, items: list, workers: int | None = None) -> list:
    """Order-preserving parallel map over independent LLM block calls
    (call_local is thread-safe; _LOCAL_SEM bounds server-side concurrency).
    Deterministic: results always in item order regardless of timing."""
    w = BLOCK_WORKERS if workers is None else workers
    if w <= 1 or len(items) <= 1:
        return [fn(it) for it in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(w, len(items))) as ex:
        return list(ex.map(fn, items))


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
