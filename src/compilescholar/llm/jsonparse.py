# -*- coding: utf-8 -*-
"""Lenient JSON extraction from LLM output (moved verbatim from kb_infra.llm: first balanced object/array,
control-character repair inside strings, greedy-regex last resort)."""
from __future__ import annotations

import json
import re
from typing import Any


def _repair_control_chars_in_strings(obj_text: str) -> str:
    """Escape literal newlines/tabs inside JSON string values (2026-09-22
    registry_growth forensics: Qwen3.8-27B occasionally emits raw newlines
    inside string values — invalid JSON that json.loads rejects. Three
    consecutive ChannelDead aborts traced to exactly this; the outputs were
    balanced and complete, just carrying control chars in strings)."""
    out = []
    in_str = False
    esc = False
    for c in obj_text:
        if esc:
            out.append(c)
            esc = False
            continue
        if c == chr(92):  # backslash
            out.append(c)
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            out.append(c)
            continue
        if in_str and c == chr(10):
            out.append(chr(92) + "n")
            continue
        if in_str and c == chr(9):
            out.append(chr(92) + "t")
            continue
        out.append(c)
    return "".join(out)


_VALID_ESC = set('"\\/bfnrt')


def repair_json_escapes(s: str) -> tuple[str, int]:
    r"""Double every invalid backslash escape (moved from kb_compiler.records.cards). LaTeX copied verbatim into a JSON
    string ('n^{\gg}', '\underbrace') makes the whole object unparseable; valid pairs (including an escaped
    backslash followed by a letter) are consumed untouched — a regex lookahead breaks those. Returns (fixed, n)."""
    out, i, n, nfix = [], 0, len(s), 0
    while i < n:
        c = s[i]
        if c == "\\":
            if i + 1 < n:
                nx = s[i + 1]
                if nx in _VALID_ESC:
                    out.append(c)
                    out.append(nx)
                    i += 2
                    continue
                if nx == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", s[i + 2:i + 6] or ""):
                    out.append(s[i:i + 6])
                    i += 6
                    continue
            out.append("\\\\")
            nfix += 1
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out), nfix


def parse_json_response(text: str | None) -> Any:
    r"""Parse JSON from LLM response, handling markdown fences and extra text.

    Long-chunk fix: the old greedy `(\[.*\]|\{.*\})` matched the FIRST `{` to
    the LAST `}` — on long LLM outputs (which drift into prose with stray
    braces) it grabbed a huge span full of non-JSON → json.loads failed → 0
    nodes/edges parsed (a whole chunk lost). This now extracts the FIRST
    BALANCED JSON object/array (counting brace depth, respecting strings), so
    even if the LLM wraps JSON in prose, the first real JSON object is found."""
    if not text:
        return None
    text = re.sub(r"```json|```", "", text, flags=re.S)
    # find the first balanced { ... } or [ ... ] (string-aware, depth-counted)
    obj = _extract_first_json(text)
    if obj is not None:
        try:
            return json.loads(obj)
        except Exception:
            pass
        # repair tier: raw newlines/tabs inside string values (LLM-emitted
        # invalid JSON — balanced but with control chars)
        try:
            return json.loads(_repair_control_chars_in_strings(obj))
        except Exception:
            pass
        # repair tier 2: invalid backslash escapes (LaTeX copied verbatim into a string)
        try:
            return json.loads(_repair_control_chars_in_strings(repair_json_escapes(obj)[0]))
        except Exception:
            pass
    # fallback: old greedy regex (last resort, may fail on long prose)
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _extract_first_json(text: str) -> str | None:
    """Extract the first balanced JSON object/array from text (string-aware,
    depth-counted). Returns the substring or None. Handles LLM output that
    drifts into prose with stray braces — grabs only the first complete {…} or
    […] block, not a greedy first-{ to last-} span."""
    start = -1
    depth = 0
    in_str = False
    esc = False
    open_ch = ""
    close_ch = ""
    for i, ch in enumerate(text):
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
            continue
        if ch in "{[":
            if depth == 0:
                start = i
                open_ch = ch
                close_ch = "}" if ch == "{" else "]"
            depth += 1
        elif ch in "}]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = text[start:i + 1]
                    # only return if the closing matches the opening type
                    if ch == close_ch:
                        return candidate
                    # mismatched (e.g. opened { closed ]) — reset, keep scanning
                    start = -1
                    open_ch = ""
                    close_ch = ""
    return None


def salvage_json_records(raw: str | None, item_key: str = "kind", wrapper_key: str = "records"):
    """Truncation salvage (moved from kb_compiler.records.common): complete brace-balanced objects containing
    `item_key`, found at the top level or one level inside a wrapper, from an unparseable or truncated reply.
    Returns {wrapper_key: [objects]} or None. Measured need: a 37k-char reply truncated mid-JSON lost a whole chunk."""
    if not raw:
        return None
    objs, depth, in_str, esc = [], 0, False, False
    stack = []
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
                if open_depth > 1:
                    continue
                blob = raw[start:i + 1]
                if f'"{item_key}"' not in blob:
                    continue
                try:
                    o = json.loads(blob)
                except ValueError:
                    continue
                if isinstance(o, dict) and o.get(item_key) is not None:
                    objs.append(o)
    return {wrapper_key: objs} if objs else None
