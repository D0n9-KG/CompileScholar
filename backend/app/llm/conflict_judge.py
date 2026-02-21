from __future__ import annotations

import json
import re
from typing import Any

from json_repair import repair_json


_TPL_RE = re.compile(r"\{\{\s*([A-Za-z][A-Za-z0-9_]*)\s*\}\}")
_JSON_BLOCK_RE = re.compile(r"```json\s*(?P<body>.*?)\s*```", re.DOTALL | re.IGNORECASE)
_ALLOWED_LABELS = {"contradict", "not_conflict", "insufficient"}


def _render_template(template: str, vars: dict[str, Any]) -> str:
    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        value = vars.get(key)
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    return _TPL_RE.sub(_sub, template or "")


def _rule_int(rules: dict[str, Any], key: str, default: int, *, lo: int, hi: int) -> int:
    try:
        value = int(rules.get(key, default))
    except Exception:
        value = int(default)
    return max(lo, min(hi, value))


def _repair_and_parse(raw_response: str) -> dict[str, Any]:
    """
    Repair and parse potentially malformed JSON using json-repair.

    Extracts JSON from code blocks and attempts repair if direct parsing fails.
    Returns empty dict on failure.
    """
    text = (raw_response or "").strip()
    if not text:
        return {}

    # Try to extract JSON from markdown code block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        text = m.group("body").strip()

    # Try to locate first { ... } if extra text exists
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]

    # First attempt: direct parse (fast path)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    # Second attempt: use json-repair (slow path for malformed JSON)
    try:
        repaired = repair_json(text)
        parsed = json.loads(repaired)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _mark_batch_insufficient(batch_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Mark all pairs in a batch as 'insufficient' when LLM call fails.

    This provides graceful degradation instead of losing all data.
    """
    return [
        {
            "pair_id": pair.get("pair_id", f"unknown_{idx}"),
            "label": "insufficient",
            "score": 0.0,
            "reason": "LLM processing failed for this batch",
        }
        for idx, pair in enumerate(batch_pairs)
    ]


def _judge_single_batch(
    *,
    batch_pairs: list[dict[str, Any]],
    system: str,
    user_template: str,
    default_user_fmt: str,
) -> list[dict[str, Any]]:
    """
    Judge a single batch of conflict pairs with JSON repair.

    Retry logic is delegated to call_text() at client layer.
    This avoids retry amplification across multiple layers.

    Args:
        batch_pairs: List of pair dicts to judge
        system: System prompt
        user_template: User prompt template (optional)
        default_user_fmt: Default user prompt format string

    Returns:
        List of judgment dicts with pair_id, label, score, reason
    """
    from app.llm.client import call_text

    if not batch_pairs:
        return []

    # Prepare user prompt
    if user_template:
        user = _render_template(
            user_template,
            {
                "pairs_json": json.dumps({"pairs": batch_pairs}, ensure_ascii=False),
                "pair_count": len(batch_pairs),
            },
        )
    else:
        user = default_user_fmt.format(pairs_json=json.dumps({"pairs": batch_pairs}, ensure_ascii=False))

    # Call LLM with retry at client layer
    try:
        raw = call_text(system, user, use_retry=True)
    except Exception:
        return _mark_batch_insufficient(batch_pairs)

    # Parse and repair JSON if needed
    out = _repair_and_parse(raw)

    # Validate with Pydantic (best-effort, fallback to raw dict)
    try:
        from app.llm.schemas import ConflictJudgeResponse
        validated = ConflictJudgeResponse.model_validate(out)
        out = validated.model_dump()
    except Exception:
        pass  # proceed with raw repaired dict

    rows = out.get("items") if isinstance(out, dict) else []
    if not isinstance(rows, list):
        return _mark_batch_insufficient(batch_pairs)

    # Parse and validate results
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        pair_id = str(row.get("pair_id") or "").strip()
        if not pair_id:
            continue
        label = str(row.get("label") or "").strip().lower()
        if label not in _ALLOWED_LABELS:
            label = "insufficient"
        try:
            score = float(row.get("score"))
        except Exception:
            score = 0.0
        score = max(0.0, min(1.0, score))
        reason = str(row.get("reason") or "").strip()
        result.append(
            {
                "pair_id": pair_id,
                "label": label,
                "score": score,
                "reason": reason,
            }
        )

    if not result:
        return _mark_batch_insufficient(batch_pairs)
    return result


def judge_conflict_pairs_batch(*, pairs: list[dict[str, Any]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Judge conflict pairs in small batches with retry and error tolerance.

    Improvements over original:
    - Small batch size (15 pairs) for better LLM performance
    - Retry with exponential backoff
    - JSON repair for malformed responses
    - Graceful degradation (mark as 'insufficient' on failure)
    """
    if not pairs:
        return []

    rules = dict(schema.get("rules") or {})
    prompts = dict(schema.get("prompts") or {})

    # Configuration
    max_pairs = _rule_int(rules, "phase2_conflict_candidate_max_pairs", 120, lo=1, hi=2000)
    batch_size = _rule_int(rules, "phase2_conflict_batch_size", 15, lo=5, hi=50)
    safe_pairs = list(pairs[:max_pairs])

    # Prompts
    default_system = (
        "You are a scientific claim contradiction judge.\n"
        "Return STRICT JSON only.\n"
        "For each pair, classify semantic relation:\n"
        "- contradict: cannot both be true under comparable conditions.\n"
        "- not_conflict: compatible or discussing different contexts.\n"
        "- insufficient: evidence insufficient to decide.\n"
        "Use score in [0,1] as confidence."
    )
    default_user_fmt = (
        "Judge contradiction for each pair:\n"
        "{pairs_json}\n\n"
        "Output JSON schema:\n"
        '{{ "items": [ {{"pair_id":"p1","label":"contradict","score":0.0,"reason":"..."}} ] }}'
    )

    system = str(prompts.get("phase2_conflict_judge_system") or "").strip() or default_system
    user_template = str(prompts.get("phase2_conflict_judge_user_template") or "").strip()

    # Process in small batches
    all_results: list[dict[str, Any]] = []
    for i in range(0, len(safe_pairs), batch_size):
        batch = safe_pairs[i : i + batch_size]
        batch_results = _judge_single_batch(
            batch_pairs=batch,
            system=system,
            user_template=user_template,
            default_user_fmt=default_user_fmt,
        )
        all_results.extend(batch_results)

    return all_results
