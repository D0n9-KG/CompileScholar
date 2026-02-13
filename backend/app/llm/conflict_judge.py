from __future__ import annotations

import json
import re
from typing import Any


_TPL_RE = re.compile(r"\{\{\s*([A-Za-z][A-Za-z0-9_]*)\s*\}\}")
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


def judge_conflict_pairs_batch(*, pairs: list[dict[str, Any]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    if not pairs:
        return []

    from app.llm.client import call_json

    rules = dict(schema.get("rules") or {})
    prompts = dict(schema.get("prompts") or {})
    max_pairs = _rule_int(rules, "phase2_conflict_candidate_max_pairs", 120, lo=1, hi=2000)
    safe_pairs = list(pairs[:max_pairs])

    default_system = (
        "You are a scientific claim contradiction judge.\n"
        "Return STRICT JSON only.\n"
        "For each pair, classify semantic relation:\n"
        "- contradict: cannot both be true under comparable conditions.\n"
        "- not_conflict: compatible or discussing different contexts.\n"
        "- insufficient: evidence insufficient to decide.\n"
        "Use score in [0,1] as confidence."
    )
    default_user = (
        "Judge contradiction for each pair:\n"
        + json.dumps({"pairs": safe_pairs}, ensure_ascii=False)
        + "\n\nOutput JSON schema:\n"
        '{ "items": [ {"pair_id":"p1","label":"contradict","score":0.0,"reason":"..."} ] }'
    )

    system = str(prompts.get("phase2_conflict_judge_system") or "").strip() or default_system
    user_template = str(prompts.get("phase2_conflict_judge_user_template") or "").strip()
    if user_template:
        user = _render_template(
            user_template,
            {
                "pairs_json": json.dumps({"pairs": safe_pairs}, ensure_ascii=False),
                "pair_count": len(safe_pairs),
            },
        )
    else:
        user = default_user

    out = call_json(system, user)
    rows = out.get("items") or []
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
    return result
