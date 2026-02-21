from __future__ import annotations

import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)

_TPL_RE = re.compile(r"\{\{\s*([A-Za-z][A-Za-z0-9_]*)\s*\}\}")
_ALLOWED_LABELS = {"supported", "weak", "unsupported", "contradicted"}


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


def _rule_float(rules: dict[str, Any], key: str, default: float, *, lo: float, hi: float) -> float:
    try:
        value = float(rules.get(key, default))
    except Exception:
        value = float(default)
    return max(lo, min(hi, value))


def _rule_int(rules: dict[str, Any], key: str, default: int) -> int:
    try:
        return max(1, int(rules.get(key, default)))
    except Exception:
        return max(1, int(default))


def _judge_one_batch(
    *,
    batch_items: list[dict[str, Any]],
    system: str,
    user_template: str,
    supported_min: float,
    weak_min: float,
) -> tuple[list[dict[str, Any]], bool]:
    """Judge a single batch. Returns (results, fallback_used)."""
    from app.llm.client import call_json, call_validated_json

    default_user = (
        "Grounding judgment payload:\n"
        + json.dumps({"items": batch_items}, ensure_ascii=False)
        + "\n\nOutput JSON schema:\n"
        '{ "items": [ {"canonical_claim_id":"...","label":"supported","score":0.0,"reason":"..."} ] }'
    )
    if user_template:
        user = _render_template(
            user_template,
            {
                "items_json": json.dumps({"items": batch_items}, ensure_ascii=False),
                "supported_min": supported_min,
                "weak_min": weak_min,
            },
        )
    else:
        user = default_user

    fallback_used = False
    try:
        from app.llm.schemas import GroundingJudgeResponse
        validated = call_validated_json(system, user, GroundingJudgeResponse)
        out = validated.model_dump()
    except Exception:
        try:
            out = call_json(system, user)
        except Exception:
            logger.warning("Grounding batch LLM call failed for %d claims", len(batch_items), exc_info=True)
            fallback_used = True
            return [], fallback_used

    rows = out.get("items") or []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        claim_id = str(row.get("canonical_claim_id") or "").strip()
        if not claim_id:
            continue
        label_raw = str(row.get("label") or "").strip().lower()
        if label_raw not in _ALLOWED_LABELS:
            label_raw = "unsupported"
        try:
            score = float(row.get("score"))
        except Exception:
            score = 0.0
        score = max(0.0, min(1.0, score))
        if label_raw == "contradicted":
            label = "unsupported"
        elif label_raw == "supported":
            label = "supported" if score >= supported_min else ("weak" if score >= weak_min else "unsupported")
        elif label_raw == "weak":
            label = "weak" if score >= weak_min else "unsupported"
        else:
            label = "unsupported"
        result.append(
            {
                "canonical_claim_id": claim_id,
                "support_label": label,
                "judge_score": score,
                "reason": str(row.get("reason") or "").strip() or f"semantic:{label_raw}",
                "judge_mode": "semantic",
            }
        )
    return result, fallback_used


def judge_claim_support_batch(
    *,
    claims: list[dict[str, Any]],
    chunk_by_id: dict[str, str],
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    if not claims:
        return []

    rules = dict(schema.get("rules") or {})
    prompts = dict(schema.get("prompts") or {})
    supported_min = _rule_float(rules, "phase1_grounding_semantic_supported_min", 0.75, lo=0.0, hi=1.0)
    weak_min = _rule_float(rules, "phase1_grounding_semantic_weak_min", 0.55, lo=0.0, hi=1.0)
    if weak_min > supported_min:
        weak_min = supported_min
    batch_size = _rule_int(rules, "phase1_grounding_batch_size", 20)

    evidence_top_k = _rule_int(rules, "phase1_grounding_evidence_top_k", 3)
    evidence_chunk_chars_max = _rule_int(rules, "phase1_grounding_evidence_chunk_chars_max", 2000)

    payload_items: list[dict[str, Any]] = []
    for claim in claims:
        claim_id = str(claim.get("canonical_claim_id") or claim.get("claim_id") or "").strip()
        if not claim_id:
            continue
        # Collect all origin chunk IDs (plural preferred, singular fallback)
        chunk_ids: list[str] = []
        for cid in claim.get("origin_chunk_ids") or []:
            s = str(cid).strip()
            if s and s not in chunk_ids:
                chunk_ids.append(s)
        if not chunk_ids:
            s = str(claim.get("origin_chunk_id") or "").strip()
            if s:
                chunk_ids.append(s)
        # Build evidence texts for top-k chunks
        evidence_texts: list[str] = []
        for cid in chunk_ids[:evidence_top_k]:
            txt = str(chunk_by_id.get(cid) or "").strip()
            if txt:
                evidence_texts.append(txt[:evidence_chunk_chars_max])
        # Backward-compatible: single chunk_text for single-chunk claims
        combined_text = "\n---\n".join(evidence_texts) if evidence_texts else ""
        payload_items.append(
            {
                "canonical_claim_id": claim_id,
                "claim_text": str(claim.get("text") or ""),
                "origin_chunk_id": chunk_ids[0] if chunk_ids else "",
                "origin_chunk_ids": chunk_ids[:evidence_top_k],
                "chunk_text": combined_text,
                "evidence_count": len(evidence_texts),
            }
        )
    if not payload_items:
        return []

    default_system = (
        "You are a scientific grounding judge.\n"
        "Return STRICT JSON only.\n"
        "For each claim, compare with the provided chunk text(s) and classify support.\n"
        "When multiple evidence chunks are provided (separated by ---), consider ALL of them jointly.\n"
        "Labels:\n"
        "- supported: claim is directly supported by the evidence.\n"
        "- weak: partially supported / ambiguous.\n"
        "- unsupported: not supported by any evidence.\n"
        "- contradicted: evidence contradicts claim.\n"
        "Provide score in [0,1] and a short reason."
    )
    system = str(prompts.get("phase1_grounding_judge_system") or "").strip() or default_system
    user_template = str(prompts.get("phase1_grounding_judge_user_template") or "").strip()

    # Split into batches
    all_results: list[dict[str, Any]] = []
    fallback_count = 0
    total_batches = (len(payload_items) + batch_size - 1) // batch_size
    for i in range(0, len(payload_items), batch_size):
        batch = payload_items[i : i + batch_size]
        batch_results, fallback_used = _judge_one_batch(
            batch_items=batch,
            system=system,
            user_template=user_template,
            supported_min=supported_min,
            weak_min=weak_min,
        )
        all_results.extend(batch_results)
        if fallback_used:
            fallback_count += 1

    if fallback_count > 0:
        logger.warning(
            "Grounding: %d/%d batches failed (fallback), %d claims judged out of %d",
            fallback_count, total_batches, len(all_results), len(payload_items),
        )

    return all_results
