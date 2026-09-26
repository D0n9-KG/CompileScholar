from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from retrieval._see_extraction import CandidateCase
from retrieval._see_llm import LlmConfig, create_openai_client
from retrieval._see_models import EvidenceUnit


class CandidateGenerator(Protocol):
    def __call__(
        self,
        *,
        paper_title: str,
        evidence_units: list[EvidenceUnit],
        llm_config: LlmConfig,
    ) -> "GeneratedCases":
        ...


@dataclass(frozen=True)
class GeneratedCases:
    candidates: list[CandidateCase]
    trace: dict[str, Any]


def generate_candidate_cases(
    *,
    paper_title: str,
    evidence_units: list[EvidenceUnit],
    llm_config: LlmConfig,
) -> GeneratedCases:
    prompt = build_candidate_prompt(
        paper_title=paper_title,
        evidence_units=evidence_units,
    )
    client = create_openai_client(llm_config)
    response = client.chat.completions.create(
        model=llm_config.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract scientific evolution cases from provided evidence only. "
                    "Return a JSON object. Do not invent evidence ids."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    candidates = parse_candidate_response(content)
    return GeneratedCases(
        candidates=candidates,
        trace={
            **llm_config.trace_metadata(),
            "status": "generated",
            "candidate_count": len(candidates),
            "response_id": getattr(response, "id", None),
        },
    )


def build_candidate_prompt(
    *,
    paper_title: str,
    evidence_units: list[EvidenceUnit],
) -> str:
    evidence_json = [
        {
            "evidence_id": item.evidence_id,
            "text": item.text[:1200],
            "source": item.source.model_dump(),
        }
        for item in evidence_units[:12]
    ]
    return (
        "Given the paper title and evidence units, produce at most one scientific evolution case.\n"
        "The case must describe a concrete shift from a prior state to a new state.\n"
        "Use only evidence_ids from the provided list.\n"
        "If the evidence is insufficient, return an empty cases array.\n\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "cases": [\n'
        "    {\n"
        '      "case_id": "case_0001",\n'
        '      "title": "...",\n'
        '      "summary": "...",\n'
        '      "evidence_ids": ["ev_0001"],\n'
        '      "delivery": {\n'
        '        "phenomenon": "...",\n'
        '        "prior_state": "...",\n'
        '        "new_state": "..."\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Paper title: {paper_title}\n"
        f"Evidence units:\n{json.dumps(evidence_json, ensure_ascii=False, indent=2)}"
    )


def parse_candidate_response(content: str) -> list[CandidateCase]:
    payload = _loads_json_object(content)
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise LlmCaseGenerationError("LLM response missing list field: cases")
    candidates: list[CandidateCase] = []
    for index, item in enumerate(raw_cases[:1], start=1):
        if not isinstance(item, dict):
            raise LlmCaseGenerationError(f"LLM case row {index} is not an object")
        candidate_payload = {
            "case_id": item.get("case_id") or f"case_{index:04d}",
            "title": item.get("title") or "",
            "summary": item.get("summary") or "",
            "evidence_ids": item.get("evidence_ids") or [],
            "delivery": item.get("delivery") or {},
            "metadata": {"generator": "live_llm"},
        }
        try:
            candidates.append(CandidateCase.model_validate(candidate_payload))
        except ValidationError as exc:
            raise LlmCaseGenerationError(f"LLM case row {index} failed schema validation") from exc
    return candidates


def _loads_json_object(content: str) -> dict[str, Any]:
    cleaned = _strip_code_fence(content.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LlmCaseGenerationError("LLM response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise LlmCaseGenerationError("LLM response must be a JSON object")
    return payload


def _strip_code_fence(content: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else content


class LlmCaseGenerationError(RuntimeError):
    pass
