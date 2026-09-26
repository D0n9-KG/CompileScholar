from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from retrieval._see_models import EvidenceUnit, EvolutionCase


class CandidateCase(BaseModel):
    case_id: str
    title: str = ""
    summary: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    delivery: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)


class GateFailure(BaseModel):
    case_id: str
    reason: str
    detail: str


class CaseTrace(BaseModel):
    case_id: str
    status: Literal["final", "rejected", "repaired"]
    failures: list[GateFailure] = Field(default_factory=list)
    repair_attempted: bool = False
    repair_result: Literal["not_needed", "accepted", "rejected"] = "not_needed"


@dataclass(frozen=True)
class FinalizationResult:
    final_cases: list[EvolutionCase]
    traces: list[CaseTrace]


RepairFn = Callable[[CandidateCase, list[GateFailure]], CandidateCase | None]


REQUIRED_DELIVERY_FIELDS = ("phenomenon", "prior_state", "new_state")


def finalize_cases(
    candidates: list[CandidateCase],
    evidence_units: list[EvidenceUnit],
    *,
    repair_fn: RepairFn | None = None,
) -> FinalizationResult:
    evidence_index = {item.evidence_id: item for item in evidence_units}
    final_cases: list[EvolutionCase] = []
    traces: list[CaseTrace] = []

    for candidate in candidates:
        failures = validate_candidate(candidate, evidence_index)
        if failures and repair_fn is not None:
            repaired = repair_fn(candidate, failures)
            if repaired is not None:
                repaired_failures = validate_candidate(repaired, evidence_index)
                if not repaired_failures:
                    final_cases.append(_to_final_case(repaired))
                    traces.append(
                        CaseTrace(
                            case_id=candidate.case_id,
                            status="repaired",
                            failures=failures,
                            repair_attempted=True,
                            repair_result="accepted",
                        )
                    )
                    continue
                traces.append(
                    CaseTrace(
                        case_id=candidate.case_id,
                        status="rejected",
                        failures=[*failures, *repaired_failures],
                        repair_attempted=True,
                        repair_result="rejected",
                    )
                )
                continue

        if failures:
            traces.append(
                CaseTrace(
                    case_id=candidate.case_id,
                    status="rejected",
                    failures=failures,
                )
            )
            continue

        final_cases.append(_to_final_case(candidate))
        traces.append(CaseTrace(case_id=candidate.case_id, status="final"))

    return FinalizationResult(final_cases=final_cases, traces=traces)


def validate_candidate(
    candidate: CandidateCase,
    evidence_index: dict[str, EvidenceUnit],
) -> list[GateFailure]:
    failures: list[GateFailure] = []
    if not candidate.title.strip():
        failures.append(_failure(candidate, "missing_required_field", "title"))
    if not candidate.summary.strip():
        failures.append(_failure(candidate, "missing_required_field", "summary"))
    if not candidate.evidence_ids:
        failures.append(_failure(candidate, "missing_required_field", "evidence_ids"))

    missing_evidence = [
        evidence_id
        for evidence_id in candidate.evidence_ids
        if evidence_id not in evidence_index
    ]
    if missing_evidence:
        failures.append(
            _failure(
                candidate,
                "missing_evidence_ref",
                ",".join(sorted(missing_evidence)),
            )
        )

    missing_delivery = [
        field
        for field in REQUIRED_DELIVERY_FIELDS
        if not candidate.delivery.get(field, "").strip()
    ]
    if missing_delivery:
        failures.append(
            _failure(
                candidate,
                "delivery_incomplete",
                ",".join(missing_delivery),
            )
        )

    return failures


def _failure(candidate: CandidateCase, reason: str, detail: str) -> GateFailure:
    return GateFailure(case_id=candidate.case_id, reason=reason, detail=detail)


def _to_final_case(candidate: CandidateCase) -> EvolutionCase:
    return EvolutionCase(
        case_id=candidate.case_id,
        title=candidate.title,
        summary=candidate.summary,
        evidence_ids=candidate.evidence_ids,
        status="final",
        metadata={"delivery": candidate.delivery, **candidate.metadata},
    )
