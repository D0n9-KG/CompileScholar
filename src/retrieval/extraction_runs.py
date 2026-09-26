from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retrieval._see_io import write_json_schema, write_jsonl
from retrieval._see_llm import load_llm_config
from retrieval._see_llm_cases import (
    CandidateGenerator,
    LlmCaseGenerationError,
    generate_candidate_cases,
)
from retrieval._see_extraction import finalize_cases
from retrieval._see_models import EvidenceUnit, EvolutionCase, RetrievalTrace, RunSummary
from retrieval._see_retrieval import RetrievalCandidate, lexical_retrieve
from retrieval.evidence_units import EvidenceUnitError, ensure_paper_evidence
from retrieval.registry import LibraryRegistry, LibraryRegistryError, stable_id, utc_now


@dataclass(frozen=True)
class ExtractionRunResult:
    run: dict[str, Any]
    job: dict[str, Any]
    status: str
    message: str | None = None


def create_extraction_run_for_paper(
    *,
    registry: LibraryRegistry,
    paper_id: str,
    force: bool = False,
    candidate_generator: CandidateGenerator | None = None,
) -> ExtractionRunResult:
    paper = registry.get_paper(paper_id)
    if not paper:
        raise ExtractionRunError(f"paper not found: {paper_id}")

    content_artifact = _latest_ready_content_list(registry, paper_id)
    if content_artifact is None:
        raise ExtractionRunBlocked("没有 ready 的 MinerU content_list，请先完成 MinerU 解析。")

    existing = registry.latest_extraction_run_for_paper(paper_id)
    if existing and existing["status"] == "ready" and not force:
        job = registry.create_api_job(
            kind="extraction",
            status="ready",
            paper_id=paper_id,
            current_stage="extraction",
        )
        return ExtractionRunResult(run=existing, job=job, status="ready", message="复用已有抽取结果。")

    try:
        evidence_result = ensure_paper_evidence(registry=registry, paper_id=paper_id)
    except EvidenceUnitError as exc:
        raise ExtractionRunError(str(exc)) from exc
    evidence_candidates = evidence_result.evidence_units
    query = str(paper.get("title") or paper.get("normalized_doi") or paper_id)
    matches = lexical_retrieve(query, _retrieval_candidates(evidence_candidates), limit=12)
    evidence_units = _matched_evidence_units(evidence_candidates, matches) if matches else evidence_candidates[:12]
    traces = [
        RetrievalTrace(
            query_id="query_0001",
            query=query,
            matched_evidence_ids=[item.evidence_id for item in evidence_units],
            metadata={"match_count": str(len(evidence_units)), "source": "multisource_evidence"},
        )
    ]

    llm_config = load_llm_config()
    final_cases: list[EvolutionCase] = []
    status = "blocked"
    failures = ["llm_config_missing: 未配置 live LLM，已保留证据候选但不生成最终科学演化案例。"]
    llm_trace: dict[str, Any] = {
        "provider": None,
        "status": "blocked",
        "reason": "missing_llm_config",
    }
    if llm_config:
        generator = candidate_generator or generate_candidate_cases
        try:
            generated = generator(
                paper_title=query,
                evidence_units=evidence_units,
                llm_config=llm_config,
            )
            finalization = finalize_cases(generated.candidates, evidence_units)
            final_cases = finalization.final_cases
            gate_failures = [
                f"{failure.case_id}:{failure.reason}:{failure.detail}"
                for trace in finalization.traces
                for failure in trace.failures
            ]
            if final_cases:
                status = "ready"
                failures = gate_failures
            else:
                status = "blocked"
                failures = gate_failures or ["llm_generated_no_final_cases: LLM 未生成可通过 evidence gate 的候选案例。"]
            llm_trace = {
                **generated.trace,
                "gate_status": "ready" if final_cases else "blocked",
                "case_traces": [trace.model_dump() for trace in finalization.traces],
            }
        except LlmCaseGenerationError as exc:
            status = "blocked"
            failures = [f"llm_generation_failed: {exc}"]
            llm_trace = {
                **llm_config.trace_metadata(),
                "status": "blocked",
                "reason": "llm_generation_failed",
                "error": str(exc),
            }

    run_id = stable_id("RUN_", paper_id, content_artifact["artifact_id"], utc_now())
    output_dir = registry.library_root / "papers" / paper_id / "extraction_runs" / run_id
    summary = RunSummary(
        run_id=run_id,
        paper_id=paper_id,
        content_blocks=evidence_result.coverage["by_source_type"].get("mineru_content_block", 0),
        kg_nodes=evidence_result.coverage["by_source_type"].get("kg_node", 0),
        kg_edges=evidence_result.coverage["by_source_type"].get("kg_edge", 0),
        evidence_units=len(evidence_units),
        final_cases=len(final_cases),
        failures=failures,
    )
    output_artifacts = _write_run_outputs(
        output_dir=output_dir,
        summary=summary,
        evidence_units=evidence_units,
        final_cases=final_cases,
        traces=traces,
        llm_trace=llm_trace,
    )
    llm_config_hash = _hash_dict(llm_trace)
    run = registry.register_extraction_run(
        run_id=run_id,
        paper_id=paper_id,
        status=status,
        input_artifact_ids=[content_artifact["artifact_id"]],
        output_artifacts=output_artifacts,
        llm_config_hash=llm_config_hash,
        error="; ".join(failures) if status != "ready" else None,
    )
    job = registry.create_api_job(
        kind="extraction",
        status=status,
        paper_id=paper_id,
        current_stage="extraction",
        error=run.get("error"),
    )
    registry.record_provenance_event(
        action="create_extraction_run",
        source="library_api",
        inputs={"paper_id": paper_id, "content_artifact_id": content_artifact["artifact_id"], "evidence_source": "multisource"},
        outputs={"run_id": run_id, "job_id": job["job_id"], "status": status},
        error_summary=run.get("error"),
    )
    return ExtractionRunResult(run=run, job=job, status=status, message=run.get("error"))


def _latest_ready_content_list(
    registry: LibraryRegistry,
    paper_id: str,
) -> dict[str, Any] | None:
    artifacts = [
        item
        for item in registry.list_paper_artifacts(paper_id)
        if item.get("kind") == "mineru_content_list" and item.get("status") == "ready"
    ]
    return artifacts[-1] if artifacts else None


def _retrieval_candidates(evidence_units: list[EvidenceUnit]) -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(
            source_type=item.source.source_type,
            source_id=item.source.source_id,
            text=item.text,
            page_idx=item.source.page_idx,
        )
        for item in evidence_units
    ]


def _matched_evidence_units(
    evidence_units: list[EvidenceUnit],
    matches: list[RetrievalCandidate],
) -> list[EvidenceUnit]:
    evidence_by_source = {
        (item.source.source_type, item.source.source_id): item
        for item in evidence_units
    }
    selected: list[EvidenceUnit] = []
    for match in matches:
        item = evidence_by_source.get((match.source_type, match.source_id))
        if item is None:
            continue
        metadata = {**item.metadata, "retrieval_score": match.score}
        selected.append(item.model_copy(update={"metadata": metadata}))
    return selected


def _write_run_outputs(
    *,
    output_dir: Path,
    summary: RunSummary,
    evidence_units: list[EvidenceUnit],
    final_cases: list[EvolutionCase],
    traces: list[RetrievalTrace],
    llm_trace: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    evidence_path = output_dir / "evidence_units.jsonl"
    cases_path = output_dir / "evolution_cases.final.jsonl"
    trace_path = output_dir / "retrieval_trace.jsonl"
    llm_path = output_dir / "llm_trace.json"
    summary_path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    write_jsonl(evidence_path, evidence_units)
    write_jsonl(cases_path, final_cases)
    write_jsonl(trace_path, traces)
    llm_path.write_text(json.dumps(llm_trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    schema_dir = output_dir / "schemas"
    write_json_schema(EvidenceUnit, schema_dir / "evidence_unit.schema.json")
    write_json_schema(EvolutionCase, schema_dir / "evolution_case.schema.json")
    write_json_schema(RetrievalTrace, schema_dir / "retrieval_trace.schema.json")
    write_json_schema(RunSummary, schema_dir / "run_summary.schema.json")
    return {
        "summary": summary_path,
        "evidence_units": evidence_path,
        "evolution_cases": cases_path,
        "retrieval_trace": trace_path,
        "llm_trace": llm_path,
    }


def _hash_dict(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ExtractionRunError(RuntimeError):
    pass


class ExtractionRunBlocked(ExtractionRunError):
    pass
