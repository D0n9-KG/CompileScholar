from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import Field

from app.ingest.rebuild import evaluate_sampled_paper_from_source

from .corpus_sampling import CorpusInventoryEntry, CorpusSamplingBatch, Neo4jLookupStatus, SamplingMode, SourceKind
from .models import ContractModel


ExecutionStatus = Literal['executed', 'source_missing', 'runtime_error']


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _read_json(path_like: str | Path) -> Any:
    path = _as_path(path_like)
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'sampled-paper artifact not found: {path}') from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid JSON in sampled-paper artifact: {path}') from exc


class SampledPaperSelectionRef(ContractModel):
    corpus_paper_id: str
    display_title: str
    cohort: SamplingMode
    selection_mode: SamplingMode
    corpus_relative_ref: str
    preferred_source_path: str | None = None
    preferred_source_kind: SourceKind | None = None
    md_path: str | None = None
    txt_path: str | None = None
    neo4j_paper_id: str | None = None
    neo4j_paper_source: str | None = None
    neo4j_ingested: bool | None = None


class SampledPaperRunResult(ContractModel):
    corpus_paper_id: str
    display_title: str
    cohort: SamplingMode
    selection_mode: SamplingMode
    corpus_relative_ref: str
    preferred_source_path: str | None = None
    preferred_source_kind: SourceKind | None = None
    iteration_label: str
    execution_status: ExecutionStatus
    paper_id: str | None = None
    trace_id: str | None = None
    source_path: str | None = None
    source_kind: SourceKind | None = None
    quality_report: dict[str, Any] | None = None
    trace_quality: dict[str, Any] | None = None
    reference_recovery: dict[str, Any] | None = None
    citation_event_recovery: dict[str, Any] | None = None
    artifacts_dir: str | None = None
    artifact_refs: dict[str, str] = Field(default_factory=dict)
    citations: dict[str, int] = Field(default_factory=dict)
    llm: dict[str, Any] = Field(default_factory=dict)
    skipped_canonical_write: bool = False
    graph_write_performed: bool = False
    error_message: str | None = None
    error_type: str | None = None


class FixedRegressionSampledPaperResult(SampledPaperRunResult):
    cohort: Literal['fixed_regression'] = 'fixed_regression'
    selection_mode: Literal['fixed_regression'] = 'fixed_regression'
    execution_status: Literal['executed'] = 'executed'


class RandomExplorationSampledPaperResult(SampledPaperRunResult):
    cohort: Literal['random_exploration'] = 'random_exploration'
    selection_mode: Literal['random_exploration'] = 'random_exploration'
    execution_status: Literal['executed'] = 'executed'


class SampledPaperAvailabilityIssue(SampledPaperRunResult):
    execution_status: Literal['source_missing', 'runtime_error']


class LoadedPhase7SamplingBundle(ContractModel):
    bundle_dir: str
    bundle_manifest_ref: str
    built_at: str
    schema_version: str = 'v1'
    fixed_manifest_ref: str | None = None
    seed: int | None = None
    neo4j_lookup_status: Neo4jLookupStatus | None = None
    neo4j_lookup_error: str | None = None
    fixed_selected: list[SampledPaperSelectionRef] = Field(default_factory=list)
    random_selected: list[SampledPaperSelectionRef] = Field(default_factory=list)


class SampledSinglePaperIterationResult(ContractModel):
    schema_version: str = 'v1'
    built_at: str = Field(default_factory=_utc_now_iso)
    iteration_label: str
    sampling_bundle_dir: str
    sampling_bundle_manifest_ref: str
    fixed_manifest_ref: str | None = None
    seed: int | None = None
    neo4j_lookup_status: Neo4jLookupStatus | None = None
    neo4j_lookup_error: str | None = None
    fixed_selected_count: int = 0
    random_selected_count: int = 0
    fixed_results: list[FixedRegressionSampledPaperResult] = Field(default_factory=list)
    random_results: list[RandomExplorationSampledPaperResult] = Field(default_factory=list)
    availability_issues: list[SampledPaperAvailabilityIssue] = Field(default_factory=list)


def _selection_from_entry(entry: CorpusInventoryEntry, *, cohort: SamplingMode) -> SampledPaperSelectionRef:
    return SampledPaperSelectionRef(
        corpus_paper_id=entry.corpus_paper_id,
        display_title=entry.display_title,
        cohort=cohort,
        selection_mode=cohort,
        corpus_relative_ref=entry.corpus_relative_ref,
        preferred_source_path=entry.preferred_source_path,
        preferred_source_kind=entry.preferred_source_kind,
        md_path=entry.md_path,
        txt_path=entry.txt_path,
        neo4j_paper_id=entry.neo4j_paper_id,
        neo4j_paper_source=entry.neo4j_paper_source,
        neo4j_ingested=entry.neo4j_ingested,
    )


def _load_batch(bundle_dir: Path, relative_path: str) -> CorpusSamplingBatch:
    return CorpusSamplingBatch.model_validate(_read_json(bundle_dir / relative_path))


def load_phase7_sampling_bundle(bundle_dir: str | Path) -> LoadedPhase7SamplingBundle:
    resolved_bundle_dir = _as_path(bundle_dir).expanduser().resolve()
    manifest_path = resolved_bundle_dir / 'bundle_manifest.json'
    manifest_payload = _read_json(manifest_path)
    files_payload = dict(manifest_payload.get('files') or {})

    fixed_batch_path = str(files_payload.get('fixed_regression_batch') or '').strip()
    random_batch_path = str(files_payload.get('random_exploration_batch') or '').strip()
    if not fixed_batch_path or not random_batch_path:
        raise ValueError(f'phase7 sampling bundle is missing batch file refs: {manifest_path}')

    fixed_batch = _load_batch(resolved_bundle_dir, fixed_batch_path)
    random_batch = _load_batch(resolved_bundle_dir, random_batch_path)
    return LoadedPhase7SamplingBundle(
        bundle_dir=str(resolved_bundle_dir),
        bundle_manifest_ref=str(manifest_path),
        built_at=str(manifest_payload.get('built_at') or ''),
        schema_version=str(manifest_payload.get('schema_version') or 'v1'),
        fixed_manifest_ref=str(manifest_payload.get('fixed_manifest_ref') or '').strip() or None,
        seed=int(manifest_payload['seed']) if manifest_payload.get('seed') is not None else None,
        neo4j_lookup_status=manifest_payload.get('neo4j_lookup_status'),
        neo4j_lookup_error=str(manifest_payload.get('neo4j_lookup_error') or '').strip() or None,
        fixed_selected=[
            _selection_from_entry(entry, cohort='fixed_regression')
            for entry in fixed_batch.selected
        ],
        random_selected=[
            _selection_from_entry(entry, cohort='random_exploration')
            for entry in random_batch.selected
        ],
    )


def _coerce_run_result(
    selection: SampledPaperSelectionRef,
    raw_result: dict[str, Any],
    *,
    iteration_label: str,
) -> SampledPaperRunResult:
    return SampledPaperRunResult.model_validate(
        {
            'corpus_paper_id': selection.corpus_paper_id,
            'display_title': selection.display_title,
            'cohort': selection.cohort,
            'selection_mode': selection.selection_mode,
            'corpus_relative_ref': selection.corpus_relative_ref,
            'preferred_source_path': raw_result.get('preferred_source_path') or selection.preferred_source_path,
            'preferred_source_kind': raw_result.get('preferred_source_kind') or selection.preferred_source_kind,
            'iteration_label': raw_result.get('iteration_label') or iteration_label,
            'execution_status': raw_result.get('execution_status'),
            'paper_id': raw_result.get('paper_id'),
            'trace_id': raw_result.get('trace_id'),
            'source_path': raw_result.get('source_path'),
            'source_kind': raw_result.get('source_kind') or selection.preferred_source_kind,
            'quality_report': raw_result.get('quality_report'),
            'trace_quality': raw_result.get('trace_quality'),
            'reference_recovery': raw_result.get('reference_recovery'),
            'citation_event_recovery': raw_result.get('citation_event_recovery'),
            'artifacts_dir': raw_result.get('artifacts_dir'),
            'artifact_refs': dict(raw_result.get('artifact_refs') or {}),
            'citations': dict(raw_result.get('citations') or {}),
            'llm': dict(raw_result.get('llm') or {}),
            'skipped_canonical_write': bool(raw_result.get('skipped_canonical_write')),
            'graph_write_performed': bool(raw_result.get('graph_write_performed')),
            'error_message': raw_result.get('error_message'),
            'error_type': raw_result.get('error_type'),
        }
    )


def run_sampled_single_paper_iteration(
    sampling_bundle_dir: str | Path,
    *,
    iteration_label: str,
    artifacts_dir: str | Path,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    allow_graph_write: bool = False,
) -> SampledSinglePaperIterationResult:
    loaded_bundle = load_phase7_sampling_bundle(sampling_bundle_dir)
    evaluation_runner = evaluator or evaluate_sampled_paper_from_source
    artifact_root = _as_path(artifacts_dir).expanduser().resolve()

    fixed_results: list[FixedRegressionSampledPaperResult] = []
    random_results: list[RandomExplorationSampledPaperResult] = []
    availability_issues: list[SampledPaperAvailabilityIssue] = []

    for selection in [*loaded_bundle.fixed_selected, *loaded_bundle.random_selected]:
        result_payload = evaluation_runner(
            corpus_paper_id=selection.corpus_paper_id,
            cohort=selection.cohort,
            corpus_relative_ref=selection.corpus_relative_ref,
            preferred_source_path=selection.preferred_source_path,
            preferred_source_kind=selection.preferred_source_kind,
            iteration_label=iteration_label,
            artifacts_dir=artifact_root / 'paper_artifacts' / selection.cohort / selection.corpus_paper_id,
            allow_graph_write=allow_graph_write,
        )
        result = _coerce_run_result(selection, result_payload, iteration_label=iteration_label)
        if result.execution_status == 'executed':
            if selection.cohort == 'fixed_regression':
                fixed_results.append(FixedRegressionSampledPaperResult.model_validate(result.model_dump(mode='python')))
            else:
                random_results.append(RandomExplorationSampledPaperResult.model_validate(result.model_dump(mode='python')))
            continue
        availability_issues.append(SampledPaperAvailabilityIssue.model_validate(result.model_dump(mode='python')))

    return SampledSinglePaperIterationResult(
        iteration_label=iteration_label,
        sampling_bundle_dir=loaded_bundle.bundle_dir,
        sampling_bundle_manifest_ref=loaded_bundle.bundle_manifest_ref,
        fixed_manifest_ref=loaded_bundle.fixed_manifest_ref,
        seed=loaded_bundle.seed,
        neo4j_lookup_status=loaded_bundle.neo4j_lookup_status,
        neo4j_lookup_error=loaded_bundle.neo4j_lookup_error,
        fixed_selected_count=len(loaded_bundle.fixed_selected),
        random_selected_count=len(loaded_bundle.random_selected),
        fixed_results=fixed_results,
        random_results=random_results,
        availability_issues=availability_issues,
    )


__all__ = [
    'ExecutionStatus',
    'FixedRegressionSampledPaperResult',
    'LoadedPhase7SamplingBundle',
    'RandomExplorationSampledPaperResult',
    'SampledPaperAvailabilityIssue',
    'SampledPaperRunResult',
    'SampledPaperSelectionRef',
    'SampledSinglePaperIterationResult',
    'load_phase7_sampling_bundle',
    'run_sampled_single_paper_iteration',
]
