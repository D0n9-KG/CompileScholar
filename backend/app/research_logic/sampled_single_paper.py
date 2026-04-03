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
FixedComparisonVerdict = Literal['stable_pass', 'recurring_failure', 'new_regression', 'improved', 'availability_only']
RandomComparisonVerdict = Literal[
    'new_edge_case',
    'repeated_random_failure',
    'random_improved',
    'stable_random_pass',
    'availability_only',
]
OwnerBucket = Literal[
    'slot_recovery',
    'relation_assembly',
    'route_seed_richness',
    'metadata_repair',
    'noise_cleanup',
    'reference_recovery',
    'citation_semantics',
]


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
    citation_semantic: dict[str, int] = Field(default_factory=dict)
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


class FixedSampledL2ComparisonRow(ContractModel):
    corpus_paper_id: str
    display_title: str
    cohort: Literal['fixed_regression'] = 'fixed_regression'
    verdict: FixedComparisonVerdict
    current_result: SampledPaperRunResult
    previous_result: SampledPaperRunResult | None = None


class RandomSampledL2ComparisonRow(ContractModel):
    corpus_paper_id: str
    display_title: str
    cohort: Literal['random_exploration'] = 'random_exploration'
    verdict: RandomComparisonVerdict
    current_result: SampledPaperRunResult
    previous_result: SampledPaperRunResult | None = None


class OwnerBucketSummaryEntry(ContractModel):
    bucket: OwnerBucket
    priority: int = 0
    fixed_count: int = 0
    random_count: int = 0
    total_count: int = 0
    exemplar_corpus_paper_ids: list[str] = Field(default_factory=list)
    verdict_counts: dict[str, int] = Field(default_factory=dict)


class SampledL2ComparisonResult(ContractModel):
    schema_version: str = 'v1'
    built_at: str = Field(default_factory=_utc_now_iso)
    iteration_label: str
    previous_iteration_label: str | None = None
    baseline_only: bool = False
    fixed_comparisons: list[FixedSampledL2ComparisonRow] = Field(default_factory=list)
    random_comparisons: list[RandomSampledL2ComparisonRow] = Field(default_factory=list)
    owner_buckets: list[OwnerBucketSummaryEntry] = Field(default_factory=list)


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


def load_sampled_l2_iteration_bundle(bundle_dir: str | Path) -> SampledSinglePaperIterationResult:
    resolved_bundle_dir = _as_path(bundle_dir).expanduser().resolve()
    summary_payload = _read_json(resolved_bundle_dir / 'iteration_summary.json')
    inspection_payload = _read_json(resolved_bundle_dir / 'iteration_inspection.json')
    return SampledSinglePaperIterationResult(
        schema_version=str(summary_payload.get('schema_version') or 'v1'),
        built_at=str(summary_payload.get('built_at') or inspection_payload.get('built_at') or ''),
        iteration_label=str(summary_payload.get('iteration_label') or ''),
        sampling_bundle_dir=str(summary_payload.get('sampling_bundle_dir') or inspection_payload.get('sampling_bundle_dir') or ''),
        sampling_bundle_manifest_ref=str(
            summary_payload.get('sampling_bundle_manifest_ref')
            or inspection_payload.get('sampling_bundle_manifest_ref')
            or ''
        ),
        fixed_manifest_ref=str(summary_payload.get('fixed_manifest_ref') or '').strip() or None,
        seed=int(summary_payload['seed']) if summary_payload.get('seed') is not None else None,
        neo4j_lookup_status=summary_payload.get('neo4j_lookup_status'),
        neo4j_lookup_error=str(inspection_payload.get('neo4j_lookup_error') or '').strip() or None,
        fixed_selected_count=int(summary_payload.get('fixed_selected_count') or 0),
        random_selected_count=int(summary_payload.get('random_selected_count') or 0),
        fixed_results=[
            FixedRegressionSampledPaperResult.model_validate(item)
            for item in inspection_payload.get('fixed_regression_results') or []
        ],
        random_results=[
            RandomExplorationSampledPaperResult.model_validate(item)
            for item in inspection_payload.get('random_exploration_results') or []
        ],
        availability_issues=[
            SampledPaperAvailabilityIssue.model_validate(item)
            for item in inspection_payload.get('availability_issues') or []
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
            'citation_semantic': dict(raw_result.get('citation_semantic') or {}),
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


def _all_iteration_results(iteration: SampledSinglePaperIterationResult) -> list[SampledPaperRunResult]:
    return [
        *iteration.fixed_results,
        *iteration.random_results,
        *iteration.availability_issues,
    ]


def _result_quality_tier(result: SampledPaperRunResult | None) -> str:
    if result is None:
        return ''
    quality_report = dict(result.quality_report or {})
    trace_quality = dict(result.trace_quality or {})
    return str(quality_report.get('quality_tier') or trace_quality.get('quality_tier') or '').strip().lower()


def _result_is_green(result: SampledPaperRunResult | None) -> bool:
    return _result_quality_tier(result) == 'green'


def classify_fixed_result_delta(
    current_result: SampledPaperRunResult,
    previous_result: SampledPaperRunResult | None = None,
) -> FixedComparisonVerdict:
    if current_result.execution_status != 'executed':
        return 'availability_only'
    if previous_result is not None and previous_result.execution_status != 'executed':
        return 'availability_only'
    if previous_result is None:
        return 'stable_pass' if _result_is_green(current_result) else 'recurring_failure'
    if _result_is_green(current_result) and _result_is_green(previous_result):
        return 'stable_pass'
    if not _result_is_green(current_result) and not _result_is_green(previous_result):
        return 'recurring_failure'
    if not _result_is_green(current_result) and _result_is_green(previous_result):
        return 'new_regression'
    return 'improved'


def _classify_random_result_delta(
    current_result: SampledPaperRunResult,
    previous_result: SampledPaperRunResult | None = None,
) -> RandomComparisonVerdict:
    if current_result.execution_status != 'executed':
        return 'availability_only'
    if previous_result is not None and previous_result.execution_status != 'executed':
        return 'availability_only'
    if previous_result is None:
        return 'stable_random_pass' if _result_is_green(current_result) else 'new_edge_case'
    if _result_is_green(current_result) and _result_is_green(previous_result):
        return 'stable_random_pass'
    if not _result_is_green(current_result) and not _result_is_green(previous_result):
        return 'repeated_random_failure'
    if not _result_is_green(current_result) and _result_is_green(previous_result):
        return 'new_edge_case'
    return 'random_improved'


def _owner_buckets_for_result(result: SampledPaperRunResult) -> set[OwnerBucket]:
    quality_report = dict(result.quality_report or {})
    trace_quality = dict(result.trace_quality or {})
    completeness_audit = dict(quality_report.get('l2_completeness_audit') or {})
    hot_path_gate_report = dict(quality_report.get('hot_path_gate_report') or {})
    quality_flags = {
        str(flag).strip()
        for flag in quality_report.get('quality_flags') or trace_quality.get('quality_flags') or []
        if str(flag).strip()
    }
    buckets: set[OwnerBucket] = set()

    if (
        completeness_audit.get('missing_expected_roles')
        or completeness_audit.get('missing_expected_slot_fields')
        or completeness_audit.get('sparse_expected_slot_fields')
        or {'missing_expected_roles', 'missing_expected_slots', 'sparse_expected_slots'} & quality_flags
    ):
        buckets.add('slot_recovery')
    relation_ratio = float(
        completeness_audit.get('relation_coverage_ratio')
        or hot_path_gate_report.get('relation_coverage_ratio')
        or 0.0
    )
    if 'weak_relation_stitching' in quality_flags or relation_ratio < 0.5:
        buckets.add('relation_assembly')
    if 'route_state_seed_thin' in quality_flags:
        buckets.add('route_seed_richness')
    if 'metadata_summary_mismatch' in quality_flags:
        buckets.add('metadata_repair')
    if 'residual_noise_moves' in quality_flags or completeness_audit.get('noise_move_ids'):
        buckets.add('noise_cleanup')

    reference_status = str((result.reference_recovery or {}).get('status') or '').strip().lower()
    citation_event_status = str((result.citation_event_recovery or {}).get('status') or '').strip().lower()
    if reference_status.startswith('recovered_heuristic') or 'error' in reference_status or 'timeout' in reference_status:
        buckets.add('reference_recovery')
    if citation_event_status in {'empty_result', 'agent_error'} or 'error' in citation_event_status:
        buckets.add('reference_recovery')

    cites_resolved = int((result.citations or {}).get('cites_resolved') or 0)
    purposes = int((result.llm or {}).get('purposes') or 0)
    citation_acts = int((result.citation_semantic or {}).get('citation_acts') or 0)
    citation_mentions = int((result.citation_semantic or {}).get('citation_mentions') or 0)
    if cites_resolved > 0 and (purposes <= 0 or citation_acts <= 0 or citation_mentions <= 0):
        buckets.add('citation_semantics')

    return buckets


def build_owner_bucket_summary(
    comparison: SampledL2ComparisonResult,
) -> list[OwnerBucketSummaryEntry]:
    bucket_data: dict[OwnerBucket, dict[str, Any]] = {}

    for row in [*comparison.fixed_comparisons, *comparison.random_comparisons]:
        current_result = row.current_result
        if current_result.execution_status != 'executed':
            continue
        if _result_is_green(current_result):
            continue
        for bucket in _owner_buckets_for_result(current_result):
            data = bucket_data.setdefault(
                bucket,
                {
                    'fixed_count': 0,
                    'random_count': 0,
                    'exemplar_corpus_paper_ids': [],
                    'verdict_counts': {},
                },
            )
            if row.cohort == 'fixed_regression':
                data['fixed_count'] += 1
            else:
                data['random_count'] += 1
            if current_result.corpus_paper_id not in data['exemplar_corpus_paper_ids']:
                data['exemplar_corpus_paper_ids'].append(current_result.corpus_paper_id)
            data['verdict_counts'][row.verdict] = data['verdict_counts'].get(row.verdict, 0) + 1

    ordered_entries: list[OwnerBucketSummaryEntry] = []
    for bucket, data in bucket_data.items():
        ordered_entries.append(
            OwnerBucketSummaryEntry(
                bucket=bucket,
                fixed_count=int(data['fixed_count']),
                random_count=int(data['random_count']),
                total_count=int(data['fixed_count']) + int(data['random_count']),
                exemplar_corpus_paper_ids=list(data['exemplar_corpus_paper_ids'])[:5],
                verdict_counts=dict(data['verdict_counts']),
            )
        )

    ordered_entries.sort(
        key=lambda entry: (
            0 if entry.fixed_count > 0 else 1,
            -entry.fixed_count,
            -entry.random_count,
            entry.bucket,
        )
    )
    for index, entry in enumerate(ordered_entries, start=1):
        entry.priority = index
    return ordered_entries


def compare_sampled_l2_iterations(
    current_iteration: SampledSinglePaperIterationResult,
    previous_iteration: SampledSinglePaperIterationResult | None = None,
) -> SampledL2ComparisonResult:
    previous_by_key = {
        (result.cohort, result.corpus_paper_id): result
        for result in _all_iteration_results(previous_iteration)
    } if previous_iteration is not None else {}

    fixed_comparisons: list[FixedSampledL2ComparisonRow] = []
    random_comparisons: list[RandomSampledL2ComparisonRow] = []
    for result in _all_iteration_results(current_iteration):
        previous_result = previous_by_key.get((result.cohort, result.corpus_paper_id))
        if result.cohort == 'fixed_regression':
            fixed_comparisons.append(
                FixedSampledL2ComparisonRow(
                    corpus_paper_id=result.corpus_paper_id,
                    display_title=result.display_title,
                    verdict=classify_fixed_result_delta(result, previous_result),
                    current_result=result,
                    previous_result=previous_result,
                )
            )
            continue
        random_comparisons.append(
            RandomSampledL2ComparisonRow(
                corpus_paper_id=result.corpus_paper_id,
                display_title=result.display_title,
                verdict=_classify_random_result_delta(result, previous_result),
                current_result=result,
                previous_result=previous_result,
            )
        )

    comparison = SampledL2ComparisonResult(
        iteration_label=current_iteration.iteration_label,
        previous_iteration_label=previous_iteration.iteration_label if previous_iteration is not None else None,
        baseline_only=previous_iteration is None,
        fixed_comparisons=fixed_comparisons,
        random_comparisons=random_comparisons,
    )
    comparison.owner_buckets = build_owner_bucket_summary(comparison)
    return comparison


__all__ = [
    'ExecutionStatus',
    'FixedComparisonVerdict',
    'FixedRegressionSampledPaperResult',
    'FixedSampledL2ComparisonRow',
    'LoadedPhase7SamplingBundle',
    'OwnerBucket',
    'OwnerBucketSummaryEntry',
    'RandomComparisonVerdict',
    'RandomExplorationSampledPaperResult',
    'RandomSampledL2ComparisonRow',
    'SampledL2ComparisonResult',
    'SampledPaperAvailabilityIssue',
    'SampledPaperRunResult',
    'SampledPaperSelectionRef',
    'SampledSinglePaperIterationResult',
    'build_owner_bucket_summary',
    'compare_sampled_l2_iterations',
    'classify_fixed_result_delta',
    'load_phase7_sampling_bundle',
    'load_sampled_l2_iteration_bundle',
    'run_sampled_single_paper_iteration',
]
