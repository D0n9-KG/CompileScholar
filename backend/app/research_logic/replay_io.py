from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from app.paper_logic_trace.models import PaperLogicTrace

from .bounded_packet_audit import (
    BoundedPacketAssemblyManifest,
    BoundedPacketAuditBundleManifest,
    BoundedPacketAuditResult,
    audit_bounded_packet_assembly,
)
from .corpus_sampling import CorpusSamplingBundle
from .decision_episode_export import DecisionEpisodeAuditExport
from .historical_environment import HistoricalEnvironmentSnapshot
from .historical_replay_compiler import HistoricalReplayCompilation
from .iteration_prioritization import (
    IterationPriorityInspection,
    IterationPriorityRecommendation,
    IterationPrioritySummary,
)
from .models import RoutePacket, RouteState
from .sampled_single_paper import (
    FixedSampledL2ComparisonRow,
    RandomSampledL2ComparisonRow,
    SampledL2ComparisonResult,
    SampledPaperRunResult,
    SampledSinglePaperIterationResult,
)

if TYPE_CHECKING:
    from .prior_induction import PriorCandidateRegistry
    from .route_state_package import RouteStatePackageValidation


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _read_json(path_like: str | Path) -> Any:
    path = _as_path(path_like)
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'replay artifact not found: {path}') from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid JSON in replay artifact: {path}') from exc


def _write_json(path_like: str | Path, payload: Any) -> Path:
    path = _as_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return path


def _write_text(path_like: str | Path, content: str) -> Path:
    path = _as_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


def _replay_quality_tier(compilation: HistoricalReplayCompilation) -> str:
    if any(record.blocking for record in compilation.failure_records):
        return 'red'
    if not compilation.quality_flags:
        return 'green'
    return 'yellow' if compilation.ready_for_pilot else 'red'


def _failure_count_map(compilation: HistoricalReplayCompilation, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in compilation.failure_records:
        value = str(getattr(record, key)).strip()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _failure_counts_by_blocking(compilation: HistoricalReplayCompilation) -> dict[str, int]:
    blocking_count = sum(1 for record in compilation.failure_records if record.blocking)
    return {
        'blocking': blocking_count,
        'nonblocking': len(compilation.failure_records) - blocking_count,
    }


def _review_status_counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        normalized = str(value or '').strip()
        if not normalized:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def _unique_strings(values: Sequence[object] | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values or []:
        normalized = str(value or '').strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


TASK_SPECIFIC_TRAINING_VIEW_SECTIONS: dict[str, tuple[str, ...]] = {
    'route_synthesis_view': ('route_synthesis',),
    'why_now_view': ('why_now',),
    'route_comparison_view': ('route_comparison',),
    'prior_antipattern_view': ('priors_antipatterns',),
    'final_decision_view': ('final_decision',),
}


def load_route_packet(path_like: str | Path) -> RoutePacket:
    return RoutePacket.model_validate(_read_json(path_like))


def load_paper_logic_trace(path_like: str | Path) -> PaperLogicTrace:
    return PaperLogicTrace.model_validate(_read_json(path_like))


def load_route_state(path_like: str | Path) -> RouteState:
    return RouteState.model_validate(_read_json(path_like))


def load_historical_environment_snapshot(path_like: str | Path) -> HistoricalEnvironmentSnapshot:
    return HistoricalEnvironmentSnapshot.model_validate(_read_json(path_like))


def _collect_trace_paths(
    *,
    trace_dir: str | Path | None = None,
    trace_files: Sequence[str | Path] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    if trace_dir is not None:
        directory = _as_path(trace_dir)
        if not directory.exists():
            raise FileNotFoundError(f'trace directory not found: {directory}')
        if not directory.is_dir():
            raise ValueError(f'trace directory is not a directory: {directory}')
        for path in sorted(directory.glob('*.json')):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)

    for trace_file in trace_files or []:
        path = _as_path(trace_file)
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(path)

    if not paths:
        raise ValueError('replay requires at least one trace source via --trace-dir or --trace-file')

    return paths


def load_paper_logic_traces(
    *,
    trace_dir: str | Path | None = None,
    trace_files: Sequence[str | Path] | None = None,
) -> list[PaperLogicTrace]:
    traces: list[PaperLogicTrace] = []
    seen_ids: set[str] = set()
    for path in _collect_trace_paths(trace_dir=trace_dir, trace_files=trace_files):
        trace = load_paper_logic_trace(path)
        if trace.trace_id in seen_ids:
            raise ValueError(f'duplicate trace_id in replay inputs: {trace.trace_id}')
        seen_ids.add(trace.trace_id)
        traces.append(trace)
    return traces


def load_route_states(paths: Sequence[str | Path] | None = None) -> list[RouteState]:
    route_states: list[RouteState] = []
    seen_ids: set[str] = set()
    for path_like in paths or []:
        route_state = load_route_state(path_like)
        if route_state.route_state_id in seen_ids:
            raise ValueError(f'duplicate route_state_id in replay inputs: {route_state.route_state_id}')
        seen_ids.add(route_state.route_state_id)
        route_states.append(route_state)
    return route_states


def ensure_packet_trace_coverage(packet: RoutePacket, traces: Sequence[PaperLogicTrace]) -> None:
    if not packet.included_items:
        raise ValueError('RoutePacket requires included_items for replay')

    missing_trace_id_papers = [
        item.paper_id
        for item in packet.included_items
        if not str(item.trace_id or '').strip()
    ]
    if missing_trace_id_papers:
        raise ValueError(
            'RoutePacket included_items missing trace_id for papers: '
            + ', '.join(sorted(dict.fromkeys(missing_trace_id_papers)))
        )

    trace_by_id = {trace.trace_id: trace for trace in traces}
    missing_trace_ids = [
        str(item.trace_id)
        for item in packet.included_items
        if str(item.trace_id) not in trace_by_id
    ]
    if missing_trace_ids:
        raise ValueError(f'missing traces for packet items: {", ".join(missing_trace_ids)}')


def build_bounded_packet_audit_summary(
    *,
    route_packet: RoutePacket,
    assembly_manifest: BoundedPacketAssemblyManifest,
    audit: BoundedPacketAuditResult | None = None,
) -> dict[str, Any]:
    audit_result = audit or audit_bounded_packet_assembly(route_packet, assembly_manifest)
    return {
        'packet_id': route_packet.packet_id,
        'topic_scope_candidate': route_packet.topic_scope_candidate,
        'cutoff_year': route_packet.cutoff_year,
        'packet_included_item_count': len(route_packet.included_items),
        'packet_excluded_item_count': len(route_packet.excluded_items),
        'support_count': audit_result.role_counts.support,
        'alternative_count': audit_result.role_counts.alternative,
        'held_out_count': audit_result.role_counts.held_out,
        'quality_tier': audit_result.quality_tier,
        'ready_for_phase10': audit_result.ready_for_phase10,
        'quality_flags': list(audit_result.quality_flags),
        'structural_errors': list(audit_result.structural_errors),
        'missing_trace_ref_paper_ids': list(audit_result.missing_trace_ref_paper_ids),
        'packet_items_missing_trace_id': list(audit_result.packet_items_missing_trace_id),
        'indistinct_alternative_paper_ids': list(audit_result.indistinct_alternative_paper_ids),
        'support_held_out_overlap_paper_ids': list(audit_result.support_held_out_overlap_paper_ids),
        'missing_exclusion_note_paper_ids': list(audit_result.missing_exclusion_note_paper_ids),
        'known_gap_note_count': len(audit_result.known_gap_notes),
        'exclusion_note_count': audit_result.exclusion_note_count,
    }


def build_bounded_packet_audit_inspection(
    *,
    route_packet: RoutePacket,
    assembly_manifest: BoundedPacketAssemblyManifest,
    audit: BoundedPacketAuditResult | None = None,
) -> dict[str, Any]:
    audit_result = audit or audit_bounded_packet_assembly(route_packet, assembly_manifest)
    return {
        'route_packet': route_packet.model_dump(mode='json', exclude_none=True),
        'assembly_manifest': assembly_manifest.model_dump(mode='json', exclude_none=True),
        'audit': audit_result.model_dump(mode='json', exclude_none=True),
    }


def write_bounded_packet_audit_bundle(
    output_dir: str | Path,
    *,
    route_packet: RoutePacket,
    assembly_manifest: BoundedPacketAssemblyManifest,
    audit: BoundedPacketAuditResult | None = None,
    metadata: dict[str, Any] | None = None,
    report_markdown: str | None = None,
) -> dict[str, Path]:
    bundle_dir = _as_path(output_dir)
    inputs_dir = bundle_dir / 'inputs'
    audit_result = audit or audit_bounded_packet_assembly(route_packet, assembly_manifest)

    summary_payload = build_bounded_packet_audit_summary(
        route_packet=route_packet,
        assembly_manifest=assembly_manifest,
        audit=audit_result,
    )
    inspection_payload = build_bounded_packet_audit_inspection(
        route_packet=route_packet,
        assembly_manifest=assembly_manifest,
        audit=audit_result,
    )

    written_files: dict[str, Path] = {}
    written_files['route_packet'] = _write_json(
        inputs_dir / 'route_packet.json',
        route_packet.model_dump(mode='json', exclude_none=True),
    )
    written_files['assembly_manifest'] = _write_json(
        inputs_dir / 'assembly_manifest.json',
        assembly_manifest.model_dump(mode='json', exclude_none=True),
    )
    written_files['audit_summary'] = _write_json(
        bundle_dir / 'audit_summary.json',
        summary_payload,
    )
    written_files['audit_inspection'] = _write_json(
        bundle_dir / 'audit_inspection.json',
        inspection_payload,
    )
    if report_markdown is not None:
        written_files['audit_report'] = _write_text(
            bundle_dir / 'audit_report.md',
            report_markdown,
        )

    bundle_manifest = BoundedPacketAuditBundleManifest(
        built_at=_utc_now_iso(),
        packet_id=route_packet.packet_id,
        topic_scope=route_packet.topic_scope_candidate,
        cutoff_year=route_packet.cutoff_year,
        route_packet_file=str(written_files['route_packet'].relative_to(bundle_dir)).replace('\\', '/'),
        assembly_manifest_file=str(written_files['assembly_manifest'].relative_to(bundle_dir)).replace('\\', '/'),
        audit_summary_file=str(written_files['audit_summary'].relative_to(bundle_dir)).replace('\\', '/'),
        audit_inspection_file=str(written_files['audit_inspection'].relative_to(bundle_dir)).replace('\\', '/'),
        audit_report_file=(
            str(written_files['audit_report'].relative_to(bundle_dir)).replace('\\', '/')
            if 'audit_report' in written_files
            else None
        ),
        metadata=dict(metadata or {}),
    )
    written_files['bundle_manifest'] = _write_json(
        bundle_dir / 'bundle_manifest.json',
        bundle_manifest.model_dump(mode='json', exclude_none=True),
    )
    return written_files


def build_replay_summary(
    *,
    route_packet: RoutePacket,
    traces: Sequence[PaperLogicTrace],
    compilation: HistoricalReplayCompilation,
    historical_environment_snapshot: HistoricalEnvironmentSnapshot | None = None,
    support_route_states: Sequence[RouteState] | None = None,
    alternative_route_states: Sequence[RouteState] | None = None,
    held_out_route_states: Sequence[RouteState] | None = None,
    reviewer_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        'packet_id': route_packet.packet_id,
        'topic_scope_candidate': route_packet.topic_scope_candidate,
        'cutoff_year': route_packet.cutoff_year,
        'trace_count': len(traces),
        'support_route_state_count': len(support_route_states or []),
        'alternative_route_state_count': len(alternative_route_states or []),
        'held_out_route_state_count': len(held_out_route_states or []),
        'reviewer_ids': [str(reviewer_id) for reviewer_id in reviewer_ids or [] if str(reviewer_id).strip()],
        'l1_snapshot_id': (
            historical_environment_snapshot.snapshot_id
            if historical_environment_snapshot is not None
            else compilation.primary_route_state.compiler_metadata.l1_snapshot_version
        ),
        'primary_route_state_id': compilation.primary_route_state.route_state_id,
        'selected_comparison_case_id': compilation.selected_comparison_case_id,
        'decision_prior_id': compilation.decision_prior_card.prior_id,
        'decision_episode_id': compilation.decision_episode.episode_id,
        'replay_quality_tier': _replay_quality_tier(compilation),
        'ready_for_pilot': compilation.ready_for_pilot,
        'quality_flags': list(compilation.quality_flags),
        'failure_record_count': len(compilation.failure_records),
        'failure_counts_by_layer': _failure_count_map(compilation, 'layer'),
        'failure_counts_by_stage': _failure_count_map(compilation, 'stage'),
        'failure_counts_by_blocking': _failure_counts_by_blocking(compilation),
    }


def build_replay_inspection(
    *,
    route_packet: RoutePacket,
    compilation: HistoricalReplayCompilation,
    traces: Sequence[PaperLogicTrace],
    route_state_package_validation: 'RouteStatePackageValidation | None' = None,
) -> dict[str, Any]:
    selected_comparison_case = next(
        (
            case
            for case in compilation.route_comparison_cases
            if case.route_comparison_case_id == compilation.selected_comparison_case_id
        ),
        None,
    )
    return {
        'packet_id': route_packet.packet_id,
        'topic_scope_candidate': route_packet.topic_scope_candidate,
        'cutoff_year': route_packet.cutoff_year,
        'trace_count': len(traces),
        'replay_quality_tier': _replay_quality_tier(compilation),
        'replay_quality_flags': list(compilation.quality_flags),
        'ready_for_pilot': compilation.ready_for_pilot,
        'failure_records': [record.model_dump(mode='json') for record in compilation.failure_records],
        'stage_quality': {
            'route_state': {
                'quality_tier': compilation.primary_route_state.quality.quality_tier,
                'quality_flags': list(compilation.primary_route_state.quality.quality_flags),
            },
            'why_now_case': {
                'quality_tier': compilation.why_now_case.quality.quality_tier,
                'quality_flags': list(compilation.why_now_case.quality.quality_flags),
            },
            'route_comparison': {
                'case_count': len(compilation.route_comparison_cases),
                'selected_case_id': compilation.selected_comparison_case_id,
                'selected_quality_tier': selected_comparison_case.quality.quality_tier if selected_comparison_case else None,
                'selected_quality_flags': list(selected_comparison_case.quality.quality_flags) if selected_comparison_case else [],
            },
            'decision_prior_card': {
                'quality_tier': compilation.decision_prior_card.quality.quality_tier,
                'quality_flags': list(compilation.decision_prior_card.quality.quality_flags),
                'supporting_route_state_count': len(compilation.decision_prior_card.supporting_route_state_ids),
                'held_out_pass_rate': compilation.decision_prior_card.held_out_consistency.pass_rate,
            },
            'decision_episode': {
                'quality_tier': compilation.decision_episode.quality.quality_tier,
                'quality_flags': list(compilation.decision_episode.quality.quality_flags),
                'ready_for_training': compilation.decision_episode.quality.ready_for_training,
                'ready_for_eval': compilation.decision_episode.quality.ready_for_eval,
            },
        },
        'minimal_attack_path': {
            'required_resources': list(compilation.decision_episode.minimal_attack_path.required_resources),
            'required_measurements': list(compilation.decision_episode.minimal_attack_path.required_measurements),
            'expected_checkpoints': list(compilation.decision_episode.minimal_attack_path.expected_checkpoints),
        },
        'route_state_package_validation': (
            route_state_package_validation.model_dump(mode='json', exclude_none=True)
            if route_state_package_validation is not None
            else None
        ),
    }


def build_prior_candidate_review_summary(
    *,
    registry: 'PriorCandidateRegistry',
) -> dict[str, Any]:
    return {
        'package_id': registry.package_id,
        'built_at': registry.built_at,
        'cluster_strategy': registry.cluster_strategy,
        'fallback_reason': registry.fallback_reason,
        'cluster_count': len(registry.clusters),
        'cluster_ids': [cluster.cluster_id for cluster in registry.clusters],
        'cluster_support_counts': {
            cluster.cluster_id: cluster.support_count for cluster in registry.clusters
        },
        'prior_candidate_count': len(registry.prior_candidates),
        'anti_pattern_candidate_count': len(registry.anti_pattern_candidates),
        'accepted_prior_ids': list(registry.accepted_prior_ids),
        'accepted_anti_pattern_ids': list(registry.accepted_anti_pattern_ids),
        'prior_review_status_counts': _review_status_counts(
            [candidate.review.review_status for candidate in registry.prior_candidates]
        ),
        'anti_pattern_review_status_counts': _review_status_counts(
            [candidate.review.review_status for candidate in registry.anti_pattern_candidates]
        ),
        'quality_flag_counts': dict(registry.quality_flag_counts),
        'prior_candidates': [
            {
                'prior_id': candidate.prior_id,
                'support_count': len(candidate.supporting_route_state_ids),
                'held_out_pass_rate': candidate.held_out_consistency.pass_rate,
                'review_status': candidate.review.review_status,
                'quality_tier': candidate.quality.quality_tier,
                'accepted': candidate.prior_id in registry.accepted_prior_ids,
            }
            for candidate in registry.prior_candidates
        ],
        'anti_pattern_candidates': [
            {
                'anti_pattern_id': candidate.anti_pattern_id,
                'failure_route_state_count': len(candidate.failure_examples.route_state_ids),
                'review_status': candidate.review.review_status,
                'quality_tier': candidate.quality.quality_tier,
                'accepted': candidate.anti_pattern_id in registry.accepted_anti_pattern_ids,
            }
            for candidate in registry.anti_pattern_candidates
        ],
    }


def build_decision_episode_export_summary(
    *,
    export: DecisionEpisodeAuditExport,
) -> dict[str, Any]:
    decision_episode = export.decision_episode
    return {
        'built_at': export.built_at,
        'episode_id': decision_episode.episode_id,
        'route_packet_id': decision_episode.observation_evidence_pack.route_packet_id,
        'route_state_id': decision_episode.route_state.route_state_id,
        'historical_cutoff_time': decision_episode.historical_cutoff_time,
        'audit_posture': 'audit_grade_pilot',
        'accepted_prior_ids': list(export.accepted_prior_ids),
        'accepted_anti_pattern_ids': list(export.accepted_anti_pattern_ids),
        'selected_prior_ids': list(decision_episode.relevant_priors.selected_prior_ids),
        'selected_antipattern_ids': list(decision_episode.relevant_priors.selected_antipattern_ids),
        'accepted_prior_count': len(export.accepted_prior_ids),
        'accepted_anti_pattern_count': len(export.accepted_anti_pattern_ids),
        'selected_prior_count': len(decision_episode.relevant_priors.selected_prior_ids),
        'selected_antipattern_count': len(decision_episode.relevant_priors.selected_antipattern_ids),
        'accepted_but_unselected_prior_count': len(export.accepted_but_unselected_priors),
        'accepted_but_unselected_antipattern_count': len(export.accepted_but_unselected_antipatterns),
        'accepted_but_unselected_priors': [
            record.model_dump(mode='json', exclude_none=True)
            for record in export.accepted_but_unselected_priors
        ],
        'accepted_but_unselected_antipatterns': [
            record.model_dump(mode='json', exclude_none=True)
            for record in export.accepted_but_unselected_antipatterns
        ],
        'visibility_bucket_counts': {
            'visible_input_refs': len(export.visible_input_refs),
            'audit_only_refs': len(export.audit_only_refs),
            'label_eval_only_refs': len(export.label_eval_only_refs),
        },
        'quality_tier': decision_episode.quality.quality_tier,
        'ready_for_training': decision_episode.quality.ready_for_training,
        'ready_for_eval': decision_episode.quality.ready_for_eval,
        'quality_flags': list(decision_episode.quality.quality_flags),
        'hindsight_input_visible': decision_episode.hindsight_outcome.input_visible,
        'prior_selection_note': export.prior_selection_note,
        'anti_pattern_selection_note': export.anti_pattern_selection_note,
        'review_status': export.review_status,
        'training_acceptance_verdict': export.training_acceptance_verdict,
        'reviewer_ids': list(export.reviewer_ids),
        'reviewed_at': export.reviewed_at,
        'rationale': export.rationale,
        'residual_defects': list(export.residual_defects),
        'section_reviews': {
            key: review.model_dump(mode='json', exclude_none=True)
            for key, review in export.section_reviews.items()
        },
    }


def build_decision_episode_export_inspection(
    *,
    export: DecisionEpisodeAuditExport,
) -> dict[str, Any]:
    decision_episode = export.decision_episode
    return {
        'built_at': export.built_at,
        'episode_id': decision_episode.episode_id,
        'route_state_id': decision_episode.route_state.route_state_id,
        'source_bundles': {
            'replay_bundle': export.source_replay_bundle_refs.model_dump(exclude_none=True),
            'review_bundle': export.source_review_bundle_refs.model_dump(exclude_none=True),
        },
        'prior_selection': {
            'accepted_prior_ids': list(export.accepted_prior_ids),
            'selected_prior_ids': list(decision_episode.relevant_priors.selected_prior_ids),
            'decision_episode_prior_selection_rationale': decision_episode.relevant_priors.prior_selection_rationale,
            'audit_note': export.prior_selection_note,
            'accepted_prior_cards': [
                prior_card.model_dump(mode='json', exclude_none=True)
                for prior_card in export.accepted_prior_cards
            ],
            'accepted_but_unselected_priors': [
                record.model_dump(mode='json', exclude_none=True)
                for record in export.accepted_but_unselected_priors
            ],
        },
        'anti_pattern_selection': {
            'accepted_anti_pattern_ids': list(export.accepted_anti_pattern_ids),
            'selected_antipattern_ids': list(decision_episode.relevant_priors.selected_antipattern_ids),
            'audit_note': export.anti_pattern_selection_note,
            'accepted_anti_pattern_cards': [
                anti_pattern.model_dump(mode='json', exclude_none=True)
                for anti_pattern in export.accepted_anti_pattern_cards
            ],
            'accepted_but_unselected_antipatterns': [
                record.model_dump(mode='json', exclude_none=True)
                for record in export.accepted_but_unselected_antipatterns
            ],
        },
        'visibility_buckets': {
            'visible_input_refs': list(export.visible_input_refs),
            'audit_only_refs': list(export.audit_only_refs),
            'label_eval_only_refs': list(export.label_eval_only_refs),
        },
        'policy': {
            'audit_posture': 'audit_grade_pilot',
            'hindsight_input_visible': decision_episode.hindsight_outcome.input_visible,
            'after_cutoff_refs_visible': any(
                ref.startswith('after_cutoff_paper:')
                for ref in export.visible_input_refs
            ),
            'label_eval_refs_visible': any(
                ref.startswith('hindsight_evidence:')
                for ref in export.visible_input_refs
            ),
        },
        'review': {
            'review_status': export.review_status,
            'training_acceptance_verdict': export.training_acceptance_verdict,
            'reviewer_ids': list(export.reviewer_ids),
            'reviewed_at': export.reviewed_at,
            'rationale': export.rationale,
            'residual_defects': list(export.residual_defects),
            'section_reviews': {
                key: review.model_dump(mode='json', exclude_none=True)
                for key, review in export.section_reviews.items()
            },
        },
        'training_sections': {
            'evidence_pack': decision_episode.observation_evidence_pack.model_dump(mode='json', exclude_none=True),
            'route_synthesis': export.route_state_snapshot.model_dump(mode='json', exclude_none=True),
            'why_now': (
                export.why_now_case.model_dump(mode='json', exclude_none=True)
                if export.why_now_case is not None
                else None
            ),
            'route_comparison': (
                export.route_comparison_case.model_dump(mode='json', exclude_none=True)
                if export.route_comparison_case is not None
                else None
            ),
            'minimal_attack_path': decision_episode.minimal_attack_path.model_dump(mode='json', exclude_none=True),
            'final_decision': decision_episode.decision_output.model_dump(mode='json', exclude_none=True),
        },
        'decision_episode_quality': decision_episode.quality.model_dump(mode='json', exclude_none=True),
    }


def _default_selected_iteration_label(bundle_dir: Path, selected_iteration_label: str | None) -> str:
    normalized = str(selected_iteration_label or '').strip()
    if normalized:
        return normalized
    if bundle_dir.name == 'export_bundle' and bundle_dir.parent.name:
        return bundle_dir.parent.name
    return bundle_dir.name


def _recommendation_payloads(
    ranked_recommendations: Sequence[IterationPriorityRecommendation | Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for recommendation in ranked_recommendations or []:
        if isinstance(recommendation, IterationPriorityRecommendation):
            payloads.append(recommendation.model_dump(mode='json', exclude_none=True))
        elif isinstance(recommendation, Mapping):
            payloads.append(dict(recommendation))
    return payloads


def build_decision_episode_training_view(
    *,
    export: DecisionEpisodeAuditExport,
    selected_iteration_label: str,
    audit_episode_ref: str | None = None,
) -> dict[str, Any]:
    decision_episode = export.decision_episode
    return {
        'schema_version': 'v1',
        'built_at': export.built_at,
        'episode_id': decision_episode.episode_id,
        'route_packet_id': decision_episode.observation_evidence_pack.route_packet_id,
        'route_state_id': export.route_state_snapshot.route_state_id,
        'selected_iteration_label': selected_iteration_label,
        'audit_episode_ref': audit_episode_ref,
        'source_bundle_refs': {
            'replay_bundle': export.source_replay_bundle_refs.model_dump(mode='json', exclude_none=True),
            'review_bundle': export.source_review_bundle_refs.model_dump(mode='json', exclude_none=True),
        },
        'review': {
            'review_status': export.review_status,
            'training_acceptance_verdict': export.training_acceptance_verdict,
            'reviewer_ids': list(export.reviewer_ids),
            'reviewed_at': export.reviewed_at,
            'rationale': export.rationale,
            'residual_defects': list(export.residual_defects),
            'section_reviews': {
                key: review.model_dump(mode='json', exclude_none=True)
                for key, review in export.section_reviews.items()
            },
        },
        'sections': {
            'evidence_pack': decision_episode.observation_evidence_pack.model_dump(mode='json', exclude_none=True),
            'route_synthesis': {
                'route_state': export.route_state_snapshot.model_dump(mode='json', exclude_none=True),
                'candidate_question': decision_episode.candidate_question.model_dump(mode='json', exclude_none=True),
                'alternative_questions': [
                    question.model_dump(mode='json', exclude_none=True)
                    for question in decision_episode.alternative_questions
                ],
            },
            'why_now': {
                'why_now_case': (
                    export.why_now_case.model_dump(mode='json', exclude_none=True)
                    if export.why_now_case is not None
                    else None
                ),
                'why_this_not_that': decision_episode.why_this_not_that.model_dump(mode='json', exclude_none=True),
                'not_now_cases': [
                    case.model_dump(mode='json', exclude_none=True)
                    for case in decision_episode.not_now_cases
                ],
            },
            'route_comparison': {
                'route_comparison_case': (
                    export.route_comparison_case.model_dump(mode='json', exclude_none=True)
                    if export.route_comparison_case is not None
                    else None
                ),
            },
            'priors_antipatterns': {
                'accepted_prior_ids': list(export.accepted_prior_ids),
                'selected_prior_ids': list(decision_episode.relevant_priors.selected_prior_ids),
                'accepted_prior_cards': [
                    prior_card.model_dump(mode='json', exclude_none=True)
                    for prior_card in export.accepted_prior_cards
                ],
                'accepted_but_unselected_priors': [
                    record.model_dump(mode='json', exclude_none=True)
                    for record in export.accepted_but_unselected_priors
                ],
                'accepted_anti_pattern_ids': list(export.accepted_anti_pattern_ids),
                'selected_antipattern_ids': list(decision_episode.relevant_priors.selected_antipattern_ids),
                'accepted_anti_pattern_cards': [
                    anti_pattern.model_dump(mode='json', exclude_none=True)
                    for anti_pattern in export.accepted_anti_pattern_cards
                ],
                'accepted_but_unselected_antipatterns': [
                    record.model_dump(mode='json', exclude_none=True)
                    for record in export.accepted_but_unselected_antipatterns
                ],
                'prior_selection_rationale': decision_episode.relevant_priors.prior_selection_rationale,
                'prior_selection_note': export.prior_selection_note,
                'anti_pattern_selection_note': export.anti_pattern_selection_note,
            },
            'minimal_attack_path': decision_episode.minimal_attack_path.model_dump(mode='json', exclude_none=True),
            'final_decision': decision_episode.decision_output.model_dump(mode='json', exclude_none=True),
            'review_labels': {
                'quality_tier': decision_episode.quality.quality_tier,
                'quality_flags': list(decision_episode.quality.quality_flags),
                'why_now_label': export.why_now_case.why_now_label if export.why_now_case is not None else None,
                'final_choice': decision_episode.decision_output.final_choice,
                'training_acceptance_verdict': export.training_acceptance_verdict,
                'review_status': export.review_status,
            },
        },
        'visibility_buckets': {
            'visible_input_refs': list(export.visible_input_refs),
            'audit_only_refs': list(export.audit_only_refs),
            'label_eval_only_refs': list(export.label_eval_only_refs),
        },
    }


def build_decision_episode_task_training_views(
    *,
    export: DecisionEpisodeAuditExport,
    selected_iteration_label: str,
    audit_episode_ref: str | None = None,
) -> dict[str, dict[str, Any]]:
    training_view_payload = build_decision_episode_training_view(
        export=export,
        selected_iteration_label=selected_iteration_label,
        audit_episode_ref=audit_episode_ref,
    )
    review_payload = dict(training_view_payload['review'])
    section_reviews = {
        str(key): value
        for key, value in dict(review_payload.get('section_reviews') or {}).items()
    }
    view_payloads: dict[str, dict[str, Any]] = {}

    for view_id, section_names in TASK_SPECIFIC_TRAINING_VIEW_SECTIONS.items():
        view_payloads[view_id] = {
            'schema_version': training_view_payload['schema_version'],
            'task_view_id': view_id,
            'built_at': training_view_payload['built_at'],
            'episode_id': training_view_payload['episode_id'],
            'route_packet_id': training_view_payload['route_packet_id'],
            'route_state_id': training_view_payload['route_state_id'],
            'selected_iteration_label': training_view_payload['selected_iteration_label'],
            'audit_episode_ref': training_view_payload['audit_episode_ref'],
            'source_training_view_ref': 'outputs/training_view.json',
            'source_bundle_refs': dict(training_view_payload['source_bundle_refs']),
            'review': {
                **{key: value for key, value in review_payload.items() if key != 'section_reviews'},
                'section_reviews': {
                    section_name: section_reviews[section_name]
                    for section_name in section_names
                    if section_name in section_reviews
                },
            },
            'sections': {
                section_name: training_view_payload['sections'][section_name]
                for section_name in section_names
                if section_name in training_view_payload['sections']
            },
            'visibility_buckets': {
                'visible_input_refs': list(training_view_payload['visibility_buckets']['visible_input_refs']),
                'audit_only_refs': list(training_view_payload['visibility_buckets']['audit_only_refs']),
                'label_eval_only_refs': list(training_view_payload['visibility_buckets']['label_eval_only_refs']),
            },
        }
    return view_payloads


def build_best_cycle_selection_payload(
    *,
    export: DecisionEpisodeAuditExport,
    selected_iteration_label: str,
    reviewed_candidate_cycles: Sequence[Mapping[str, Any]] | None = None,
    primary_recommendation_id: str | None = None,
    recommendation_evidence_refs: Sequence[str] | None = None,
    ranked_recommendations: Sequence[IterationPriorityRecommendation | Mapping[str, Any]] | None = None,
    source_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    normalized_reviewed_candidate_cycles = (
        [dict(candidate) for candidate in reviewed_candidate_cycles]
        if reviewed_candidate_cycles
        else [
            {
                'iteration_label': selected_iteration_label,
                'review_status': export.review_status,
                'training_acceptance_verdict': export.training_acceptance_verdict,
                'reviewer_ids': list(export.reviewer_ids),
                'reviewed_at': export.reviewed_at,
                'rationale': export.rationale,
                'residual_defects': list(export.residual_defects),
                'section_reviews': {
                    key: review.model_dump(mode='json', exclude_none=True)
                    for key, review in export.section_reviews.items()
                },
            }
        ]
    )
    return {
        'schema_version': 'v1',
        'built_at': export.built_at,
        'episode_id': export.decision_episode.episode_id,
        'route_state_id': export.route_state_snapshot.route_state_id,
        'selected_iteration_label': selected_iteration_label,
        'review_status': export.review_status,
        'training_acceptance_verdict': export.training_acceptance_verdict,
        'reviewer_ids': list(export.reviewer_ids),
        'reviewed_at': export.reviewed_at,
        'rationale': export.rationale,
        'residual_defects': list(export.residual_defects),
        'primary_recommendation_id': primary_recommendation_id,
        'recommendation_evidence_refs': _unique_strings(recommendation_evidence_refs),
        'ranked_recommendations': _recommendation_payloads(ranked_recommendations),
        'reviewed_candidate_cycles': normalized_reviewed_candidate_cycles,
        'source_artifacts': {str(key): str(value) for key, value in dict(source_artifacts or {}).items()},
    }


def build_corpus_sampling_summary(
    *,
    bundle: CorpusSamplingBundle,
) -> dict[str, Any]:
    selected_entries = list(bundle.fixed_regression_batch.selected if bundle.fixed_regression_batch else [])
    if bundle.random_exploration_batch is not None:
        selected_entries.extend(bundle.random_exploration_batch.selected)

    selected_with_neo4j_metadata_count = sum(
        1
        for entry in selected_entries
        if str(entry.neo4j_paper_id or '').strip() and entry.neo4j_ingested is not None
    )
    return {
        'schema_version': bundle.schema_version,
        'corpus_root': bundle.corpus_root,
        'inventory_entry_count': len(bundle.inventory_entries),
        'eligible_entry_count': sum(1 for entry in bundle.inventory_entries if entry.eligibility_status == 'eligible'),
        'fixed_selected_count': len(bundle.fixed_regression_batch.selected if bundle.fixed_regression_batch else []),
        'random_selected_count': len(bundle.random_exploration_batch.selected if bundle.random_exploration_batch else []),
        'corpus_health_failure_count': len(bundle.corpus_health_failures),
        'selected_with_neo4j_metadata_count': selected_with_neo4j_metadata_count,
        'selected_without_neo4j_metadata_count': len(selected_entries) - selected_with_neo4j_metadata_count,
        'neo4j_lookup_status': bundle.neo4j_lookup_status,
        'seed': bundle.seed,
        'fixed_manifest_ref': bundle.fixed_manifest_ref,
    }


def build_corpus_sampling_inspection(
    *,
    bundle: CorpusSamplingBundle,
) -> dict[str, Any]:
    fixed_batch = bundle.fixed_regression_batch
    random_batch = bundle.random_exploration_batch
    selected_entries = list(fixed_batch.selected if fixed_batch else [])
    if random_batch is not None:
        selected_entries.extend(random_batch.selected)

    selected_but_downstream_unavailable = []
    for entry in selected_entries:
        if str(entry.neo4j_paper_id or '').strip() and entry.neo4j_ingested is not None:
            continue
        selected_but_downstream_unavailable.append(
            {
                'corpus_paper_id': entry.corpus_paper_id,
                'display_title': entry.display_title,
                'corpus_relative_ref': entry.corpus_relative_ref,
                'reason': (
                    f'neo4j_lookup_{bundle.neo4j_lookup_status}'
                    if bundle.neo4j_lookup_status in {'skipped', 'unavailable'}
                    else 'neo4j_metadata_missing'
                ),
            }
        )

    return {
        'schema_version': bundle.schema_version,
        'corpus_root': bundle.corpus_root,
        'fixed_regression_ids': list(fixed_batch.selected_ids if fixed_batch else []),
        'random_exploration_ids': list(random_batch.selected_ids if random_batch else []),
        'seed': bundle.seed,
        'eligible_selected': {
            'fixed_regression': [
                entry.model_dump(mode='json', exclude_none=True)
                for entry in fixed_batch.selected
            ] if fixed_batch is not None else [],
            'random_exploration': [
                entry.model_dump(mode='json', exclude_none=True)
                for entry in random_batch.selected
            ] if random_batch is not None else [],
        },
        'selection_exclusions': {
            'fixed_regression': [
                exclusion.model_dump(mode='json', exclude_none=True)
                for exclusion in fixed_batch.exclusions
            ] if fixed_batch is not None else [],
            'random_exploration': [
                exclusion.model_dump(mode='json', exclude_none=True)
                for exclusion in random_batch.exclusions
            ] if random_batch is not None else [],
        },
        'corpus_health_failures': [
            issue.model_dump(mode='json', exclude_none=True)
            for issue in bundle.corpus_health_failures
        ],
        'selected_but_downstream_unavailable': selected_but_downstream_unavailable,
    }


def _sampled_l2_quality_tier(result: SampledPaperRunResult) -> str:
    quality_report = dict(result.quality_report or {})
    trace_quality = dict(result.trace_quality or {})
    return str(quality_report.get('quality_tier') or trace_quality.get('quality_tier') or '').strip().lower()


def _sampled_l2_quality_count(results: Sequence[SampledPaperRunResult], tier: str) -> int:
    return sum(1 for result in results if _sampled_l2_quality_tier(result) == tier)


def build_sampled_l2_iteration_summary(
    *,
    iteration: SampledSinglePaperIterationResult,
) -> dict[str, Any]:
    return {
        'schema_version': iteration.schema_version,
        'built_at': iteration.built_at,
        'iteration_label': iteration.iteration_label,
        'sampling_bundle_dir': iteration.sampling_bundle_dir,
        'sampling_bundle_manifest_ref': iteration.sampling_bundle_manifest_ref,
        'fixed_manifest_ref': iteration.fixed_manifest_ref,
        'seed': iteration.seed,
        'neo4j_lookup_status': iteration.neo4j_lookup_status,
        'fixed_selected_count': iteration.fixed_selected_count,
        'random_selected_count': iteration.random_selected_count,
        'executed_count': len(iteration.fixed_results) + len(iteration.random_results),
        'availability_issue_count': len(iteration.availability_issues),
        'fixed_green_count': _sampled_l2_quality_count(iteration.fixed_results, 'green'),
        'fixed_yellow_count': _sampled_l2_quality_count(iteration.fixed_results, 'yellow'),
        'fixed_red_count': _sampled_l2_quality_count(iteration.fixed_results, 'red'),
        'random_green_count': _sampled_l2_quality_count(iteration.random_results, 'green'),
        'random_yellow_count': _sampled_l2_quality_count(iteration.random_results, 'yellow'),
        'random_red_count': _sampled_l2_quality_count(iteration.random_results, 'red'),
    }


def build_sampled_l2_iteration_inspection(
    *,
    iteration: SampledSinglePaperIterationResult,
) -> dict[str, Any]:
    return {
        'schema_version': iteration.schema_version,
        'built_at': iteration.built_at,
        'iteration_label': iteration.iteration_label,
        'sampling_bundle_dir': iteration.sampling_bundle_dir,
        'sampling_bundle_manifest_ref': iteration.sampling_bundle_manifest_ref,
        'fixed_manifest_ref': iteration.fixed_manifest_ref,
        'seed': iteration.seed,
        'neo4j_lookup_status': iteration.neo4j_lookup_status,
        'neo4j_lookup_error': iteration.neo4j_lookup_error,
        'fixed_selected_count': iteration.fixed_selected_count,
        'random_selected_count': iteration.random_selected_count,
        'fixed_regression_results': [result.model_dump(mode='json', exclude_none=True) for result in iteration.fixed_results],
        'random_exploration_results': [result.model_dump(mode='json', exclude_none=True) for result in iteration.random_results],
        'availability_issues': [issue.model_dump(mode='json', exclude_none=True) for issue in iteration.availability_issues],
    }


def _comparison_verdict_counts(
    rows: Sequence[FixedSampledL2ComparisonRow] | Sequence[RandomSampledL2ComparisonRow],
    *,
    verdicts: Sequence[str],
) -> dict[str, int]:
    counts = {verdict: 0 for verdict in verdicts}
    for row in rows:
        verdict = str(row.verdict)
        if verdict in counts:
            counts[verdict] += 1
    return counts


def build_sampled_l2_comparison_summary(
    *,
    comparison: SampledL2ComparisonResult,
) -> dict[str, Any]:
    return {
        'schema_version': comparison.schema_version,
        'built_at': comparison.built_at,
        'iteration_label': comparison.iteration_label,
        'previous_iteration_label': comparison.previous_iteration_label,
        'baseline_only': comparison.baseline_only,
        'fixed_verdict_counts': _comparison_verdict_counts(
            comparison.fixed_comparisons,
            verdicts=['stable_pass', 'recurring_failure', 'new_regression', 'improved', 'availability_only'],
        ),
        'random_verdict_counts': _comparison_verdict_counts(
            comparison.random_comparisons,
            verdicts=['new_edge_case', 'repeated_random_failure', 'random_improved', 'stable_random_pass', 'availability_only'],
        ),
        'owner_buckets': [bucket.model_dump(mode='json', exclude_none=True) for bucket in comparison.owner_buckets],
    }


def build_sampled_l2_comparison_inspection(
    *,
    comparison: SampledL2ComparisonResult,
) -> dict[str, Any]:
    return {
        'schema_version': comparison.schema_version,
        'built_at': comparison.built_at,
        'iteration_label': comparison.iteration_label,
        'previous_iteration_label': comparison.previous_iteration_label,
        'baseline_only': comparison.baseline_only,
        'fixed_comparisons': [row.model_dump(mode='json', exclude_none=True) for row in comparison.fixed_comparisons],
        'random_comparisons': [row.model_dump(mode='json', exclude_none=True) for row in comparison.random_comparisons],
        'owner_buckets': [bucket.model_dump(mode='json', exclude_none=True) for bucket in comparison.owner_buckets],
    }


def _model_payload(model: Any) -> Any:
    if hasattr(model, 'model_dump'):
        return model.model_dump(mode='json', exclude_none=True)
    return model


def write_replay_bundle(
    output_dir: str | Path,
    *,
    route_packet: RoutePacket,
    traces: Sequence[PaperLogicTrace],
    compilation: HistoricalReplayCompilation,
    historical_environment_snapshot: HistoricalEnvironmentSnapshot | None = None,
    support_route_states: Sequence[RouteState] | None = None,
    alternative_route_states: Sequence[RouteState] | None = None,
    held_out_route_states: Sequence[RouteState] | None = None,
    reviewer_ids: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
    route_state_package_validation: 'RouteStatePackageValidation | None' = None,
) -> dict[str, Path]:
    bundle_dir = _as_path(output_dir)
    inputs_dir = bundle_dir / 'inputs'
    outputs_dir = bundle_dir / 'outputs'
    built_at = _utc_now_iso()

    summary_payload = build_replay_summary(
        route_packet=route_packet,
        traces=traces,
        compilation=compilation,
        historical_environment_snapshot=historical_environment_snapshot,
        support_route_states=support_route_states,
        alternative_route_states=alternative_route_states,
        held_out_route_states=held_out_route_states,
        reviewer_ids=reviewer_ids,
    )
    inspection_payload = build_replay_inspection(
        route_packet=route_packet,
        compilation=compilation,
        traces=traces,
        route_state_package_validation=route_state_package_validation,
    )

    written_files = {
        'route_packet': _write_json(inputs_dir / 'route_packet.json', _model_payload(route_packet)),
        'paper_logic_traces': _write_json(
            inputs_dir / 'paper_logic_traces.json',
            [_model_payload(trace) for trace in traces],
        ),
        'support_route_states': _write_json(
            inputs_dir / 'support_route_states.json',
            [_model_payload(route_state) for route_state in support_route_states or []],
        ),
        'alternative_route_states': _write_json(
            inputs_dir / 'alternative_route_states.json',
            [_model_payload(route_state) for route_state in alternative_route_states or []],
        ),
        'held_out_route_states': _write_json(
            inputs_dir / 'held_out_route_states.json',
            [_model_payload(route_state) for route_state in held_out_route_states or []],
        ),
        'primary_route_state': _write_json(outputs_dir / 'primary_route_state.json', _model_payload(compilation.primary_route_state)),
        'why_now_case': _write_json(outputs_dir / 'why_now_case.json', _model_payload(compilation.why_now_case)),
        'route_comparison_cases': _write_json(
            outputs_dir / 'route_comparison_cases.json',
            [_model_payload(case) for case in compilation.route_comparison_cases],
        ),
        'decision_prior_card': _write_json(outputs_dir / 'decision_prior_card.json', _model_payload(compilation.decision_prior_card)),
        'decision_episode': _write_json(outputs_dir / 'decision_episode.json', _model_payload(compilation.decision_episode)),
        'replay_summary': _write_json(bundle_dir / 'replay_summary.json', summary_payload),
        'replay_inspection': _write_json(bundle_dir / 'replay_inspection.json', inspection_payload),
    }
    if historical_environment_snapshot is not None:
        written_files['historical_environment_snapshot'] = _write_json(
            inputs_dir / 'historical_environment_snapshot.json',
            _model_payload(historical_environment_snapshot),
        )

    manifest_payload = {
        'schema_version': 'v1',
        'built_at': built_at,
        'packet_id': route_packet.packet_id,
        'cutoff_year': route_packet.cutoff_year,
        'ready_for_pilot': compilation.ready_for_pilot,
        'quality_flags': list(compilation.quality_flags),
        'metadata': dict(metadata or {}),
        'files': {
            name: str(path.relative_to(bundle_dir)).replace('\\', '/')
            for name, path in written_files.items()
        },
    }
    written_files['bundle_manifest'] = _write_json(bundle_dir / 'bundle_manifest.json', manifest_payload)
    return written_files


def write_prior_candidate_review_bundle(
    output_dir: str | Path,
    *,
    registry: 'PriorCandidateRegistry',
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    bundle_dir = _as_path(output_dir)
    summary_payload = build_prior_candidate_review_summary(registry=registry)
    written_files = {
        'prior_candidates': _write_json(
            bundle_dir / 'prior_candidates.json',
            [candidate.model_dump(mode='json', exclude_none=True) for candidate in registry.prior_candidates],
        ),
        'anti_pattern_candidates': _write_json(
            bundle_dir / 'anti_pattern_candidates.json',
            [candidate.model_dump(mode='json', exclude_none=True) for candidate in registry.anti_pattern_candidates],
        ),
        'clusters': _write_json(
            bundle_dir / 'clusters.json',
            [cluster.model_dump(mode='json', exclude_none=True) for cluster in registry.clusters],
        ),
        'candidate_review_summary': _write_json(
            bundle_dir / 'candidate_review_summary.json',
            summary_payload,
        ),
    }
    written_files['bundle_manifest'] = _write_json(
        bundle_dir / 'bundle_manifest.json',
        {
            'schema_version': 'v1',
            'built_at': registry.built_at,
            'package_id': registry.package_id,
            'accepted_prior_ids': list(registry.accepted_prior_ids),
            'accepted_anti_pattern_ids': list(registry.accepted_anti_pattern_ids),
            'metadata': dict(metadata or {}),
            'files': {
                name: str(path.relative_to(bundle_dir)).replace('\\', '/')
                for name, path in written_files.items()
            },
        },
    )
    return written_files


def write_decision_episode_export_bundle(
    output_dir: str | Path,
    *,
    export: DecisionEpisodeAuditExport,
    metadata: dict[str, Any] | None = None,
    selected_iteration_label: str | None = None,
    reviewed_candidate_cycles: Sequence[Mapping[str, Any]] | None = None,
    primary_recommendation_id: str | None = None,
    recommendation_evidence_refs: Sequence[str] | None = None,
    ranked_recommendations: Sequence[IterationPriorityRecommendation | Mapping[str, Any]] | None = None,
    selection_source_artifacts: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    bundle_dir = _as_path(output_dir)
    outputs_dir = bundle_dir / 'outputs'
    iteration_label = _default_selected_iteration_label(bundle_dir, selected_iteration_label)

    summary_payload = build_decision_episode_export_summary(export=export)
    inspection_payload = build_decision_episode_export_inspection(export=export)
    training_view_payload = build_decision_episode_training_view(
        export=export,
        selected_iteration_label=iteration_label,
        audit_episode_ref='outputs/decision_episode.json',
    )
    task_training_view_payloads = build_decision_episode_task_training_views(
        export=export,
        selected_iteration_label=iteration_label,
        audit_episode_ref='outputs/decision_episode.json',
    )

    written_files = {
        'decision_episode': _write_json(
            outputs_dir / 'decision_episode.json',
            _model_payload(export.decision_episode),
        ),
        'export_summary': _write_json(bundle_dir / 'export_summary.json', summary_payload),
        'export_inspection': _write_json(bundle_dir / 'export_inspection.json', inspection_payload),
        'training_view': _write_json(outputs_dir / 'training_view.json', training_view_payload),
    }
    for view_id, payload in task_training_view_payloads.items():
        written_files[view_id] = _write_json(outputs_dir / f'{view_id}.json', payload)

    selection_evidence_refs = _unique_strings(
        list(recommendation_evidence_refs or [])
        + [
            f'training_view:{written_files["training_view"].resolve()}',
            f'export_summary:{written_files["export_summary"].resolve()}',
            f'export_inspection:{written_files["export_inspection"].resolve()}',
            *[
                f'{view_id}:{written_files[view_id].resolve()}'
                for view_id in TASK_SPECIFIC_TRAINING_VIEW_SECTIONS
            ],
            *(
                [
                    (
                        'review_bundle:candidate_review_summary:'
                        f'{export.source_review_bundle_refs.files["candidate_review_summary"]}'
                    )
                ]
                if str(export.source_review_bundle_refs.files.get('candidate_review_summary') or '').strip()
                else []
            ),
        ]
    )
    selection_payload = build_best_cycle_selection_payload(
        export=export,
        selected_iteration_label=iteration_label,
        reviewed_candidate_cycles=reviewed_candidate_cycles,
        primary_recommendation_id=primary_recommendation_id,
        recommendation_evidence_refs=selection_evidence_refs,
        ranked_recommendations=ranked_recommendations,
        source_artifacts={
            'decision_episode': str(written_files['decision_episode'].resolve()),
            'training_view': str(written_files['training_view'].resolve()),
            'export_summary': str(written_files['export_summary'].resolve()),
            'export_inspection': str(written_files['export_inspection'].resolve()),
            **{
                view_id: str(written_files[view_id].resolve())
                for view_id in TASK_SPECIFIC_TRAINING_VIEW_SECTIONS
            },
            **{str(key): str(value) for key, value in dict(selection_source_artifacts or {}).items()},
        },
    )
    written_files['best_cycle_selection'] = _write_json(
        bundle_dir / 'best_cycle_selection.json',
        selection_payload,
    )
    manifest_payload = {
        'schema_version': 'v1',
        'built_at': export.built_at,
        'exported_episode_id': export.decision_episode.episode_id,
        'route_state_id': export.decision_episode.route_state.route_state_id,
        'accepted_prior_ids': list(export.accepted_prior_ids),
        'accepted_anti_pattern_ids': list(export.accepted_anti_pattern_ids),
        'selected_iteration_label': iteration_label,
        'review_status': export.review_status,
        'training_acceptance_verdict': export.training_acceptance_verdict,
        'reviewer_ids': list(export.reviewer_ids),
        'reviewed_at': export.reviewed_at,
        'primary_recommendation_id': primary_recommendation_id,
        'recommendation_evidence_refs': selection_evidence_refs,
        'source_replay_bundle_refs': export.source_replay_bundle_refs.model_dump(exclude_none=True),
        'source_review_bundle_refs': export.source_review_bundle_refs.model_dump(exclude_none=True),
        'metadata': dict(metadata or {}),
        'files': {
            name: str(path.relative_to(bundle_dir)).replace('\\', '/')
            for name, path in written_files.items()
        },
    }
    written_files['bundle_manifest'] = _write_json(bundle_dir / 'bundle_manifest.json', manifest_payload)
    return written_files
def write_corpus_sampling_bundle(
    output_dir: str | Path,
    *,
    bundle: CorpusSamplingBundle,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    bundle_dir = _as_path(output_dir)
    outputs_dir = bundle_dir / 'outputs'

    summary_payload = build_corpus_sampling_summary(bundle=bundle)
    inspection_payload = build_corpus_sampling_inspection(bundle=bundle)
    written_files = {
        'sampling_summary': _write_json(bundle_dir / 'sampling_summary.json', summary_payload),
        'sampling_inspection': _write_json(bundle_dir / 'sampling_inspection.json', inspection_payload),
        'fixed_regression_batch': _write_json(
            outputs_dir / 'fixed_regression_batch.json',
            _model_payload(bundle.fixed_regression_batch),
        ),
        'random_exploration_batch': _write_json(
            outputs_dir / 'random_exploration_batch.json',
            _model_payload(bundle.random_exploration_batch),
        ),
        'corpus_health_failures': _write_json(
            outputs_dir / 'corpus_health_failures.json',
            [_model_payload(issue) for issue in bundle.corpus_health_failures],
        ),
    }
    manifest_payload = {
        'schema_version': bundle.schema_version,
        'built_at': bundle.built_at,
        'corpus_root': bundle.corpus_root,
        'fixed_selected_count': len(bundle.fixed_regression_batch.selected if bundle.fixed_regression_batch else []),
        'random_selected_count': len(bundle.random_exploration_batch.selected if bundle.random_exploration_batch else []),
        'seed': bundle.seed,
        'fixed_manifest_ref': bundle.fixed_manifest_ref,
        'neo4j_lookup_status': bundle.neo4j_lookup_status,
        'neo4j_lookup_error': bundle.neo4j_lookup_error,
        'metadata': dict(metadata or {}),
        'files': {
            name: str(path.relative_to(bundle_dir)).replace('\\', '/')
            for name, path in written_files.items()
        },
    }
    written_files['bundle_manifest'] = _write_json(bundle_dir / 'bundle_manifest.json', manifest_payload)
    return written_files


def write_sampled_l2_iteration_bundle(
    output_dir: str | Path,
    *,
    iteration: SampledSinglePaperIterationResult,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    bundle_dir = _as_path(output_dir)
    outputs_dir = bundle_dir / 'outputs'

    summary_payload = build_sampled_l2_iteration_summary(iteration=iteration)
    inspection_payload = build_sampled_l2_iteration_inspection(iteration=iteration)
    written_files = {
        'iteration_summary': _write_json(bundle_dir / 'iteration_summary.json', summary_payload),
        'iteration_inspection': _write_json(bundle_dir / 'iteration_inspection.json', inspection_payload),
        'fixed_regression_results': _write_json(
            outputs_dir / 'fixed_regression_results.json',
            [result.model_dump(mode='json', exclude_none=True) for result in iteration.fixed_results],
        ),
        'random_exploration_results': _write_json(
            outputs_dir / 'random_exploration_results.json',
            [result.model_dump(mode='json', exclude_none=True) for result in iteration.random_results],
        ),
        'availability_issues': _write_json(
            outputs_dir / 'availability_issues.json',
            [issue.model_dump(mode='json', exclude_none=True) for issue in iteration.availability_issues],
        ),
    }
    manifest_payload = {
        'schema_version': iteration.schema_version,
        'built_at': iteration.built_at,
        'iteration_label': iteration.iteration_label,
        'sampling_bundle_manifest_ref': iteration.sampling_bundle_manifest_ref,
        'fixed_selected_count': iteration.fixed_selected_count,
        'random_selected_count': iteration.random_selected_count,
        'metadata': dict(metadata or {}),
        'files': {
            name: str(path.relative_to(bundle_dir)).replace('\\', '/')
            for name, path in written_files.items()
        },
    }
    written_files['bundle_manifest'] = _write_json(bundle_dir / 'bundle_manifest.json', manifest_payload)
    return written_files


def write_sampled_l2_comparison_bundle(
    output_dir: str | Path,
    *,
    comparison: SampledL2ComparisonResult,
) -> dict[str, Path]:
    bundle_dir = _as_path(output_dir)
    written_files = {
        'comparison_summary': _write_json(
            bundle_dir / 'comparison_summary.json',
            build_sampled_l2_comparison_summary(comparison=comparison),
        ),
        'comparison_inspection': _write_json(
            bundle_dir / 'comparison_inspection.json',
            build_sampled_l2_comparison_inspection(comparison=comparison),
        ),
    }
    return written_files


def build_iteration_priority_summary_payload(
    *,
    summary: IterationPrioritySummary,
) -> dict[str, Any]:
    return summary.model_dump(mode='json', exclude_none=True)


def build_iteration_priority_inspection_payload(
    *,
    inspection: IterationPriorityInspection,
) -> dict[str, Any]:
    return inspection.model_dump(mode='json', exclude_none=True)


def write_iteration_priority_bundle(
    output_dir: str | Path,
    *,
    summary: IterationPrioritySummary,
    inspection: IterationPriorityInspection,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    bundle_dir = _as_path(output_dir)
    outputs_dir = bundle_dir / 'outputs'
    summary_payload = build_iteration_priority_summary_payload(summary=summary)
    inspection_payload = build_iteration_priority_inspection_payload(inspection=inspection)

    written_files = {
        'prioritization_summary': _write_json(outputs_dir / 'prioritization_summary.json', summary_payload),
        'prioritization_inspection': _write_json(outputs_dir / 'prioritization_inspection.json', inspection_payload),
    }
    written_files['bundle_manifest'] = _write_json(
        bundle_dir / 'bundle_manifest.json',
        {
            'schema_version': summary.schema_version,
            'built_at': summary.built_at,
            'packet_id': summary.packet_id,
            'cutoff_year': summary.cutoff_year,
            'primary_recommendation_id': summary.primary_recommendation_id,
            'source_refs': summary.source_refs.model_dump(mode='json', exclude_none=True),
            'metadata': dict(metadata or {}),
            'files': {
                name: str(path.relative_to(bundle_dir)).replace('\\', '/')
                for name, path in written_files.items()
            },
        },
    )
    return written_files


__all__ = [
    'build_best_cycle_selection_payload',
    'build_corpus_sampling_inspection',
    'build_corpus_sampling_summary',
    'build_decision_episode_export_inspection',
    'build_decision_episode_export_summary',
    'build_decision_episode_task_training_views',
    'build_decision_episode_training_view',
    'build_iteration_priority_inspection_payload',
    'build_iteration_priority_summary_payload',
    'build_prior_candidate_review_summary',
    'build_replay_summary',
    'build_replay_inspection',
    'build_sampled_l2_comparison_inspection',
    'build_sampled_l2_comparison_summary',
    'build_sampled_l2_iteration_inspection',
    'build_sampled_l2_iteration_summary',
    'ensure_packet_trace_coverage',
    'load_historical_environment_snapshot',
    'load_paper_logic_trace',
    'load_paper_logic_traces',
    'load_route_packet',
    'load_route_state',
    'load_route_states',
    'write_corpus_sampling_bundle',
    'write_decision_episode_export_bundle',
    'write_iteration_priority_bundle',
    'write_prior_candidate_review_bundle',
    'write_replay_bundle',
    'write_sampled_l2_comparison_bundle',
    'write_sampled_l2_iteration_bundle',
]
