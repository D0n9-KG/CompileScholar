from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.paper_logic_trace.derived_views import build_derived_views
from app.paper_logic_trace.models import PaperLogicTrace

from .historical_environment import HistoricalEnvironmentSnapshot
from .decision_episode_builder import DecisionEpisodeBuilder
from .decision_prior_builder import DecisionPriorBuilder
from .models import (
    AntiPatternCard,
    DecisionEpisode,
    DecisionPriorCard,
    HindsightOutcome,
    RouteComparisonCase,
    RoutePacket,
    RouteState,
    WhyNowCase,
)
from .route_comparison_builder import RouteComparisonBuilder
from .route_state_synthesizer import RouteStateSynthesizer
from .why_now_builder import WhyNowCaseBuilder

ReplayFailureLayer = Literal['packet', 'l1', 'l2', 'l3_l4']
ReplayFailureStage = Literal['route_state', 'why_now_case', 'route_comparison', 'decision_prior_card', 'decision_episode']
ReplayFailureSeverity = Literal['low', 'medium', 'high', 'blocking']


class ReplayFailureRecord(BaseModel):
    model_config = ConfigDict(extra='forbid')

    failure_id: str
    layer: ReplayFailureLayer
    stage: ReplayFailureStage
    severity: ReplayFailureSeverity
    blocking: bool = False
    failure_code: str
    owner: str
    repair_target: str
    source_flags: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class HistoricalReplayCompilation(BaseModel):
    model_config = ConfigDict(extra='forbid')

    primary_route_state: RouteState
    why_now_case: WhyNowCase
    route_comparison_cases: list[RouteComparisonCase] = Field(default_factory=list)
    selected_comparison_case_id: str | None = None
    decision_prior_card: DecisionPriorCard
    decision_episode: DecisionEpisode
    quality_flags: list[str] = Field(default_factory=list)
    failure_records: list[ReplayFailureRecord] = Field(default_factory=list)
    ready_for_pilot: bool = False


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or '').strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _replay_quality_tier(quality_flags: list[str], ready_for_pilot: bool) -> Literal['green', 'yellow', 'red']:
    if not quality_flags:
        return 'green'
    return 'yellow' if ready_for_pilot else 'red'


def _dedupe_route_states(
    primary_route_state: RouteState,
    route_states: list[RouteState] | None,
    *,
    label: str,
) -> list[RouteState]:
    route_states = route_states or []
    seen_ids = {primary_route_state.route_state_id}
    deduped: list[RouteState] = []
    for route_state in route_states:
        if route_state.cutoff_year != primary_route_state.cutoff_year:
            raise ValueError(f'HistoricalReplayCompiler {label} must share cutoff_year with the primary route state')
        if route_state.route_state_id in seen_ids:
            raise ValueError(f'HistoricalReplayCompiler {label} contains duplicate route_state_id {route_state.route_state_id}')
        seen_ids.add(route_state.route_state_id)
        deduped.append(route_state)
    return deduped


def _select_comparison_case(comparison_cases: list[RouteComparisonCase]) -> RouteComparisonCase | None:
    if not comparison_cases:
        return None

    def sort_key(case: RouteComparisonCase) -> tuple[int, int, int, int]:
        preference_score = 2 if case.preference_label == 'prefer_a' else 1 if case.preference_label == 'tie' else 0
        quality_score = {'green': 2, 'yellow': 1, 'red': 0}.get(case.quality.quality_tier, 0)
        decisive_score = len(case.why_a_not_b.decisive_dimensions)
        evidence_score = len(case.why_a_not_b.decisive_evidence_ids)
        return (preference_score, quality_score, decisive_score, evidence_score)

    return sorted(comparison_cases, key=sort_key, reverse=True)[0]


def _trusted_mentions_count(move: Any, field: str) -> int:
    trusted_extraction_modes = {'direct', 'normalized'}
    trusted_support_strengths = {'strong', 'exact'}
    mentions = list(getattr(move, field, []) or [])
    if not mentions:
        return 0

    trusted_count = 0
    for index, mention in enumerate(mentions):
        if getattr(mention, 'inferred', False):
            continue
        provenance_rows = [
            provenance
            for provenance in list(getattr(move, 'slot_provenance', []) or [])
            if str(getattr(provenance, 'field', '')).strip() == field and int(getattr(provenance, 'value_index', -1)) == index
        ]
        if provenance_rows:
            extraction_mode = str(getattr(provenance_rows[0], 'extraction_mode', '') or '').strip().lower()
            support_strength = str(getattr(provenance_rows[0], 'support_strength', '') or '').strip().lower()
            if extraction_mode not in trusted_extraction_modes or support_strength not in trusted_support_strengths:
                continue
        trusted_count += 1
    return trusted_count


def _l2_gap_signals(traces: list[PaperLogicTrace]) -> dict[str, Any]:
    comparator_gap_move_ids: list[str] = []
    comparator_gap_evidence: list[str] = []
    completeness_gap_fields: list[str] = []
    completeness_gap_evidence: list[str] = []
    relation_gap_move_ids: list[str] = []
    relation_gap_evidence: list[str] = []

    expected_slot_fields = ('metrics', 'comparators', 'conditions', 'limitation_types', 'resource_mentions')
    grounded_roles = {'method', 'experiment', 'result', 'interpretation', 'future_work', 'limitation'}
    comparator_roles = {'experiment', 'result', 'interpretation'}

    for trace in traces:
        views = build_derived_views(trace)
        route_compiler_contract = dict(views.get('route_compiler_contract') or {})
        route_state_seed = dict(views.get('route_state_seed') or {})
        comparison_move_ids = {
            str(entry.get('move_id')).strip()
            for entry in list(route_compiler_contract.get('comparison_signals') or [])
            if str(entry.get('move_id')).strip()
        }
        future_signal_move_ids = {
            str(entry.get('move_id')).strip()
            for entry in list(route_compiler_contract.get('future_direction_signals') or [])
            if str(entry.get('move_id')).strip()
        }
        alternative_signal_entries = list(route_state_seed.get('alternative_route_signal_entries') or [])

        for move in trace.canonical_core.moves:
            trusted_counts = {field: _trusted_mentions_count(move, field) for field in expected_slot_fields}
            method_count = _trusted_mentions_count(move, 'methods')
            research_object_count = _trusted_mentions_count(move, 'research_objects')
            grounded = (
                move.role in grounded_roles
                and (
                    any(trusted_counts.values())
                    or method_count
                    or research_object_count
                    or bool(move.effects)
                    or bool(str(move.summary or '').strip())
                )
            )
            if grounded:
                missing_fields = [field for field, count in trusted_counts.items() if count == 0]
                if missing_fields:
                    completeness_gap_fields.extend(missing_fields)
                    completeness_gap_evidence.extend(str(anchor_id).strip() for anchor_id in list(move.anchor_ids) if str(anchor_id).strip())

            if move.role in comparator_roles:
                has_comparative_content = bool(trusted_counts['metrics'] or move.effects or method_count)
                if has_comparative_content and move.move_id not in comparison_move_ids:
                    comparator_gap_move_ids.append(move.move_id)
                    comparator_gap_evidence.extend(str(anchor_id).strip() for anchor_id in list(move.anchor_ids) if str(anchor_id).strip())

            if move.role in {'future_work', 'limitation'}:
                has_relation_tokens = bool(
                    trusted_counts['conditions']
                    or trusted_counts['limitation_types']
                    or trusted_counts['resource_mentions']
                    or method_count
                    or research_object_count
                )
                if not has_relation_tokens:
                    continue
                if move.role == 'future_work' and move.move_id not in future_signal_move_ids:
                    relation_gap_move_ids.append(move.move_id)
                    relation_gap_evidence.extend(str(anchor_id).strip() for anchor_id in list(move.anchor_ids) if str(anchor_id).strip())

        if alternative_signal_entries and not any(list(entry.get('evidence_ids') or []) for entry in alternative_signal_entries):
            relation_gap_move_ids.append(trace.paper_metadata.paper_id)
            for entry in alternative_signal_entries:
                relation_gap_evidence.extend(str(anchor_id).strip() for anchor_id in list(entry.get('anchor_ids') or []) if str(anchor_id).strip())

    return {
        'comparator_gap_count': len(_unique(comparator_gap_move_ids)),
        'comparator_gap_evidence': _unique(comparator_gap_evidence)[:8],
        'completeness_gap_fields': _unique(completeness_gap_fields),
        'completeness_gap_evidence': _unique(completeness_gap_evidence)[:8],
        'relation_gap_count': len(_unique(relation_gap_move_ids)),
        'relation_gap_evidence': _unique(relation_gap_evidence)[:8],
    }


def _failure_record(
    *,
    failure_code: str,
    layer: ReplayFailureLayer,
    stage: ReplayFailureStage,
    severity: ReplayFailureSeverity,
    blocking: bool,
    owner: str,
    repair_target: str,
    source_flags: list[str] | None = None,
    evidence_refs: list[str] | None = None,
) -> ReplayFailureRecord:
    return ReplayFailureRecord(
        failure_id=f'{stage}:{failure_code}',
        layer=layer,
        stage=stage,
        severity=severity,
        blocking=blocking,
        failure_code=failure_code,
        owner=owner,
        repair_target=repair_target,
        source_flags=_unique(list(source_flags or [])),
        evidence_refs=_unique(list(evidence_refs or [])),
    )


def _build_failure_records(
    *,
    route_packet: RoutePacket,
    traces: list[PaperLogicTrace],
    compilation: HistoricalReplayCompilation,
    support_route_states: list[RouteState],
    alternative_route_states: list[RouteState],
    held_out_route_states: list[RouteState],
    reviewer_ids: list[str],
) -> list[ReplayFailureRecord]:
    trace_by_id = {trace.trace_id: trace for trace in traces}
    missing_trace_ids = [
        str(item.trace_id).strip()
        for item in route_packet.included_items
        if str(item.trace_id or '').strip() and str(item.trace_id).strip() not in trace_by_id
    ]
    missing_trace_papers = [
        item.paper_id
        for item in route_packet.included_items
        if not str(item.trace_id or '').strip()
    ]
    l2_gap_signals = _l2_gap_signals(traces)
    failure_records: list[ReplayFailureRecord] = []

    if missing_trace_ids or missing_trace_papers:
        failure_records.append(
            _failure_record(
                failure_code='packet_trace_coverage_missing',
                layer='packet',
                stage='route_state',
                severity='blocking',
                blocking=True,
                owner='route_packet',
                repair_target='fill trace_id coverage for every included packet item before replay',
                source_flags=[*missing_trace_ids, *missing_trace_papers],
                evidence_refs=[str(item.paper_id).strip() for item in route_packet.included_items if item.paper_id],
            )
        )

    if (not route_packet.packet_composition.coverage_ok) or route_packet.packet_composition.missing_roles:
        packet_role_blocking = not route_packet.packet_quality.ready_for_route_state
        failure_records.append(
            _failure_record(
                failure_code='packet_role_coverage_missing',
                layer='packet',
                stage='route_state',
                severity='high' if packet_role_blocking else 'medium',
                blocking=packet_role_blocking,
                owner='route_packet',
                repair_target='restore the missing packet roles before relying on replay quality',
                source_flags=list(route_packet.packet_composition.missing_roles),
                evidence_refs=[item.paper_id for item in route_packet.included_items],
            )
        )

    if route_packet.l1_snapshot_ref is None:
        failure_records.append(
            _failure_record(
                failure_code='l1_snapshot_missing',
                layer='l1',
                stage='route_state',
                severity='high',
                blocking=True,
                owner='historical_environment',
                repair_target='attach a cutoff-matched L1 snapshot to the packet before replay',
                source_flags=['missing_l1_snapshot_ref'],
                evidence_refs=[route_packet.packet_id],
            )
        )

    if 'missing_l1_support' in compilation.primary_route_state.quality.quality_flags:
        failure_records.append(
            _failure_record(
                failure_code='l1_support_missing',
                layer='l1',
                stage='route_state',
                severity='high',
                blocking=True,
                owner='historical_environment',
                repair_target='add benchmark, protocol, or toolchain support refs that survive into the route state',
                source_flags=list(compilation.primary_route_state.quality.quality_flags),
                evidence_refs=list(compilation.primary_route_state.evidence_bundle.l1_support_refs),
            )
        )

    if l2_gap_signals['comparator_gap_count'] > 0:
        failure_records.append(
            _failure_record(
                failure_code='l2_comparator_sparse',
                layer='l2',
                stage='route_comparison',
                severity='medium',
                blocking=False,
                owner='paper_logic_trace.derived_views',
                repair_target='preserve comparator-bearing evidence into comparison_signals and replay-facing route outputs',
                source_flags=[f'gap_moves:{l2_gap_signals["comparator_gap_count"]}'],
                evidence_refs=l2_gap_signals['comparator_gap_evidence'],
            )
        )

    if l2_gap_signals['completeness_gap_fields']:
        failure_records.append(
            _failure_record(
                failure_code='l2_expected_slot_missing',
                layer='l2',
                stage='route_state',
                severity='medium',
                blocking=False,
                owner='paper_logic_trace.derived_views',
                repair_target='surface completeness signals for missing metrics, comparators, conditions, limitation, or resource slots',
                source_flags=[f'missing_{field}' for field in l2_gap_signals['completeness_gap_fields']],
                evidence_refs=l2_gap_signals['completeness_gap_evidence'],
            )
        )

    if l2_gap_signals['relation_gap_count'] > 0 or any(
        not alternative.evidence_ids for alternative in compilation.primary_route_state.route_landscape.alternative_routes
    ):
        failure_records.append(
            _failure_record(
                failure_code='l2_relation_stitch_missing',
                layer='l2',
                stage='route_state',
                severity='medium',
                blocking=False,
                owner='route_state_synthesizer',
                repair_target='carry future-work and critique signals into alternative-route candidates with route-level evidence ids',
                source_flags=[f'gap_moves:{l2_gap_signals["relation_gap_count"]}'],
                evidence_refs=l2_gap_signals['relation_gap_evidence']
                or [
                    evidence_id
                    for alternative in compilation.primary_route_state.route_landscape.alternative_routes
                    for evidence_id in list(alternative.evidence_ids or [])
                ][:8],
            )
        )

    if not compilation.primary_route_state.evidence_bundle.challenging_evidence_ids:
        failure_records.append(
            _failure_record(
                failure_code='l2_challenging_evidence_missing',
                layer='l2',
                stage='route_state',
                severity='high',
                blocking=True,
                owner='route_state_synthesizer',
                repair_target='preserve limitation and critique anchors as route-level challenging evidence',
                source_flags=[
                    *list(compilation.primary_route_state.quality.quality_flags),
                    *list(compilation.why_now_case.quality.quality_flags),
                ],
                evidence_refs=list(compilation.primary_route_state.evidence_bundle.supporting_evidence_ids)[:8],
            )
        )

    if not alternative_route_states or not compilation.route_comparison_cases:
        failure_records.append(
            _failure_record(
                failure_code='route_comparison_missing',
                layer='l3_l4',
                stage='route_comparison',
                severity='high',
                blocking=True,
                owner='route_comparison_builder',
                repair_target='supply a distinct alternative route and enough evidence to build a comparison case',
                source_flags=list(compilation.quality_flags),
                evidence_refs=list(compilation.primary_route_state.evidence_bundle.supporting_evidence_ids)[:8],
            )
        )

    prior_support_cluster_size = 1 + len(support_route_states)
    if prior_support_cluster_size < 3:
        failure_records.append(
            _failure_record(
                failure_code='prior_support_cluster_too_small',
                layer='l3_l4',
                stage='decision_prior_card',
                severity='high',
                blocking=True,
                owner='decision_prior_builder',
                repair_target='add support route states until the prior cluster is stable enough for reuse',
                source_flags=[f'support_cluster_size:{prior_support_cluster_size}'],
                evidence_refs=list(compilation.decision_prior_card.supporting_route_state_ids),
            )
        )

    if not held_out_route_states:
        failure_records.append(
            _failure_record(
                failure_code='held_out_routes_missing',
                layer='l3_l4',
                stage='decision_prior_card',
                severity='high',
                blocking=True,
                owner='decision_prior_builder',
                repair_target='add held-out route states so the prior can be audited for consistency',
                source_flags=list(compilation.quality_flags),
                evidence_refs=list(compilation.decision_prior_card.supporting_route_state_ids),
            )
        )

    if not reviewer_ids:
        failure_records.append(
            _failure_record(
                failure_code='reviewer_missing',
                layer='l3_l4',
                stage='decision_prior_card',
                severity='medium',
                blocking=True,
                owner='decision_prior_review',
                repair_target='attach reviewer metadata before treating the prior as audited',
                source_flags=list(compilation.quality_flags),
                evidence_refs=list(compilation.decision_prior_card.supporting_route_state_ids),
            )
        )

    return failure_records


class HistoricalReplayCompiler:
    def __init__(
        self,
        *,
        route_state_builder_version: str = 'route_state_synthesizer_v1',
        why_now_builder_version: str = 'why_now_case_builder_v1',
        route_comparison_builder_version: str = 'route_comparison_builder_v1',
        decision_prior_builder_version: str = 'decision_prior_builder_v1',
        decision_episode_builder_version: str = 'decision_episode_builder_v1',
        compile_mode: str = 'rule_only',
    ) -> None:
        self.route_state_synthesizer = RouteStateSynthesizer(
            compiler_version=route_state_builder_version,
            compile_mode=compile_mode,
        )
        self.why_now_builder = WhyNowCaseBuilder(builder_version=why_now_builder_version)
        self.route_comparison_builder = RouteComparisonBuilder(builder_version=route_comparison_builder_version)
        self.decision_prior_builder = DecisionPriorBuilder(builder_version=decision_prior_builder_version)
        self.decision_episode_builder = DecisionEpisodeBuilder(
            builder_version=decision_episode_builder_version,
            compile_mode=compile_mode,
        )
        self.compile_mode = compile_mode

    def compile(
        self,
        route_packet: RoutePacket | dict[str, Any],
        traces: list[PaperLogicTrace],
        *,
        l1_snapshot: HistoricalEnvironmentSnapshot | dict[str, Any] | None = None,
        support_route_states: list[RouteState] | None = None,
        alternative_route_states: list[RouteState] | None = None,
        held_out_route_states: list[RouteState] | None = None,
        reviewer_ids: list[str] | None = None,
        prior_cards: list[DecisionPriorCard] | None = None,
        anti_pattern_cards: list[AntiPatternCard] | None = None,
        route_state_ref: str | None = None,
        historical_cutoff_time: str | None = None,
        hindsight_outcome: HindsightOutcome | dict[str, Any] | None = None,
        built_at: str | None = None,
        route_state_id: str | None = None,
        prior_id: str | None = None,
        episode_id: str | None = None,
    ) -> HistoricalReplayCompilation:
        packet_model = route_packet if isinstance(route_packet, RoutePacket) else RoutePacket.model_validate(route_packet)
        primary_route_state = self.route_state_synthesizer.synthesize(
            packet_model,
            traces,
            l1_snapshot=l1_snapshot,
            built_at=built_at,
            route_state_id=route_state_id,
        )
        why_now_case = self.why_now_builder.build(primary_route_state, built_at=built_at)

        support_route_states = _dedupe_route_states(primary_route_state, support_route_states, label='support_route_states')
        alternative_route_states = _dedupe_route_states(primary_route_state, alternative_route_states, label='alternative_route_states')
        held_out_route_states = _dedupe_route_states(primary_route_state, held_out_route_states, label='held_out_route_states')

        route_comparison_cases = [
            self.route_comparison_builder.build(primary_route_state, alternative_route_state, built_at=built_at)
            for alternative_route_state in alternative_route_states
        ]
        selected_comparison_case = _select_comparison_case(route_comparison_cases)

        prior_support_cluster = [primary_route_state] + support_route_states
        cluster_why_now_cases = [why_now_case]
        cluster_why_now_cases.extend(self.why_now_builder.build(route_state, built_at=built_at) for route_state in support_route_states)
        decision_prior_card = self.decision_prior_builder.build(
            prior_support_cluster,
            why_now_cases=cluster_why_now_cases,
            comparison_cases=route_comparison_cases,
            held_out_route_states=held_out_route_states,
            reviewer_ids=reviewer_ids,
            built_at=built_at,
            prior_id=prior_id,
        )
        using_prebuilt_candidate_lists = prior_cards is not None or anti_pattern_cards is not None
        accepted_prior_cards = [card for card in prior_cards or [] if card.quality.quality_tier == 'green']
        accepted_anti_pattern_cards = [card for card in anti_pattern_cards or [] if card.quality.quality_tier == 'green']
        representative_prebuilt_prior = next(
            (
                card
                for card in [*accepted_prior_cards, *(prior_cards or [])]
                if primary_route_state.route_state_id in card.supporting_route_state_ids
            ),
            None,
        )
        representative_prior_card = (
            representative_prebuilt_prior
            or (accepted_prior_cards[0] if accepted_prior_cards else None)
            or ((prior_cards or [None])[0] if prior_cards else None)
            or decision_prior_card
        )
        decision_episode = self.decision_episode_builder.build(
            primary_route_state,
            route_packet=packet_model,
            why_now_case=why_now_case,
            comparison_case=selected_comparison_case,
            prior_cards=accepted_prior_cards if using_prebuilt_candidate_lists else [decision_prior_card],
            anti_pattern_cards=accepted_anti_pattern_cards if using_prebuilt_candidate_lists else anti_pattern_cards,
            route_state_ref=route_state_ref,
            historical_cutoff_time=historical_cutoff_time,
            hindsight_outcome=hindsight_outcome,
            prior_layer_version=self.decision_prior_builder.builder_version,
            built_at=built_at,
            episode_id=episode_id,
        )

        quality_flags: list[str] = []
        if len(prior_support_cluster) < 3:
            quality_flags.append('support_cluster_too_small')
        if not alternative_route_states:
            quality_flags.append('no_alternative_route_states')
        if not held_out_route_states:
            quality_flags.append('held_out_routes_missing')
        if not reviewer_ids:
            quality_flags.append('reviewer_missing')
        if selected_comparison_case and selected_comparison_case.preference_label == 'prefer_b':
            quality_flags.append('selected_comparison_not_primary')
        if representative_prior_card.quality.quality_tier == 'red':
            quality_flags.append('prior_card_not_usable')
        if decision_episode.quality.quality_tier == 'red':
            quality_flags.append('decision_episode_not_usable')

        preliminary_compilation = HistoricalReplayCompilation(
            primary_route_state=primary_route_state,
            why_now_case=why_now_case,
            route_comparison_cases=route_comparison_cases,
            selected_comparison_case_id=selected_comparison_case.route_comparison_case_id if selected_comparison_case else None,
            decision_prior_card=representative_prior_card,
            decision_episode=decision_episode,
            quality_flags=_unique(quality_flags),
            ready_for_pilot=False,
        )
        failure_records = _build_failure_records(
            route_packet=packet_model,
            traces=traces,
            compilation=preliminary_compilation,
            support_route_states=support_route_states,
            alternative_route_states=alternative_route_states,
            held_out_route_states=held_out_route_states,
            reviewer_ids=_unique([str(reviewer_id) for reviewer_id in reviewer_ids or [] if str(reviewer_id).strip()]),
        )
        ready_for_pilot = (
            primary_route_state.quality.ready_for_route_comparison
            and representative_prior_card.quality.quality_tier != 'red'
            and decision_episode.quality.ready_for_eval
        )
        if any(record.blocking for record in failure_records):
            ready_for_pilot = False

        return HistoricalReplayCompilation(
            primary_route_state=primary_route_state,
            why_now_case=why_now_case,
            route_comparison_cases=route_comparison_cases,
            selected_comparison_case_id=selected_comparison_case.route_comparison_case_id if selected_comparison_case else None,
            decision_prior_card=representative_prior_card,
            decision_episode=decision_episode,
            quality_flags=_unique(quality_flags),
            failure_records=failure_records,
            ready_for_pilot=ready_for_pilot,
        )


def compile_historical_replay(
    route_packet: RoutePacket | dict[str, Any],
    traces: list[PaperLogicTrace],
    *,
    l1_snapshot: HistoricalEnvironmentSnapshot | dict[str, Any] | None = None,
    support_route_states: list[RouteState] | None = None,
    alternative_route_states: list[RouteState] | None = None,
    held_out_route_states: list[RouteState] | None = None,
    reviewer_ids: list[str] | None = None,
    prior_cards: list[DecisionPriorCard] | None = None,
    anti_pattern_cards: list[AntiPatternCard] | None = None,
    route_state_ref: str | None = None,
    historical_cutoff_time: str | None = None,
    hindsight_outcome: HindsightOutcome | dict[str, Any] | None = None,
    built_at: str | None = None,
    route_state_id: str | None = None,
    prior_id: str | None = None,
    episode_id: str | None = None,
    route_state_builder_version: str = 'route_state_synthesizer_v1',
    why_now_builder_version: str = 'why_now_case_builder_v1',
    route_comparison_builder_version: str = 'route_comparison_builder_v1',
    decision_prior_builder_version: str = 'decision_prior_builder_v1',
    decision_episode_builder_version: str = 'decision_episode_builder_v1',
    compile_mode: str = 'rule_only',
) -> HistoricalReplayCompilation:
    compiler = HistoricalReplayCompiler(
        route_state_builder_version=route_state_builder_version,
        why_now_builder_version=why_now_builder_version,
        route_comparison_builder_version=route_comparison_builder_version,
        decision_prior_builder_version=decision_prior_builder_version,
        decision_episode_builder_version=decision_episode_builder_version,
        compile_mode=compile_mode,
    )
    return compiler.compile(
        route_packet,
        traces,
        l1_snapshot=l1_snapshot,
        support_route_states=support_route_states,
        alternative_route_states=alternative_route_states,
        held_out_route_states=held_out_route_states,
        reviewer_ids=reviewer_ids,
        prior_cards=prior_cards,
        anti_pattern_cards=anti_pattern_cards,
        route_state_ref=route_state_ref,
        historical_cutoff_time=historical_cutoff_time,
        hindsight_outcome=hindsight_outcome,
        built_at=built_at,
        route_state_id=route_state_id,
        prior_id=prior_id,
        episode_id=episode_id,
    )


__all__ = [
    'HistoricalReplayCompilation',
    'HistoricalReplayCompiler',
    'ReplayFailureRecord',
    'compile_historical_replay',
]
