from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from .models import (
    AntiPatternCard,
    CandidateQuestion,
    DecisionEpisode,
    DecisionEpisodeCompilerMetadata,
    DecisionEpisodeQuality,
    DecisionOutput,
    DecisionPriorCard,
    EpisodeComparisonDimension,
    HindsightOutcome,
    MinimalAttackPath,
    NotNowCase,
    ObservationEvidencePack,
    PaperLogicTraceRefs,
    RelevantPriors,
    RouteComparisonCase,
    RoutePacket,
    RouteState,
    RouteStateRef,
    WhyNowCase,
    WhyThisNotThat,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _slug(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', str(value or '').strip().lower())
    return slug.strip('_') or 'episode'


def _normalize(value: str) -> str:
    return str(value or '').strip().lower()


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


def _comparison_role(route_state: RouteState, comparison_case: RouteComparisonCase | None) -> str | None:
    if comparison_case is None:
        return None
    if comparison_case.route_a_state_id == route_state.route_state_id:
        return 'a'
    if comparison_case.route_b_state_id == route_state.route_state_id:
        return 'b'
    return None


def _selected_prior_cards(route_state: RouteState, prior_cards: list[DecisionPriorCard] | None) -> list[DecisionPriorCard]:
    if not prior_cards:
        return []
    return [
        prior_card
        for prior_card in prior_cards
        if route_state.route_state_id in prior_card.supporting_route_state_ids
    ]


def _selected_antipattern_cards(route_state: RouteState, anti_pattern_cards: list[AntiPatternCard] | None) -> list[AntiPatternCard]:
    if not anti_pattern_cards:
        return []
    route_family_id = str(route_state.route_family_id or '').strip()
    return [
        anti_pattern
        for anti_pattern in anti_pattern_cards
        if (
            route_state.route_state_id in anti_pattern.failure_examples.route_state_ids
            or (route_family_id and route_family_id in anti_pattern.failure_examples.route_family_ids)
        )
    ]


def _historical_cutoff_time(cutoff_year: int, value: str | None) -> str:
    if value:
        return value
    return f'{cutoff_year}-12-31T23:59:59Z'


def _validate_packet(route_state: RouteState, route_packet: RoutePacket | dict[str, Any] | None) -> RoutePacket | None:
    if route_packet is None:
        return None
    packet_model = route_packet if isinstance(route_packet, RoutePacket) else RoutePacket.model_validate(route_packet)
    if packet_model.packet_id != route_state.source_packet.packet_id:
        raise ValueError('DecisionEpisodeBuilder route_packet must match route_state.source_packet.packet_id')
    if packet_model.cutoff_year != route_state.cutoff_year:
        raise ValueError('DecisionEpisodeBuilder route_packet cutoff_year must match route_state cutoff_year')
    return packet_model


def _validate_comparison(route_state: RouteState, comparison_case: RouteComparisonCase | None) -> str | None:
    role = _comparison_role(route_state, comparison_case)
    if comparison_case is not None and role is None:
        raise ValueError('DecisionEpisodeBuilder comparison_case must reference the supplied route_state')
    return role


def _validate_hindsight(hindsight_outcome: HindsightOutcome | dict[str, Any] | None) -> HindsightOutcome:
    if hindsight_outcome is None:
        return HindsightOutcome()
    outcome_model = hindsight_outcome if isinstance(hindsight_outcome, HindsightOutcome) else HindsightOutcome.model_validate(hindsight_outcome)
    if outcome_model.input_visible:
        raise ValueError('DecisionEpisodeBuilder must not accept hindsight_outcome with input_visible=true')
    return outcome_model


def _primary_candidate_question(
    route_state: RouteState,
    *,
    why_now_case: WhyNowCase | None,
    comparison_case: RouteComparisonCase | None,
    comparison_role: str | None,
    selected_prior_cards: list[DecisionPriorCard],
) -> CandidateQuestion:
    evidence_ids = _unique(
        list(route_state.evidence_bundle.supporting_evidence_ids)
        + (list(comparison_case.evidence_chain.cross_route_comparison_ids) if comparison_case else [])
    )[:8]
    if comparison_case and comparison_role:
        other_route_id = comparison_case.route_b_state_id if comparison_role == 'a' else comparison_case.route_a_state_id
        if comparison_role == 'a':
            decisive_dimensions = comparison_case.why_a_not_b.decisive_dimensions
        else:
            decisive_dimensions = comparison_case.why_b_not_a.decisive_dimensions
        target_feature = decisive_dimensions[0] if decisive_dimensions else 'strategic_value'
        return CandidateQuestion(
            question_text=(
                f'Should the route around {route_state.topic_scope} be prioritized over nearby alternative route '
                f'{other_route_id} under the current historical cutoff?'
            ),
            question_type='route_choice',
            target_route_feature=target_feature,
            justification_evidence_ids=evidence_ids,
        )
    if route_state.readiness_scores.measurement is not None and route_state.readiness_scores.measurement < 0.65:
        return CandidateQuestion(
            question_text=f'What measurement setup would make {route_state.topic_scope} testable under the current cutoff?',
            question_type='measurement_gap',
            target_route_feature='measurement',
            justification_evidence_ids=evidence_ids,
        )
    if route_state.readiness_scores.data_resource is not None and route_state.readiness_scores.data_resource < 0.65:
        question_type = 'benchmark_gap' if not route_state.route_landscape.active_benchmarks else 'resource_gap'
        target_feature = 'data_resource'
        return CandidateQuestion(
            question_text=f'What missing benchmark or resource would make {route_state.topic_scope} feasible enough to test?',
            question_type=question_type,
            target_route_feature=target_feature,
            justification_evidence_ids=evidence_ids,
        )
    if why_now_case and why_now_case.why_now_label in {'now', 'almost_now'}:
        return CandidateQuestion(
            question_text=f'Should we pursue a scoped version of {route_state.topic_scope} now?',
            question_type='hypothesis',
            target_route_feature='overall',
            justification_evidence_ids=evidence_ids,
        )
    if selected_prior_cards:
        return CandidateQuestion(
            question_text=f'How should we narrow the scope of {route_state.topic_scope} before full commitment?',
            question_type='scope_refinement',
            target_route_feature='overall',
            justification_evidence_ids=evidence_ids,
        )
    return CandidateQuestion(
        question_text=f'What should we do next for {route_state.topic_scope}?',
        question_type='unknown',
        target_route_feature=None,
        justification_evidence_ids=evidence_ids,
    )


def _alternative_questions(
    route_state: RouteState,
    *,
    comparison_case: RouteComparisonCase | None,
    comparison_role: str | None,
) -> list[CandidateQuestion]:
    alternatives: list[CandidateQuestion] = []
    if comparison_case and comparison_role:
        other_route_id = comparison_case.route_b_state_id if comparison_role == 'a' else comparison_case.route_a_state_id
        if comparison_role == 'a':
            opposing_dimensions = comparison_case.why_b_not_a.decisive_dimensions
        else:
            opposing_dimensions = comparison_case.why_a_not_b.decisive_dimensions
        alternatives.append(
            CandidateQuestion(
                question_text=f'Should effort remain focused on alternative route {other_route_id} under the same cutoff?',
                question_type='route_choice',
                target_route_feature=(opposing_dimensions[0] if opposing_dimensions else 'infrastructure'),
                justification_evidence_ids=_unique(list(comparison_case.evidence_chain.cross_route_comparison_ids))[:6],
            )
        )
    if route_state.readiness_scores.measurement is not None and route_state.readiness_scores.measurement < 0.72:
        alternatives.append(
            CandidateQuestion(
                question_text=f'Should we delay route expansion until measurement quality improves for {route_state.topic_scope}?',
                question_type='measurement_gap',
                target_route_feature='measurement',
                justification_evidence_ids=_unique(list(route_state.evidence_bundle.challenging_evidence_ids))[:6],
            )
        )
    return alternatives[:2]


def _why_this_not_that(
    route_state: RouteState,
    *,
    why_now_case: WhyNowCase | None,
    comparison_case: RouteComparisonCase | None,
    comparison_role: str | None,
) -> WhyThisNotThat:
    if comparison_case and comparison_role:
        dimensions: list[EpisodeComparisonDimension] = []
        for score in comparison_case.comparison_dimension_scores[:6]:
            if score.preferred_route == 'tie':
                preferred_candidate = 'tie'
            elif (
                (comparison_role == 'a' and score.preferred_route == 'a')
                or (comparison_role == 'b' and score.preferred_route == 'b')
            ):
                preferred_candidate = 'primary'
            elif score.preferred_route in {'a', 'b'}:
                preferred_candidate = 'alternative_1'
            else:
                preferred_candidate = 'unknown'
            dimensions.append(
                EpisodeComparisonDimension(
                    dimension=score.dimension if score.dimension != 'cost_cycle' else 'feasibility',
                    preferred_candidate=preferred_candidate,
                    rationale=score.rationale,
                )
            )
        primary_summary = comparison_case.why_a_not_b.summary if comparison_role == 'a' else comparison_case.why_b_not_a.summary
        return WhyThisNotThat(
            primary_reasoning=primary_summary,
            comparison_dimensions=dimensions,
            evidence_chain=_unique(list(comparison_case.evidence_chain.cross_route_comparison_ids))[:10],
        )
    if why_now_case and why_now_case.why_now_label in {'now', 'almost_now'}:
        return WhyThisNotThat(
            primary_reasoning=f'The route is actionable enough to test now because {why_now_case.unlocking_factors[0].label.lower()} is visible.',
            comparison_dimensions=[],
            evidence_chain=_unique(list(why_now_case.evidence_chain.supporting_evidence_ids) + list(why_now_case.evidence_chain.challenging_evidence_ids))[:8],
        )
    return WhyThisNotThat(
        primary_reasoning='Current evidence does not yet support a decisive route choice without stronger comparison or prior support.',
        comparison_dimensions=[],
        evidence_chain=_unique(list(route_state.evidence_bundle.supporting_evidence_ids) + list(route_state.evidence_bundle.challenging_evidence_ids))[:8],
    )


def _not_now_cases(route_state: RouteState) -> list[NotNowCase]:
    not_now_cases: list[NotNowCase] = []
    for feature in route_state.not_now_features.blocking_factors[:2]:
        not_now_cases.append(
            NotNowCase(
                rejected_question_text=f'Can {route_state.topic_scope} be pursued without first resolving the dominant blocker?',
                blocker_summary=feature.label,
                evidence_ids=list(feature.evidence_ids),
            )
        )
    for feature in route_state.not_now_features.missing_prerequisites[:1]:
        not_now_cases.append(
            NotNowCase(
                rejected_question_text=f'Can {route_state.topic_scope} scale before the missing prerequisite is satisfied?',
                blocker_summary=feature.label,
                evidence_ids=list(feature.evidence_ids),
            )
        )
    return not_now_cases


def _minimal_attack_path(
    route_state: RouteState,
    *,
    selected_prior_cards: list[DecisionPriorCard],
    candidate_question: CandidateQuestion,
    comparison_case: RouteComparisonCase | None,
    comparison_role: str | None,
) -> MinimalAttackPath:
    prerequisite_steps = _unique(
        [
            action.action_text
            for prior_card in selected_prior_cards
            for action in prior_card.recommended_actions
        ]
        + [f'Constrain the first experiment to the scope implied by {candidate_question.target_route_feature}.']
        + (
            ['Run an explicit side-by-side comparison against the nearby alternative route.']
            if comparison_case and comparison_role
            else []
        )
    )[:4]
    required_resources = _unique(
        [benchmark.label for benchmark in route_state.route_landscape.active_benchmarks]
        + [infra.label for infra in route_state.route_landscape.toolchains_and_infrastructure]
    )[:4]
    required_measurements = _unique(
        [protocol.label for protocol in route_state.route_landscape.measurement_protocols]
        + [
            metric_signal
            for capability in route_state.route_landscape.known_capabilities
            for metric_signal in capability.metric_signals
        ]
    )[:4]
    expected_checkpoints = _unique(
        [
            'Show repeatable evidence under historically visible conditions.',
            (
                f'Beat the nearby alternative on {candidate_question.target_route_feature}.'
                if comparison_case and comparison_role and candidate_question.target_route_feature
                else ''
            ),
            (
                f'Reduce blocker pressure from {route_state.not_now_features.blocking_factors[0].label}.'
                if route_state.not_now_features.blocking_factors
                else ''
            ),
        ]
    )[:4]
    return MinimalAttackPath(
        prerequisite_steps=prerequisite_steps,
        required_resources=required_resources,
        required_measurements=required_measurements,
        expected_checkpoints=expected_checkpoints,
    )


def _final_choice(
    route_state: RouteState,
    *,
    why_now_case: WhyNowCase | None,
    comparison_case: RouteComparisonCase | None,
    comparison_role: str | None,
    selected_prior_cards: list[DecisionPriorCard],
    not_now_cases: list[NotNowCase],
) -> tuple[str, str, float | None]:
    if comparison_case and comparison_role:
        route_preferred = (
            (comparison_role == 'a' and comparison_case.preference_label == 'prefer_a')
            or (comparison_role == 'b' and comparison_case.preference_label == 'prefer_b')
        )
        if route_preferred:
            return (
                'primary',
                'Prioritize the primary route, but keep the scope historically bounded and comparison-aware.',
                min(round((route_state.readiness_scores.overall or 0.65) + 0.08, 2), 0.95),
            )
        if comparison_case.preference_label in {'prefer_a', 'prefer_b'}:
            return (
                'alternative_1',
                'Prefer the alternative route for now because the comparison evidence is stronger at this cutoff.',
                min(round((route_state.readiness_scores.overall or 0.55), 2), 0.9),
            )
    if why_now_case and why_now_case.why_now_label in {'now', 'almost_now'} and selected_prior_cards:
        return (
            'primary',
            'Pursue the primary route with a scoped first attack path rather than a full commitment jump.',
            min(round((route_state.readiness_scores.overall or 0.6) + 0.05, 2), 0.9),
        )
    if not_now_cases:
        return (
            'defer',
            'Defer the route until the blockers are reduced enough to make the next step interpretable.',
            round(max((route_state.readiness_scores.overall or 0.45) - 0.1, 0.2), 2),
        )
    return (
        'reject_all',
        'Do not commit to any candidate yet because the current evidence is still too underconstrained.',
        round(max((route_state.readiness_scores.overall or 0.35) - 0.1, 0.15), 2),
    )


class DecisionEpisodeBuilder:
    def __init__(
        self,
        *,
        builder_version: str = 'decision_episode_builder_v1',
        compile_mode: str = 'rule_only',
    ) -> None:
        self.builder_version = builder_version
        self.compile_mode = compile_mode

    def build(
        self,
        route_state: RouteState,
        *,
        route_packet: RoutePacket | dict[str, Any] | None = None,
        why_now_case: WhyNowCase | None = None,
        comparison_case: RouteComparisonCase | None = None,
        prior_cards: list[DecisionPriorCard] | None = None,
        anti_pattern_cards: list[AntiPatternCard] | None = None,
        route_state_ref: str | None = None,
        historical_cutoff_time: str | None = None,
        hindsight_outcome: HindsightOutcome | dict[str, Any] | None = None,
        prior_layer_version: str | None = None,
        built_at: str | None = None,
        episode_id: str | None = None,
    ) -> DecisionEpisode:
        packet_model = _validate_packet(route_state, route_packet)
        comparison_role = _validate_comparison(route_state, comparison_case)
        hindsight_model = _validate_hindsight(hindsight_outcome)

        selected_prior_cards = _selected_prior_cards(route_state, prior_cards)
        selected_antipattern_cards = _selected_antipattern_cards(route_state, anti_pattern_cards)

        candidate_question = _primary_candidate_question(
            route_state,
            why_now_case=why_now_case,
            comparison_case=comparison_case,
            comparison_role=comparison_role,
            selected_prior_cards=selected_prior_cards,
        )
        alternative_questions = _alternative_questions(
            route_state,
            comparison_case=comparison_case,
            comparison_role=comparison_role,
        )
        why_this_not_that = _why_this_not_that(
            route_state,
            why_now_case=why_now_case,
            comparison_case=comparison_case,
            comparison_role=comparison_role,
        )
        not_now_cases = _not_now_cases(route_state)
        attack_path = _minimal_attack_path(
            route_state,
            selected_prior_cards=selected_prior_cards,
            candidate_question=candidate_question,
            comparison_case=comparison_case,
            comparison_role=comparison_role,
        )
        final_choice, final_decision_text, confidence = _final_choice(
            route_state,
            why_now_case=why_now_case,
            comparison_case=comparison_case,
            comparison_role=comparison_role,
            selected_prior_cards=selected_prior_cards,
            not_now_cases=not_now_cases,
        )

        quality_flags: list[str] = []
        if not selected_prior_cards or not any(prior_card.quality.quality_tier == 'green' for prior_card in selected_prior_cards):
            quality_flags.append('weak_prior_support')
        if not not_now_cases:
            quality_flags.append('missing_not_now_case')
        if not alternative_questions:
            quality_flags.append('weak_alternative_set')
        if candidate_question.question_type in {'unknown', 'hypothesis'} and not candidate_question.target_route_feature:
            quality_flags.append('candidate_question_too_open_ended')
        if (
            not attack_path.prerequisite_steps
            or not attack_path.required_resources
            or not attack_path.required_measurements
            or not attack_path.expected_checkpoints
        ):
            quality_flags.append('minimal_attack_path_missing')

        if route_state.quality.quality_tier == 'green' and not quality_flags:
            quality_tier = 'green'
        elif route_state.quality.quality_tier != 'red':
            quality_tier = 'yellow'
        else:
            quality_tier = 'red'

        return DecisionEpisode(
            episode_id=episode_id or f'episode:{_slug(route_state.topic_scope)}:{route_state.cutoff_year}:{_slug(route_state.route_state_id)}',
            built_at=built_at or _utc_now_iso(),
            historical_cutoff_time=_historical_cutoff_time(route_state.cutoff_year, historical_cutoff_time),
            observation_evidence_pack=ObservationEvidencePack(
                route_packet_id=route_state.source_packet.packet_id,
                l1_snapshot_ref=route_state.source_packet.l1_snapshot_ref,
                visible_paper_ids=list(route_state.source_packet.included_paper_ids),
                visible_trace_ids=list(route_state.source_packet.included_trace_ids),
                excluded_after_cutoff_ids=(
                    [
                        excluded_item.paper_id
                        for excluded_item in packet_model.excluded_items
                        if excluded_item.exclusion_reason == 'after_cutoff'
                    ]
                    if packet_model
                    else []
                ),
                evidence_refs=_unique(
                    list(route_state.evidence_bundle.supporting_evidence_ids)
                    + list(route_state.evidence_bundle.challenging_evidence_ids)
                )[:12],
            ),
            paper_logic_traces=PaperLogicTraceRefs(
                trace_ids=list(route_state.source_packet.included_trace_ids),
                representative_trace_ids=list(route_state.source_packet.included_trace_ids[:2]),
            ),
            route_state=RouteStateRef(
                route_state_id=route_state.route_state_id,
                route_family_id=route_state.route_family_id,
                route_state_ref=route_state_ref,
            ),
            relevant_priors=RelevantPriors(
                selected_prior_ids=[prior_card.prior_id for prior_card in selected_prior_cards],
                selected_antipattern_ids=[anti_pattern.anti_pattern_id for anti_pattern in selected_antipattern_cards],
                prior_selection_rationale=(
                    f'Selected {len(selected_prior_cards)} prior card(s) because they match the route support cluster.'
                    if selected_prior_cards
                    else None
                ),
            ),
            candidate_question=candidate_question,
            alternative_questions=alternative_questions,
            why_this_not_that=why_this_not_that,
            not_now_cases=not_now_cases,
            minimal_attack_path=attack_path,
            decision_output=DecisionOutput(
                final_choice=final_choice,
                final_decision_text=final_decision_text,
                confidence=confidence,
            ),
            hindsight_outcome=hindsight_model,
            compiler_metadata=DecisionEpisodeCompilerMetadata(
                episode_builder_version=self.builder_version,
                route_state_version=route_state.compiler_metadata.compiler_version,
                prior_layer_version=prior_layer_version or ('decision_prior_builder_v1' if selected_prior_cards else None),
                compile_mode=self.compile_mode,
                notes='Rule-built episode object with constrained decision fields.',
            ),
            quality=DecisionEpisodeQuality(
                quality_tier=quality_tier,
                ready_for_training=quality_tier == 'green',
                ready_for_eval=quality_tier != 'red',
                quality_flags=_unique(quality_flags),
                audit_status='reviewed' if quality_tier == 'green' else 'eligible' if quality_tier == 'yellow' else 'hot_path',
            ),
        )


def build_decision_episode(
    route_state: RouteState,
    *,
    route_packet: RoutePacket | dict[str, Any] | None = None,
    why_now_case: WhyNowCase | None = None,
    comparison_case: RouteComparisonCase | None = None,
    prior_cards: list[DecisionPriorCard] | None = None,
    anti_pattern_cards: list[AntiPatternCard] | None = None,
    route_state_ref: str | None = None,
    historical_cutoff_time: str | None = None,
    hindsight_outcome: HindsightOutcome | dict[str, Any] | None = None,
    prior_layer_version: str | None = None,
    built_at: str | None = None,
    episode_id: str | None = None,
    builder_version: str = 'decision_episode_builder_v1',
    compile_mode: str = 'rule_only',
) -> DecisionEpisode:
    builder = DecisionEpisodeBuilder(
        builder_version=builder_version,
        compile_mode=compile_mode,
    )
    return builder.build(
        route_state,
        route_packet=route_packet,
        why_now_case=why_now_case,
        comparison_case=comparison_case,
        prior_cards=prior_cards,
        anti_pattern_cards=anti_pattern_cards,
        route_state_ref=route_state_ref,
        historical_cutoff_time=historical_cutoff_time,
        hindsight_outcome=hindsight_outcome,
        prior_layer_version=prior_layer_version,
        built_at=built_at,
        episode_id=episode_id,
    )


__all__ = ['DecisionEpisodeBuilder', 'build_decision_episode']
