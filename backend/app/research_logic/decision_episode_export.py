from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .decision_episode_builder import build_decision_episode
from .models import (
    AntiPatternCard,
    DecisionEpisode,
    DecisionPriorCard,
    HindsightOutcome,
    RouteComparisonCase,
    RoutePacket,
    RouteState,
    TrainingAcceptanceVerdict,
    TrainingReviewStatus,
    TrainingSectionReview,
    WhyNowCase,
)

ModelT = TypeVar('ModelT', bound=BaseModel)
TrainingPriorExclusionReasonCode = Literal[
    'route_state_not_supported',
    'accepted_prior_candidate_missing',
]
TrainingAntiPatternExclusionReasonCode = Literal[
    'route_state_not_supported',
    'accepted_anti_pattern_candidate_missing',
]
REQUIRED_SECTION_REVIEW_KEYS = [
    'evidence_pack',
    'route_synthesis',
    'why_now',
    'route_comparison',
    'priors_antipatterns',
    'minimal_attack_path',
    'final_decision',
    'review_labels',
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or '').strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _model_from(value: ModelT | Mapping[str, Any] | None, model_type: type[ModelT]) -> ModelT | None:
    if value is None:
        return None
    if isinstance(value, model_type):
        return value
    return model_type.model_validate(value)


def _models_from(values: Sequence[BaseModel | Mapping[str, Any]] | None, model_type: type[ModelT]) -> list[ModelT]:
    models: list[ModelT] = []
    for value in values or []:
        model = _model_from(value, model_type)
        if model is None:
            continue
        models.append(model)
    return models


def _normalize_ids(values: Sequence[str] | None) -> list[str]:
    return _unique([str(value or '').strip() for value in values or []])


def _default_section_reviews() -> dict[str, TrainingSectionReview]:
    return {key: TrainingSectionReview() for key in REQUIRED_SECTION_REVIEW_KEYS}


def _normalize_section_reviews(
    value: Mapping[str, TrainingSectionReview | Mapping[str, Any]] | None,
) -> dict[str, TrainingSectionReview]:
    normalized = _default_section_reviews()
    for key, section_value in (value or {}).items():
        section_key = str(key or '').strip()
        if not section_key:
            continue
        normalized[section_key] = (
            section_value
            if isinstance(section_value, TrainingSectionReview)
            else TrainingSectionReview.model_validate(section_value)
        )
    return normalized


def _allowlisted_cards(
    cards: Sequence[ModelT],
    accepted_ids: Sequence[str],
    *,
    id_attr: str,
) -> list[ModelT]:
    cards_by_id = {
        str(getattr(card, id_attr) or '').strip(): card
        for card in cards
        if str(getattr(card, id_attr) or '').strip()
    }
    return [
        cards_by_id[accepted_id]
        for accepted_id in _normalize_ids(accepted_ids)
        if accepted_id in cards_by_id
    ]


class DecisionEpisodeExportSourceRefs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    bundle_ref: str | None = None
    manifest_ref: str | None = None
    files: dict[str, str] = Field(default_factory=dict)


class AcceptedButUnselectedPrior(BaseModel):
    model_config = ConfigDict(extra='forbid')

    prior_id: str
    exclusion_reason_code: TrainingPriorExclusionReasonCode
    rationale: str
    evidence_refs: list[str] = Field(default_factory=list)
    supporting_route_state_ids: list[str] = Field(default_factory=list)
    held_out_route_state_ids: list[str] = Field(default_factory=list)
    review_status: str | None = None
    reviewer_ids: list[str] = Field(default_factory=list)
    reviewer_notes: str | None = None


class AcceptedButUnselectedAntiPattern(BaseModel):
    model_config = ConfigDict(extra='forbid')

    anti_pattern_id: str
    exclusion_reason_code: TrainingAntiPatternExclusionReasonCode
    failure_route_state_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class DecisionEpisodeAuditExport(BaseModel):
    model_config = ConfigDict(extra='forbid')

    built_at: str
    decision_episode: DecisionEpisode
    route_state_snapshot: RouteState
    why_now_case: WhyNowCase | None = None
    route_comparison_case: RouteComparisonCase | None = None
    source_replay_bundle_refs: DecisionEpisodeExportSourceRefs = Field(default_factory=DecisionEpisodeExportSourceRefs)
    source_review_bundle_refs: DecisionEpisodeExportSourceRefs = Field(default_factory=DecisionEpisodeExportSourceRefs)
    accepted_prior_ids: list[str] = Field(default_factory=list)
    accepted_anti_pattern_ids: list[str] = Field(default_factory=list)
    accepted_prior_cards: list[DecisionPriorCard] = Field(default_factory=list)
    accepted_anti_pattern_cards: list[AntiPatternCard] = Field(default_factory=list)
    accepted_but_unselected_priors: list[AcceptedButUnselectedPrior] = Field(default_factory=list)
    accepted_but_unselected_antipatterns: list[AcceptedButUnselectedAntiPattern] = Field(default_factory=list)
    visible_input_refs: list[str] = Field(default_factory=list)
    audit_only_refs: list[str] = Field(default_factory=list)
    label_eval_only_refs: list[str] = Field(default_factory=list)
    prior_selection_note: str | None = None
    anti_pattern_selection_note: str | None = None
    review_status: TrainingReviewStatus = 'not_started'
    training_acceptance_verdict: TrainingAcceptanceVerdict = 'pending'
    reviewer_ids: list[str] = Field(default_factory=list)
    reviewed_at: str | None = None
    rationale: str | None = None
    residual_defects: list[str] = Field(default_factory=list)
    section_reviews: dict[str, TrainingSectionReview] = Field(default_factory=_default_section_reviews)

    @model_validator(mode='after')
    def validate_section_reviews(self) -> 'DecisionEpisodeAuditExport':
        missing_keys = [key for key in REQUIRED_SECTION_REVIEW_KEYS if key not in self.section_reviews]
        if missing_keys:
            raise ValueError(f'missing required section reviews: {", ".join(missing_keys)}')
        return self


def _normalize_source_refs(
    value: DecisionEpisodeExportSourceRefs | Mapping[str, Any] | None,
) -> DecisionEpisodeExportSourceRefs:
    if isinstance(value, DecisionEpisodeExportSourceRefs):
        return value
    if value is None:
        return DecisionEpisodeExportSourceRefs()
    return DecisionEpisodeExportSourceRefs.model_validate(value)


def _source_bundle_refs(prefix: str, source_refs: DecisionEpisodeExportSourceRefs) -> list[str]:
    refs: list[str] = []
    if source_refs.bundle_ref:
        refs.append(f'{prefix}:bundle:{source_refs.bundle_ref}')
    if source_refs.manifest_ref:
        refs.append(f'{prefix}:manifest:{source_refs.manifest_ref}')
    refs.extend(
        f'{prefix}:{name}:{value}'
        for name, value in sorted(source_refs.files.items())
        if str(value or '').strip()
    )
    return refs


def _accepted_prior_evidence_refs(
    prior_card: DecisionPriorCard,
    review_refs: DecisionEpisodeExportSourceRefs,
) -> list[str]:
    review_summary_ref = str(review_refs.files.get('candidate_review_summary') or '').strip()
    refs = [
        f'accepted_prior:{prior_card.prior_id}',
        *(f'supporting_route_state:{route_state_id}' for route_state_id in prior_card.supporting_route_state_ids),
        *(
            f'held_out_route_state:{route_state_id}'
            for route_state_id in prior_card.held_out_consistency.held_out_route_state_ids
        ),
        *(f'counterexample:{counterexample_id}' for counterexample_id in prior_card.counterexample_ids),
        *([f'review_bundle:candidate_review_summary:{review_summary_ref}'] if review_summary_ref else []),
        *(
            [f'review_bundle:manifest:{review_refs.manifest_ref}']
            if str(review_refs.manifest_ref or '').strip()
            else []
        ),
    ]
    return _unique(refs)


def _accepted_but_unselected_priors(
    *,
    route_state: RouteState,
    accepted_prior_ids: Sequence[str],
    selected_prior_ids: Sequence[str],
    prior_cards: Sequence[DecisionPriorCard],
    review_refs: DecisionEpisodeExportSourceRefs,
) -> list[AcceptedButUnselectedPrior]:
    selected_prior_id_set = set(_normalize_ids(selected_prior_ids))
    prior_cards_by_id = {
        prior_card.prior_id: prior_card
        for prior_card in prior_cards
        if str(prior_card.prior_id or '').strip()
    }
    records: list[AcceptedButUnselectedPrior] = []
    for prior_id in _normalize_ids(accepted_prior_ids):
        if prior_id in selected_prior_id_set:
            continue

        prior_card = prior_cards_by_id.get(prior_id)
        if prior_card is None:
            records.append(
                AcceptedButUnselectedPrior(
                    prior_id=prior_id,
                    exclusion_reason_code='accepted_prior_candidate_missing',
                    rationale=(
                        f'Review bundle accepted prior {prior_id}, but its candidate payload was unavailable during '
                        'audit export assembly, so selected_prior_ids stayed unchanged.'
                    ),
                    evidence_refs=_unique(
                        [
                            f'accepted_prior:{prior_id}',
                            *(
                                [f'review_bundle:manifest:{review_refs.manifest_ref}']
                                if str(review_refs.manifest_ref or '').strip()
                                else []
                            ),
                        ]
                    ),
                )
            )
            continue

        route_supported = route_state.route_state_id in prior_card.supporting_route_state_ids
        records.append(
            AcceptedButUnselectedPrior(
                prior_id=prior_card.prior_id,
                exclusion_reason_code=(
                    'route_state_not_supported' if not route_supported else 'accepted_prior_candidate_missing'
                ),
                rationale=(
                    f'Accepted prior {prior_card.prior_id} remains review-backed, but it does not list exported '
                    f'route_state {route_state.route_state_id} in supporting_route_state_ids, so selected_prior_ids '
                    'stayed empty to keep the audited export route-backed.'
                    if not route_supported
                    else (
                        f'Accepted prior {prior_card.prior_id} matched exported route_state {route_state.route_state_id}, '
                        'but the audited export did not preserve it in selected_prior_ids; keep the mismatch explicit '
                        'until the selection contract is resolved.'
                    )
                ),
                evidence_refs=_accepted_prior_evidence_refs(prior_card, review_refs),
                supporting_route_state_ids=list(prior_card.supporting_route_state_ids),
                held_out_route_state_ids=list(prior_card.held_out_consistency.held_out_route_state_ids),
                review_status=prior_card.review.review_status,
                reviewer_ids=list(prior_card.review.reviewer_ids),
                reviewer_notes=prior_card.review.reviewer_notes,
            )
        )
    return records


def _anti_pattern_matches_route(route_state: RouteState, anti_pattern_card: AntiPatternCard) -> bool:
    route_family_id = str(route_state.route_family_id or '').strip()
    return route_state.route_state_id in anti_pattern_card.failure_examples.route_state_ids or (
        route_family_id and route_family_id in anti_pattern_card.failure_examples.route_family_ids
    )


def _accepted_anti_pattern_evidence_refs(
    anti_pattern_card: AntiPatternCard,
    review_refs: DecisionEpisodeExportSourceRefs,
) -> list[str]:
    review_summary_ref = str(review_refs.files.get('candidate_review_summary') or '').strip()
    refs = [
        f'accepted_antipattern:{anti_pattern_card.anti_pattern_id}',
        *(f'failure_route_state:{route_state_id}' for route_state_id in anti_pattern_card.failure_examples.route_state_ids),
        *(f'failure_route_family:{route_family_id}' for route_family_id in anti_pattern_card.failure_examples.route_family_ids),
        *([f'review_bundle:candidate_review_summary:{review_summary_ref}'] if review_summary_ref else []),
        *(
            [f'review_bundle:manifest:{review_refs.manifest_ref}']
            if str(review_refs.manifest_ref or '').strip()
            else []
        ),
    ]
    return _unique(refs)


def _accepted_but_unselected_antipatterns(
    *,
    route_state: RouteState,
    accepted_anti_pattern_ids: Sequence[str],
    selected_antipattern_ids: Sequence[str],
    anti_pattern_cards: Sequence[AntiPatternCard],
    review_refs: DecisionEpisodeExportSourceRefs,
) -> list[AcceptedButUnselectedAntiPattern]:
    selected_antipattern_id_set = set(_normalize_ids(selected_antipattern_ids))
    anti_pattern_cards_by_id = {
        anti_pattern_card.anti_pattern_id: anti_pattern_card
        for anti_pattern_card in anti_pattern_cards
        if str(anti_pattern_card.anti_pattern_id or '').strip()
    }
    records: list[AcceptedButUnselectedAntiPattern] = []
    for anti_pattern_id in _normalize_ids(accepted_anti_pattern_ids):
        if anti_pattern_id in selected_antipattern_id_set:
            continue

        anti_pattern_card = anti_pattern_cards_by_id.get(anti_pattern_id)
        if anti_pattern_card is None:
            records.append(
                AcceptedButUnselectedAntiPattern(
                    anti_pattern_id=anti_pattern_id,
                    exclusion_reason_code='accepted_anti_pattern_candidate_missing',
                    failure_route_state_ids=[],
                    evidence_refs=_unique(
                        [
                            f'accepted_antipattern:{anti_pattern_id}',
                            *(
                                [f'review_bundle:manifest:{review_refs.manifest_ref}']
                                if str(review_refs.manifest_ref or '').strip()
                                else []
                            ),
                        ]
                    ),
                )
            )
            continue

        route_supported = _anti_pattern_matches_route(route_state, anti_pattern_card)
        records.append(
            AcceptedButUnselectedAntiPattern(
                anti_pattern_id=anti_pattern_card.anti_pattern_id,
                exclusion_reason_code=(
                    'route_state_not_supported' if not route_supported else 'accepted_anti_pattern_candidate_missing'
                ),
                failure_route_state_ids=list(anti_pattern_card.failure_examples.route_state_ids),
                evidence_refs=_accepted_anti_pattern_evidence_refs(anti_pattern_card, review_refs),
            )
        )
    return records


def _prior_selection_note(
    *,
    route_state: RouteState,
    accepted_prior_ids: Sequence[str],
    selected_prior_ids: Sequence[str],
) -> str:
    accepted_prior_ids = _normalize_ids(accepted_prior_ids)
    selected_prior_ids = _normalize_ids(selected_prior_ids)
    if not accepted_prior_ids:
        return 'Review bundle accepted no prior ids, so the audited export preserved empty selected_prior_ids.'
    if selected_prior_ids:
        return (
            f'Selected {len(selected_prior_ids)} reviewed accepted prior card(s) because they support '
            f'route_state {route_state.route_state_id}.'
        )
    return (
        f'Review bundle accepted prior ids, but none support route_state {route_state.route_state_id}; '
        'selected_prior_ids stayed empty.'
    )


def _anti_pattern_selection_note(
    *,
    route_state: RouteState,
    accepted_anti_pattern_ids: Sequence[str],
    selected_antipattern_ids: Sequence[str],
    accepted_but_unselected_antipatterns: Sequence[AcceptedButUnselectedAntiPattern] | None = None,
) -> str:
    accepted_anti_pattern_ids = _normalize_ids(accepted_anti_pattern_ids)
    selected_antipattern_ids = _normalize_ids(selected_antipattern_ids)
    accepted_but_unselected_antipattern_count = len(accepted_but_unselected_antipatterns or [])
    route_match_target = f'route_state {route_state.route_state_id}'
    if str(route_state.route_family_id or '').strip():
        route_match_target = (
            f'route_state {route_state.route_state_id} or route_family_id {route_state.route_family_id}'
        )
    if not accepted_anti_pattern_ids:
        return 'Review bundle accepted no anti-pattern ids, so the audited export carried no anti-pattern ids.'
    if selected_antipattern_ids:
        message = (
            f'Carried {len(selected_antipattern_ids)} reviewed accepted anti-pattern id(s) because their failure '
            f'examples match {route_match_target}.'
        )
        if accepted_but_unselected_antipattern_count > 0:
            return (
                f'{message[:-1]} and preserved {accepted_but_unselected_antipattern_count} '
                'accepted_but_unselected_antipatterns record(s) for accepted anti-patterns that did not match.'
            )
        return message
    return (
        f'Review bundle accepted anti-pattern ids, but none match {route_match_target}; '
        'selected_antipattern_ids stayed empty and accepted_but_unselected_antipatterns records capture the mismatch.'
    )


def _visible_input_refs(decision_episode: DecisionEpisode) -> list[str]:
    refs = [f'route_packet:{decision_episode.observation_evidence_pack.route_packet_id}']
    if decision_episode.observation_evidence_pack.l1_snapshot_ref:
        refs.append(f'l1_snapshot:{decision_episode.observation_evidence_pack.l1_snapshot_ref}')
    refs.extend(f'paper:{paper_id}' for paper_id in decision_episode.observation_evidence_pack.visible_paper_ids)
    refs.extend(f'trace:{trace_id}' for trace_id in decision_episode.observation_evidence_pack.visible_trace_ids)
    refs.extend(f'evidence:{evidence_id}' for evidence_id in decision_episode.observation_evidence_pack.evidence_refs)
    return _unique(refs)


def _audit_only_refs(
    *,
    decision_episode: DecisionEpisode,
    route_state: RouteState,
    why_now_case: WhyNowCase | None,
    comparison_case: RouteComparisonCase | None,
    route_state_ref: str | None,
    source_replay_bundle_refs: DecisionEpisodeExportSourceRefs,
    source_review_bundle_refs: DecisionEpisodeExportSourceRefs,
    accepted_prior_ids: Sequence[str],
    accepted_anti_pattern_ids: Sequence[str],
) -> list[str]:
    refs = [
        f'route_state:{route_state.route_state_id}',
        *(
            [f'route_state_ref:{route_state_ref}']
            if str(route_state_ref or '').strip()
            else []
        ),
        *(
            f'after_cutoff_paper:{paper_id}'
            for paper_id in decision_episode.observation_evidence_pack.excluded_after_cutoff_ids
        ),
        *(
            f'accepted_prior:{prior_id}'
            for prior_id in _normalize_ids(accepted_prior_ids)
        ),
        *(
            f'accepted_antipattern:{anti_pattern_id}'
            for anti_pattern_id in _normalize_ids(accepted_anti_pattern_ids)
        ),
        *_source_bundle_refs('replay_bundle', source_replay_bundle_refs),
        *_source_bundle_refs('review_bundle', source_review_bundle_refs),
    ]
    if why_now_case is not None:
        refs.append(f'why_now_case:{why_now_case.why_now_case_id}')
    if comparison_case is not None:
        refs.append(f'route_comparison_case:{comparison_case.route_comparison_case_id}')
    return _unique(refs)


def _label_eval_only_refs(hindsight_outcome: HindsightOutcome) -> list[str]:
    return _unique(
        [f'hindsight_evidence:{evidence_ref}' for evidence_ref in hindsight_outcome.later_evidence_refs]
    )


def build_decision_episode_audit_export(
    *,
    route_packet: RoutePacket | Mapping[str, Any],
    route_state: RouteState | Mapping[str, Any],
    why_now_case: WhyNowCase | Mapping[str, Any] | None = None,
    comparison_case: RouteComparisonCase | Mapping[str, Any] | None = None,
    hindsight_outcome: HindsightOutcome | Mapping[str, Any] | None = None,
    prior_cards: Sequence[DecisionPriorCard | Mapping[str, Any]] | None = None,
    anti_pattern_cards: Sequence[AntiPatternCard | Mapping[str, Any]] | None = None,
    accepted_prior_ids: Sequence[str] | None = None,
    accepted_anti_pattern_ids: Sequence[str] | None = None,
    route_state_ref: str | None = None,
    source_replay_bundle_refs: DecisionEpisodeExportSourceRefs | Mapping[str, Any] | None = None,
    source_review_bundle_refs: DecisionEpisodeExportSourceRefs | Mapping[str, Any] | None = None,
    review_status: TrainingReviewStatus = 'not_started',
    training_acceptance_verdict: TrainingAcceptanceVerdict = 'pending',
    reviewer_ids: Sequence[str] | None = None,
    reviewed_at: str | None = None,
    rationale: str | None = None,
    residual_defects: Sequence[str] | None = None,
    section_reviews: Mapping[str, TrainingSectionReview | Mapping[str, Any]] | None = None,
    built_at: str | None = None,
    episode_id: str | None = None,
) -> DecisionEpisodeAuditExport:
    route_packet_model = _model_from(route_packet, RoutePacket)
    route_state_model = _model_from(route_state, RouteState)
    why_now_model = _model_from(why_now_case, WhyNowCase)
    comparison_model = _model_from(comparison_case, RouteComparisonCase)
    hindsight_model = _model_from(hindsight_outcome, HindsightOutcome) or HindsightOutcome()
    replay_refs = _normalize_source_refs(source_replay_bundle_refs)
    review_refs = _normalize_source_refs(source_review_bundle_refs)
    built_at_value = built_at or _utc_now_iso()

    normalized_prior_cards = _models_from(prior_cards, DecisionPriorCard)
    normalized_anti_pattern_cards = _models_from(anti_pattern_cards, AntiPatternCard)
    allowlisted_prior_cards = _allowlisted_cards(
        normalized_prior_cards,
        accepted_prior_ids or [],
        id_attr='prior_id',
    )
    allowlisted_anti_pattern_cards = _allowlisted_cards(
        normalized_anti_pattern_cards,
        accepted_anti_pattern_ids or [],
        id_attr='anti_pattern_id',
    )

    decision_episode = build_decision_episode(
        route_state_model,
        route_packet=route_packet_model,
        why_now_case=why_now_model,
        comparison_case=comparison_model,
        prior_cards=allowlisted_prior_cards,
        anti_pattern_cards=allowlisted_anti_pattern_cards,
        route_state_ref=route_state_ref,
        hindsight_outcome=hindsight_model,
        built_at=built_at_value,
        episode_id=episode_id,
    )

    selected_prior_ids = list(decision_episode.relevant_priors.selected_prior_ids)
    selected_antipattern_ids = list(decision_episode.relevant_priors.selected_antipattern_ids)
    accepted_but_unselected_antipatterns = _accepted_but_unselected_antipatterns(
        route_state=route_state_model,
        accepted_anti_pattern_ids=accepted_anti_pattern_ids or [],
        selected_antipattern_ids=selected_antipattern_ids,
        anti_pattern_cards=allowlisted_anti_pattern_cards,
        review_refs=review_refs,
    )

    return DecisionEpisodeAuditExport(
        built_at=built_at_value,
        decision_episode=decision_episode,
        route_state_snapshot=route_state_model,
        why_now_case=why_now_model,
        route_comparison_case=comparison_model,
        source_replay_bundle_refs=replay_refs,
        source_review_bundle_refs=review_refs,
        accepted_prior_ids=_normalize_ids(accepted_prior_ids),
        accepted_anti_pattern_ids=_normalize_ids(accepted_anti_pattern_ids),
        accepted_prior_cards=allowlisted_prior_cards,
        accepted_anti_pattern_cards=allowlisted_anti_pattern_cards,
        accepted_but_unselected_priors=_accepted_but_unselected_priors(
            route_state=route_state_model,
            accepted_prior_ids=accepted_prior_ids or [],
            selected_prior_ids=selected_prior_ids,
            prior_cards=allowlisted_prior_cards,
            review_refs=review_refs,
        ),
        accepted_but_unselected_antipatterns=accepted_but_unselected_antipatterns,
        visible_input_refs=_visible_input_refs(decision_episode),
        audit_only_refs=_audit_only_refs(
            decision_episode=decision_episode,
            route_state=route_state_model,
            why_now_case=why_now_model,
            comparison_case=comparison_model,
            route_state_ref=route_state_ref,
            source_replay_bundle_refs=replay_refs,
            source_review_bundle_refs=review_refs,
            accepted_prior_ids=accepted_prior_ids or [],
            accepted_anti_pattern_ids=accepted_anti_pattern_ids or [],
        ),
        label_eval_only_refs=_label_eval_only_refs(hindsight_model),
        prior_selection_note=_prior_selection_note(
            route_state=route_state_model,
            accepted_prior_ids=accepted_prior_ids or [],
            selected_prior_ids=selected_prior_ids,
        ),
        anti_pattern_selection_note=_anti_pattern_selection_note(
            route_state=route_state_model,
            accepted_anti_pattern_ids=accepted_anti_pattern_ids or [],
            selected_antipattern_ids=selected_antipattern_ids,
            accepted_but_unselected_antipatterns=accepted_but_unselected_antipatterns,
        ),
        review_status=review_status,
        training_acceptance_verdict=training_acceptance_verdict,
        reviewer_ids=_normalize_ids(reviewer_ids),
        reviewed_at=reviewed_at,
        rationale=rationale,
        residual_defects=_unique([str(value or '').strip() for value in residual_defects or [] if str(value or '').strip()]),
        section_reviews=_normalize_section_reviews(section_reviews),
    )


__all__ = [
    'AcceptedButUnselectedAntiPattern',
    'AcceptedButUnselectedPrior',
    'DecisionEpisodeAuditExport',
    'DecisionEpisodeExportSourceRefs',
    'build_decision_episode_audit_export',
]
