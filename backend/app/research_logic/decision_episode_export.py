from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .decision_episode_builder import build_decision_episode
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

ModelT = TypeVar('ModelT', bound=BaseModel)


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


class DecisionEpisodeAuditExport(BaseModel):
    model_config = ConfigDict(extra='forbid')

    built_at: str
    decision_episode: DecisionEpisode
    source_replay_bundle_refs: DecisionEpisodeExportSourceRefs = Field(default_factory=DecisionEpisodeExportSourceRefs)
    source_review_bundle_refs: DecisionEpisodeExportSourceRefs = Field(default_factory=DecisionEpisodeExportSourceRefs)
    accepted_prior_ids: list[str] = Field(default_factory=list)
    accepted_anti_pattern_ids: list[str] = Field(default_factory=list)
    visible_input_refs: list[str] = Field(default_factory=list)
    audit_only_refs: list[str] = Field(default_factory=list)
    label_eval_only_refs: list[str] = Field(default_factory=list)
    prior_selection_note: str | None = None
    anti_pattern_selection_note: str | None = None


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
) -> str:
    accepted_anti_pattern_ids = _normalize_ids(accepted_anti_pattern_ids)
    selected_antipattern_ids = _normalize_ids(selected_antipattern_ids)
    route_match_target = f'route_state {route_state.route_state_id}'
    if str(route_state.route_family_id or '').strip():
        route_match_target = (
            f'route_state {route_state.route_state_id} or route_family_id {route_state.route_family_id}'
        )
    if not accepted_anti_pattern_ids:
        return 'Review bundle accepted no anti-pattern ids, so the audited export carried no anti-pattern ids.'
    if selected_antipattern_ids:
        return (
            f'Carried {len(selected_antipattern_ids)} reviewed accepted anti-pattern id(s) because their failure '
            f'examples match {route_match_target}.'
        )
    return (
        f'Review bundle accepted anti-pattern ids, but none match {route_match_target}; '
        'selected_antipattern_ids stayed empty.'
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

    return DecisionEpisodeAuditExport(
        built_at=built_at_value,
        decision_episode=decision_episode,
        source_replay_bundle_refs=replay_refs,
        source_review_bundle_refs=review_refs,
        accepted_prior_ids=_normalize_ids(accepted_prior_ids),
        accepted_anti_pattern_ids=_normalize_ids(accepted_anti_pattern_ids),
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
        ),
    )


__all__ = [
    'DecisionEpisodeAuditExport',
    'DecisionEpisodeExportSourceRefs',
    'build_decision_episode_audit_export',
]
