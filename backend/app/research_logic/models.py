from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


QualityTier = Literal['red', 'yellow', 'green']
AuditStatus = Literal['hot_path', 'eligible', 'reviewed']
CompileMode = Literal['rule_only', 'rule_plus_llm', 'llm_assisted_audit']


def _has_items(values: list[str]) -> bool:
    return any(str(value or '').strip() for value in values)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class PacketRoleCounts(ContractModel):
    core_method: int = 0
    resource_or_benchmark: int = 0
    limitation_or_critique: int = 0
    survey_or_review: int = 0
    alternative_route: int = 0


class PacketComposition(ContractModel):
    target_size: int
    actual_size: int
    role_counts: PacketRoleCounts = Field(default_factory=PacketRoleCounts)
    coverage_ok: bool = False
    missing_roles: list[str] = Field(default_factory=list)


class L1SnapshotRef(ContractModel):
    snapshot_id: str
    resource_registry_ref: str | None = None
    resource_timeline_ref: str | None = None
    benchmark_timeline_ref: str | None = None
    toolchain_timeline_ref: str | None = None
    protocol_registry_ref: str | None = None


PacketItemRole = Literal[
    'core_method',
    'resource_or_benchmark',
    'limitation_or_critique',
    'survey_or_review',
    'alternative_route',
]
SourceSelector = Literal['manual', 'rule', 'retrieval', 'mixed']
PacketStatus = Literal['draft', 'reviewed', 'frozen']
ExclusionReason = Literal[
    'after_cutoff',
    'off_topic',
    'duplicate_signal',
    'low_quality_trace',
    'unresolved_metadata',
    'missing_source',
    'other',
]
LeakageRisk = Literal['low', 'medium', 'high', 'unknown']
ManualReviewStatus = Literal['not_started', 'partial', 'completed']


class IncludedPacketItem(ContractModel):
    paper_id: str
    trace_id: str | None = None
    paper_year: int | None = None
    item_role: PacketItemRole
    inclusion_reason: str
    source_selector: SourceSelector = 'rule'
    evidence_for_inclusion: list[str] = Field(default_factory=list)
    title: str | None = None


class ExcludedPacketItem(ContractModel):
    paper_id: str
    paper_year: int | None = None
    exclusion_reason: ExclusionReason
    notes: str | None = None


class AcceptedYearRange(ContractModel):
    min_year: int | None = None
    max_year: int


class InclusionRules(ContractModel):
    scope_definition: str
    scope_aliases: list[str] = Field(default_factory=list)
    accepted_year_range: AcceptedYearRange
    hard_exclusion_rules: list[str] = Field(default_factory=list)
    role_assignment_rules: list[str] = Field(default_factory=list)
    leakage_policy: str


class PacketQuality(ContractModel):
    quality_tier: QualityTier = 'red'
    ready_for_route_state: bool = False
    quality_flags: list[str] = Field(default_factory=list)
    topic_boundary_confidence: float | None = None
    leakage_risk: LeakageRisk = 'unknown'
    manual_review_status: ManualReviewStatus = 'not_started'


class RouteCompilerHints(ContractModel):
    preferred_scope_label: str | None = None
    preferred_method_labels: list[str] = Field(default_factory=list)
    preferred_benchmark_labels: list[str] = Field(default_factory=list)
    preferred_bottleneck_labels: list[str] = Field(default_factory=list)
    expected_alternative_routes: list[str] = Field(default_factory=list)
    notes_for_route_state_compiler: str | None = None


class RoutePacket(ContractModel):
    packet_id: str
    schema_version: str = 'v1'
    built_at: str
    topic_scope_candidate: str
    cutoff_year: int
    packet_status: PacketStatus = 'draft'
    packet_composition: PacketComposition
    l1_snapshot_ref: L1SnapshotRef | None = None
    inclusion_rules: InclusionRules
    included_items: list[IncludedPacketItem] = Field(default_factory=list)
    excluded_items: list[ExcludedPacketItem] = Field(default_factory=list)
    packet_quality: PacketQuality = Field(default_factory=PacketQuality)
    compiler_hints: RouteCompilerHints = Field(default_factory=RouteCompilerHints)

    @model_validator(mode='after')
    def validate_green_packet(self) -> 'RoutePacket':
        if self.packet_quality.quality_tier != 'green':
            return self
        if not self.packet_composition.coverage_ok:
            raise ValueError('green RoutePacket requires complete role coverage')
        if any(not str(item.trace_id or '').strip() for item in self.included_items):
            raise ValueError('green RoutePacket requires trace coverage for all included items')
        if self.l1_snapshot_ref is None:
            raise ValueError('green RoutePacket requires an L1 snapshot reference')
        return self


AdoptionLevel = Literal['emerging', 'workable', 'established', 'dominant', 'unknown']
BenchmarkType = Literal['dataset', 'benchmark', 'task_suite', 'unknown']
BenchmarkAdoptionLevel = Literal['absent', 'emerging', 'active', 'dominant', 'unknown']
ProtocolType = Literal['measurement', 'evaluation', 'simulation', 'experiment', 'unknown']
InfrastructureType = Literal['software', 'hardware', 'platform', 'instrument', 'compute', 'unknown']
AvailabilityLevel = Literal['unavailable', 'limited', 'usable', 'abundant', 'unknown']
CapabilityType = Literal['prediction', 'measurement', 'optimization', 'control', 'explanation', 'synthesis', 'unknown']
CapabilityStatus = Literal['tentative', 'repeatable', 'scalable', 'limited', 'unknown']
BottleneckType = Literal['data', 'measurement', 'theory', 'compute', 'evaluation', 'engineering', 'resource', 'unknown']
Severity = Literal['low', 'medium', 'high', 'blocking', 'unknown']
BlockingScope = Literal['local', 'route_level', 'packet_level', 'unknown']
ConditionType = Literal['data', 'resource', 'tool', 'protocol', 'theory', 'cost', 'coordination', 'unknown']
ConditionStatus = Literal['unmet', 'partially_met', 'met', 'unstable', 'unknown']
RouteRelation = Literal['competing', 'complementary', 'precursor', 'fallback', 'unknown']
FeatureType = Literal[
    'bottleneck_relief',
    'new_resource',
    'benchmark_availability',
    'method_maturity',
    'infrastructure',
    'protocol',
    'comparative_gain',
    'blocker',
    'fragility',
    'missing_prerequisite',
    'unknown',
]
FeatureDirection = Literal['unlock', 'accelerate', 'block', 'warn']


class RouteStateSourcePacket(ContractModel):
    packet_id: str
    included_trace_ids: list[str] = Field(default_factory=list)
    included_paper_ids: list[str] = Field(default_factory=list)
    packet_role_counts: PacketRoleCounts = Field(default_factory=PacketRoleCounts)
    l1_snapshot_ref: str | None = None


class ScopeResolution(ContractModel):
    topic_scope_candidates: list[str] = Field(default_factory=list)
    accepted_scope_label: str
    rejected_scope_labels: list[str] = Field(default_factory=list)
    resolution_rationale: str | None = None
    resolution_evidence_ids: list[str] = Field(default_factory=list)


class MethodState(ContractModel):
    label: str
    family: str | None = None
    maturity_score: float | None = None
    adoption_level: AdoptionLevel = 'unknown'
    source_move_ids: list[str] = Field(default_factory=list)
    source_paper_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class BenchmarkState(ContractModel):
    label: str
    benchmark_type: BenchmarkType = 'unknown'
    adoption_level: BenchmarkAdoptionLevel = 'unknown'
    source_paper_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class ProtocolState(ContractModel):
    label: str
    protocol_type: ProtocolType = 'unknown'
    maturity_score: float | None = None
    source_paper_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class InfrastructureState(ContractModel):
    label: str
    infra_type: InfrastructureType = 'unknown'
    availability_level: AvailabilityLevel = 'unknown'
    source_paper_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class CapabilityState(ContractModel):
    label: str
    capability_type: CapabilityType = 'unknown'
    status: CapabilityStatus = 'unknown'
    metric_signals: list[str] = Field(default_factory=list)
    condition_signals: list[str] = Field(default_factory=list)
    source_paper_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class BottleneckState(ContractModel):
    label: str
    bottleneck_type: BottleneckType = 'unknown'
    severity: Severity = 'unknown'
    blocking_scope: BlockingScope = 'unknown'
    source_paper_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    counterevidence_ids: list[str] = Field(default_factory=list)


class ConditionState(ContractModel):
    label: str
    condition_type: ConditionType = 'unknown'
    status: ConditionStatus = 'unknown'
    source_paper_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class AlternativeRouteState(ContractModel):
    label: str
    route_family: str | None = None
    relation_to_main_route: RouteRelation = 'unknown'
    distinguishing_features: list[str] = Field(default_factory=list)
    source_paper_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class RouteLandscape(ContractModel):
    dominant_methods: list[MethodState] = Field(default_factory=list)
    active_benchmarks: list[BenchmarkState] = Field(default_factory=list)
    measurement_protocols: list[ProtocolState] = Field(default_factory=list)
    toolchains_and_infrastructure: list[InfrastructureState] = Field(default_factory=list)
    known_capabilities: list[CapabilityState] = Field(default_factory=list)
    known_bottlenecks: list[BottleneckState] = Field(default_factory=list)
    enabling_conditions: list[ConditionState] = Field(default_factory=list)
    alternative_routes: list[AlternativeRouteState] = Field(default_factory=list)


class ReadinessScores(ContractModel):
    theory: float | None = None
    method: float | None = None
    measurement: float | None = None
    data_resource: float | None = None
    infrastructure: float | None = None
    community: float | None = None
    cost_cycle: float | None = None
    overall: float | None = None
    score_rationale: str | None = None


class RouteFeature(ContractModel):
    label: str
    feature_type: FeatureType = 'unknown'
    direction: FeatureDirection
    source_paper_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    l1_refs: list[str] = Field(default_factory=list)
    confidence: float | None = None


class WhyNowFeatures(ContractModel):
    unlocking_factors: list[RouteFeature] = Field(default_factory=list)
    acceleration_factors: list[RouteFeature] = Field(default_factory=list)
    positive_comparison_signals: list[RouteFeature] = Field(default_factory=list)


class NotNowFeatures(ContractModel):
    blocking_factors: list[RouteFeature] = Field(default_factory=list)
    fragility_factors: list[RouteFeature] = Field(default_factory=list)
    missing_prerequisites: list[RouteFeature] = Field(default_factory=list)


class RouteEvidenceBundle(ContractModel):
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    challenging_evidence_ids: list[str] = Field(default_factory=list)
    representative_move_ids: list[str] = Field(default_factory=list)
    representative_paper_ids: list[str] = Field(default_factory=list)
    l1_support_refs: list[str] = Field(default_factory=list)
    l1_constraint_refs: list[str] = Field(default_factory=list)


class RouteUncertaintyPoints(ContractModel):
    open_questions: list[str] = Field(default_factory=list)
    unresolved_conflicts: list[str] = Field(default_factory=list)
    weak_fields: list[str] = Field(default_factory=list)
    low_confidence_clusters: list[str] = Field(default_factory=list)


class RouteCompilerMetadata(ContractModel):
    compiler_version: str
    packet_builder_version: str | None = None
    l1_snapshot_version: str | None = None
    trace_versions: dict[str, str] = Field(default_factory=dict)
    compile_mode: CompileMode = 'rule_only'
    llm_usage_notes: str | None = None


class RouteStateQuality(ContractModel):
    quality_tier: QualityTier = 'red'
    ready_for_why_now: bool = False
    ready_for_route_comparison: bool = False
    ready_for_prior_selection: bool = False
    quality_flags: list[str] = Field(default_factory=list)
    audit_status: AuditStatus = 'hot_path'


class RouteState(ContractModel):
    route_state_id: str
    schema_version: str = 'v1'
    built_at: str
    topic_scope: str
    cutoff_year: int
    source_packet: RouteStateSourcePacket
    scope_resolution: ScopeResolution
    route_landscape: RouteLandscape = Field(default_factory=RouteLandscape)
    readiness_scores: ReadinessScores = Field(default_factory=ReadinessScores)
    why_now_features: WhyNowFeatures = Field(default_factory=WhyNowFeatures)
    not_now_features: NotNowFeatures = Field(default_factory=NotNowFeatures)
    evidence_bundle: RouteEvidenceBundle = Field(default_factory=RouteEvidenceBundle)
    uncertainty_points: RouteUncertaintyPoints = Field(default_factory=RouteUncertaintyPoints)
    compiler_metadata: RouteCompilerMetadata
    quality: RouteStateQuality = Field(default_factory=RouteStateQuality)

    @model_validator(mode='after')
    def validate_green_route_state(self) -> 'RouteState':
        if self.quality.quality_tier != 'green':
            return self
        if not _has_items(self.evidence_bundle.supporting_evidence_ids):
            raise ValueError('green RouteState requires supporting evidence')
        if not _has_items(self.evidence_bundle.challenging_evidence_ids):
            raise ValueError('green RouteState requires challenging evidence')
        return self


WhyNowLabel = Literal['now', 'almost_now', 'not_now', 'unclear']
WhyNowFactorType = Literal[
    'method_maturity',
    'benchmark_availability',
    'data_resource',
    'infrastructure',
    'protocol',
    'comparative_advantage',
    'community_shift',
    'bottleneck',
    'resource_gap',
    'fragility',
    'cost_cycle',
    'unknown',
]
WhyNowStrength = Literal['low', 'medium', 'high', 'unknown']


class WhyNowFactor(ContractModel):
    label: str
    factor_type: WhyNowFactorType = 'unknown'
    strength: WhyNowStrength = 'unknown'
    source_route_fields: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    l1_refs: list[str] = Field(default_factory=list)


class WhyNowEvidenceChain(ContractModel):
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    challenging_evidence_ids: list[str] = Field(default_factory=list)
    representative_route_fields: list[str] = Field(default_factory=list)


class WhyNowUncertaintyPoints(ContractModel):
    unresolved_conflicts: list[str] = Field(default_factory=list)
    weak_signals: list[str] = Field(default_factory=list)
    ambiguous_enablers: list[str] = Field(default_factory=list)


class TrainingQuality(ContractModel):
    quality_tier: QualityTier = 'red'
    ready_for_training: bool = False
    quality_flags: list[str] = Field(default_factory=list)


class WhyNowCase(ContractModel):
    why_now_case_id: str
    schema_version: str = 'v1'
    built_at: str
    route_state_id: str
    why_now_label: WhyNowLabel
    unlocking_factors: list[WhyNowFactor] = Field(default_factory=list)
    blocking_factors: list[WhyNowFactor] = Field(default_factory=list)
    evidence_chain: WhyNowEvidenceChain = Field(default_factory=WhyNowEvidenceChain)
    uncertainty_points: WhyNowUncertaintyPoints = Field(default_factory=WhyNowUncertaintyPoints)
    quality: TrainingQuality = Field(default_factory=TrainingQuality)

    @model_validator(mode='after')
    def validate_positive_label_grounding(self) -> 'WhyNowCase':
        if self.why_now_label in {'now', 'almost_now'} and not self.unlocking_factors:
            raise ValueError('positive WhyNowCase labels require unlocking factors')
        return self


PreferenceLabel = Literal['prefer_a', 'prefer_b', 'tie', 'unclear']
PreferredRoute = Literal['a', 'b', 'tie', 'unknown']
ComparisonDimension = Literal[
    'method_maturity',
    'measurement',
    'data_resource',
    'infrastructure',
    'bottleneck',
    'feasibility',
    'strategic_value',
    'novelty',
    'cost_cycle',
    'unknown',
]


class ComparisonDimensionScore(ContractModel):
    dimension: ComparisonDimension
    route_a_score: float | None = None
    route_b_score: float | None = None
    preferred_route: PreferredRoute = 'unknown'
    rationale: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class RoutePreferenceExplanation(ContractModel):
    summary: str
    decisive_dimensions: list[str] = Field(default_factory=list)
    decisive_evidence_ids: list[str] = Field(default_factory=list)


class RouteComparisonEvidenceChain(ContractModel):
    route_a_support_ids: list[str] = Field(default_factory=list)
    route_b_support_ids: list[str] = Field(default_factory=list)
    cross_route_comparison_ids: list[str] = Field(default_factory=list)


class RouteComparisonUncertaintyPoints(ContractModel):
    incomparable_dimensions: list[str] = Field(default_factory=list)
    weak_dimensions: list[str] = Field(default_factory=list)
    unresolved_conflicts: list[str] = Field(default_factory=list)


class RouteComparisonCase(ContractModel):
    route_comparison_case_id: str
    schema_version: str = 'v1'
    built_at: str
    cutoff_year: int
    route_a_state_id: str
    route_b_state_id: str
    comparison_dimension_scores: list[ComparisonDimensionScore] = Field(default_factory=list)
    preference_label: PreferenceLabel = 'unclear'
    why_a_not_b: RoutePreferenceExplanation
    why_b_not_a: RoutePreferenceExplanation
    evidence_chain: RouteComparisonEvidenceChain = Field(default_factory=RouteComparisonEvidenceChain)
    uncertainty_points: RouteComparisonUncertaintyPoints = Field(default_factory=RouteComparisonUncertaintyPoints)
    quality: TrainingQuality = Field(default_factory=TrainingQuality)

    @model_validator(mode='after')
    def validate_distinct_routes(self) -> 'RouteComparisonCase':
        if self.route_a_state_id == self.route_b_state_id:
            raise ValueError('RouteComparisonCase must compare distinct route states')
        return self


PriorConditionType = Literal['readiness', 'bottleneck', 'route_shape', 'environment', 'comparison', 'unknown']
PriorPolarity = Literal['present', 'absent', 'high', 'low', 'improving', 'degrading', 'unknown']
ThresholdOperator = Literal['gte', 'lte', 'eq', 'contains', 'absent']
ActionType = Literal[
    'pursue_question',
    'defer_route',
    'gather_resource',
    'improve_measurement',
    'compare_routes',
    'test_assumption',
    'narrow_scope',
    'unknown',
]
ReviewStatus = Literal['draft', 'candidate', 'reviewed', 'approved', 'rejected']


class PriorCondition(ContractModel):
    label: str
    condition_type: PriorConditionType = 'unknown'
    polarity: PriorPolarity = 'unknown'
    required: bool = True


class PriorThreshold(ContractModel):
    field: str
    operator: ThresholdOperator
    value: str | float | int | bool


class PriorAppliesWhen(ContractModel):
    readiness_pattern: list[PriorCondition] = Field(default_factory=list)
    bottleneck_pattern: list[PriorCondition] = Field(default_factory=list)
    route_pattern: list[PriorCondition] = Field(default_factory=list)
    evidence_thresholds: list[PriorThreshold] = Field(default_factory=list)


class PriorDoesNotApplyWhen(ContractModel):
    blocker_pattern: list[PriorCondition] = Field(default_factory=list)
    fragility_pattern: list[PriorCondition] = Field(default_factory=list)
    mismatch_pattern: list[PriorCondition] = Field(default_factory=list)


class RecommendedAction(ContractModel):
    action_type: ActionType = 'unknown'
    action_text: str
    target_route_feature: str | None = None
    confidence: float | None = None


class ExpectedFailureMode(ContractModel):
    label: str
    linked_bottleneck_types: list[str] = Field(default_factory=list)
    warning_signals: list[str] = Field(default_factory=list)
    mitigation_hint: str | None = None


class HeldOutConsistency(ContractModel):
    held_out_route_state_ids: list[str] = Field(default_factory=list)
    pass_rate: float | None = None
    failure_notes: str | None = None


class ReviewMetadata(ContractModel):
    review_status: ReviewStatus = 'draft'
    reviewer_notes: str | None = None
    reviewer_ids: list[str] = Field(default_factory=list)


class CardQuality(ContractModel):
    quality_tier: QualityTier = 'red'
    quality_flags: list[str] = Field(default_factory=list)
    audit_status: AuditStatus = 'hot_path'


class DecisionPriorCard(ContractModel):
    prior_id: str
    schema_version: str = 'v1'
    built_at: str
    prior_text: str
    applies_when: PriorAppliesWhen = Field(default_factory=PriorAppliesWhen)
    does_not_apply_when: PriorDoesNotApplyWhen = Field(default_factory=PriorDoesNotApplyWhen)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    expected_failure_modes: list[ExpectedFailureMode] = Field(default_factory=list)
    supporting_route_state_ids: list[str] = Field(default_factory=list)
    counterexample_ids: list[str] = Field(default_factory=list)
    held_out_consistency: HeldOutConsistency = Field(default_factory=HeldOutConsistency)
    review: ReviewMetadata = Field(default_factory=ReviewMetadata)
    quality: CardQuality = Field(default_factory=CardQuality)

    @model_validator(mode='after')
    def validate_green_prior(self) -> 'DecisionPriorCard':
        if self.quality.quality_tier != 'green':
            return self
        if not _has_items(self.supporting_route_state_ids):
            raise ValueError('green DecisionPriorCard requires supporting route states')
        if not _has_items(self.held_out_consistency.held_out_route_state_ids) or self.held_out_consistency.pass_rate is None:
            raise ValueError('green DecisionPriorCard requires held-out results')
        if self.review.review_status not in {'reviewed', 'approved'} or not _has_items(self.review.reviewer_ids):
            raise ValueError('green DecisionPriorCard requires completed review metadata')
        if 'missing_counterexample_search' in self.quality.quality_flags:
            raise ValueError('green DecisionPriorCard must not skip counterexample search')
        return self


WarningSignalType = Literal[
    'bottleneck',
    'contradiction',
    'weak_comparison',
    'weak_measurement',
    'resource_gap',
    'route_fragility',
    'cost_spike',
    'unknown',
]
TriggerLogic = Literal['all', 'any', 'weighted']


class WarningSignal(ContractModel):
    label: str
    signal_type: WarningSignalType = 'unknown'
    severity: Severity = 'unknown'
    source_field: str | None = None


class WarningSignalPattern(ContractModel):
    signals: list[WarningSignal] = Field(default_factory=list)
    trigger_logic: TriggerLogic = 'any'


class FailureExamples(ContractModel):
    route_state_ids: list[str] = Field(default_factory=list)
    decision_episode_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class CounterexampleSet(ContractModel):
    route_state_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class AntiPatternCard(ContractModel):
    anti_pattern_id: str
    schema_version: str = 'v1'
    built_at: str
    anti_pattern_text: str
    warning_signal_pattern: WarningSignalPattern = Field(default_factory=WarningSignalPattern)
    failure_examples: FailureExamples = Field(default_factory=FailureExamples)
    corrective_checklist: list[str] = Field(default_factory=list)
    counterexamples: CounterexampleSet = Field(default_factory=CounterexampleSet)
    review: ReviewMetadata = Field(default_factory=ReviewMetadata)
    quality: CardQuality = Field(default_factory=CardQuality)


QuestionType = Literal[
    'hypothesis',
    'route_choice',
    'scope_refinement',
    'measurement_gap',
    'benchmark_gap',
    'resource_gap',
    'unknown',
]
EpisodeComparisonDimensionType = Literal[
    'method_maturity',
    'measurement',
    'data_resource',
    'infrastructure',
    'bottleneck',
    'novelty',
    'feasibility',
    'strategic_value',
    'unknown',
]
PreferredCandidate = Literal['primary', 'alternative_1', 'alternative_2', 'tie', 'unknown']
FinalChoice = Literal['primary', 'alternative_1', 'alternative_2', 'reject_all', 'defer']
OutcomeLabel = Literal['success', 'partial_success', 'failure', 'still_open', 'unknown']


class ObservationEvidencePack(ContractModel):
    route_packet_id: str
    l1_snapshot_ref: str | None = None
    visible_paper_ids: list[str] = Field(default_factory=list)
    visible_trace_ids: list[str] = Field(default_factory=list)
    excluded_after_cutoff_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class PaperLogicTraceRefs(ContractModel):
    trace_ids: list[str] = Field(default_factory=list)
    representative_trace_ids: list[str] = Field(default_factory=list)


class RouteStateRef(ContractModel):
    route_state_id: str
    route_state_ref: str | None = None


class RelevantPriors(ContractModel):
    selected_prior_ids: list[str] = Field(default_factory=list)
    selected_antipattern_ids: list[str] = Field(default_factory=list)
    prior_selection_rationale: str | None = None


class CandidateQuestion(ContractModel):
    question_text: str
    question_type: QuestionType = 'unknown'
    target_route_feature: str | None = None
    justification_evidence_ids: list[str] = Field(default_factory=list)


class EpisodeComparisonDimension(ContractModel):
    dimension: EpisodeComparisonDimensionType = 'unknown'
    preferred_candidate: PreferredCandidate = 'unknown'
    rationale: str | None = None


class WhyThisNotThat(ContractModel):
    primary_reasoning: str
    comparison_dimensions: list[EpisodeComparisonDimension] = Field(default_factory=list)
    evidence_chain: list[str] = Field(default_factory=list)


class NotNowCase(ContractModel):
    rejected_question_text: str
    blocker_summary: str
    evidence_ids: list[str] = Field(default_factory=list)


class MinimalAttackPath(ContractModel):
    prerequisite_steps: list[str] = Field(default_factory=list)
    required_resources: list[str] = Field(default_factory=list)
    required_measurements: list[str] = Field(default_factory=list)
    expected_checkpoints: list[str] = Field(default_factory=list)


class DecisionOutput(ContractModel):
    final_choice: FinalChoice
    final_decision_text: str
    confidence: float | None = None


class HindsightOutcome(ContractModel):
    outcome_label: OutcomeLabel = 'unknown'
    later_evidence_refs: list[str] = Field(default_factory=list)
    retrospective_notes: str | None = None
    input_visible: bool = False


class DecisionEpisodeCompilerMetadata(ContractModel):
    episode_builder_version: str
    route_state_version: str | None = None
    prior_layer_version: str | None = None
    compile_mode: CompileMode = 'rule_only'
    notes: str | None = None


class DecisionEpisodeQuality(ContractModel):
    quality_tier: QualityTier = 'red'
    ready_for_training: bool = False
    ready_for_eval: bool = False
    quality_flags: list[str] = Field(default_factory=list)
    audit_status: AuditStatus = 'hot_path'


class DecisionEpisode(ContractModel):
    episode_id: str
    schema_version: str = 'v1'
    built_at: str
    historical_cutoff_time: str
    observation_evidence_pack: ObservationEvidencePack
    paper_logic_traces: PaperLogicTraceRefs = Field(default_factory=PaperLogicTraceRefs)
    route_state: RouteStateRef
    relevant_priors: RelevantPriors = Field(default_factory=RelevantPriors)
    candidate_question: CandidateQuestion
    alternative_questions: list[CandidateQuestion] = Field(default_factory=list)
    why_this_not_that: WhyThisNotThat
    not_now_cases: list[NotNowCase] = Field(default_factory=list)
    minimal_attack_path: MinimalAttackPath = Field(default_factory=MinimalAttackPath)
    decision_output: DecisionOutput
    hindsight_outcome: HindsightOutcome = Field(default_factory=HindsightOutcome)
    compiler_metadata: DecisionEpisodeCompilerMetadata
    quality: DecisionEpisodeQuality = Field(default_factory=DecisionEpisodeQuality)

    @model_validator(mode='after')
    def validate_hindsight_visibility(self) -> 'DecisionEpisode':
        if self.hindsight_outcome.input_visible:
            raise ValueError('DecisionEpisode hindsight_outcome must not be visible to the input side')
        return self


__all__ = [
    'AntiPatternCard',
    'DecisionEpisode',
    'DecisionPriorCard',
    'RouteComparisonCase',
    'RoutePacket',
    'RouteState',
    'WhyNowCase',
]
