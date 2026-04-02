# LogicKG RouteState Schema

## 1. Background

The project has already completed most of the conceptual shift from:

1. paper extraction for graph browsing

to:

1. paper extraction for downstream scientific-reasoning compilation

At the moment, the repository has:

1. `PaperLogicTrace` as the canonical `L2` export
2. `route_compiler_contract` as a structured signal bundle
3. `route_state_seed` as a single-paper readiness-oriented candidate object

But it still does not have a real `RouteState`.

This is now the most important missing object in the stack, because it is the first object that:

1. leaves the single-paper perspective
2. introduces `topic_scope + cutoff_year`
3. joins `L1` environment signals with `L2` logic signals
4. can be consumed directly by `WhyNowCase`, `RouteComparisonCase`, and `DecisionPriorCard`

## 2. Purpose

This spec defines the minimum viable but training-usable `RouteState` schema for `L3`.

It should be the canonical output of:

1. `route packet` compilation
2. `topic x year` reconstruction
3. `L1 + L2` structured aggregation

## 3. Non-Goals

1. `RouteState` is not a free-form route summary.
2. `RouteState` is not a proposal or hypothesis.
3. `RouteState` is not a `DecisionPriorCard`.
4. This spec does not define the full `WhyNowCase` or `RouteComparisonCase` schema.
5. This spec does not define the whole `L1` storage format.

## 4. Design Principles

### 4.1 State, not story

`RouteState` must describe the state of a research route under a historical cutoff, not tell a persuasive story.

### 4.2 Multi-paper by construction

No valid `RouteState` should be treated as complete if it is only a lightly rewritten single-paper view.

### 4.3 Time-sliced by default

Every field in `RouteState` must respect `cutoff_year`.

### 4.4 Evidence before explanation

The compiler should first aggregate structured evidence, then allow an LLM to produce tightly constrained explanations.

### 4.5 Environment matters

`RouteState` must combine:

1. `L2` paper logic
2. `L1` research environment

Without `L1`, `RouteState` can at best be partial.

## 5. Position in the Stack

The intended flow is:

```text
HistoricalEnvironmentPack (L1)
        +
PaperLogicTrace[] (L2)
        +
route packet
        v
RouteState (L3)
        v
WhyNowCase / RouteComparisonCase
        v
DecisionPriorCard / AntiPatternCard (L4)
        v
DecisionEpisode
```

## 6. Input Contract

### 6.1 Required inputs

`RouteStateSynthesizer` should accept:

1. one `route packet`
2. one `HistoricalEnvironmentPack` or equivalent `L1` snapshot
3. all included `PaperLogicTrace` objects

### 6.2 Minimum route packet fields

```text
RoutePacket
- packet_id
- topic_scope_candidate
- cutoff_year
- included_paper_ids
- included_trace_ids
- packet_roles
- excluded_after_cutoff_ids
- l1_snapshot_ref
```

`packet_roles` should at least distinguish:

1. core-method papers
2. benchmark / dataset / instrument / software papers
3. limitation / critique papers
4. survey / review materials
5. alternative-route papers

### 6.3 Minimum L1 fields consumed by RouteState

The `L1` side does not need to be fully complete yet, but `RouteState` should be able to consume:

1. `resource_registry`
2. `resource_timeline`
3. `benchmark_adoption_timeline`
4. `toolchain_timeline`
5. `measurement_protocol_registry`
6. `paper_resource_usage`
7. optional `community_maturity` or equivalent route-level signals

## 7. Canonical RouteState Object

`RouteState` becomes the canonical `L3` output.

```text
RouteState
- route_state_id
- route_family_id
- schema_version
- built_at
- topic_scope
- cutoff_year
- source_packet
- scope_resolution
- route_landscape
- readiness_scores
- why_now_features
- not_now_features
- evidence_bundle
- uncertainty_points
- compiler_metadata
- quality
```

## 8. Core Fields

### 8.1 Top-level identity

```text
route_state_id: str
route_family_id: str | null
schema_version: str
built_at: str
topic_scope: str
cutoff_year: int
```

Rules:

1. `topic_scope` must be resolved to a single canonical scope string.
2. `cutoff_year` must be explicit and non-optional.
3. `route_state_id` should be deterministic from `topic_scope + cutoff_year + packet_id + schema_version`.
4. `route_family_id` should be stable across support/core/context and runtime-subset variants that represent the same bounded route family under the same cutoff.

### 8.2 `source_packet`

```text
source_packet
- packet_id: str
- included_trace_ids: str[]
- included_paper_ids: str[]
- packet_role_counts: {
    core_method: int,
    resource_or_benchmark: int,
    limitation_or_critique: int,
    survey_or_review: int,
    alternative_route: int
  }
- l1_snapshot_ref: str | null
```

This field exists to make replay, auditing, and leakage control explicit.

### 8.3 `scope_resolution`

```text
scope_resolution
- topic_scope_candidates: str[]
- accepted_scope_label: str
- rejected_scope_labels: str[]
- resolution_rationale: str | null
- resolution_evidence_ids: str[]
```

This field is important because many later failures are really topic-boundary failures.

### 8.4 `route_landscape`

This is the main semantic body of the object.

```text
route_landscape
- dominant_methods: MethodState[]
- active_benchmarks: BenchmarkState[]
- measurement_protocols: ProtocolState[]
- toolchains_and_infrastructure: InfrastructureState[]
- known_capabilities: CapabilityState[]
- known_bottlenecks: BottleneckState[]
- enabling_conditions: ConditionState[]
- alternative_routes: AlternativeRouteState[]
```

## 9. Sub-Objects

### 9.1 `MethodState`

```text
MethodState
- label: str
- family: str | null
- maturity_score: float | null
- adoption_level: emerging | workable | established | dominant | unknown
- source_move_ids: str[]
- source_paper_ids: str[]
- evidence_ids: str[]
- notes: str | null
```

### 9.2 `BenchmarkState`

```text
BenchmarkState
- label: str
- benchmark_type: dataset | benchmark | task_suite | unknown
- adoption_level: absent | emerging | active | dominant | unknown
- source_paper_ids: str[]
- evidence_ids: str[]
```

### 9.3 `ProtocolState`

```text
ProtocolState
- label: str
- protocol_type: measurement | evaluation | simulation | experiment | unknown
- maturity_score: float | null
- source_paper_ids: str[]
- evidence_ids: str[]
```

### 9.4 `InfrastructureState`

```text
InfrastructureState
- label: str
- infra_type: software | hardware | platform | instrument | compute | unknown
- availability_level: unavailable | limited | usable | abundant | unknown
- source_paper_ids: str[]
- evidence_ids: str[]
```

### 9.5 `CapabilityState`

```text
CapabilityState
- label: str
- capability_type: prediction | measurement | optimization | control | explanation | synthesis | unknown
- status: tentative | repeatable | scalable | limited | unknown
- metric_signals: str[]
- condition_signals: str[]
- source_paper_ids: str[]
- evidence_ids: str[]
```

### 9.6 `BottleneckState`

```text
BottleneckState
- label: str
- bottleneck_type: data | measurement | theory | compute | evaluation | engineering | resource | unknown
- severity: low | medium | high | blocking | unknown
- blocking_scope: local | route_level | packet_level | unknown
- source_paper_ids: str[]
- evidence_ids: str[]
- counterevidence_ids: str[]
```

### 9.7 `ConditionState`

```text
ConditionState
- label: str
- condition_type: data | resource | tool | protocol | theory | cost | coordination | unknown
- status: unmet | partially_met | met | unstable | unknown
- source_paper_ids: str[]
- evidence_ids: str[]
```

### 9.8 `AlternativeRouteState`

```text
AlternativeRouteState
- label: str
- route_family: str | null
- relation_to_main_route: competing | complementary | precursor | fallback | unknown
- distinguishing_features: str[]
- source_paper_ids: str[]
- evidence_ids: str[]
```

## 10. Readiness Layer

`RouteState` should include a normalized readiness block.

```text
readiness_scores
- theory: float | null
- method: float | null
- measurement: float | null
- data_resource: float | null
- infrastructure: float | null
- community: float | null
- cost_cycle: float | null
- overall: float | null
- score_rationale: str | null
```

Rules:

1. These scores should be compiler outputs, not LLM guesses.
2. An LLM may explain the scores, but should not invent them.
3. Each score should be grounded in explicit aggregated signals from `L1` or `L2`.

## 11. Why-Now Layer

`RouteState` should store the features needed by `WhyNowCase`, even if the final why-now label is produced by another object.

```text
why_now_features
- unlocking_factors: RouteFeature[]
- acceleration_factors: RouteFeature[]
- positive_comparison_signals: RouteFeature[]
```

```text
not_now_features
- blocking_factors: RouteFeature[]
- fragility_factors: RouteFeature[]
- missing_prerequisites: RouteFeature[]
```

```text
RouteFeature
- label: str
- feature_type: bottleneck_relief | new_resource | benchmark_availability | method_maturity | infrastructure | protocol | comparative_gain | blocker | fragility | missing_prerequisite | unknown
- direction: unlock | accelerate | block | warn
- source_paper_ids: str[]
- evidence_ids: str[]
- l1_refs: str[]
- confidence: float | null
```

## 12. Evidence Layer

`RouteState` should never rely on implicit evidence.

```text
evidence_bundle
- supporting_evidence_ids: str[]
- challenging_evidence_ids: str[]
- representative_move_ids: str[]
- representative_paper_ids: str[]
- l1_support_refs: str[]
- l1_constraint_refs: str[]
```

Rules:

1. `supporting_evidence_ids` and `challenging_evidence_ids` must not collapse into one undifferentiated pool.
2. Route-level confidence must be auditable through these references.
3. Every non-trivial route field should be traceable to at least one evidence list.

## 13. Uncertainty Layer

```text
uncertainty_points
- open_questions: str[]
- unresolved_conflicts: str[]
- weak_fields: str[]
- low_confidence_clusters: str[]
```

This layer is important because `RouteState` should support conservative decision-making, not overclaiming.

## 14. Compiler Metadata

```text
compiler_metadata
- compiler_version: str
- packet_builder_version: str | null
- l1_snapshot_version: str | null
- trace_versions: { trace_id: schema_version }
- compile_mode: rule_only | rule_plus_llm | llm_assisted_audit
- llm_usage_notes: str | null
```

This must make it obvious where a state came from.

## 15. Quality Block

```text
quality
- quality_tier: red | yellow | green
- ready_for_why_now: bool
- ready_for_route_comparison: bool
- ready_for_prior_selection: bool
- quality_flags: str[]
- audit_status: hot_path | eligible | reviewed
```

### 15.1 Minimum green conditions

A `green` `RouteState` should usually require:

1. resolved `topic_scope`
2. explicit `cutoff_year`
3. at least one dominant method with multi-paper support
4. at least one bottleneck or challenge signal
5. at least one enabling condition or positive environment signal
6. at least one alternative route or an explicit reason it is absent
7. both supporting and challenging evidence
8. at least partial `L1` support

### 15.2 Yellow conditions

A `yellow` state is still useful if:

1. route core is visible
2. but `L1` support is thin
3. or alternative routes are unclear
4. or why-now features are incomplete

### 15.3 Red conditions

A `red` state should be used only for debugging if:

1. it is still effectively a single-paper restatement
2. topic resolution failed
3. cutoff leakage is suspected
4. evidence support is missing

## 16. Relationship to Current Code

### 16.1 Current `route_state_seed` to future `RouteState`

The current `route_state_seed` should be treated as a seed object, not the final `RouteState`.

Suggested mapping:

| Current seed field | Future RouteState field |
|---|---|
| `topic_scope_candidates` | `scope_resolution.topic_scope_candidates` |
| `dominant_method_candidates` | `route_landscape.dominant_methods[].label` |
| `active_benchmark_candidates` | `route_landscape.active_benchmarks[].label` |
| `known_bottleneck_candidates` | `route_landscape.known_bottlenecks[].label` |
| `enabling_condition_candidates` | `route_landscape.enabling_conditions[].label` |
| `alternative_route_candidates` | `route_landscape.alternative_routes[].label` |
| `supporting_evidence_ids` | `evidence_bundle.supporting_evidence_ids` |
| `challenging_evidence_ids` | `evidence_bundle.challenging_evidence_ids` |
| `readiness_feature_inputs.*` | `readiness_scores` input signals |

### 16.2 Current missing transformations

The following transformations still need to be implemented:

1. topic resolution from packet-wide evidence
2. multi-paper method family merging
3. benchmark and protocol normalization
4. bottleneck deduplication and severity calibration
5. alternative-route clustering
6. `L1` join and time-slice enforcement
7. readiness score computation

## 17. Compiler Outline

Recommended `RouteStateSynthesizer` pipeline:

1. validate packet and cutoff boundary
2. load all `PaperLogicTrace` objects
3. load `L1` snapshot
4. collect route compiler contracts and bridge hints
5. normalize labels and merge aliases
6. resolve topic scope
7. aggregate route-landscape sub-objects
8. compute readiness scores
9. derive why-now and not-now feature pools
10. attach evidence and uncertainty
11. run constrained LLM explanation only for rationale strings
12. emit quality block

## 18. Minimal JSON Example

```json
{
  "route_state_id": "cnn_image_recognition_2011_packet_01_v1",
  "schema_version": "v1",
  "built_at": "2026-04-01T18:00:00Z",
  "topic_scope": "large-scale image recognition with deep neural networks",
  "cutoff_year": 2011,
  "source_packet": {
    "packet_id": "packet_01",
    "included_trace_ids": ["p1:trace", "p2:trace"],
    "included_paper_ids": ["p1", "p2"],
    "packet_role_counts": {
      "core_method": 8,
      "resource_or_benchmark": 4,
      "limitation_or_critique": 3,
      "survey_or_review": 2,
      "alternative_route": 3
    },
    "l1_snapshot_ref": "imagenet_2011_snapshot"
  },
  "scope_resolution": {
    "topic_scope_candidates": [
      "cnn image recognition",
      "large-scale image recognition with deep neural networks"
    ],
    "accepted_scope_label": "large-scale image recognition with deep neural networks",
    "rejected_scope_labels": ["generic object recognition"],
    "resolution_rationale": "Packet evidence converges on large-scale supervised image recognition rather than object recognition in general.",
    "resolution_evidence_ids": ["a-1", "a-2", "a-3"]
  },
  "route_landscape": {
    "dominant_methods": [
      {
        "label": "cnn",
        "family": "deep neural networks",
        "maturity_score": 0.62,
        "adoption_level": "workable",
        "source_move_ids": ["m-1", "m-8"],
        "source_paper_ids": ["p1", "p2"],
        "evidence_ids": ["a-2", "a-7"],
        "notes": null
      }
    ],
    "active_benchmarks": [
      {
        "label": "ImageNet",
        "benchmark_type": "dataset",
        "adoption_level": "active",
        "source_paper_ids": ["p1", "p3"],
        "evidence_ids": ["a-5", "a-9"]
      }
    ],
    "measurement_protocols": [],
    "toolchains_and_infrastructure": [],
    "known_capabilities": [],
    "known_bottlenecks": [
      {
        "label": "insufficient gpu throughput",
        "bottleneck_type": "compute",
        "severity": "high",
        "blocking_scope": "route_level",
        "source_paper_ids": ["p4"],
        "evidence_ids": ["a-12"],
        "counterevidence_ids": []
      }
    ],
    "enabling_conditions": [
      {
        "label": "ImageNet available",
        "condition_type": "resource",
        "status": "met",
        "source_paper_ids": ["p3"],
        "evidence_ids": ["a-11"]
      }
    ],
    "alternative_routes": [
      {
        "label": "hand-crafted vision pipelines",
        "route_family": "classical computer vision",
        "relation_to_main_route": "competing",
        "distinguishing_features": ["manual features", "shallower models"],
        "source_paper_ids": ["p5"],
        "evidence_ids": ["a-20"]
      }
    ]
  },
  "readiness_scores": {
    "theory": 0.58,
    "method": 0.62,
    "measurement": 0.85,
    "data_resource": 0.9,
    "infrastructure": 0.66,
    "community": 0.55,
    "cost_cycle": 0.48,
    "overall": 0.66,
    "score_rationale": "Data and benchmark availability are strong, method readiness is moderate, and compute remains a major bottleneck."
  },
  "why_now_features": {
    "unlocking_factors": [],
    "acceleration_factors": [],
    "positive_comparison_signals": []
  },
  "not_now_features": {
    "blocking_factors": [],
    "fragility_factors": [],
    "missing_prerequisites": []
  },
  "evidence_bundle": {
    "supporting_evidence_ids": ["a-1", "a-2"],
    "challenging_evidence_ids": ["a-12"],
    "representative_move_ids": ["m-1", "m-8"],
    "representative_paper_ids": ["p1", "p4"],
    "l1_support_refs": ["imagenet_registry"],
    "l1_constraint_refs": ["gpu_timeline_2011"]
  },
  "uncertainty_points": {
    "open_questions": ["generalization stability under larger models"],
    "unresolved_conflicts": [],
    "weak_fields": ["toolchains_and_infrastructure"],
    "low_confidence_clusters": []
  },
  "compiler_metadata": {
    "compiler_version": "route_state_synthesizer_v1",
    "packet_builder_version": "packet_builder_v1",
    "l1_snapshot_version": "imagenet_snapshot_v1",
    "trace_versions": {
      "p1:trace": "v2",
      "p2:trace": "v2"
    },
    "compile_mode": "rule_plus_llm",
    "llm_usage_notes": "LLM used only for rationale phrasing and uncertainty wording."
  },
  "quality": {
    "quality_tier": "green",
    "ready_for_why_now": true,
    "ready_for_route_comparison": true,
    "ready_for_prior_selection": true,
    "quality_flags": [],
    "audit_status": "eligible"
  }
}
```

## 19. Decision

The next engineering step after this spec should be:

1. define `route packet` schema
2. implement `RouteStateSynthesizer`
3. only then continue targeted `L2` fixes based on compiler failures

This preserves the intended system direction:

1. `L2` stays a stable middle layer
2. `L3` becomes a true route-state reconstruction layer
3. `L4` can be built from structured states rather than open-ended generation
