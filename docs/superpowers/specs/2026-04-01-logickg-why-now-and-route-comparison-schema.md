# LogicKG WhyNowCase And RouteComparisonCase Schema

## 1. Purpose

This spec defines the two reasoning objects that sit directly on top of `RouteState`:

1. `WhyNowCase`
2. `RouteComparisonCase`

These are not generic explanations.

They are training-oriented reasoning objects that convert route-state structure into explicit judgment:

1. why a route is becoming timely or remains premature
2. why one route should be preferred over another under the same historical cutoff

## 2. Position in the Stack

```text
HistoricalEnvironmentPack (L1)
        +
PaperLogicTrace[] (L2)
        +
RoutePacket
        v
RouteState (L3)
        v
WhyNowCase / RouteComparisonCase
        v
DecisionPriorCard / AntiPatternCard (L4)
        v
DecisionEpisode
```

## 3. Non-Goals

1. `WhyNowCase` is not a route-state restatement.
2. `RouteComparisonCase` is not a final decision episode.
3. These objects are not free-form chain-of-thought dumps.

## 4. Design Principles

### 4.1 Explicit reasons only

Every claim in these objects should be tied to explicit route-state or evidence fields.

### 4.2 Comparative judgment is structured

The output should expose dimensions, not just a final preference sentence.

### 4.3 Timeliness is conditional

A route can be promising and still be "not now."

### 4.4 Uncertainty must stay visible

These objects should preserve uncertainty rather than hide it behind fluent prose.

## 5. WhyNowCase

### 5.1 Purpose

`WhyNowCase` answers:

1. why this route is becoming more actionable now
2. or why it is still not actionable yet

under a fixed historical cutoff.

### 5.2 Canonical Object

```text
WhyNowCase
- why_now_case_id
- schema_version
- built_at
- route_state_id
- why_now_label
- unlocking_factors
- blocking_factors
- evidence_chain
- uncertainty_points
- quality
```

### 5.3 Core fields

```text
why_now_case_id: str
schema_version: str
built_at: str
route_state_id: str
why_now_label: now | almost_now | not_now | unclear
```

The label should be derived from structured route-state evidence, not directly guessed by an LLM.

### 5.4 `unlocking_factors`

```text
unlocking_factors: WhyNowFactor[]
```

```text
WhyNowFactor
- label: str
- factor_type: method_maturity | benchmark_availability | data_resource | infrastructure | protocol | comparative_advantage | community_shift | unknown
- strength: low | medium | high | unknown
- source_route_fields: str[]
- evidence_ids: str[]
- l1_refs: str[]
```

### 5.5 `blocking_factors`

```text
blocking_factors: WhyNowFactor[]
```

Here `factor_type` may also include:

1. `bottleneck`
2. `resource_gap`
3. `fragility`
4. `cost_cycle`

### 5.6 `evidence_chain`

```text
evidence_chain
- supporting_evidence_ids: str[]
- challenging_evidence_ids: str[]
- representative_route_fields: str[]
```

### 5.7 `uncertainty_points`

```text
uncertainty_points
- unresolved_conflicts: str[]
- weak_signals: str[]
- ambiguous_enablers: str[]
```

### 5.8 Quality

```text
quality
- quality_tier: red | yellow | green
- ready_for_training: bool
- quality_flags: str[]
```

Common flags:

1. `weak_unlocking_factors`
2. `blocking_factors_missing`
3. `label_not_grounded`
4. `evidence_chain_thin`

## 6. RouteComparisonCase

### 6.1 Purpose

`RouteComparisonCase` answers:

1. under the same cutoff year
2. between two plausible routes
3. which route is strategically preferable and why

### 6.2 Canonical Object

```text
RouteComparisonCase
- route_comparison_case_id
- schema_version
- built_at
- cutoff_year
- route_a_state_id
- route_b_state_id
- comparison_dimension_scores
- preference_label
- why_a_not_b
- why_b_not_a
- evidence_chain
- uncertainty_points
- quality
```

### 6.3 Core fields

```text
route_comparison_case_id: str
schema_version: str
built_at: str
cutoff_year: int
route_a_state_id: str
route_b_state_id: str
preference_label: prefer_a | prefer_b | tie | unclear
```

### 6.4 `comparison_dimension_scores`

```text
comparison_dimension_scores: ComparisonDimensionScore[]
```

```text
ComparisonDimensionScore
- dimension: method_maturity | measurement | data_resource | infrastructure | bottleneck | feasibility | strategic_value | novelty | cost_cycle | unknown
- route_a_score: float | null
- route_b_score: float | null
- preferred_route: a | b | tie | unknown
- rationale: str | null
- evidence_ids: str[]
```

### 6.5 `why_a_not_b`

```text
why_a_not_b
- summary: str
- decisive_dimensions: str[]
- decisive_evidence_ids: str[]
```

### 6.6 `why_b_not_a`

```text
why_b_not_a
- summary: str
- decisive_dimensions: str[]
- decisive_evidence_ids: str[]
```

Keeping both directions explicit prevents one-sided comparisons from hiding asymmetry.

### 6.7 `evidence_chain`

```text
evidence_chain
- route_a_support_ids: str[]
- route_b_support_ids: str[]
- cross_route_comparison_ids: str[]
```

### 6.8 `uncertainty_points`

```text
uncertainty_points
- incomparable_dimensions: str[]
- weak_dimensions: str[]
- unresolved_conflicts: str[]
```

### 6.9 Quality

```text
quality
- quality_tier: red | yellow | green
- ready_for_training: bool
- quality_flags: str[]
```

Common flags:

1. `alternative_route_not_real`
2. `comparison_dimensions_too_sparse`
3. `preference_not_grounded`
4. `same_route_disguised_as_two`

## 7. Builder Contracts

### 7.1 WhyNowCaseBuilder

Inputs:

1. one `RouteState`

Workflow:

1. read `readiness_scores`
2. read `why_now_features`
3. read `not_now_features`
4. aggregate unlocking vs blocking balance
5. emit grounded label and factor lists

### 7.2 RouteComparisonBuilder

Inputs:

1. two `RouteState` objects under the same cutoff

Workflow:

1. align dimensions
2. compare route readiness and bottlenecks
3. identify decisive dimensions
4. emit bidirectional comparison rationale

## 8. Minimal JSON Example

### 8.1 WhyNowCase

```json
{
  "why_now_case_id": "cnn_2011_why_now_01",
  "schema_version": "v1",
  "built_at": "2026-04-01T19:15:00Z",
  "route_state_id": "cnn_image_recognition_2011_packet_01_v1",
  "why_now_label": "almost_now",
  "unlocking_factors": [
    {
      "label": "ImageNet availability",
      "factor_type": "benchmark_availability",
      "strength": "high",
      "source_route_fields": ["route_landscape.active_benchmarks"],
      "evidence_ids": ["a-1"],
      "l1_refs": ["benchmark_timeline_imagenet"]
    }
  ],
  "blocking_factors": [
    {
      "label": "GPU throughput remains limited",
      "factor_type": "infrastructure",
      "strength": "high",
      "source_route_fields": ["route_landscape.known_bottlenecks"],
      "evidence_ids": ["a-12"],
      "l1_refs": ["gpu_timeline_2011"]
    }
  ],
  "evidence_chain": {
    "supporting_evidence_ids": ["a-1", "a-2"],
    "challenging_evidence_ids": ["a-12"],
    "representative_route_fields": ["active_benchmarks", "known_bottlenecks", "readiness_scores"]
  },
  "uncertainty_points": {
    "unresolved_conflicts": [],
    "weak_signals": ["community readiness"],
    "ambiguous_enablers": []
  },
  "quality": {
    "quality_tier": "green",
    "ready_for_training": true,
    "quality_flags": []
  }
}
```

### 8.2 RouteComparisonCase

```json
{
  "route_comparison_case_id": "cnn_vs_classical_2011_cmp_01",
  "schema_version": "v1",
  "built_at": "2026-04-01T19:18:00Z",
  "cutoff_year": 2011,
  "route_a_state_id": "cnn_image_recognition_2011_packet_01_v1",
  "route_b_state_id": "classical_vision_2011_packet_02_v1",
  "comparison_dimension_scores": [
    {
      "dimension": "data_resource",
      "route_a_score": 0.9,
      "route_b_score": 0.6,
      "preferred_route": "a",
      "rationale": "Large-scale benchmark availability disproportionately benefits the deep route.",
      "evidence_ids": ["a-1", "b-2"]
    },
    {
      "dimension": "infrastructure",
      "route_a_score": 0.66,
      "route_b_score": 0.85,
      "preferred_route": "b",
      "rationale": "The classical route has lower compute requirements.",
      "evidence_ids": ["a-12", "b-5"]
    }
  ],
  "preference_label": "prefer_a",
  "why_a_not_b": {
    "summary": "Route A has better strategic upside under the newly available benchmark regime.",
    "decisive_dimensions": ["data_resource", "strategic_value"],
    "decisive_evidence_ids": ["a-1", "a-2"]
  },
  "why_b_not_a": {
    "summary": "Route B remains safer under current infrastructure constraints, but appears less scalable.",
    "decisive_dimensions": ["infrastructure", "method_maturity"],
    "decisive_evidence_ids": ["b-5", "a-12"]
  },
  "evidence_chain": {
    "route_a_support_ids": ["a-1", "a-2"],
    "route_b_support_ids": ["b-2", "b-5"],
    "cross_route_comparison_ids": ["a-12", "b-5"]
  },
  "uncertainty_points": {
    "incomparable_dimensions": [],
    "weak_dimensions": ["novelty"],
    "unresolved_conflicts": []
  },
  "quality": {
    "quality_tier": "green",
    "ready_for_training": true,
    "quality_flags": []
  }
}
```

## 9. Decision

With this spec in place, the five key higher-layer objects from the strategy are now materially frozen in schema form:

1. `RouteState`
2. `WhyNowCase`
3. `RouteComparisonCase`
4. `DecisionPriorCard`
5. `AntiPatternCard`

The next logical implementation step is no longer schema design.

It is:

1. packet-level pilot compilation
2. `RouteStateSynthesizer`
3. `WhyNowCaseBuilder`
4. `RouteComparisonBuilder`
5. `DecisionPriorBuilder`
