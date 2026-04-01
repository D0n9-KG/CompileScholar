# LogicKG Decision Prior Layer And DecisionEpisode Schema

## 1. Background

The project strategy already makes a strong early-stage choice:

1. the system should not jump directly from papers to open-ended hypothesis generation
2. the fourth layer should first be narrowed into a decision-prior layer
3. the final training object should be a cross-layer decision sample rather than a free-form proposal

At this point, the schema chain is partially frozen:

1. `PaperLogicTrace` for `L2`
2. `RoutePacket` for packetized historical compilation
3. `RouteState` for `L3`

The next missing objects are:

1. `DecisionPriorCard`
2. `AntiPatternCard`
3. `DecisionEpisode`

These objects are critical because they are what make the project become a scientific-judgment training pipeline rather than a better literature QA system.

## 2. Purpose

This spec defines:

1. the canonical objects of the `Decision Prior Layer`
2. the minimum contract for the final `DecisionEpisode`
3. the auditing and leakage constraints that make these objects training-usable

## 3. Non-Goals

1. These objects are not open-ended hypothesis generators.
2. This spec does not define the full natural-language prompting format for model training.
3. This spec does not define the whole prior-building clustering algorithm.
4. This spec does not define the full inference-time decision policy.

## 4. Design Principles

### 4.1 Priors are conditional, not universal

A `DecisionPriorCard` must always say when it applies and when it does not.

### 4.2 Every prior needs support and counterexample

If a prior cannot be tied to supporting route states and counterexamples, it must not enter training.

### 4.3 Anti-patterns are first-class objects

Failure patterns should not be hidden inside free-form notes.

### 4.4 Final decisions are cross-layer objects

`DecisionEpisode` must combine:

1. `L1` environment
2. `L2` paper logic
3. `L3` route state
4. `L4` priors

### 4.5 Hindsight is label-only

`hindsight_outcome` can be used for labeling and evaluation, but must never enter the training input side.

## 5. Position in the Stack

```text
HistoricalEnvironmentPack (L1)
        +
PaperLogicTrace[] (L2)
        +
RoutePacket
        v
RouteState (L3)
        v
DecisionPriorCard / AntiPatternCard (L4)
        v
DecisionEpisode
```

## 6. Decision Prior Layer

The fourth layer contains two object families:

1. `DecisionPriorCard`
2. `AntiPatternCard`

They are not paper-level objects and not route-state summaries.

They are reusable decision constraints distilled from multiple route states.

## 7. DecisionPriorCard

### 7.1 Purpose

`DecisionPriorCard` captures a reusable positive judgment pattern of the form:

1. when a route shows a certain configuration of readiness and bottlenecks
2. then a certain kind of next move is usually reasonable
3. unless known failure conditions apply

### 7.2 Canonical Object

```text
DecisionPriorCard
- prior_id
- schema_version
- built_at
- prior_text
- applies_when
- does_not_apply_when
- recommended_actions
- expected_failure_modes
- supporting_route_state_ids
- counterexample_ids
- held_out_consistency
- review
- quality
```

### 7.3 Core fields

```text
prior_id: str
schema_version: str
built_at: str
prior_text: str
```

Rules:

1. `prior_text` should be short, abstract, and reusable.
2. It must not name a single paper-specific action unless the prior is explicitly domain-locked.

### 7.4 `applies_when`

```text
applies_when
- readiness_pattern: PriorCondition[]
- bottleneck_pattern: PriorCondition[]
- route_pattern: PriorCondition[]
- evidence_thresholds: PriorThreshold[]
```

```text
PriorCondition
- label: str
- condition_type: readiness | bottleneck | route_shape | environment | comparison | unknown
- polarity: present | absent | high | low | improving | degrading | unknown
- required: bool
```

```text
PriorThreshold
- field: str
- operator: gte | lte | eq | contains | absent
- value: str | float | int | bool
```

This block should encode when the prior is supposed to fire.

### 7.5 `does_not_apply_when`

```text
does_not_apply_when
- blocker_pattern: PriorCondition[]
- fragility_pattern: PriorCondition[]
- mismatch_pattern: PriorCondition[]
```

This field is mandatory. A prior without a non-application boundary is not safe enough.

### 7.6 `recommended_actions`

```text
recommended_actions
- action_type: pursue_question | defer_route | gather_resource | improve_measurement | compare_routes | test_assumption | narrow_scope | unknown
- action_text: str
- target_route_feature: str | null
- confidence: float | null
```

These are not final questions yet. They are recommended next-move types.

### 7.7 `expected_failure_modes`

```text
expected_failure_modes
- label: str
- linked_bottleneck_types: str[]
- warning_signals: str[]
- mitigation_hint: str | null
```

This field keeps the prior honest by making failure legible.

### 7.8 Support and counterexamples

```text
supporting_route_state_ids: str[]
counterexample_ids: str[]
```

Rules:

1. A prior should usually not be accepted with fewer than 3 supporting route states.
2. At least one counterexample should be searched for even if none are found.
3. "No counterexample found" is not the same as "counterexample impossible."

### 7.9 Held-out consistency

```text
held_out_consistency
- held_out_route_state_ids: str[]
- pass_rate: float | null
- failure_notes: str | null
```

This field records whether the prior generalizes beyond the cluster it was generated from.

### 7.10 Review

```text
review
- review_status: draft | candidate | reviewed | approved | rejected
- reviewer_notes: str | null
- reviewer_ids: str[]
```

### 7.11 Quality

```text
quality
- quality_tier: red | yellow | green
- quality_flags: str[]
- audit_status: hot_path | eligible | reviewed
```

Common flags:

1. `missing_counterexample_search`
2. `too_paper_specific`
3. `weak_support_cluster`
4. `held_out_failure`
5. `overlapping_with_existing_prior`

## 8. AntiPatternCard

### 8.1 Purpose

`AntiPatternCard` captures a reusable negative pattern:

1. routes that look promising on the surface
2. but systematically fail under certain warning signals
3. and should trigger caution or rejection

### 8.2 Canonical Object

```text
AntiPatternCard
- anti_pattern_id
- schema_version
- built_at
- anti_pattern_text
- warning_signal_pattern
- failure_examples
- corrective_checklist
- counterexamples
- review
- quality
```

### 8.3 Core fields

```text
anti_pattern_id: str
schema_version: str
built_at: str
anti_pattern_text: str
```

### 8.4 `warning_signal_pattern`

```text
warning_signal_pattern
- signals: WarningSignal[]
- trigger_logic: all | any | weighted
```

```text
WarningSignal
- label: str
- signal_type: bottleneck | contradiction | weak_comparison | weak_measurement | resource_gap | route_fragility | cost_spike | unknown
- severity: low | medium | high | blocking | unknown
- source_field: str | null
```

### 8.5 `failure_examples`

```text
failure_examples
- route_state_ids: str[]
- decision_episode_ids: str[]
- notes: str | null
```

The examples should include real failures or retrospectively poor choices under historical replay.

### 8.6 `corrective_checklist`

```text
corrective_checklist: str[]
```

This should contain concrete checks like:

1. verify missing benchmark support
2. test whether enabling condition is actually met
3. inspect route comparison against alternative paths

### 8.7 `counterexamples`

```text
counterexamples
- route_state_ids: str[]
- notes: str | null
```

Anti-patterns also need counterexamples because some warning signals may be survivable under specific conditions.

### 8.8 Review and quality

Use the same structure as `DecisionPriorCard`, with additional common flags:

1. `warning_pattern_too_generic`
2. `failure_examples_too_few`
3. `missing_corrective_checklist`
4. `counterexample_gap`

## 9. Decision Prior Builder Contract

### 9.1 Input

The prior builder should consume:

1. multiple `RouteState` objects
2. optional `WhyNowCase` objects
3. optional `RouteComparisonCase` objects
4. optional historical replay outcomes

### 9.2 Recommended workflow

```text
1. cluster route states by readiness and bottleneck profile
2. generate candidate priors and anti-patterns per cluster
3. search for held-out route states
4. search for counterexamples
5. rank cards by support, stability, and distinctiveness
6. send only top cards to expert review
```

### 9.3 Strict rule

If support, held-out check, or counterexample search is missing, the card must stay below `green`.

## 10. DecisionEpisode

### 10.1 Purpose

`DecisionEpisode` is the final cross-layer decision sample used for training and evaluation.

It should represent a historically constrained decision situation:

1. what was observable at the cutoff
2. what route state could be reconstructed
3. what priors should have been invoked
4. what candidate question or action should have been preferred
5. what later outcome happened

### 10.2 Canonical Object

```text
DecisionEpisode
- episode_id
- schema_version
- built_at
- historical_cutoff_time
- observation_evidence_pack
- paper_logic_traces
- route_state
- relevant_priors
- candidate_question
- alternative_questions
- why_this_not_that
- not_now_cases
- minimal_attack_path
- decision_output
- hindsight_outcome
- compiler_metadata
- quality
```

### 10.3 Top-level identity

```text
episode_id: str
schema_version: str
built_at: str
historical_cutoff_time: str
```

Rules:

1. `historical_cutoff_time` must be precise enough to enforce visibility.
2. This object must be replayable without hidden context.

### 10.4 `observation_evidence_pack`

```text
observation_evidence_pack
- route_packet_id: str
- l1_snapshot_ref: str | null
- visible_paper_ids: str[]
- visible_trace_ids: str[]
- excluded_after_cutoff_ids: str[]
- evidence_refs: str[]
```

This explicitly defines what the model is allowed to see.

### 10.5 `paper_logic_traces`

```text
paper_logic_traces
- trace_ids: str[]
- representative_trace_ids: str[]
```

This field should avoid copying whole traces into the final object; references are enough.

### 10.6 `route_state`

```text
route_state
- route_state_id: str
- route_state_ref: str | null
```

### 10.7 `relevant_priors`

```text
relevant_priors
- selected_prior_ids: str[]
- selected_antipattern_ids: str[]
- prior_selection_rationale: str | null
```

This is the bridge between `L3` and `L4`.

### 10.8 Candidate question layer

```text
candidate_question
- question_text: str
- question_type: hypothesis | route_choice | scope_refinement | measurement_gap | benchmark_gap | resource_gap | unknown
- target_route_feature: str | null
- justification_evidence_ids: str[]
```

```text
alternative_questions: CandidateQuestion[]
```

Rules:

1. The candidate should be historically feasible enough to be worth evaluation.
2. Alternatives should be plausible, not strawman distractors.

### 10.9 Comparison layer

```text
why_this_not_that
- primary_reasoning: str
- comparison_dimensions: EpisodeComparisonDimension[]
- evidence_chain: str[]
```

```text
EpisodeComparisonDimension
- dimension: method_maturity | measurement | data_resource | infrastructure | bottleneck | novelty | feasibility | strategic_value | unknown
- preferred_candidate: primary | alternative_1 | alternative_2 | tie | unknown
- rationale: str | null
```

### 10.10 `not_now_cases`

```text
not_now_cases
- rejected_question_text: str
- blocker_summary: str
- evidence_ids: str[]
```

This field is critical because valuable scientific judgment includes knowing when not to pursue a question yet.

### 10.11 `minimal_attack_path`

```text
minimal_attack_path
- prerequisite_steps: str[]
- required_resources: str[]
- required_measurements: str[]
- expected_checkpoints: str[]
```

This field converts judgment into an actionable attack plan.

### 10.12 `decision_output`

```text
decision_output
- final_choice: primary | alternative_1 | alternative_2 | reject_all | defer
- final_decision_text: str
- confidence: float | null
```

### 10.13 `hindsight_outcome`

```text
hindsight_outcome
- outcome_label: success | partial_success | failure | still_open | unknown
- later_evidence_refs: str[]
- retrospective_notes: str | null
- input_visible: bool
```

Rules:

1. `input_visible` must always be `false` for training inputs.
2. This field exists only for labels and evaluation.

### 10.14 Compiler metadata

```text
compiler_metadata
- episode_builder_version: str
- route_state_version: str | null
- prior_layer_version: str | null
- compile_mode: rule_only | rule_plus_llm | llm_assisted_audit
- notes: str | null
```

### 10.15 Quality

```text
quality
- quality_tier: red | yellow | green
- ready_for_training: bool
- ready_for_eval: bool
- quality_flags: str[]
- audit_status: hot_path | eligible | reviewed
```

Common flags:

1. `hindsight_leakage_risk`
2. `weak_prior_support`
3. `missing_not_now_case`
4. `candidate_question_too_open_ended`
5. `minimal_attack_path_missing`
6. `weak_alternative_set`

## 11. Minimal JSON Example

```json
{
  "episode_id": "image_recognition_2011_ep_01",
  "schema_version": "v1",
  "built_at": "2026-04-01T19:00:00Z",
  "historical_cutoff_time": "2011-12-31T23:59:59Z",
  "observation_evidence_pack": {
    "route_packet_id": "image_recognition_2011_packet_01",
    "l1_snapshot_ref": "imagenet_2011_snapshot",
    "visible_paper_ids": ["p1", "p2", "p3"],
    "visible_trace_ids": ["p1:trace", "p2:trace", "p3:trace"],
    "excluded_after_cutoff_ids": ["p99"],
    "evidence_refs": ["a-1", "a-2", "a-12"]
  },
  "paper_logic_traces": {
    "trace_ids": ["p1:trace", "p2:trace", "p3:trace"],
    "representative_trace_ids": ["p1:trace", "p3:trace"]
  },
  "route_state": {
    "route_state_id": "cnn_image_recognition_2011_packet_01_v1",
    "route_state_ref": "route_state_store/cnn_image_recognition_2011_packet_01_v1.json"
  },
  "relevant_priors": {
    "selected_prior_ids": ["prior_07"],
    "selected_antipattern_ids": ["anti_03"],
    "prior_selection_rationale": "The route shows strong data/benchmark readiness but still has compute and stability fragility."
  },
  "candidate_question": {
    "question_text": "Should the route shift from hand-crafted pipelines toward deep convolutional architectures for large-scale supervised image recognition?",
    "question_type": "route_choice",
    "target_route_feature": "method_maturity",
    "justification_evidence_ids": ["a-1", "a-2"]
  },
  "alternative_questions": [
    {
      "question_text": "Should effort remain focused on hand-crafted vision pipelines under current compute limits?",
      "question_type": "route_choice",
      "target_route_feature": "infrastructure",
      "justification_evidence_ids": ["a-12"]
    }
  ],
  "why_this_not_that": {
    "primary_reasoning": "The deep route now benefits from newly available benchmark scale and improving infrastructure, while the classical route appears more mature but less strategically scalable.",
    "comparison_dimensions": [
      {
        "dimension": "data_resource",
        "preferred_candidate": "primary",
        "rationale": "ImageNet availability changes the ceiling for the deep route."
      }
    ],
    "evidence_chain": ["a-1", "a-2", "a-12"]
  },
  "not_now_cases": [
    {
      "rejected_question_text": "Can fully unrestricted very-deep models be trained reliably at this cutoff?",
      "blocker_summary": "Compute and training stability remain too fragile.",
      "evidence_ids": ["a-12"]
    }
  ],
  "minimal_attack_path": {
    "prerequisite_steps": [
      "verify benchmark-scale data pipeline",
      "stabilize training protocol",
      "compare against classical baselines"
    ],
    "required_resources": ["ImageNet", "GPU throughput"],
    "required_measurements": ["top-1 accuracy", "top-5 accuracy"],
    "expected_checkpoints": ["baseline parity", "scaling gain visible"]
  },
  "decision_output": {
    "final_choice": "primary",
    "final_decision_text": "Prioritize the deep route, but narrow scope to historically feasible training and evaluation conditions.",
    "confidence": 0.76
  },
  "hindsight_outcome": {
    "outcome_label": "success",
    "later_evidence_refs": ["future_ref_1"],
    "retrospective_notes": "Later history supports the route choice, but this evidence was not visible at cutoff.",
    "input_visible": false
  },
  "compiler_metadata": {
    "episode_builder_version": "decision_episode_builder_v1",
    "route_state_version": "route_state_synthesizer_v1",
    "prior_layer_version": "decision_prior_builder_v1",
    "compile_mode": "rule_plus_llm",
    "notes": "LLM used only for constrained rationale phrasing."
  },
  "quality": {
    "quality_tier": "green",
    "ready_for_training": true,
    "ready_for_eval": true,
    "quality_flags": [],
    "audit_status": "eligible"
  }
}
```

## 12. Build Order

Recommended order after this spec:

1. implement `WhyNowCase` and `RouteComparisonCase` schemas if needed for training datasets
2. implement `DecisionPriorBuilder`
3. implement `DecisionEpisodeBuilder`
4. run small historical replay pilots

## 13. Decision

The project should now treat the fourth layer and the final decision sample as constrained, auditable objects.

This preserves the intended system behavior:

1. no direct jump from papers to open-ended idea generation
2. priors are conditional and reviewable
3. final decisions are historically replayable
