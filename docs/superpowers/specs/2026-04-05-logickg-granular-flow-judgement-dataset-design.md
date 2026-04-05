# LogicKG Granular-Flow Judgement Dataset Design

## 1. Purpose

This design defines the first serious research version of LogicKG as a data engine for training scientific judgement models in mechanics, with granular flow as the pilot domain.

The goal is not to build a general "AI scientist" system in one step. The goal is to produce a high-quality, paper-grounded, historically replayable training dataset that improves a model's ability to:

1. judge whether a candidate scientific question is worth doing under a historical cutoff
2. explain why it is worth doing now, premature, not competitive, or under-specified
3. compare competing routes under the same topic and cutoff
4. diagnose bottlenecks and identify the minimal attack path

This design also keeps a clean upgrade path toward the harder second-stage goal: generating high-value domain hypotheses.

## 2. Scope

### 2.1 In scope

1. granular-flow mechanics as the first pilot domain
2. markdown papers as the first source corpus
3. single-paper extraction into a canonical statement graph
4. citation-aware cross-paper synthesis
5. paper-grounded historical replay labeling
6. multi-view dataset export for SFT, preference learning, and later reward modeling / RL

### 2.2 Out of scope for v1

1. a frontend product or graph browser
2. full real-world environment reconstruction beyond literature-visible evidence
3. open-ended hypothesis generation as the main task
4. graph-native RL agents acting directly over the knowledge graph
5. multi-domain generalization beyond mechanics before the pilot closes

## 3. Main Design Decision

The repository should move from a route-first downstream schema to a canonical, replay-first evidence architecture.

The recommended architecture is:

```text
markdown papers
    -> single-paper statement graph
    -> citation mention + citation act layers
    -> canonical argumentative temporal research KG
    -> historical replay compiler
    -> multi-view judgement dataset
```

This means:

1. the knowledge graph is required
2. the raw graph is not the training dataset
3. route objects and question objects are compiled objects, not primary facts
4. citation information is a core cross-paper signal, not a side attribute

## 4. Alternatives Considered

### 4.1 Route-first downstream schema

This approach compiles early into objects like `RouteState`, `WhyNowCase`, and `DecisionEpisode`.

Advantages:

1. close to the current repository direction
2. short engineering path
3. easy to explain in early demos

Disadvantages:

1. compresses evidence too early
2. makes citation signals too shallow
3. binds the project to route-centric tasks
4. makes later judgement and hypothesis tasks harder to support cleanly

Decision:

Rejected as the canonical architecture. Some current downstream objects may remain as legacy experiments or later exports.

### 4.2 Graph-native RL / agent-first architecture

This approach treats the graph as the environment and directly optimizes policies over graph actions.

Advantages:

1. attractive for long-term scientific-agent work
2. naturally connects to later hypothesis generation

Disadvantages:

1. too risky for the first paper
2. training, evaluation, and system complexity all rise together
3. would obscure the main contribution, which should be the dataset and judgement task

Decision:

Rejected for v1. Keep reward-oriented exports so this remains possible later.

### 4.3 Canonical argumentative temporal KG plus replay compiler

Advantages:

1. supports single-paper, cross-paper, and macro-route information together
2. makes citation a first-class evidence source
3. keeps the evidence layer reusable across multiple training views
4. fits the first-paper goal of dataset plus model improvement

Decision:

Accepted.

## 5. Research Boundary

Version 1 is explicitly **paper-grounded**.

The system is allowed to claim:

1. reconstruction of literature-visible historical research conditions
2. historical scientific judgement under paper-visible evidence
3. replayable, audit-ready labels from later paper evolution

The system is not allowed to claim:

1. complete reconstruction of true real-world environment constraints
2. full access to unpublished failures or hidden operational conditions
3. final scientific hypothesis generation performance

To make this boundary explicit, the environment layer is defined as:

1. `LiteratureVisibleEnvironmentSnapshot` in v1
2. `ExtendedEnvironmentSnapshot` reserved for future non-paper sources

### 5.1 Temporal boundary rule

Historical replay in v1 is governed by a canonical `cutoff_date`, not only a coarse year bucket.

Rules:

1. every replay object keeps `cutoff_year` for reporting and grouping
2. every replay build also computes an internal `cutoff_date`
3. a paper or event is visible only if `publication_date <= cutoff_date`
4. if exact publication date is unknown and only year is known, the system uses a conservative fallback date of `YYYY-12-31` and marks the record as `time_precision=year`
5. same-year papers are therefore only visible automatically when their effective publication date does not exceed the canonical cutoff date
6. if time precision is too weak to support a stable replay decision, the sample may be downgraded to `underspecified`

This decision is mandatory because leakage control cannot rely on `cutoff_year` alone.

### 5.2 Publication date source-of-truth rule

When multiple publication timestamps exist for one paper, v1 uses the earliest public research exposure that can be independently verified.

Priority order:

1. first publicly accessible preprint date
2. online-first journal or proceedings publication date
3. official journal issue or conference publication date
4. year-only fallback with `time_precision=year`

Version rule:

1. if a preprint is visible before the cutoff and the formal publication is after the cutoff, only preprint-visible content is allowed into replay inputs
2. later revised text, supplementary files, or formal-publication additions after the cutoff are audit-only
3. `future_work` or open-problem statements that appear in cutoff-visible papers are valid replay inputs, even if later papers eventually pursue them

## 6. Canonical Storage Decision

The source of truth should be file-based canonical assets, not Neo4j.

Recommended storage roles:

1. `JSONL` for record-like canonical objects
2. `Parquet` for large tabular event stores and dataset exports
3. `Neo4j` only as a projection for querying, debugging, and visualization

This is required because the main output is a training dataset, not a graph product.

Recommended canonical asset families:

```text
papers.parquet
statements.jsonl
anchors.jsonl
slot_mentions.jsonl
statement_edges.jsonl
citation_mentions.jsonl
citation_acts.parquet
evaluation_events.parquet
resource_usage_events.parquet
problem_signal_events.parquet
topic_scopes.jsonl
route_families.jsonl
bottleneck_clusters.jsonl
question_candidates.jsonl
decision_units.jsonl
evidence_packs.jsonl
historical_outcome_traces.jsonl
dataset_views/*.jsonl
```

### 6.1 Versioning and freeze policy

The first paper must be built from a frozen corpus and a frozen rule set.

Rules:

1. every official dataset release binds:
   - corpus manifest version
   - compiler version set
   - label-policy version set
   - topic-scope manifest
   - cutoff manifest
2. the first paper uses full rebuilds from frozen inputs, not ad hoc incremental mutation
3. newly ingested papers may only enter later dataset versions, never an already frozen release
4. audit metadata should record the exact replay build version for every exported sample

### 6.2 Partitioning strategy

Canonical compiled assets should be partitioned by:

1. `topic_scope_id`
2. `cutoff_year`
3. dataset release version when relevant

This is required for replay reproducibility and sample traceability.

## 7. Canonical Object Taxonomy

The knowledge layer must be separated into four object classes:

1. `Node`
2. `Edge`
3. `Event`
4. `CompiledObject`

This is a hard design rule. No downstream training view should redefine this taxonomy ad hoc.

## 8. Node Schema

### 8.1 Required nodes

1. `Paper`
2. `Statement`
3. `EvidenceAnchor`
4. `ResearchObject`
5. `Method`
6. `Variable`
7. `Metric`
8. `Condition`
9. `Resource`
10. `TopicConcept`
11. `BottleneckType`

### 8.2 `Paper`

Minimum fields:

```text
paper_id
title
year
publication_date
publication_version
time_precision
authors
venue
doi
paper_type
source_path
```

Allowed `publication_version` values:

1. `preprint`
2. `online_first`
3. `official_publication`
4. `unknown`

Allowed `time_precision` values:

1. `day`
2. `month`
3. `year`
4. `unknown`

### 8.3 `Statement`

`Statement` replaces task-specific naming like move-only or claim-only objects at the canonical layer.

Minimum fields:

```text
statement_id
paper_id
statement_type
secondary_tags
summary
anchor_ids
confidence
```

Allowed `statement_type` values:

1. `problem`
2. `method`
3. `result`
4. `interpretation`
5. `limitation`
6. `future_work`
7. `background`

Design rule:

1. each `Statement` has exactly one primary `statement_type`
2. `EvidenceAnchor` objects may ground multiple statements when one passage contains mixed functions
3. `secondary_tags` may record additional semantic hints such as `interpretive`, `comparative`, or `negative_result`, but they do not replace the primary type

### 8.4 `EvidenceAnchor`

Minimum fields:

```text
anchor_id
paper_id
modality
section_path
span_start
span_end
quote
source_ref
```

Allowed `modality` values:

1. `text`
2. `figure`
3. `table`
4. `caption`

### 8.5 Canonical concept nodes

For `ResearchObject`, `Method`, `Variable`, `Metric`, `Condition`, `Resource`, `TopicConcept`, and `BottleneckType`, every normalized node must keep:

```text
canonical_id
canonical_name
aliases
entity_type
source_mentions
confidence
```

The first version should use conservative linking. False merges are more harmful than missed merges.

For `Resource`, the canonical schema must also keep a subtype:

1. `dataset`
2. `benchmark`
3. `software`
4. `hardware`
5. `instrument`
6. `protocol`
7. `other`

### 8.6 Granular-flow pilot lexicon

The first implementation plan should include a controlled pilot lexicon for granular flow so extraction and linking stay domain-precise.

Minimum curated families:

1. `Method`
   - `DEM`
   - `CFD-DEM`
   - `LBM`
   - `Eulerian-Eulerian`
   - `PTV / PIV`
   - `tomography-based reconstruction`
2. `ResearchObject`
   - particle phase
   - fluid phase
   - packed bed
   - fluidized bed
   - hopper / silo flow
   - slurry or suspension system
3. `Metric`
   - pressure drop
   - packing fraction
   - velocity field
   - mixing index
   - drag correlation error
4. `BottleneckType`
   - contact-model fidelity
   - turbulence coupling uncertainty
   - multiscale compute cost
   - measurement resolution
   - wall-effect uncertainty

The lexicon should be versioned and extendable, but v1 should not attempt a domain-agnostic ontology from the start.

## 9. Edge Schema

Edges represent stable semantic relations, not context-rich events.

Required edge types:

1. `PAPER_HAS_STATEMENT`
2. `STATEMENT_GROUNDED_BY_ANCHOR`
3. `STATEMENT_ABOUT_OBJECT`
4. `STATEMENT_USES_METHOD`
5. `STATEMENT_OBSERVES_VARIABLE`
6. `STATEMENT_EVALUATED_BY_METRIC`
7. `STATEMENT_UNDER_CONDITION`
8. `STATEMENT_MENTIONS_RESOURCE`
9. `STATEMENT_HAS_BOTTLENECK`
10. `STATEMENT_RELATES_TO_STATEMENT`

For `STATEMENT_RELATES_TO_STATEMENT`, the relation subtype should stay close to the current repository:

1. `motivates`
2. `tests`
3. `supports`
4. `compares`
5. `limits`
6. `extends`
7. `explains`

## 10. Event Schema

Events are the most important object class in v1. Citation and evaluation information must live here.

### 10.1 `CitationMentionEvent`

This is the raw citation context layer.

Minimum fields:

```text
citation_mention_id
citing_paper_id
cited_paper_id
source_anchor_id
source_statement_id
ref_marker
context_text
section_path
span_start
span_end
```

One textual mention should create one event.

### 10.2 `CitationActEvent`

This is the semantic citation layer. It aggregates one or more mentions.

Minimum fields:

```text
citation_act_id
citing_paper_id
citing_statement_id
cited_paper_id
cited_statement_candidates
intent_labels
stance
target_content_type
evidence_mention_ids
confidence
```

Allowed `intent_labels` in v1:

1. `background`
2. `problem_motivation`
3. `method_use`
4. `data_or_tool_use`
5. `baseline_compare`
6. `support_claim`
7. `critique_or_limit`
8. `extend_or_improve`
9. `future_direction`

Allowed `stance` values:

1. `support`
2. `contrast`
3. `neutral`

Allowed `target_content_type` values:

1. `problem`
2. `method`
3. `result`
4. `resource`
5. `limitation`
6. `paper`

Design rule:

The hard target for v1 is `citing_statement -> cited_paper`.  
`cited_statement_candidates` are supported but not required for compiler correctness.

`confidence` fields in all v1 event schemas are normalized to `[0, 1]` and should be interpreted as compiler confidence, not model probability.

Escalation rule:

1. for `background` and broad `data_or_tool_use`, paper-level targets are acceptable
2. for `support_claim` and `critique_or_limit`, the system should attempt statement-level candidate resolution
3. if statement-level candidates cannot be resolved for these high-value intents, the event remains usable but should carry a downgrade flag and lower confidence

### 10.3 `EvaluationEvent`

This records measurable performance or comparison facts.

Minimum fields:

```text
evaluation_event_id
statement_id
paper_id
method_id
research_object_id
metric_id
condition_ids
comparison_target
effect_direction
effect_size
anchor_ids
```

### 10.4 `ResourceUsageEvent`

Minimum fields:

```text
resource_usage_event_id
paper_id
statement_id
resource_id
usage_role
year
anchor_ids
```

Allowed `usage_role` values:

1. `introduced`
2. `used`
3. `baseline`
4. `required`
5. `limited_by`

### 10.5 `ProblemSignalEvent`

Minimum fields:

```text
problem_signal_event_id
paper_id
statement_id
signal_type
target_topic_id
anchor_ids
```

Allowed `signal_type` values:

1. `future_direction`
2. `limitation`
3. `open_problem`
4. `unresolved_conflict`
5. `question_seed`

## 11. Compiled Object Schema

Compiled objects are not canonical facts. They are replay-ready syntheses.

Required compiled objects in v1:

1. `TopicScope`
1. `LiteratureVisibleEnvironmentSnapshot`
2. `SupportConflictCluster`
3. `RouteFamily`
4. `BottleneckCluster`
5. `QuestionCandidate`
6. `AlternativeRouteCandidate`
7. `ResearchDecisionUnit`
8. `EvidencePack`
9. `HistoricalOutcomeTrace`

For all compiled objects in v1:

1. `confidence` fields are normalized to `[0, 1]`
2. they represent compiler confidence from evidence sufficiency and agreement, not calibrated model probability

## 12. `TopicScope`

This object freezes the bounded topic under which replay happens.

Minimum fields:

```text
topic_scope_id
canonical_label
aliases
seed_statement_ids
seed_citation_act_ids
inclusion_notes
exclusion_notes
confidence
```

`TopicScope` is compiled from paper-level topic signals and then treated as a stable replay boundary.

## 13. `LiteratureVisibleEnvironmentSnapshot`

This is the v1 environment object. It is intentionally paper-grounded.

Minimum fields:

```text
snapshot_id
topic_scope_id
cutoff_year
visible_resource_ids
visible_benchmark_ids
visible_protocol_ids
resource_usage_refs
protocol_signal_refs
cost_proxy_refs
quality_flags
```

This object is allowed to be incomplete relative to true real-world conditions. It must not imply full external grounding.

Reference semantics:

1. `visible_benchmark_ids` are `Resource.canonical_id` values with subtype `benchmark` or `dataset`
2. `visible_protocol_ids` are `Resource.canonical_id` values with subtype `protocol`
3. `resource_usage_refs` are `ResourceUsageEvent.resource_usage_event_id` values
4. `protocol_signal_refs` are `Statement.statement_id` or `EvaluationEvent.evaluation_event_id` values that justify protocol visibility
5. `cost_proxy_refs` are `Statement.statement_id` or `ResourceUsageEvent.resource_usage_event_id` values carrying literature-visible cost or cycle proxies

## 14. `SupportConflictCluster`

This object captures multi-paper support and critique patterns around a target statement family, method family, route family, or question candidate.

Minimum fields:

```text
cluster_id
topic_scope_id
cutoff_year
target_type
target_ids
support_statement_ids
challenge_statement_ids
citation_support_act_ids
citation_critique_act_ids
confidence
```

Allowed `target_type` values:

1. `statement_family`
2. `method_family`
3. `route_family`
4. `question_candidate`
5. `bottleneck_cluster`

`citation_support_act_ids` and `citation_critique_act_ids` are ordered `CitationActEvent.citation_act_id` lists. v1 does not require a separate `CitationChain` object.

## 15. `RouteFamily`

Routes are compiled, not extracted directly.

Minimum fields:

```text
route_family_id
topic_scope_id
cutoff_year
core_method_ids
supporting_statement_ids
supporting_citation_act_ids
competing_route_ids
bottleneck_cluster_ids
environment_snapshot_refs
confidence
```

Reference semantics:

1. `competing_route_ids` may reference `AlternativeRouteCandidate.alternative_route_candidate_id` values in early synthesis, or stable `RouteFamily.route_family_id` values after route consolidation
2. `environment_snapshot_refs` are `LiteratureVisibleEnvironmentSnapshot.snapshot_id` values

## 16. `BottleneckCluster`

Minimum fields:

```text
bottleneck_cluster_id
topic_scope_id
cutoff_year
bottleneck_type_ids
source_statement_ids
source_citation_act_ids
affected_route_family_ids
primary_severity
severity_evidence_count
confidence
```

Allowed `primary_severity` values:

1. `low`
2. `medium`
3. `high`
4. `blocking`

## 17. `QuestionCandidate`

This object must be semi-structured. The text is only one view.

Minimum fields:

```text
candidate_id
topic_scope_id
cutoff_year
origin_type
candidate_action_type
canonical_frame
surface_forms
source_statement_ids
supporting_citation_act_ids
opposing_citation_act_ids
related_route_family_ids
related_bottleneck_cluster_ids
confidence
dedup_signature
```

Allowed `origin_type` values:

1. `future_work_derived`
2. `limitation_derived`
3. `citation_gap_derived`
4. `route_competition_derived`
5. `conflict_resolution_derived`

Allowed `candidate_action_type` values:

1. `test_method`
2. `improve_measurement`
3. `build_resource`
4. `resolve_conflict`
5. `compare_routes`
6. `narrow_scope_probe`
7. `mechanism_check`

Minimum `canonical_frame` fields:

```text
research_object_id | null
method_or_intervention_id | null
target_variable_id | null
target_metric_id | null
comparator_id | null
condition_ids
target_bottleneck_ids
```

Minimum `surface_forms` fields:

```text
short_question
judgement_prompt_text
```

Rule:

`canonical_frame` is task-shaped, not universally dense. Fields may be null when the `candidate_action_type` does not require them, but every candidate must keep enough structured fields to support deduplication and replay auditing.

`dedup_signature` is a deterministic normalized hash over:

1. `topic_scope_id`
2. `candidate_action_type`
3. populated `canonical_frame` fields
4. normalized route or bottleneck references when present

## 18. `AlternativeRouteCandidate`

This object makes route competition explicit even when no final route family has yet stabilized.

Minimum fields:

```text
alternative_route_candidate_id
topic_scope_id
cutoff_year
candidate_route_label
supporting_method_ids
supporting_statement_ids
supporting_citation_act_ids
distinguishing_features
related_route_family_ids
status
retirement_reason
confidence
```

This object is produced by `AlternativeRouteCompiler` from route-competition signals, comparison statements, and citation transfer / critique patterns.

Allowed `status` values:

1. `emerging`
2. `candidate`
3. `consolidated`
4. `expired`

Lifecycle rules:

1. create as `emerging` when cross-paper evidence suggests a distinct route shape but support is still thin
2. promote to `candidate` when the route has:
   - at least 2 supporting papers or statements
   - at least 1 distinguishing method or comparison signal
   - at least 1 citation-based competition, transfer, or critique link
3. promote to `consolidated` when the route is stable enough to be materialized as a `RouteFamily`, meaning:
   - multi-paper support remains consistent
   - distinguishing features remain non-trivial after deduplication
   - the route can be separated from the main route and other alternatives without ambiguity
4. mark `expired` when:
   - the candidate collapses into an existing route family
   - the evidence is too weak after replay slicing
   - or the route is no longer meaningfully distinct

Handoff rule:

`AlternativeRouteCompiler` owns creation and early-state updates. `RouteFamilyCompiler` may only promote a candidate into a stable `RouteFamily` when the `consolidated` gate passes.

## 19. `ResearchDecisionUnit`

This is the unified replay task object that keeps judgement, route choice, bottleneck diagnosis, and feasibility probe tasks in one contract.

Minimum fields:

```text
decision_unit_id
unit_type
topic_scope_id
cutoff_year
primary_candidate_ids
alternative_candidate_ids
route_family_ids
bottleneck_cluster_ids
evidence_pack_id
label_policy_id
historical_outcome_trace_id
quality_status
```

Allowed `unit_type` values:

1. `question_judgement`
2. `route_choice`
3. `bottleneck_diagnosis`
4. `feasibility_probe`

Allowed `quality_status` values:

1. `draft`
2. `eligible`
3. `reviewed`
4. `rejected`

## 20. `EvidencePack`

This object makes sample composition auditable and enforceable.

Minimum fields:

```text
evidence_pack_id
topic_scope_id
cutoff_year
focal_statement_ids
support_statement_ids
challenge_statement_ids
citation_support_act_ids
citation_critique_act_ids
resource_usage_event_ids
evaluation_event_ids
problem_signal_event_ids
alternative_route_candidate_ids
bottleneck_cluster_ids
environment_snapshot_refs
excluded_post_cutoff_refs
pack_quality
```

The v1 evidence-pack quality gate has two levels.

Base gates for every eligible pack:

1. at least 2 papers
2. at least 1 citation support chain
3. at least 1 citation critique chain or limitation chain
4. at least 1 environment signal
5. enough evidence diversity to support a non-random judgement

Unit-type-specific gates:

1. `question_judgement`
   - at least 1 alternative-route or comparator signal when available
2. `route_choice`
   - at least 2 route-linked evidence slices and at least 1 explicit comparison signal
3. `bottleneck_diagnosis`
   - at least 1 bottleneck cluster and at least 1 supporting limitation or critique chain
4. `feasibility_probe`
   - at least 1 resource or protocol signal and at least 1 attack-path-relevant comparison or limitation signal

If these minimums are not met, the default label path should allow `underspecified`.

`pack_quality` must include:

```text
paper_count
support_statement_count
challenge_statement_count
citation_support_count
citation_critique_count
environment_signal_count
alternative_signal_count
status
quality_flags
```

Allowed `status` values:

1. `draft`
2. `eligible`
3. `insufficient`

Reference semantics:

1. `alternative_route_candidate_ids` are `AlternativeRouteCandidate.alternative_route_candidate_id` values
2. `bottleneck_cluster_ids` are `BottleneckCluster.bottleneck_cluster_id` values
3. `environment_snapshot_refs` are `LiteratureVisibleEnvironmentSnapshot.snapshot_id` values

## 21. `HistoricalOutcomeTrace`

This object stores how replay labels were derived from post-cutoff literature evolution.

Minimum fields:

```text
historical_outcome_trace_id
topic_scope_id
target_decision_refs
post_cutoff_support_refs
post_cutoff_conflict_refs
post_cutoff_route_shift_refs
outcome_summary
label_assignment_rationale
visibility
```

`visibility` must default to `eval_only`.

`target_decision_refs` may point to question candidates, route families, bottleneck clusters, or full decision units depending on the replay task type.

Allowed `visibility` values:

1. `eval_only`
2. `audit_only`

## 22. `LabelPolicy`

Main labels in v1 must be rule-driven, not LLM-driven.

Minimum fields:

```text
label_policy_id
unit_type
policy_version
required_inputs
decision_rules
fallback_rules
leakage_constraints
```

For `question_judgement`, the primary labels are:

1. `worth_doing_now`
2. `promising_but_premature`
3. `not_competitive`
4. `underspecified`

Rule intent:

1. `worth_doing_now`
   - later support shows real follow-through
   - no clearly superior alternative already dominates at the cutoff
   - key prerequisites are literature-visible at the cutoff
2. `promising_but_premature`
   - later work validates the general direction
   - but a key prerequisite is still missing at the cutoff
3. `not_competitive`
   - later history does not support the route strongly
   - or a stronger alternative route is already visible
4. `underspecified`
   - the evidence pack does not justify a stable decision

LLMs may help phrase rationales. They must not assign the primary gold label by themselves.

Minimum executable rule shape for `question_judgement`:

1. label assignment must combine:
   - post-cutoff follow-through evidence
   - alternative-route competitiveness evidence
   - prerequisite visibility evidence at the cutoff
   - critique versus support balance
2. bibliometric signals such as citation count may be used only as weak supporting signals, never as the sole gold-label rule
3. the first paper should publish the exact threshold table used for the frozen release

For `route_choice`, the primary labels are:

1. `prefer_primary`
2. `prefer_alternative`
3. `tie`
4. `underspecified`

For `bottleneck_diagnosis`, `LabelPolicy` should define:

1. the gold primary bottleneck cluster
2. optional secondary bottleneck clusters
3. the expected first-fix action type

For `feasibility_probe`, the primary labels are:

1. `feasible_now`
2. `feasible_if_narrowed`
3. `not_feasible_now`
4. `underspecified`

`decision_rules` must be executable against canonical refs and compiler outputs. Free-text-only policies are not sufficient.

## 23. Extraction Pipeline

### 23.1 Step 1: markdown to `DocumentIR`

Normalize the markdown paper into a stable document representation.

### 23.2 Step 2: `DocumentIR` to statement graph

For each paper, extract:

1. `Paper`
2. `EvidenceAnchor`
3. `Statement`
4. `SlotMentionEvent`
5. statement-to-statement relations

This stage is responsible for producing a stable single-paper `Statement graph`.

### 23.3 Step 3: citation mention extraction

Detect all in-text citation mentions and emit `CitationMentionEvent`.

### 23.4 Step 4: citation act classification

Aggregate mentions into `CitationActEvent`.

### 23.5 Step 5: lightweight entity linking

Normalize methods, metrics, resources, bottlenecks, and topic concepts using:

1. rule-based normalization
2. embedding similarity
3. LLM disambiguation only for uncertain cases

v1 linking policy:

1. auto-merge on exact normalized match plus identical entity type
2. auto-merge on known alias-table match
3. candidate merge only when embedding similarity is high and no explicit type or scope conflict exists
4. no-merge when resource subtype conflicts, metric semantics conflict, or route-family semantics conflict
5. uncertain cases remain split and are flagged for review rather than merged aggressively

### 23.6 Step 6: event builders

Emit:

1. `SlotMentionEvent`
2. `EvaluationEvent`
3. `ResourceUsageEvent`
4. `ProblemSignalEvent`

`SlotMentionEvent` is the raw mention-level bridge between `Statement` and normalized concept nodes.

Minimum fields:

```text
slot_mention_event_id
paper_id
statement_id
anchor_ids
slot_type
surface
normalized_candidate
confidence
```

### 23.7 Step 7: cross-paper synthesis

Run exactly these compilers in v1:

1. `TopicScopeCompiler`
2. `SupportConflictCompiler`
3. `RouteFamilyCompiler`
4. `AlternativeRouteCompiler`
5. `BottleneckCompiler`
6. `QuestionCandidateCompiler`
7. `EnvironmentSnapshotCompiler`

Compiler dependency order for v1:

```text
TopicScopeCompiler
    -> EnvironmentSnapshotCompiler
    -> SupportConflictCompiler
    -> AlternativeRouteCompiler
    -> RouteFamilyCompiler
    -> BottleneckCompiler
    -> QuestionCandidateCompiler
```

Failure downgrade rules:

1. if `TopicScopeCompiler` confidence is below the release threshold, stop and keep the sample `draft`
2. if `EnvironmentSnapshotCompiler` fails, downstream replay may continue only for samples that can still be labeled `underspecified`
3. if `AlternativeRouteCompiler` yields no valid alternatives, `route_choice` units are skipped but `question_judgement` may continue
4. if `RouteFamilyCompiler` cannot consolidate a route cleanly, retain `AlternativeRouteCandidate` objects and block route-choice exports for that slice
5. if `QuestionCandidateCompiler` cannot produce a stable deduplicated candidate, no `question_judgement` unit is exported for that slice

### 23.8 Step 8: historical replay labeling

For each topic scope and cutoff:

1. compile `ResearchDecisionUnit`
2. build `EvidencePack`
3. compute `HistoricalOutcomeTrace`
4. assign the gold label using `LabelPolicy`
5. generate rationale text after the label is fixed

### 23.9 Step 9: dataset view export

Export:

1. `judgement_sft.jsonl`
2. `preference_pairs.jsonl`
3. `reward_records.jsonl`

## 24. Training Views

### 24.1 `judgement_sft.jsonl`

Main task export.

Input:

1. `ResearchDecisionUnit`
2. `EvidencePack`
3. topic scope
4. cutoff year

Output:

1. primary label
2. rationale
3. key bottlenecks
4. next attack path

### 24.2 `preference_pairs.jsonl`

Input:

1. two decision units or two candidate options under the same topic and cutoff

Output:

1. `chosen`
2. `rejected`
3. `why_this_not_that`

### 24.3 `reward_records.jsonl`

Reserved for later reward modeling or RL.

Reward decomposition:

1. groundedness
2. feasibility
3. competitiveness
4. novelty proxy
5. uncertainty penalty

## 25. Dataset Quality Gates

A sample is high quality only if it passes all five dimensions below:

1. `Trace fidelity`
   - single-paper extraction is grounded and faithful to anchors
2. `Cross-paper grounding`
   - the sample uses citation or multi-paper evidence, not only one paper
3. `Historical integrity`
   - no post-cutoff leakage appears in the visible input
4. `Decision sufficiency`
   - the evidence pack is strong enough to support a non-random judgement
5. `Expert acceptability`
   - spot-check reviewers agree that the sample resembles real scientific judgement

If a sample fails any required gate, it must not enter the main training export.

Operational acceptance criteria for v1:

1. `Cross-paper grounding` requires at least one support and one challenge path, unless the unit is explicitly marked `underspecified`
2. `Decision sufficiency` requires all base evidence-pack gates to pass
3. `Expert acceptability` means at least two domain reviewers agree the sample is acceptable, or one reviewer approves and no reviewer rejects in the pilot phase
4. eligible non-underspecified judgement samples should typically include:
   - at least 10 evidence-carrying statements
   - at least 4 statement-function categories across problem, method, result, limitation, future-work, or interpretation
   - at least 3 distinct papers

## 26. Evaluation Boundary for the First Paper

The first paper should evaluate only judgement-centric tasks:

1. judgement classification
2. pairwise route or question preference
3. bottleneck diagnosis
4. feasibility / minimal attack path quality

The first paper should not rely on open-ended hypothesis generation as the main result.

The first implementation plan should target the minimum shippable slice:

1. `question_judgement`
2. `judgement_sft.jsonl`
3. replay labeling for the four question-judgement labels
4. enough citation-aware synthesis to support those samples well

`route_choice`, `bottleneck_diagnosis`, and `feasibility_probe` remain first-paper tasks, but they are second-wave implementation targets after the question-judgement path is stable.

Recommended first-paper evaluation bundle:

1. baselines
   - a strong general-purpose frontier model
   - a strong open model without domain fine-tuning
   - a retrieval-augmented baseline using the same paper set
   - the paper's trained judgement model
2. metrics
   - classification: accuracy and macro-F1
   - pairwise preference: win rate and rank correlation
   - bottleneck diagnosis: hit@k or mAP against expert references
   - groundedness: expert-scored evidence support
3. split policy
   - split by topic scope and replay slice
   - keep the test split frozen and excluded from compiler-rule tuning

## 27. First-Paper Positioning

The first paper should differentiate itself from generic scholarly KG or scientific QA work in three ways:

1. it treats citation as a first-order argumentative signal rather than a side relation
2. it builds labels through historical replay under strict cutoff constraints rather than retrospective free-form annotation
3. it targets scientific judgement tasks directly: worth-doing-now, route choice, bottleneck diagnosis, and feasibility
## 28. Relationship to Current Repository

### 28.1 Keep and upgrade

These parts are directionally useful and should be retained as foundations:

1. `backend/app/paper_logic_trace/models.py`
2. `backend/app/paper_logic_trace/compiler.py`
3. `backend/app/paper_logic_trace/derived_views.py` as a source of reusable extraction logic
4. `backend/app/citations/models.py`
5. `backend/app/llm/citation_purpose.py`

### 28.2 Downgrade to legacy experimental downstream objects

These current research-layer objects should no longer define canonical truth:

1. current route-first contracts in `backend/app/research_logic/models.py`
2. early direct exports that jump from paper traces to route / prior / episode style training objects

They may remain as later compiled exports or regression references.

### 28.3 New package boundary

Create a new primary package for the redesign:

```text
backend/app/research_kg/schema/
backend/app/research_kg/extract/
backend/app/research_kg/linking/
backend/app/research_kg/events/
backend/app/research_kg/synthesis/
backend/app/research_kg/replay/
backend/app/research_kg/datasets/
backend/app/research_kg/eval/
```

This package should own the new canonical contracts.

## 29. Risks and Controls

### 29.1 Risk: over-compression into route-only objects

Control:

Keep `QuestionCandidate`, `ResearchDecisionUnit`, and `EvidencePack` as explicit objects.

### 29.2 Risk: citation extracted but not consumed

Control:

Require citation support and critique chains in eligible evidence packs.

### 29.3 Risk: paper-only environment overclaim

Control:

Name the environment object `LiteratureVisibleEnvironmentSnapshot` and reserve a future `ExtendedEnvironmentSnapshot`.

### 29.4 Risk: label drift from weak replay rules

Control:

Bind every label to `LabelPolicy` and `HistoricalOutcomeTrace`.

### 29.5 Risk: beautiful schema, weak training signal

Control:

Require expert spot-checks and explicit sample-quality gates before main export.

## 30. Implementation Sequence

The recommended implementation order is:

1. freeze canonical schema
2. implement single-paper statement graph export
3. implement citation mention and citation act pipeline
4. implement lightweight entity linking
5. implement the seven cross-paper compilers
6. implement replay labeling
7. implement the three dataset views
8. iterate with the pilot granular-flow paper set

This order favors dataset quality over raw build speed.

## 31. Decision

The project should proceed as a paper-grounded historical scientific judgement data engine for granular flow, built on top of a canonical argumentative temporal research KG.

The first paper should be framed as:

1. a new judgement-centric task definition
2. a high-quality paper-grounded replay dataset
3. a citation-aware cross-paper knowledge compilation pipeline
4. a model-training result showing improved scientific judgement

That scope is ambitious enough for a strong paper while still narrow enough to execute rigorously.
