# LogicKG PaperLogicTrace L2 Hard-Cut Redesign

## 1. Background

The current LogicKG mainline is already strong at single-paper structured extraction, but its core schema is still shaped around `LogicStep`, `Claim`, and Neo4j-facing internal objects. That is good enough for graph browsing and retrieval, but it is not yet a stable second-layer asset for the four-layer scientific reasoning system described in [科学家思维AI项目技术文档.md](C:/Users/D0n9/Desktop/LogicKG/docs/科学家思维AI项目技术文档.md).

The main problem is no longer "extract more claims." The main problem is that the current extraction result is still an internal representation, not a canonical middle layer that can be:

1. versioned
2. recompiled into higher-layer objects
3. time-sliced
4. audited against evidence
5. reused without rerunning the most expensive paper extraction path

At the same time, the current extraction chain has accumulated too many stages around the old schema. If the schema is redesigned but the extraction chain is left structurally unchanged, the project will simply end up with a better schema on top of an increasingly expensive pipeline.

This design therefore hard-cuts both:

1. the old second-layer abstraction
2. the old extraction-chain shape

and replaces them with a new canonical export centered on `PaperLogicTrace` plus a rebuilt, throughput-conscious extraction compiler.

## 2. Goals

1. Redefine L2 around a stable canonical export: `PaperLogicTrace`.
2. Remove `LogicStep / Claim` as canonical second-layer schema.
3. Preserve strong evidence traceability from every important L2 field back to paper evidence.
4. Rebuild the L2 extraction chain together with the schema, instead of adapting the old pipeline in place.
5. Keep the extraction hot path fast enough for large-scale paper processing.
6. Make L2 naturally consumable by:
   1. future L1 environment linking
   2. community construction
   3. L3 route-state reconstruction
   4. L4 decision-prior induction

## 3. Non-Goals

1. This spec does not define the full L1 schema.
2. This spec does not implement L3 or L4 objects directly.
3. This spec does not preserve backward compatibility for `LogicStep / Claim`.
4. This spec does not finalize the new community algorithm; it only defines what L2 exports to community builders.
5. This spec does not require Neo4j to remain the canonical storage format for L2.
6. This spec does not preserve the existing extraction orchestration shape just because it already exists.

## 4. Layer Alignment

The official scientific layers remain:

1. `L1`: Historical Research Environment Layer
2. `L2`: PaperLogicTrace
3. `L3`: Route-State Reconstruction Layer
4. `L4`: Decision Prior Layer

To support these layers, the implementation may keep a non-layer engineering substrate for evidence and source localization. This substrate is not part of the scientific four-layer model. It exists only to support provenance, locating evidence, and ingestion plumbing.

This spec covers only `L2`, while leaving explicit interface slots for `L1`, `L3`, and `L4`.

## 5. Approaches Considered

### Approach A: Continue patching `LogicStep / Claim`

Keep the current schema and keep adding missing L2.5 fields around it.

Pros:

1. Lowest migration cost.
2. Minimal short-term breakage.

Cons:

1. Keeps the wrong canonical abstraction.
2. Preserves the existing mismatch between paper logic units and sentence-like claims.
3. Encourages more patch layers instead of a stable export contract.

### Approach B: Replace L2 with a raw `ResearchMove` store only

Make `ResearchMove` the only top-level L2 export object.

Pros:

1. Much cleaner than `LogicStep / Claim`.
2. Closer to the real unit of scientific reasoning.

Cons:

1. Too implementation-shaped for the scientific architecture.
2. Does not give downstream systems a paper-level stable export boundary.

### Approach C: Make `PaperLogicTrace` the canonical export, with `ResearchMove` as the internal core unit

Pros:

1. Matches the scientific architecture in the research design document.
2. Gives downstream builders a stable paper-level contract.
3. Still allows a much better internal unit than `LogicStep / Claim`.
4. Keeps derived views separable from canonical truths.

Cons:

1. Requires a real schema migration.
2. Breaks old APIs and storage assumptions.

### Recommendation

Choose Approach C.

## 6. L2 Design Principles

### 6.1 Canonical before convenient

The canonical L2 export must describe stable paper logic, not mirror whatever the current database schema happens to store.

### 6.2 Summary is never the only truth

Summaries help humans read the trace, but machine truth must live in structured fields with provenance.

### 6.3 Every important field must trace back to evidence

If a value enters canonical L2, it must be possible to identify where it came from in the paper.

### 6.4 Hot path stays narrow

Expensive enrichment and deep audits must not block the first usable `PaperLogicTrace`.

### 6.5 Derived views may evolve faster than canonical core

Community inputs, L1 bridge hints, and route feature candidates should be derived from canonical L2, not mixed into the truth layer itself.

### 6.6 Extraction redesign is in scope

Schema redesign and extraction-chain redesign are coupled in this project. The extraction flow should be simplified around `PaperLogicTrace`, not forced to preserve the stage boundaries created by the old `LogicStep / Claim` pipeline.

## 7. Canonical L2 Export

`PaperLogicTrace` becomes the single canonical output of second-layer extraction.

```text
PaperLogicTrace
- trace_id
- schema_version
- built_at
- paper_metadata
- canonical_core
- derived_views
- quality
```

### 7.1 `paper_metadata`

```text
PaperMetadata
- paper_id: str
- canonical_doi: str | null
- title: str
- year: int | null
- authors: str[]
- venue: str | null
- paper_type: empirical | theoretical | review | software | benchmark | case_study | unknown
- source_refs: str[]
```

### 7.2 `canonical_core`

This is the stable truth layer of L2.

```text
canonical_core
- evidence_anchors: EvidenceAnchor[]
- moves: ResearchMove[]
- move_relations: MoveRelation[]
- citation_acts: CitationAct[]
- figure_refs: FigureRef[]
- table_refs: TableRef[]
```

The most important units are:

1. `EvidenceAnchor`
2. `ResearchMove`
3. `MoveRelation`

Everything else in L2 either supports these units or is derived from them.

### 7.3 `derived_views`

This is the compiler-friendly helper layer.

```text
derived_views
- l2_5_slot_inventory
- l1_bridge_hints
- community_signatures
- route_feature_candidates
- paper_summaries
```

`derived_views` is not canonical truth. It is allowed to evolve faster than `canonical_core`, as long as every derived item points back to canonical objects.

### 7.4 `quality`

```text
quality
- quality_tier
- hot_path_gate_report
- audit_status
- audit_findings[]
```

`quality` records whether the trace is:

1. usable for storage
2. usable for downstream compilers
3. fully audited or still only hot-path validated

## 8. Core Internal Objects

### 8.1 `EvidenceAnchor`

`EvidenceAnchor` is the stable evidence reference object used by the canonical core.

```text
EvidenceAnchor
- anchor_id: str
- paper_id: str
- source_ref: str
- modality: text | figure | table | citation
- section_path: str[]
- locator: {
    chunk_id?: str,
    start_line?: int,
    end_line?: int,
    start_char?: int,
    end_char?: int,
    page?: int
  }
- quote: str
- citation_ids: str[]
- support_type: direct | contextual | indirect
- weak: bool
```

### 8.2 `MentionValue`

All major slot values should use one shared structure rather than raw strings.

```text
MentionValue
- surface: str
- normalized: str | null
- type: str | null
- anchor_ids: str[]
- confidence: float
- inferred: bool
```

### 8.3 `EffectValue`

Results and comparisons should use a dedicated structure rather than free-form text.

```text
EffectValue
- direction: increase | decrease | improve | worsen | mixed | none | unknown
- magnitude_text: str | null
- magnitude_numeric: float | null
- unit: str | null
- comparator_surface: str | null
- anchor_ids: str[]
- confidence: float
```

### 8.4 `SlotProvenance`

Canonical slot values require explicit provenance.

```text
SlotProvenance
- field: str
- value_index: int | null
- anchor_ids: str[]
- extraction_mode: direct | normalized | inferred
- support_strength: exact | strong | weak
- confidence: float
- notes: str | null
```

### 8.5 `ResearchMove`

`ResearchMove` is the core scientific unit inside L2.

```text
ResearchMove
- move_id: str
- sequence_no: int
- role: problem | background | hypothesis | method | experiment | result | interpretation | limitation | future_work
- act_type: identify_gap | define_task | formulate_hypothesis | propose_method | adapt_method | build_resource | set_condition | run_experiment | measure_outcome | compare_baseline | report_effect | explain_mechanism | diagnose_failure | state_limitation | suggest_extension
- summary: str
- research_objects: MentionValue[]
- methods: MentionValue[]
- observed_variables: MentionValue[]
- metrics: MentionValue[]
- comparators: MentionValue[]
- conditions: MentionValue[]
- effects: EffectValue[]
- limitation_types: MentionValue[]
- resource_mentions: MentionValue[]
- anchor_ids: str[]
- slot_provenance: SlotProvenance[]
- confidence: float
- audit_state: hot_path | audited
```

`ResearchMove` replaces the old `LogicStep / Claim` pairing as the canonical paper-logic unit.

### 8.6 `MoveRelation`

```text
MoveRelation
- relation_id: str
- source_move_id: str
- target_move_id: str
- relation_type: motivates | addresses | implements | evaluates | yields | explains | limits | extends
- anchor_ids: str[]
- confidence: float
```

### 8.7 `CitationAct`

The current citation enrichment remains useful, but it should be exported as a structured side object, not entangled with graph-specific storage.

```text
CitationAct
- citation_act_id
- source_move_id: str | null
- target_paper_id: str | null
- purpose
- polarity
- semantic_signal
- target_scope
- anchor_ids[]
- confidence
```

### 8.8 `FigureRef` and `TableRef`

Figures and tables remain paper-level evidence helpers.

```text
FigureRef
- figure_id
- caption
- anchor_ids[]
```

```text
TableRef
- table_id
- caption
- anchor_ids[]
```

## 9. Field-Level Rules

### 9.1 Required hot-path fields

Every canonical `ResearchMove` must have:

1. `move_id`
2. `sequence_no`
3. `role`
4. `act_type`
5. `summary`
6. `anchor_ids`
7. `confidence`

If a candidate move lacks `role`, `act_type`, or at least one anchor, it does not enter canonical L2.

### 9.2 Hot-path preferred fields

The hot path should try to populate:

1. `research_objects`
2. `methods`
3. `metrics`
4. `comparators`
5. `conditions`
6. `limitation_types`
7. `resource_mentions`

### 9.3 Audit/enrichment fields

The following are allowed to be improved or completed later:

1. `observed_variables`
2. `effects`
3. deeper normalization of values
4. extra provenance refinement
5. citation-act enrichment
6. advanced figure/table interpretation

### 9.4 Single-value vs multi-value

Single-value:

1. `move_id`
2. `sequence_no`
3. `role`
4. `act_type`
5. `summary`
6. `confidence`
7. `audit_state`

Multi-value:

1. `research_objects`
2. `methods`
3. `observed_variables`
4. `metrics`
5. `comparators`
6. `conditions`
7. `effects`
8. `limitation_types`
9. `resource_mentions`
10. `anchor_ids`
11. `slot_provenance`

## 10. End-to-End L2 Extraction Flow

The new L2 flow is split into a synchronous hot path and a deferred audit/enrichment path.

This is not merely a scheduling tweak. It is a structural rewrite of the second-layer extraction compiler.

### 10.1 Hot path

1. `Evidence substrate preparation`
   1. paper metadata
   2. section structure
   3. source references
   4. figure/table refs
   5. citation events
2. `Document framing`
   1. coarse paper type
   2. section-function framing
   3. high-value extraction zones
3. `Semantic windowing`
   1. section-driven windows
   2. figure/table-adjacent windows
   3. discussion/result windows
4. `Window-level ResearchMove extraction`
5. `Cross-window dedup and merge`
6. `MoveRelation stitching`
7. `Minimal hot-path gate`
8. `Canonical PaperLogicTrace export`

The hot path is expected to replace the old "logic first, claim second, then packaging and accumulated extras" shape with a much tighter `PaperLogicTrace` compiler path.

### 10.2 Audit/enrichment path

Run later, outside the critical first-result path:

1. semantic audit refinement
2. multi-evidence aggregation
3. deeper citation-act enrichment
4. effect normalization
5. figure/table deeper interpretation
6. derived-view compilation refresh

## 11. Recommended Extraction Strategy

### 11.1 Rejected strategies

Rejected:

1. whole-paper one-shot extraction
2. sentence-level extraction followed by heavy merge

The first is too lossy and too hard to ground. The second is too fragmented and reproduces the current claim-level problems.

### 11.2 Recommended strategy

Use:

1. `Document framing`
2. `Semantic windowing`
3. `Window-level move extraction`
4. `Cross-window dedup + merge`
5. `Relation stitching`
6. `Canonical export`

This gives a middle ground between stability, evidence locality, and throughput.

### 11.3 Pipeline simplification principle

When an old pipeline step exists only to support the previous schema, it should be removed rather than adapted.

Examples:

1. old intermediate objects that only exist to materialize `LogicStep / Claim`
2. packaging stages that mirror old database structures instead of canonical L2
3. synchronous enrichments that do not affect first valid `PaperLogicTrace`

## 12. Hot-Path vs Audit Split

### 12.1 Hot-path responsibilities

The hot path is responsible for making a paper usable as an L2 asset.

It must produce:

1. a valid `PaperLogicTrace`
2. canonical moves
3. move relations
4. evidence anchors
5. minimum slot inventory
6. quality tier and hot-path gate report

The hot path is not responsible for preserving old extraction stage boundaries if those boundaries hurt throughput.

### 12.2 Audit responsibilities

Audit exists to improve confidence and downstream eligibility, not to define first availability.

Audit may:

1. increase completeness
2. refine slot normalization
3. strengthen provenance
4. attach more citation semantics
5. decide whether the trace is fit for stricter training subsets

### 12.3 Performance constraint

No audit-only work may be required for the first usable `PaperLogicTrace`.

### 12.4 Budget targets

The redesign should keep explicit hot-path budgets so that richer structure does not silently destroy throughput.

Recommended initial budgets:

1. per-paper hot-path target: `P50 <= 3 minutes`
2. per-paper hot-path target: `P95 <= 6 minutes`
3. no extra audit-only LLM round may be required before first export
4. the default extraction path should avoid unbounded sentence-level fanout
5. slot refinement that materially increases latency belongs in audit unless it is required for canonical validity
6. schema migration must not be implemented as "old pipeline plus new export appended at the end"

## 13. Derived Views

### 13.1 `l2_5_slot_inventory`

Paper-level inventory of the L2.5 slot values present in the trace.

Purpose:

1. auditing
2. filtering
3. quick statistics
4. downstream feature compilers

### 13.2 `l1_bridge_hints`

This is the only L1-facing bridge defined in this spec.

It does not define L1 itself. It only exports hints that later L1 compilers may consume.

Suggested contents:

1. `resource_candidates`
2. `benchmark_candidates`
3. `metric_candidates`
4. `protocol_candidates`
5. `toolchain_candidates`

All hints must point back to canonical provenance.

### 13.3 `community_signatures`

Community building must not read long raw summaries directly. It should consume normalized per-move signatures derived from canonical L2.

```text
CommunitySignature
- move_id
- paper_id
- role
- act_type
- method_tokens[]
- object_tokens[]
- metric_tokens[]
- condition_tokens[]
- comparator_tokens[]
- effect_direction
- limitation_tokens[]
- resource_tokens[]
```

### 13.4 `route_feature_candidates`

L3 builders should consume precompiled candidates rather than reparsing paper text.

Suggested outputs:

1. `dominant_method_candidates`
2. `benchmark_mentions`
3. `known_bottleneck_candidates`
4. `enabling_condition_candidates`
5. `comparison_signals`

### 13.5 `paper_summaries`

Human-readable summaries may exist here, but they are not canonical truth.

## 14. Interfaces to Other Layers

### 14.1 Interface to L1

This spec does not define L1, but L2 must expose enough stable bridge points for future L1 integration:

1. `resource_mentions`
2. `metrics`
3. `conditions`
4. `comparators`
5. protocol/toolchain hints in `l1_bridge_hints`

### 14.2 Interface to community builders

Community builders should read only:

1. `derived_views.community_signatures`
2. optionally selected canonical provenance for explanation

They should not directly consume raw paper text as the canonical clustering input.

### 14.3 Interface to L3

L3 should read:

1. `canonical_core.moves`
2. `canonical_core.move_relations`
3. `derived_views.route_feature_candidates`
4. future L1 environment packs

L3 must not reconstruct route state by reparsing the full paper corpus from scratch.

### 14.4 Interface to L4

L4 should consume L3 objects as its primary input, while tracing back to L2 evidence for:

1. effects
2. limitations
3. comparisons
4. conditions
5. supporting paper logic evidence

## 15. Failure Handling and Degraded Output

The new L2 pipeline must degrade cleanly instead of turning every extraction shortcoming into a total paper failure.

### 15.1 Hard failure conditions

A paper fails canonical export only when one of the following is true:

1. the source cannot be parsed into usable evidence references
2. no valid `ResearchMove` survives with `role`, `act_type`, `summary`, and anchors
3. provenance is so broken that canonical fields cannot be traced back to source evidence

### 15.2 Degraded but exportable conditions

A paper may still export a lower-confidence `PaperLogicTrace` when:

1. some preferred L2.5 slots are sparse
2. citation enrichment is incomplete
3. figures or tables are only partially resolved
4. audit has not yet run

In this case, the trace remains exportable but must carry:

1. downgraded `quality_tier`
2. explicit `audit_status`
3. missing-slot visibility in `l2_5_slot_inventory`

### 15.3 Downstream gating

Downstream builders must be able to choose stricter subsets without redefining L2 validity.

Examples:

1. community builders may accept hot-path-valid traces
2. L3 builders may require stronger slot coverage
3. high-trust training subsets may require audited traces only

## 16. Breaking Changes

This redesign is a hard cut.

1. `LogicStep / Claim` stop being canonical L2 objects.
2. Old second-layer persistence and API assumptions may be removed rather than adapted.
3. Existing paper-level outputs should be replaced with `PaperLogicTrace`.
4. Existing community inputs must be rebuilt against `community_signatures`.

Short-term migration compatibility is not required by this design.

## 17. Engineering Impact

The following areas will require rewrite or major refactoring:

1. extraction orchestration
2. canonical export pipeline
3. quality gate logic
4. community input generation
5. paper-level read APIs
6. frontend views that currently assume `LogicStep / Claim`

In practice, this is an extraction-pipeline refactor as much as it is a schema refactor. Old staging, batching assumptions, and packaging boundaries should be treated as replaceable rather than preserved by default.

The following areas remain useful and should be reused where possible:

1. evidence extraction and quote grounding primitives
2. citation event recovery
3. figure extraction
4. paper identity resolution
5. ingest hot-path optimization discipline

## 18. Risks and Controls

### 17.1 Over-fragmented moves

Risk:

`ResearchMove` degenerates into sentence-level fragments.

Control:

Use semantic windowing and cross-window merge as first-class stages.

### 17.2 Under-filled slots

Risk:

The new schema looks richer on paper but remains sparse in practice.

Control:

Prioritize a narrow hot-path field set and measure slot coverage explicitly.

### 17.3 Provenance drift

Risk:

Derived values lose their link to evidence.

Control:

Require all major canonical slots to carry `slot_provenance`.

### 17.4 Throughput collapse

Risk:

A richer schema makes single-paper extraction too slow.

Control:

Keep hot-path requirements narrow and push refinement into audit.

### 17.5 Layer leakage

Risk:

L2 starts silently reimplementing L1 or L3 concerns.

Control:

Keep only bridge hints to L1 and feature candidates to L3. Do not let L2 own higher-layer objects.

## 19. Testing and Acceptance

### 18.1 L2 schema acceptance

A valid `PaperLogicTrace` must satisfy:

1. every `ResearchMove` has `role`, `act_type`, `summary`, and anchors
2. every non-empty major slot has provenance
3. every derived-view item points back to canonical objects

### 18.2 Extraction acceptance

For an acceptance packet of papers:

1. move coverage is stable across paper types
2. slot coverage improves over the current L2.5 baseline
3. evidence links remain auditable

### 18.3 Throughput acceptance

The redesigned hot path must not regress into a workflow that is too slow for future large-scale processing. Audit-only improvements must remain deferrable.

### 18.4 Downstream acceptance

The new L2 export is accepted when it is sufficient to:

1. build community signatures without `LogicStep / Claim`
2. compile route feature candidates without reparsing raw papers
3. expose L1 bridge hints without redefining L1 inside L2

## 20. Recommendation

Proceed with a hard-cut L2 redesign centered on `PaperLogicTrace`.

The implementation should:

1. replace `LogicStep / Claim` with `ResearchMove`-based canonical exports
2. rebuild the extraction chain as a throughput-conscious `PaperLogicTrace` compiler
3. split canonical truths from derived compiler views
4. keep the hot path narrow
5. expose explicit interfaces to future L1, community, L3, and L4 builders

This gives LogicKG a real second-layer identity: not a graph workbench with paper extraction inside it, but a stable paper-logic compiler that can feed the rest of the scientific reasoning system.
