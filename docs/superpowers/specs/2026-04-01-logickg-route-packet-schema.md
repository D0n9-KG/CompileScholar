# LogicKG Route Packet Schema

## 1. Purpose

`route packet` is the minimum historical compilation unit for `L3`.

It exists to replace the wrong scaling target:

1. "ingest as many papers as possible"

with the correct target:

1. "compile a high-quality topic x year state from a bounded and auditable packet"

Without a packet contract, the system cannot stably guarantee:

1. topic boundary consistency
2. cutoff-year discipline
3. role coverage
4. hindsight leakage control
5. reproducible route-state compilation

## 2. Position in the Pipeline

```text
paper corpus
    +
L1 environment assets
    +
topic_scope candidate
    +
cutoff_year
    v
RoutePacket
    v
RouteStateSynthesizer
    v
RouteState
```

## 3. Non-Goals

1. `RoutePacket` is not a training sample by itself.
2. `RoutePacket` is not a route summary.
3. `RoutePacket` is not a retrieval cache.
4. `RoutePacket` does not decide the final route ranking.

## 4. Design Principles

### 4.1 Historical discipline first

If a paper, resource, benchmark, or tool became visible only after the cutoff, it must be excluded from the packet.

### 4.2 Multi-role composition

A packet must not be made only of core method papers.

### 4.3 Auditable inclusion

Every included paper should have a reason for inclusion.

### 4.4 Small but sufficient

A packet should be large enough to support `RouteState`, but small enough to audit manually.

## 5. Canonical Object

```text
RoutePacket
- packet_id
- schema_version
- built_at
- topic_scope_candidate
- cutoff_year
- packet_status
- packet_composition
- l1_snapshot_ref
- inclusion_rules
- included_items
- excluded_items
- packet_quality
- compiler_hints
```

## 6. Core Fields

### 6.1 Identity

```text
packet_id: str
schema_version: str
built_at: str
topic_scope_candidate: str
cutoff_year: int
packet_status: draft | reviewed | frozen
```

Rules:

1. `packet_id` should be deterministic from `topic_scope_candidate + cutoff_year + packet_version`.
2. `cutoff_year` is mandatory.
3. `packet_status=frozen` means it is eligible for training compilation.

### 6.2 `packet_composition`

```text
packet_composition
- target_size: int
- actual_size: int
- role_counts: {
    core_method: int,
    resource_or_benchmark: int,
    limitation_or_critique: int,
    survey_or_review: int,
    alternative_route: int
  }
- coverage_ok: bool
- missing_roles: str[]
```

Recommended early-stage size:

1. 20 to 30 papers per packet

Recommended minimum role coverage:

1. `core_method >= 6`
2. `resource_or_benchmark >= 2`
3. `limitation_or_critique >= 2`
4. `survey_or_review >= 1`
5. `alternative_route >= 2`

These thresholds should be configurable by domain.

### 6.3 `l1_snapshot_ref`

```text
l1_snapshot_ref
- snapshot_id: str
- resource_registry_ref: str | null
- resource_timeline_ref: str | null
- benchmark_timeline_ref: str | null
- toolchain_timeline_ref: str | null
- protocol_registry_ref: str | null
```

This field makes the packet the bridge between `L2` papers and `L1` environment.

## 7. Included and Excluded Items

### 7.1 `included_items`

```text
IncludedPacketItem
- paper_id: str
- trace_id: str | null
- paper_year: int | null
- item_role: core_method | resource_or_benchmark | limitation_or_critique | survey_or_review | alternative_route
- inclusion_reason: str
- source_selector: manual | rule | retrieval | mixed
- evidence_for_inclusion: str[]
- title: str | null
```

Rules:

1. Each item must have exactly one primary `item_role`.
2. `inclusion_reason` must explain why the item belongs in the packet.
3. If `trace_id` is missing, the packet is not yet compile-ready.

### 7.2 `excluded_items`

```text
ExcludedPacketItem
- paper_id: str
- paper_year: int | null
- exclusion_reason: after_cutoff | off_topic | duplicate_signal | low_quality_trace | unresolved_metadata | missing_source | other
- notes: str | null
```

This field is important because packet boundaries must be inspectable.

## 8. Inclusion Rules

```text
inclusion_rules
- scope_definition: str
- scope_aliases: str[]
- accepted_year_range: {
    min_year: int | null,
    max_year: int
  }
- hard_exclusion_rules: str[]
- role_assignment_rules: str[]
- leakage_policy: str
```

`leakage_policy` should always explicitly mention:

1. papers after cutoff are excluded
2. later surveys can only be used for audit, never for training input
3. hindsight labels cannot be used in packet compilation

## 9. Packet Quality

```text
packet_quality
- quality_tier: red | yellow | green
- ready_for_route_state: bool
- quality_flags: str[]
- topic_boundary_confidence: float | null
- leakage_risk: low | medium | high | unknown
- manual_review_status: not_started | partial | completed
```

### 9.1 Green packet conditions

A packet should usually be `green` only if:

1. role coverage is complete
2. trace coverage is complete
3. cutoff discipline is checked
4. topic boundary is reviewed
5. at least one `L1` snapshot reference is attached

### 9.2 Common quality flags

Suggested flags:

1. `missing_core_method_coverage`
2. `missing_alternative_routes`
3. `missing_limitation_papers`
4. `no_l1_snapshot`
5. `trace_missing`
6. `after_cutoff_candidate_detected`
7. `topic_boundary_blurry`
8. `packet_too_small`

## 10. Compiler Hints

`RoutePacket` should include optional hints for the downstream compiler.

```text
compiler_hints
- preferred_scope_label: str | null
- preferred_method_labels: str[]
- preferred_benchmark_labels: str[]
- preferred_bottleneck_labels: str[]
- expected_alternative_routes: str[]
- notes_for_route_state_compiler: str | null
```

These hints are not truth. They are constrained bootstrap signals.

## 11. Selection Workflow

Recommended packet-building workflow:

1. choose `topic_scope_candidate`
2. choose `cutoff_year`
3. retrieve candidate papers
4. assign packet roles
5. remove after-cutoff and off-topic items
6. attach `PaperLogicTrace`
7. attach `L1` snapshot refs
8. run packet quality audit
9. freeze packet

## 12. Minimal JSON Example

```json
{
  "packet_id": "image_recognition_2011_packet_01",
  "schema_version": "v1",
  "built_at": "2026-04-01T18:20:00Z",
  "topic_scope_candidate": "large-scale image recognition with deep neural networks",
  "cutoff_year": 2011,
  "packet_status": "reviewed",
  "packet_composition": {
    "target_size": 24,
    "actual_size": 22,
    "role_counts": {
      "core_method": 9,
      "resource_or_benchmark": 3,
      "limitation_or_critique": 3,
      "survey_or_review": 2,
      "alternative_route": 5
    },
    "coverage_ok": true,
    "missing_roles": []
  },
  "l1_snapshot_ref": {
    "snapshot_id": "imagenet_2011_snapshot",
    "resource_registry_ref": "resource_registry_v1",
    "resource_timeline_ref": "resource_timeline_v1",
    "benchmark_timeline_ref": "benchmark_timeline_v1",
    "toolchain_timeline_ref": "toolchain_timeline_v1",
    "protocol_registry_ref": "protocol_registry_v1"
  },
  "inclusion_rules": {
    "scope_definition": "supervised large-scale image recognition using deep neural networks",
    "scope_aliases": ["cnn image classification", "deep image recognition"],
    "accepted_year_range": {
      "min_year": 2006,
      "max_year": 2011
    },
    "hard_exclusion_rules": [
      "exclude papers published after 2011",
      "exclude generic object recognition papers with no large-scale benchmark relevance"
    ],
    "role_assignment_rules": [
      "benchmark papers mention shared large-scale evaluation data",
      "alternative routes are non-deep or non-cnn competitive approaches"
    ],
    "leakage_policy": "No papers after cutoff may enter the packet. Later retrospectives are audit-only."
  },
  "included_items": [
    {
      "paper_id": "p1",
      "trace_id": "p1:trace",
      "paper_year": 2011,
      "item_role": "core_method",
      "inclusion_reason": "Defines the main cnn route under the cutoff.",
      "source_selector": "mixed",
      "evidence_for_inclusion": ["topic_match", "method_signal"],
      "title": "Example Core CNN Paper"
    }
  ],
  "excluded_items": [
    {
      "paper_id": "p99",
      "paper_year": 2012,
      "exclusion_reason": "after_cutoff",
      "notes": "Relevant but not historically visible at the cutoff."
    }
  ],
  "packet_quality": {
    "quality_tier": "green",
    "ready_for_route_state": true,
    "quality_flags": [],
    "topic_boundary_confidence": 0.86,
    "leakage_risk": "low",
    "manual_review_status": "completed"
  },
  "compiler_hints": {
    "preferred_scope_label": "large-scale image recognition with deep neural networks",
    "preferred_method_labels": ["cnn"],
    "preferred_benchmark_labels": ["ImageNet"],
    "preferred_bottleneck_labels": ["gpu throughput", "training instability"],
    "expected_alternative_routes": ["svm pipelines", "hand-crafted vision pipelines"],
    "notes_for_route_state_compiler": "Use benchmark availability and gpu constraints as major readiness factors."
  }
}
```

## 13. Decision

The packet is the real unit of progress for the project.

From this point on, the project should prefer:

1. packet count
2. packet quality
3. packet replayability

over:

1. raw paper count
2. extraction volume
3. graph size
