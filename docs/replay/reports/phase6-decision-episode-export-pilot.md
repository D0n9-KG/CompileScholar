# Phase 6 Decision Episode Export Pilot - 2026-04-02

## Source Bundles

This pilot consumed the existing Phase 5 bounded jamming artifacts and wrote a separate Phase 6 export bundle.

Source inputs:

- replay bundle: `tmp/phase5_multi_route_prior_induction/replay_bundle/`
- prior review bundle: `tmp/phase5_multi_route_prior_induction/review_bundle/`

Generated Phase 6 export bundle:

- `tmp/phase6_decision_episode_audit_export/`
- `tmp/phase6_decision_episode_audit_export/bundle_manifest.json`
- `tmp/phase6_decision_episode_audit_export/export_summary.json`
- `tmp/phase6_decision_episode_audit_export/export_inspection.json`
- `tmp/phase6_decision_episode_audit_export/outputs/decision_episode.json`

The export kept the original replay episode id:

- `episode:jamming_transition_in_frictionless_sphere_packings_near_point_j:2010:jamming_transition_in_frictionless_sphere_packings_near_point_j_2010_jamming_frictionless_packings_2010_packet_01_runtime_subset_v1`

The export also preserved explicit source bundle refs in both the Phase 6 manifest and inspection payload so later audit work can trace every replay and review artifact without mutating either Phase 5 source bundle.

## Accepted Card Carryover

The Phase 5 review bundle remained the acceptance truth for this export:

- accepted prior ids: `[]`
- accepted anti-pattern ids: `5`

The replay bundle still carried one replay-time selected prior:

- replay `selected_prior_ids`:
  - `prior:jamming_transition_in_frictionless_sphere_packings_near_point_j:2010`

The Phase 6 export did not copy that replay-time prior through. It rebuilt the episode from the review allowlists and truthfully produced:

- exported `selected_prior_ids`: `[]`
- exported `selected_antipattern_ids`: `5`

This means the pilot preserved the most important Phase 6 truth boundary:

- empty reviewed prior acceptance stayed empty in the export, even though the live replay episode had a selected prior

The export also preserved and selected the accepted anti-pattern inventory:

- exported `accepted_anti_pattern_ids` contains the same five accepted review ids
- exported `selected_antipattern_ids` now contains the same five reviewed anti-pattern ids

This now works through stable route-family matching rather than exact route-id equality alone. The support-route examples and the runtime-subset replay route share:

- `route_family:jamming_transition_in_frictionless_sphere_packings_near_point_j:2010:point_j`

So the export keeps the reviewed anti-pattern carryover auditable without requiring the support-route ids to equal the runtime-subset replay route id.

## Leakage Boundary

The export bundle made the visibility contract explicit:

- `visible_input_refs`: `24`
- `audit_only_refs`: `34`
- `label_eval_only_refs`: `0`

Leakage-sensitive state stayed outside the visible-input surface:

- `hindsight_outcome.input_visible = false`
- after-cutoff paper ids `1628`, `1654`, `1632`, and `770` appear under `audit_only_refs`, not visible inputs
- source replay and review bundle paths are recorded under `audit_only_refs`, not visible inputs

This bounded slice did not have any later-evidence hindsight refs, so `label_eval_only_refs` is empty. That is still the correct audit result. The bucket exists, remains separate from visible inputs, and would hold later-evidence refs when the source replay episode contains them.

## Audit Findings

The real pilot produced a dedicated audited export bundle without modifying the Phase 5 replay or review bundles.

The most important findings are:

- the export posture is explicit: `audit_grade_pilot`
- the exported episode kept `ready_for_eval = true` but dropped to `ready_for_training = false`
- export quality is `yellow` with `quality_flags = ["weak_prior_support"]`
- the export preserved reviewed acceptance truth instead of copying the replay-time prior
- the export carried all five reviewed anti-pattern ids into the episode through shared `route_family_id`, while still keeping the underlying support-route ids visible in the review bundle for audit traceability

## Pilot Limits

This output is an audit-grade pilot, not a generalized production dataset.

What this pilot shows:

- one bounded jamming replay/review slice can be rebuilt into a separate audited export bundle
- review allowlists can override replay-time `L4` selections without leaking hindsight
- the artifact boundary is inspectable through manifest, summary, and inspection surfaces

What this pilot does not show:

- broad training-dataset readiness
- cross-topic stability
- cross-topic stability of the new `route_family_id` carryover rule beyond the bounded jamming slice
- canonical committed source corpora beyond the current `tmp/` runtime artifacts

The right interpretation is that Phase 6 now has a reproducible, inspectable, leakage-safe export pilot for downstream eval or training research, including reviewed anti-pattern carryover across support/core/context and runtime-subset variants of the same route family, while still remaining conservative about maturity and about what the current jamming slice can claim.
