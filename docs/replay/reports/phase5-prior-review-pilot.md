# Phase 5 Prior Review Pilot - 2026-04-02

## Candidate Inventory

This pilot used the existing real package at `tmp/phase3_route_state_package/bundle/` plus the bounded jamming replay packet to test the first multi-route prior-induction workflow.

Generated Phase 5 artifacts:

- review bundle: `tmp/phase5_multi_route_prior_induction/review_bundle/`
- replay bundle: `tmp/phase5_multi_route_prior_induction/replay_bundle/`

Candidate counts from `candidate_review_summary.json`:

- support route states: `2`
- alternative route states: `1`
- held-out route states: `1`
- support clusters: `1`
- prior candidates: `1`
- anti-pattern candidates: `5`

The single prior candidate was induced from the two support route states in the jamming package and kept the package boundary explicit:

- support ids:
  - `jamming_transition_in_frictionless_sphere_packings_near_point_j_2010_jamming_support_core_packet_v1_v1`
  - `jamming_transition_in_frictionless_sphere_packings_near_point_j_2010_jamming_support_context_packet_v1_v1`
- held-out id:
  - `random_packing_definition_route_around_maximally_random_jammed_state_2010_jamming_heldout_random_packing_packet_v1_v1`

The accepted anti-patterns centered on repeated blocker signals across the support cluster:

- `anisotropic packing`
- `friction`
- `nonspherically symmetric potentials`
- `not sharp jammed surface`
- `relaxation time exceeding experimental time scales`

## Acceptance Gates

Phase 5 kept acceptance conservative on purpose.

Prior acceptance requires:

- explicit support route ids
- held-out consistency
- reviewer metadata
- a quality tier of `green`

In this pilot, the prior candidate recorded:

- `held_out_pass_rate = 1.0`
- `review_status = reviewed`
- `quality_tier = yellow`
- `quality_flags = ["weak_support_cluster"]`

The prior stayed below acceptance because the package-only induction cluster contained only two support route states. That is enough for a reviewable candidate, but not enough for an accepted reusable prior under the current gate.

Anti-pattern acceptance used the same typed review metadata plus counterexample coverage. All five anti-pattern candidates reached `green` because each one:

- repeated across both support routes
- kept explicit failure-example route ids
- recorded a counterexample route id from the frictional alternative
- carried reviewer metadata

## Held-out Findings

The package-level prior candidate passed the held-out check on the single held-out route:

- held-out route count: `1`
- pass rate: `1.0`

That result is useful but still not sufficient to promote the prior from candidate to accepted because the support cluster itself is still intentionally small. The held-out check now acts as an explicit audit gate rather than an implicit note.

The live replay run remained healthy at the same time:

- replay quality tier: `green`
- ready for pilot: `true`
- package validation quality tier: `green`

The important distinction is:

- package-only prior induction stays conservative and keeps the support-only cluster yellow
- replay can still produce a green live prior when it combines the current primary route with the two support routes

That is the right Phase 5 behavior. The review bundle is a candidate-audit surface, not a shortcut to promote sparse package evidence into a dataset-quality prior.

## Review Decisions

Review outcomes for this pilot:

- accepted prior ids: `[]`
- accepted anti-pattern ids:
  - `anti:jamming_transition_in_frictionless_sphere_packings_near_point_j:01:anisotropic_packing`
  - `anti:jamming_transition_in_frictionless_sphere_packings_near_point_j:03:friction`
  - `anti:jamming_transition_in_frictionless_sphere_packings_near_point_j:07:nonspherically_symmetric_potentials`
  - `anti:jamming_transition_in_frictionless_sphere_packings_near_point_j:08:not_sharp_jammed_surface`
  - `anti:jamming_transition_in_frictionless_sphere_packings_near_point_j:10:relaxation_time_exceeding_experimental_time_scales`

Interpretation:

- Phase 5 now has a real auditable candidate-review workflow for priors and anti-patterns.
- The workflow stays JSON / Markdown / CLI based and does not require a new frontend surface.
- The project can now preserve accepted card ids separately from candidate generation.

What this pilot does not claim:

- it does not claim a training-ready exported prior dataset
- it does not claim cross-topic induction stability
- it does not replace Phase 6

Phase 6 still owns audited export, packaging policy, and any claim that accepted cards are ready to become long-lived training or evaluation assets.
