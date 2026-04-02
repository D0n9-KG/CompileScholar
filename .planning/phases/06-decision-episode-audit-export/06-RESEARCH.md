# Phase 6: Decision Episode Audit Export - Research

**Researched:** 2026-04-02
**Domain:** leakage-safe audited `DecisionEpisode` export from replay bundles plus reviewed `L4` cards
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Phase 6 must create a dedicated audited export bundle, separate from the existing replay bundle and prior-review bundle.
- The export must carry both the final `DecisionEpisode` sample and enough audit context to reconstruct where it came from later.
- The first deliverable is a small inspected jamming pilot, not a generalized multi-topic export system.
- Audited exports must consume reviewed accepted `L4` cards from the review bundle manifest/registry rather than raw candidates or the live replay-time prior by default.
- If `accepted_prior_ids` is empty, the export must preserve that truth instead of silently promoting a yellow replay prior.
- Accepted anti-pattern ids are eligible to populate `DecisionEpisode.relevant_priors.selected_antipattern_ids` when they match the exported route.
- Hindsight remains label/eval-only. Exported samples must keep `hindsight_outcome.input_visible = false`.
- Visibility boundaries must be explicit at the artifact level rather than hidden only inside nested fields.
- Phase 6 stays inside `backend/app/research_logic/`, adjacent scripts, and file-based JSON bundles; no frontend review UI or database-backed export pipeline.
- The phase must preserve the distinction between live replay reasoning and audit-grade packaged outputs.

### the agent's Discretion
- Exact export bundle filenames beyond the required manifest / summary / inspection surfaces.
- Whether anti-pattern carryover is driven by route-state matching, accepted id filtering, or both with audit notes.
- How much convenience data belongs in `export_summary.json` versus `export_inspection.json`.

### Deferred Ideas (OUT OF SCOPE)
- Building a frontend export/review UI.
- Declaring broad dataset maturity or cross-topic generalization.
- Promoting the current runtime jamming assets into canonical committed corpora.
- Prompt-format or training-format work beyond the typed audit artifact.
</user_constraints>

<research_summary>
## Summary

Phase 6 should add an export assembly layer above the existing replay and review bundles, not introduce a new decision compiler. The codebase already has the critical seam:

1. `HistoricalReplayCompiler.compile(...)` can accept injected `prior_cards` and `anti_pattern_cards`;
2. `DecisionEpisodeBuilder` already keeps `hindsight_outcome.input_visible = false`, can emit empty `selected_prior_ids`, and can populate `selected_antipattern_ids` from route-matching anti-pattern cards;
3. `replay_io.py` already defines the repo's standard `bundle_manifest.json` plus summary/inspection bundle pattern;
4. Phase 5 already emits a review bundle manifest with accepted ids and a replay bundle with a live `decision_episode.json`.

The most important Phase 6 finding is the current mismatch between those two artifact families:

- the Phase 5 review bundle truthfully records `accepted_prior_ids = []`;
- the same review bundle records five accepted anti-pattern ids;
- the current replay `outputs/decision_episode.json` still contains a replay-time `selected_prior_ids` entry and no anti-pattern ids.

That means audited export cannot simply copy the replay-time `decision_episode.json` as-is. It must rebuild the episode from replay context plus reviewed acceptance state. Otherwise the exported sample would overstate `L4` support and understate the strongest reviewed negative signal now available.

The safest design is:

1. load the replay bundle context that is still historically visible at cutoff;
2. load reviewed candidate files plus accepted ids from the Phase 5 review bundle;
3. rebuild the exported `DecisionEpisode` through `DecisionEpisodeBuilder` using only accepted cards;
4. write a new dedicated export bundle with explicit visibility buckets:
   - `visible_input_refs`
   - `audit_only_refs`
   - `label_eval_only_refs`
5. preserve source bundle refs and acceptance-policy explanations in inspection payloads so the pilot remains auditable.

This architecture lets the export stay truthful even when accepted priors are empty. It also lets accepted anti-patterns become the main audited `L4` carryover in the first jamming export, which is exactly what the real Phase 5 artifacts currently support.

**Primary recommendation:** create a dedicated `decision_episode_export.py` assembly module plus bundle-writer support in `replay_io.py`, then expose it through a standalone export CLI such as `backend/scripts/export_decision_episode_pilot.py`.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Asset | Status | Purpose | Why Standard Here |
|-------|--------|---------|-------------------|
| `backend/app/research_logic/decision_episode_builder.py` | Existing | Rebuild the exported episode from reviewed accepted cards | Already enforces anti-hindsight rules and prior/anti-pattern selection behavior |
| `backend/app/research_logic/historical_replay_compiler.py` | Existing | Defines the current replay/runtime seam between live priors and injected reviewed cards | Shows where Phase 6 must diverge from runtime replay behavior without rewriting the compiler |
| `backend/app/research_logic/replay_io.py` | Existing | Bundle writer / loader conventions for replay and review artifacts | Best place to add export bundle manifest, summary, and inspection payloads |
| `backend/app/research_logic/prior_induction.py` | Existing | Accepted-card registry semantics and helper functions | Current source of truth for reviewed card selection behavior |
| `backend/app/research_logic/models.py` | Existing | Typed `DecisionEpisode`, `DecisionPriorCard`, `AntiPatternCard`, and hindsight contracts | Prevents Phase 6 from hand-rolling a second schema for export |

### Supporting
| Asset | Status | Purpose | When to Use |
|-------|--------|---------|-------------|
| `tmp/phase5_multi_route_prior_induction/review_bundle/bundle_manifest.json` | Existing | Records accepted prior and anti-pattern ids | Use as the acceptance gate for audited export |
| `tmp/phase5_multi_route_prior_induction/review_bundle/anti_pattern_candidates.json` | Existing | Reviewed anti-pattern candidate objects | Use to reconstruct accepted anti-pattern cards by id |
| `tmp/phase5_multi_route_prior_induction/replay_bundle/outputs/decision_episode.json` | Existing | Current replay-time episode artifact | Use as the comparison baseline, not as the final audited truth |
| `tmp/phase5_multi_route_prior_induction/replay_bundle/replay_summary.json` | Existing | Runtime replay quality state | Use to explain why replay green is not enough for audited export |
| `backend/tests/test_decision_episode_builder.py` | Existing | Downstream selection and leakage regression boundary | Use when accepted-card carryover changes selected ids or quality flags |
| `backend/tests/test_historical_replay_compiler.py` | Existing | Replay compatibility boundary | Use to ensure Phase 6 does not accidentally mutate runtime replay semantics |
| `backend/tests/test_replay_io.py` | Existing | Bundle contract regression boundary | Use when adding export bundle writers/loaders |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| a dedicated export assembly layer | mutating the existing replay bundle in place | Faster short-term, but destroys the distinction between runtime replay and audited export |
| review manifest as acceptance truth | inferring acceptance from `quality_tier == green` alone | Simpler, but can drift from human-reviewed decisions recorded in Phase 5 |
| explicit visibility buckets | relying only on nested `hindsight_outcome.input_visible` | Too implicit for later audit and dataset inspection |
| standalone export CLI | overloading `run_replay_pilot.py` further | Reuses less cleanly and mixes replay generation with audited packaging concerns |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Export Assembly Above Replay And Review
**What:** Treat Phase 6 as a post-replay assembly layer that consumes one replay bundle plus one review bundle and emits a new audited export bundle.
**When to use:** When runtime replay is already useful, but reviewed acceptance state must control what is safe to export.
**Why recommended:** It preserves backward compatibility and keeps export policy isolated from the live compiler path.

### Pattern 2: Review Manifest As The Acceptance Gate
**What:** Use `accepted_prior_ids` and `accepted_anti_pattern_ids` from the review manifest as the authoritative allowlist for export.
**When to use:** Whenever a candidate file contains more cards than the audited export is allowed to carry forward.
**Why recommended:** Human review outcomes remain the truth source instead of silently re-deriving acceptance from card fields.

### Pattern 3: Visibility-Tiered Reference Surfaces
**What:** Split export refs into `visible_input_refs`, `audit_only_refs`, and `label_eval_only_refs`.
**When to use:** Whenever the export needs to remain trainable/evaluable without leaking after-cutoff or hindsight evidence into visible inputs.
**Why recommended:** The existing `input_visible=false` rule becomes inspectable at the artifact level rather than only inside one nested field.

### Pattern 4: Truthful Empty Prior Carryover
**What:** Allow the export to carry `selected_prior_ids = []` when no reviewed prior was accepted, while still carrying accepted anti-patterns if they match the route.
**When to use:** In early bounded pilots where negative knowledge is stronger than positive reusable priors.
**Why recommended:** It matches the actual Phase 5 review result instead of pretending the replay-time green prior was already audited.

### Anti-Patterns To Avoid
- Copying the replay `decision_episode.json` unchanged into the export bundle.
- Treating a green replay-time prior as audited just because it matches the route.
- Dropping accepted anti-patterns because the export path only thinks in terms of priors.
- Mixing hindsight refs into visible evidence packs or summary convenience fields.
- Calling the first bounded jamming export a generalized production-ready dataset.
</architecture_patterns>

<validation_architecture>
## Validation Architecture

Phase 6 should validate at four levels:

1. **export-assembly contract tests**
   - prove exported `selected_prior_ids` come only from reviewed accepted priors,
   - prove accepted anti-pattern ids can populate the exported episode,
   - prove hindsight stays label/eval-only.
2. **bundle I/O contract tests**
   - prove the new export bundle writes a manifest, summary, inspection payload, and output episode using the same audit-friendly conventions as replay and review bundles,
   - prove source bundle refs and accepted ids are preserved in machine-readable form.
3. **CLI smoke tests**
   - prove a dedicated export script can consume the real replay/review bundle directory layout,
   - prevent Phase 6 from depending on ad hoc notebook or one-off local wiring.
4. **bounded runtime artifact audit**
   - inspect the first export bundle built from the current jamming Phase 5 artifacts,
   - confirm that exported prior ids remain empty, accepted anti-patterns are carried through when route-matching, and visibility buckets remain truthful.

The phase should not claim cross-topic validity. The runtime audit only needs to prove that the export policy is auditable and leakage-safe on the one bounded pilot slice already used across Phases 3-5.
</validation_architecture>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Export schema | a second free-form JSON shape unrelated to replay bundles | `DecisionEpisode` plus manifest / summary / inspection bundle pattern | Preserves downstream reuse and auditability |
| Acceptance policy | ad hoc filters based only on card fields | review bundle manifest accepted-id lists | Human review already established the allowlist |
| Hindsight boundary | manual string comments about leakage | explicit `label_eval_only_refs` plus `input_visible=false` checks | Makes leakage inspectable and testable |
| Pilot export workflow | notebooks or manual file copying | a dedicated backend CLI script | Keeps the workflow reproducible inside the repo |
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Replay truth and review truth drift apart
**What goes wrong:** The export inherits the replay-time selected prior even though the review manifest accepted no prior ids.
**Why it happens:** The replay bundle already contains a plausible `decision_episode.json`, so it is tempting to reuse it directly.
**How to avoid:** Rebuild the episode from accepted cards instead of patching the existing JSON.
**Warning signs:** `selected_prior_ids` in the export do not match `accepted_prior_ids` from the review bundle.

### Pitfall 2: Accepted anti-patterns disappear from the export
**What goes wrong:** The exported episode carries no negative `L4` signal even though Phase 5 accepted several anti-pattern cards.
**Why it happens:** Export assembly treats priors as the only reusable `L4` surface.
**How to avoid:** Select anti-patterns by accepted id and route-state membership in `failure_examples.route_state_ids`.
**Warning signs:** The review bundle has accepted anti-pattern ids, but the exported `selected_antipattern_ids` is always empty.

### Pitfall 3: Leakage sneaks in through convenience metadata
**What goes wrong:** Later evidence refs or after-cutoff artifacts appear in visible summary fields even though `input_visible=false` is preserved on the nested hindsight object.
**Why it happens:** Summary and inspection payloads are often treated as "just metadata."
**How to avoid:** Give every ref bucket an explicit visibility label and test them separately.
**Warning signs:** `later_evidence_refs` or after-cutoff ids appear under visible-input keys.

### Pitfall 4: Replay readiness gets mistaken for export readiness
**What goes wrong:** Because the replay episode says `ready_for_training=true`, the Phase 6 export is assumed to be automatically training-ready.
**Why it happens:** Replay quality already looks green on the bounded slice.
**How to avoid:** Let export summary state its own audit-grade policy and carry acceptance-state explanations.
**Warning signs:** The export summary does not explain why empty prior ids are still acceptable in an audit-grade pilot.

### Pitfall 5: The pilot report overclaims maturity
**What goes wrong:** The report implies the repo now produces generalized decision-episode datasets.
**Why it happens:** The new export bundle feels like a final milestone.
**How to avoid:** Keep the report language anchored to the bounded jamming pilot and explicit Phase 6 limits.
**Warning signs:** The report stops mentioning "audit-grade pilot" and starts describing production dataset readiness.
</common_pitfalls>

<open_questions>
## Open Questions

1. **Should `export_inspection.json` duplicate replay summary fields or mostly reference the source replay bundle?**
   - What we know: replay already stores quality counts and inspection detail.
   - What's unclear: how much duplication is worth it for later export-only consumers.
   - Recommendation: keep only export-specific derived counts and explicit source bundle refs; do not duplicate the full replay inspection payload.

2. **Should accepted anti-pattern carryover require both accepted-id membership and route-state matching?**
   - What we know: accepted ids alone are too broad, while route matching alone can ignore review decisions.
   - What's unclear: whether future exports need broader mapping logic.
   - Recommendation: require both in the first Phase 6 pilot.

3. **Should the export bundle copy raw replay inputs or only reference them?**
   - What we know: full copying increases audit convenience but duplicates already large replay artifacts.
   - What's unclear: whether downstream consumers need standalone portability immediately.
   - Recommendation: keep the export bundle lightweight for Phase 6 by writing explicit source refs in the manifest and inspection payloads, while storing the rebuilt episode and export-specific audit metadata locally.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- `.planning/phases/06-decision-episode-audit-export/06-CONTEXT.md`
- `.planning/PROJECT.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`
- `.planning/phases/05-multi-route-prior-induction/05-CONTEXT.md`
- `.planning/phases/05-multi-route-prior-induction/05-VERIFICATION.md`
- `docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md`
- `docs/replay/reports/phase5-prior-review-pilot.md`
- `docs/replay/reports/route-state-package-pilot-2026-04-02.md`
- `backend/app/research_logic/models.py`
- `backend/app/research_logic/decision_episode_builder.py`
- `backend/app/research_logic/historical_replay_compiler.py`
- `backend/app/research_logic/replay_io.py`
- `backend/app/research_logic/prior_induction.py`
- `tmp/phase5_multi_route_prior_induction/review_bundle/bundle_manifest.json`
- `tmp/phase5_multi_route_prior_induction/review_bundle/candidate_review_summary.json`
- `tmp/phase5_multi_route_prior_induction/replay_bundle/replay_summary.json`
- `tmp/phase5_multi_route_prior_induction/replay_bundle/outputs/decision_episode.json`

### Secondary (MEDIUM confidence)
- `backend/tests/test_decision_episode_builder.py`
- `backend/tests/test_historical_replay_compiler.py`
- `backend/tests/test_replay_io.py`
- `backend/tests/test_route_state_pilot_cli.py`
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: audited `DecisionEpisode` export assembly from replay and review bundles
- Ecosystem: `research_logic`, Phase 5 review bundles, replay bundle JSON artifacts, pytest regression tests
- Patterns: export assembly layer, accepted-id allowlists, explicit visibility buckets, bounded pilot reporting
- Pitfalls: replay/review drift, anti-pattern loss, hindsight leakage, maturity overclaim

**Confidence breakdown:**
- Standard stack: HIGH
- Architecture: HIGH
- Export policy: HIGH
- Cross-topic generalization: LOW

**Research date:** 2026-04-02
**Valid until:** 2026-05-02
</metadata>

---
