# Phase 6: Decision Episode Audit Export - Context

**Gathered:** 2026-04-02 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 6 turns the current replay-time `DecisionEpisode` object into a leakage-safe, auditable export asset that assembles `L1/L2/L3` context with reviewed `L4` cards and preserves enough manifest/inspection context for later training and eval research. It does not claim broad dataset maturity, add a new product surface, or relax the project's anti-hindsight boundary.

</domain>

<decisions>
## Implementation Decisions

### Export Boundary And Artifact Shape
- **D-01:** Phase 6 should introduce a dedicated file-based decision-episode export bundle, separate from the existing replay bundle and prior-review bundle. The export should reuse the repo's existing manifest/summary/inspection pattern instead of inventing a new storage surface.
- **D-02:** The export bundle should carry both the final `DecisionEpisode` sample and the audit context required to replay or inspect it later: source packet refs, route-state refs, selected review-approved card ids, and an inspection summary that makes leakage and readiness visible.
- **D-03:** The first Phase 6 deliverable is a small inspected pilot batch built from the current jamming runtime-subset artifacts, not a generalized multi-topic export system.

### L4 Card Selection Policy
- **D-04:** Audited exports should consume reviewed accepted `L4` cards from the prior-review registry/manifest, not raw candidate lists or the live replay-time prior candidate by default.
- **D-05:** When accepted prior ids are empty, the export should preserve that fact rather than silently promoting a yellow candidate into the audited sample.
- **D-06:** Accepted anti-pattern ids should be eligible to populate `DecisionEpisode.relevant_priors.selected_antipattern_ids` when their failure-example route ids or review-approved registry entries match the exported route.

### Leakage And Visibility Discipline
- **D-07:** Hindsight remains label/eval-only. Exported samples must preserve `hindsight_outcome.input_visible = false` and keep any later evidence outside the visible-input side of the artifact.
- **D-08:** The export path should make visibility boundaries explicit at the artifact level, not just inside nested fields: the sample should make clear which refs are observable at cutoff versus audit/eval-only attachments.
- **D-09:** The phase should keep the current audit-first posture: a green replay bundle does not automatically mean the exported sample is safe to describe as training-ready without the extra Phase 6 export checks.

### Implementation Path And Reuse
- **D-10:** Phase 6 should extend `backend/app/research_logic/` and adjacent scripts/JSON artifacts rather than adding a frontend review UI or a separate database-backed export pipeline.
- **D-11:** Existing bundle writers/loaders in `replay_io.py`, the accepted-card helpers in `prior_induction.py`, and the typed contracts in `models.py` are the default reuse points for export assembly.
- **D-12:** The export plan should preserve the distinction between live replay reasoning and audit-grade packaged outputs: runtime replay may still build a green prior from the support cluster, while audited exports must faithfully reflect reviewed acceptance state.

### the agent's Discretion
- Exact export bundle filenames and whether the export summary/inspection live under a new top-level bundle or a dedicated subdirectory under replay outputs
- Whether accepted anti-pattern selection is driven strictly by route-id matching, by registry manifest ids, or by both with audit notes
- How much derived convenience data belongs in the export summary versus the full inspection artifact

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project And Phase Framing
- `.planning/PROJECT.md` - Project mission, layer boundaries, anti-hindsight discipline, and current Phase 6 positioning
- `.planning/REQUIREMENTS.md` - `L4-02` requirement and the v1 boundary for training/eval-ready decision samples
- `.planning/ROADMAP.md` - Phase 6 goal, success criteria, and plan slots `06-01` / `06-02`
- `.planning/STATE.md` - Current milestone handoff and unresolved runtime/canonicalization caveats

### Prior Phase Context That Constrains Phase 6
- `.planning/phases/03-grounded-route-replay-compilation/03-CONTEXT.md` - Packaged route-state/replay contract that Phase 6 must build on
- `.planning/phases/04-replay-failure-taxonomy-and-l2-surgical-loop/04-CONTEXT.md` - Audit-first failure-record/reporting boundary reused by later export work
- `.planning/phases/05-multi-route-prior-induction/05-CONTEXT.md` - Locked Phase 5 decisions about candidate vs accepted cards and the audit-first handoff to Phase 6
- `.planning/phases/05-multi-route-prior-induction/05-VERIFICATION.md` - Verified Phase 5 behavior and explicit note that Phase 6 owns audited export policy

### Product And Spec References
- `docs/superpowers/specs/2026-04-01-logickg-decision-prior-and-episode-schema.md` - Canonical `DecisionPriorCard`, `AntiPatternCard`, and `DecisionEpisode` contracts, especially hindsight and quality rules
- `docs/replay/reports/phase5-prior-review-pilot.md` - Human-readable interpretation of the current Phase 5 review results and what they do not yet claim
- `docs/replay/reports/route-state-package-pilot-2026-04-02.md` - Current replay/package pilot baseline that Phase 6 export should not contradict

### Current Pilot Artifacts To Preserve
- `tmp/phase5_multi_route_prior_induction/review_bundle/bundle_manifest.json` - Accepted prior/anti-pattern ids recorded by the review bundle
- `tmp/phase5_multi_route_prior_induction/review_bundle/candidate_review_summary.json` - Current acceptance counts, support sizes, and quality flags
- `tmp/phase5_multi_route_prior_induction/review_bundle/anti_pattern_candidates.json` - Reviewed accepted anti-pattern cards available for audited export selection
- `tmp/phase5_multi_route_prior_induction/replay_bundle/replay_summary.json` - Current live replay quality/readiness snapshot
- `tmp/phase5_multi_route_prior_induction/replay_bundle/replay_inspection.json` - Failure-record surface and stage-quality audit details
- `tmp/phase5_multi_route_prior_induction/replay_bundle/outputs/decision_episode.json` - Current replay-time episode artifact that Phase 6 needs to harden into an audit export

### Implementation Refs
- `backend/app/research_logic/models.py` - Typed contracts and validation rules for packets, priors, anti-patterns, and decision episodes
- `backend/app/research_logic/decision_episode_builder.py` - Current episode assembly logic, quality flags, and hindsight guardrails
- `backend/app/research_logic/historical_replay_compiler.py` - How replay currently combines route-state synthesis, prior selection, and episode generation
- `backend/app/research_logic/replay_io.py` - Existing bundle writer/loader patterns for replay and review artifacts
- `backend/app/research_logic/prior_induction.py` - Review registry structure and accepted-card helpers that should drive audited export selection
- `backend/tests/test_decision_episode_builder.py` - Episode behavior and leakage/quality expectations
- `backend/tests/test_historical_replay_compiler.py` - Current injection path for accepted prior cards and replay bundle quality behavior
- `backend/tests/test_replay_io.py` - Existing summary/inspection/manifest bundle conventions

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/app/research_logic/replay_io.py`: already writes file-based replay bundles and prior-review bundles with manifest, summary, and inspection surfaces
- `backend/app/research_logic/prior_induction.py`: exposes `PriorCandidateRegistry`, plus `accepted_prior_cards()` and `accepted_anti_pattern_cards()` helpers that fit Phase 6 selection needs
- `backend/app/research_logic/decision_episode_builder.py`: already enforces `hindsight_outcome.input_visible = false`, produces audit quality flags, and can emit episodes with empty or partial prior selection
- `backend/app/research_logic/historical_replay_compiler.py`: already distinguishes between live generated prior cards and injected accepted cards, which is the key seam for Phase 6 export policy
- `tmp/phase5_multi_route_prior_induction/review_bundle/` and `tmp/phase5_multi_route_prior_induction/replay_bundle/`: current real pilot artifacts to reuse as the first export batch source

### Established Patterns
- Research-logic contracts are typed Pydantic models with strict validation and explicit green/yellow/red quality tiers
- Auditable work products are JSON-first, file-based bundles with manifests rather than database-only state
- Review acceptance is separated from candidate generation: accepted ids are recorded independently from raw candidate lists
- The project consistently treats runtime green outputs as pilot signals, not blanket permission to claim dataset maturity

### Integration Points
- Phase 6 export logic should live in `backend/app/research_logic/` alongside `replay_io.py` and the current builders
- The export assembler should load reviewed cards from the Phase 5-style review bundle/registry and pair them with a replay-produced route/episode context
- Any new CLI or script should follow the existing artifact-driven pattern used by replay/report scripts rather than creating a new frontend surface
- Regression coverage should extend the existing `decision_episode_builder`, `historical_replay_compiler`, and `replay_io` test suites with export-specific assertions

</code_context>

<specifics>
## Specific Ideas

- The audited export should preserve the difference between "live replay thinks this route is good" and "reviewed cards are safe to package as reusable training/eval context"
- The current jamming pilot is the right first export slice because it already has accepted anti-pattern ids, a replay-time `DecisionEpisode`, and bundle manifests on both sides
- Phase 6 should be comfortable exporting a sample with empty accepted prior ids if that is the truthful audit state
- Accepted anti-patterns are currently the strongest reviewed `L4` asset in the workspace and should inform the first export design instead of being dropped on the floor

</specifics>

<deferred>
## Deferred Ideas

- New frontend review/export UI for priors, anti-patterns, or episodes
- Cross-topic export batching and generalized dataset production claims
- Promoting the current runtime-subset packet assets into permanent canonical committed corpora
- Broader LLM-assisted export phrasing or training prompt design beyond the typed audit artifact boundary

</deferred>

---

*Phase: 06-decision-episode-audit-export*
*Context gathered: 2026-04-02*
