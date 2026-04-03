# Phase 10: Multi-Paper L3 And L4 Validation - Context

**Gathered:** 2026-04-03 (assumptions mode)
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 10 takes the committed Phase 9 bounded corpus packet and companion role mapping, compiles route-state-package, replay, prior-review, and audited export artifacts against that fixed boundary, and makes the real multi-paper `L3/L4` failure surface inspectable. It does not redefine the packet topic, add new packet members just to hide blocker flags, or collapse package/replay/review/export into one opaque success claim.

</domain>

<decisions>
## Implementation Decisions

### Orchestration And Operator Surface
- **D-01:** Phase 10 should add one dedicated backend workflow for the committed Phase 9 packet, but it should compose the existing file-based CLIs and bundle writers (`run_route_state_package.py`, `run_replay_pilot.py`, `export_decision_episode_pilot.py`) rather than introduce a UI flow, database object, or a new artifact family.
- **D-02:** The workflow should keep the repo's current audit shape: machine-readable bundles under `tmp/` plus a committed human-readable report under `docs/replay/reports/` that summarizes package validation, replay inspection, prior/review outcomes, export posture, and the main blocker queue.

### Canonical Input Boundary
- **D-03:** The committed Phase 9 packet (`phase9-route-packet.json`) and companion assembly manifest (`phase9-assembly-manifest.json`) are the canonical multi-paper boundary for Phase 10. Planning should assume the topic scope, cutoff year, and `support / alternative / held_out` membership are frozen unless Phase 10 explicitly reports a blocker for a later cycle.
- **D-04:** Phase 10 should derive whatever runtime package inputs the current package compiler needs from the committed Phase 9 packet plus assembly manifest, not by redesigning the `RoutePacket` schema or overwriting the committed Phase 9 artifacts.
- **D-05:** The Phase 9 placeholder `l1_snapshot_ref` must be replaced by a real historical environment snapshot that matches the packet cutoff before any replay-quality claim is treated as meaningful.

### Validation And Blocker Semantics
- **D-06:** Success in Phase 10 means producing inspectable `L3` replay/package-validation artifacts and `L4` prior/review/export artifacts, or explicit blockers at the correct stage. A yellow or red outcome is acceptable if the blocking layer, flags, and evidence are recorded clearly enough for Phase 11 prioritization.
- **D-07:** The thin support cluster, single-paper alternative depth, single-paper held-out depth, and fallback-heavy source mix carried forward from Phase 9 should remain first-class blocker hypotheses in Phase 10 outputs rather than being normalized away or silently fixed by expanding scope.
- **D-08:** Structural package validation, replay compilation, prior review, and audited export should remain separate artifact boundaries. The phase should not collapse them into one combined pass/fail artifact that hides whether the problem was package structure, replay quality, prior support, or export visibility policy.

### Review And Export Truth Boundary
- **D-09:** When Phase 10 reaches `L4`, replay-time priors and anti-patterns are provisional. The audited export must rebuild from the prior-review bundle's accepted ids, preserving the existing review-allowlist truth boundary instead of copying raw replay selections straight into the exported episode.
- **D-10:** Export outputs must preserve the existing visibility split (`visible_input_refs`, `audit_only_refs`, `label_eval_only_refs`) and the route-family-aware anti-pattern carryover logic already verified in Phase 6, so Phase 10 remains audit-grade even if multi-paper support stays weak.

### Comparison And Closeout Framing
- **D-11:** Phase 10 should compare the new corpus-backed package, replay, and export results against the existing jamming packaged baseline to separate generic compiler regressions from packet-specific weaknesses in the computational-mechanics slice.
- **D-12:** The closeout artifact should translate the run into a short blocker queue organized by stage/layer (`package validation`, `route-state replay`, `prior induction`, `export`) so Phase 11 can decide whether the next cycle belongs in `L2`, packet construction, or `L4` aggregation.

### the agent's Discretion
- Exact naming for the Phase 10 runner, output directory, and report filenames as long as they follow the repo's `tmp/...` plus `docs/replay/reports/...` audit pattern
- Whether runtime subset packets, a generated route-state-package manifest, or both are used internally to bridge the Phase 9 assembly manifest into the current package compiler
- Exact comparison table format between the Phase 10 computational-mechanics run and the earlier jamming baseline

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project And Milestone Framing
- `.planning/PROJECT.md` - Current `v1.1` strategy, bounded multi-paper philosophy, and active Phase 10 focus
- `.planning/REQUIREMENTS.md` - `PACK-02` and `AGGR-01`, including the rule that explicit blockers are valid Phase 10 outputs
- `.planning/ROADMAP.md` - Phase 10 goal, success criteria, and the requirement to compare against the jamming baseline
- `.planning/STATE.md` - Current handoff state and carry-forward notes from Phase 9 into Phase 10

### Phase 9 Handoff Inputs
- `.planning/phases/09-bounded-packet-construction-from-corpus/09-CONTEXT.md` - Frozen packet boundary, schema-preservation rule, and Phase 10 handoff expectations
- `.planning/phases/09-bounded-packet-construction-from-corpus/09-VERIFICATION.md` - Fresh verification that the committed packet and assembly manifest still audit cleanly
- `docs/replay/pilot_packets/phase9-route-packet.json` - Canonical bounded packet input for Phase 10
- `docs/replay/pilot_packets/phase9-assembly-manifest.json` - Canonical `support / alternative / held_out` mapping, trace refs, and known gaps
- `docs/replay/pilot_packets/phase9-selection-notes.md` - Topic boundary rationale, exclusions, cutoff discipline, and carry-forward caveats
- `docs/replay/reports/phase9-bounded-packet-audit.md` - Structural Phase 10 handoff status plus the blocker notes that Phase 10 must keep explicit

### Existing Package And Replay Pipeline
- `docs/replay/reports/route-state-package-pilot-2026-04-02.md` - Proven packaged replay baseline and the artifact-quality target for comparison
- `backend/app/research_logic/route_state_package.py` - Current package manifest, compilation, validation, and bundle format
- `backend/app/research_logic/historical_replay_compiler.py` - Replay compiler that consumes grouped `support / alternative / held_out` route states
- `backend/app/research_logic/replay_io.py` - Replay summary, inspection, bundle-writing, and trace-coverage contract
- `backend/scripts/run_route_state_package.py` - Operator-facing package compiler CLI
- `backend/scripts/run_replay_pilot.py` - Replay CLI that already accepts a route-state package and can emit a prior review bundle
- `backend/tests/test_route_state_package.py` - Regression expectations for package validation, replay consumption, and review-bundle emission

### L4 Review And Export Boundary
- `docs/replay/reports/phase6-decision-episode-export-pilot.md` - Existing audit-grade export baseline and review-truth boundary
- `backend/app/research_logic/prior_induction.py` - Prior and anti-pattern candidate registry flow from package groups
- `backend/app/research_logic/decision_episode_export.py` - Audited export builder, allowlist truth, and visibility-bucket logic
- `backend/scripts/export_decision_episode_pilot.py` - Replay-bundle plus review-bundle to export-bundle CLI
- `backend/tests/test_decision_episode_export.py` - Export truth, route-family anti-pattern carryover, and visibility-boundary expectations
- `backend/tests/test_decision_episode_export_cli.py` - CLI behavior and source-bundle immutability checks

### Core Contracts
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md` - Compiler-layer intent and what `L3/L4` are supposed to prove
- `docs/superpowers/specs/2026-04-01-logickg-route-packet-schema.md` - Canonical packet contract, cutoff discipline, and leakage boundaries

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `backend/app/research_logic/bounded_packet_audit.py`: existing structural handoff audit and ready-for-Phase-10 summary surface from Phase 9
- `backend/app/research_logic/route_state_package.py`: compiles, validates, and writes grouped route-state package bundles from packet-plus-trace inputs
- `backend/scripts/run_route_state_package.py`: operator CLI for package compilation and bundle emission
- `backend/scripts/run_replay_pilot.py`: already chains route-state package inputs into replay and can emit a prior-review bundle in the same run
- `backend/app/research_logic/prior_induction.py`: builds prior and anti-pattern candidate registries directly from grouped package roles
- `backend/app/research_logic/decision_episode_export.py`: converts replay plus review outputs into a separate audited export bundle
- `backend/scripts/export_decision_episode_pilot.py`: operator CLI for replay-plus-review to export orchestration
- `backend/app/research_logic/replay_io.py`: standardized summary, inspection, manifest, and review-bundle writers

### Established Patterns
- The repo prefers file-based, auditable workflow bundles under `tmp/` plus committed narrative reports under `docs/replay/reports/`
- Preflight contract checks fail fast on missing trace ids, cutoff mismatches, and invalid bundle references
- Structural validation and semantic quality remain explicit through quality tiers and quality flags instead of silent fallback behavior
- Review allowlists override replay-time `L4` selections during export, and visibility buckets keep leakage boundaries explicit

### Integration Points
- The Phase 9 packet and assembly manifest, together with the reused Phase 8 trace refs they already name, are the direct input bridge into Phase 10 package compilation
- A real `HistoricalEnvironmentSnapshot` must be attached before Phase 10 can treat replay output as more than a structural dry run
- The route-state package bundle feeds both replay compilation and prior/anti-pattern review generation
- The export flow consumes replay and review bundles without mutating either source bundle, which is the correct audit boundary for Phase 10 as well

</code_context>

<specifics>
## Specific Ideas

- The first Phase 10 run should stay on `phase9_comp_mech_2021_packet_01` and treat the committed `support / alternative / held_out` mapping as fixed input rather than reopening packet curation
- Because there is no dedicated Phase 10 runner or test yet, the most natural implementation is a new orchestration layer that composes the existing package, replay, and export tooling instead of new compiler primitives
- The first run should explicitly test whether the placeholder `L1` ref, thin support cluster, one-paper alternative depth, and one-paper held-out depth degrade package validation, replay quality, prior support, or export readiness
- Comparison output should reuse the fields the code already emits: package quality flags, replay failure counts by stage and layer, prior held-out pass rate, export visibility buckets, and ready-for-training or ready-for-eval posture

</specifics>

<deferred>
## Deferred Ideas

- Broadening the packet with extra corpus papers just to paper over thin support, alternative, or held-out coverage
- UI productization of packet, replay, review, and export orchestration
- Automatic corpus-wide multi-packet validation or topic discovery inside the same phase
- Reworking the core `RoutePacket`, replay, or export schemas during Phase 10

</deferred>

---

*Phase: 10-multi-paper-l3-and-l4-validation*
*Context gathered: 2026-04-03*
