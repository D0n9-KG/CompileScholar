# Phase 10: Multi-Paper L3 And L4 Validation - Research

**Date:** 2026-04-03
**Status:** Ready for planning

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Phase 10 must stay on the committed Phase 9 packet boundary and companion `support / alternative / held_out` mapping instead of redefining the topic or mutating the canonical `RoutePacket`.
- The phase should reuse the existing backend CLI and bundle-writing stack (`run_route_state_package.py`, `run_replay_pilot.py`, `export_decision_episode_pilot.py`) rather than creating a UI flow or a new artifact family.
- A real `HistoricalEnvironmentSnapshot` must replace the Phase 9 placeholder `l1_snapshot_ref` before any replay-quality claim is treated as meaningful.
- Structural package validation, replay compilation, prior review, and audited export must remain separate artifact boundaries so blockers stay attributable to the correct stage.
- Phase 10 succeeds either by producing inspectable `L3/L4` artifacts or by making blockers explicit; it does not require an artificially green result.
- Export must preserve the existing review-allowlist truth boundary and the visibility split between `visible_input_refs`, `audit_only_refs`, and `label_eval_only_refs`.
- The final output should compare the computational-mechanics run against the jamming packaged baseline to distinguish generic compiler behavior from packet-specific limitations.

### the agent's Discretion
- Exact filename and CLI shape for the Phase 10 orchestration entrypoint
- Whether the bridge from Phase 9 packet + assembly manifest into `RouteStatePackageManifest` is materialized as generated runtime subset packets, a generated package manifest, or both
- Exact report layout and comparison-table structure as long as machine-readable bundle outputs remain the source of truth

### Deferred / Out Of Scope
- Recurating the packet with more corpus papers just to hide support-density or held-out-depth weaknesses
- UI productization of package / replay / review / export orchestration
- Schema rewrites for `RoutePacket`, replay compilation, or audited export
- General multi-packet or full-corpus validation beyond this one bounded computational-mechanics slice

## Summary

Phase 10 is best treated as a conservative orchestration-and-audit phase built on top of already-shipped primitives, not as a greenfield compiler phase. The codebase already has:

- a committed bounded packet and role mapping from Phase 9,
- a route-state package compiler and validator from the Phase 3 work,
- replay bundle generation that already accepts grouped `support / alternative / held_out` route states,
- prior/anti-pattern induction from grouped package roles,
- and an audited export flow that rebuilds from replay + review bundles instead of trusting replay-time `L4` selections.

What is missing is the Phase 10 glue:

- a reproducible way to turn the committed Phase 9 packet boundary into the route-state package inputs the existing compiler expects,
- a real historical snapshot for this packet,
- a dedicated runtime workflow that executes package -> replay -> prior-review -> export on the same packet,
- comparison/reporting that surfaces where the computational-mechanics slice breaks relative to the jamming baseline,
- and focused regression tests that prove blockers remain explicit instead of being flattened into prose.

The most robust planning direction is therefore:

1. build one backend Phase 10 runner/orchestration seam around existing CLIs/helpers,
2. preserve machine-readable summaries and inspections at every stage,
3. add explicit comparison and blocker-report surfaces,
4. then run the real bounded packet and commit the human-readable report plus verification notes.

## Standard Stack

### Core
- `backend/app/research_logic/route_state_package.py` - typed package manifest, grouped route-state compilation, and package validation
- `backend/app/research_logic/historical_replay_compiler.py` - replay compiler consuming grouped route states plus optional `L1`
- `backend/app/research_logic/replay_io.py` - canonical replay summary / inspection / bundle-writer surface
- `backend/app/research_logic/prior_induction.py` - prior and anti-pattern candidate registry from support/alternative/held-out groups
- `backend/app/research_logic/decision_episode_export.py` - audited export builder with review-allowlist truth and visibility buckets
- `backend/scripts/run_route_state_package.py` - operator CLI for route-state package compilation
- `backend/scripts/run_replay_pilot.py` - operator CLI for replay and optional prior-review bundle emission
- `backend/scripts/export_decision_episode_pilot.py` - operator CLI for audited export from replay + review bundles

### Supporting
- `docs/replay/pilot_packets/phase9-route-packet.json`
- `docs/replay/pilot_packets/phase9-assembly-manifest.json`
- `docs/replay/reports/phase9-bounded-packet-audit.md`
- `docs/replay/reports/route-state-package-pilot-2026-04-02.md`
- `docs/replay/reports/phase6-decision-episode-export-pilot.md`
- `backend/tests/test_bounded_packet_audit.py`
- `backend/tests/test_route_state_package.py`
- `backend/tests/test_historical_replay_compiler.py`
- `backend/tests/test_decision_prior_builder.py`
- `backend/tests/test_decision_episode_builder.py`
- `backend/tests/test_decision_episode_export.py`
- `backend/tests/test_decision_episode_export_cli.py`
- `backend/tests/test_replay_io.py`

### Alternatives Considered
- New Phase 10 UI or database workflow: rejected because the milestone is explicitly backend/audit focused and existing CLIs already encode the artifact boundaries we need.
- Rewriting `RoutePacket` to carry package grouping inline: rejected because Phase 9 locked the packet contract and treated package grouping as a layer on top.
- Manual docs-only execution without a reusable runner: rejected because Phase 11 needs repeatable evidence, not one-off prose.

## Architecture Patterns

### Pattern 1: Phase 9 Packet And Assembly Manifest Stay Canonical
The committed packet plus assembly manifest should remain the authoritative Phase 10 inputs. Any runtime adapter must read those files and derive package compilation inputs from them; it should not overwrite or reinterpret the packet boundary in code comments or ad hoc scripts.

### Pattern 2: Bridge To `RouteStatePackageManifest` Instead Of Reworking Core Contracts
`route_state_package.py` expects entries with packet paths, trace files, and optional L1 snapshot paths. The cleanest Phase 10 seam is a generated runtime package manifest or equivalent adapter that resolves each role group into the existing package compiler format while keeping the Phase 9 packet docs untouched.

### Pattern 3: Real `L1` Snapshot Before Replay Claims
Phase 9 explicitly marked the current packet as structurally ready but replay-blocked by a placeholder `L1` reference. Phase 10 should therefore source or generate one real snapshot matching the `2021` cutoff before treating replay output as a meaningful `L3` signal.

### Pattern 4: Package -> Replay -> Prior Review -> Export Must Stay Layered
The codebase already enforces distinct artifacts:
- package validation is separate from replay summary,
- replay-time prior/anti-pattern candidates are separate from accepted review ids,
- export rebuilds from replay + review and preserves audit-only refs.

Phase 10 should keep those layers independent so the blocker queue can say exactly which stage failed.

### Pattern 5: Machine-Readable Bundles Are The Source Of Truth
The repo’s durable workflow pattern is:
- runtime JSON bundles under `tmp/`,
- committed markdown report under `docs/replay/reports/`,
- tests targeting the JSON contract rather than prose.

Phase 10 should follow the same pattern and compute comparisons from `bundle_manifest.json`, summaries, and inspections instead of diffing markdown narratives.

### Pattern 6: Baseline Comparison Needs Stage-Level Surfaces
The roadmap asks for comparison with the existing jamming baseline. The code already exposes stage-oriented signals such as:
- route-state package validation flags,
- replay failure counts by stage/layer,
- prior held-out pass rate,
- decision episode quality flags,
- export visibility buckets and training/eval posture.

Those exact surfaces should drive the comparison report.

### Anti-Patterns To Avoid
- Generating a “successful” report by enlarging the packet rather than exposing weak support/alternative/held-out depth.
- Letting a Phase 10 runner manually serialize bespoke payloads instead of calling existing bundle writers.
- Treating replay-time selected priors as if they were reviewed accepted priors in the export.
- Writing the report first and retrofitting JSON summaries afterward.

## Recommended Execution Boundary

### Recommended Orchestration Shape
Create one Phase 10 backend runner that orchestrates these stages in order:

1. Load Phase 9 packet and assembly manifest.
2. Resolve or generate a valid `HistoricalEnvironmentSnapshot` for the packet cutoff.
3. Generate runtime package inputs compatible with `run_route_state_package.py` or `compile_route_state_package(...)`.
4. Compile the route-state package bundle and capture package validation output.
5. Run replay against the same packet with the generated package bundle and write a replay bundle.
6. Emit a prior-review bundle from the route-state package / replay path.
7. Rebuild an audited export bundle from replay + review.
8. Compare the resulting package/replay/export surfaces against the jamming baseline and synthesize one committed report.

### Expected Runtime Outputs
- `tmp/phase10_multi_paper_validation/<run-id>/route_state_package/`
- `tmp/phase10_multi_paper_validation/<run-id>/replay_bundle/`
- `tmp/phase10_multi_paper_validation/<run-id>/prior_review_bundle/`
- `tmp/phase10_multi_paper_validation/<run-id>/export_bundle/`
- `tmp/phase10_multi_paper_validation/<run-id>/comparison_summary.json`
- `docs/replay/reports/phase10-multi-paper-l3-l4-validation.md`

### Expected Comparison Surfaces
- Package: role counts, validation tier, validation flags, reused-packet or scope-drift flags
- Replay: quality tier, `ready_for_pilot`, failure counts by stage and layer, selected comparison case, quality flags
- Prior/review: prior candidate count, anti-pattern candidate count, accepted ids, held-out consistency
- Export: selected prior ids vs accepted prior ids, selected anti-pattern ids, visibility buckets, `ready_for_training`, `ready_for_eval`

## Validation Architecture

Phase 10 should validate at five levels:

1. **Phase 9 input-bridge tests**
   - prove the committed `phase9-route-packet.json` and `phase9-assembly-manifest.json` can be turned into valid route-state package inputs,
   - prove every generated package entry preserves cutoff alignment and trace coverage,
   - prove missing `L1` or trace refs fail clearly before replay.

2. **Package and replay contract tests**
   - prove the Phase 10 runner produces a route-state package bundle and replay bundle with the expected filenames,
   - prove the replay bundle retains explicit support/alternative/held-out counts and stage/layer failure surfaces,
   - prove package validation signals remain visible in replay inspection rather than being swallowed by a single success/fail bit.

3. **Prior-review and export truth tests**
   - prove the prior-review bundle is emitted from the multi-paper package path,
   - prove the audited export still rebuilds from accepted review ids rather than replay-time priors,
   - prove route-family anti-pattern carryover and visibility-bucket rules remain intact for the new packet.

4. **Comparison and report tests**
   - prove the comparison summary reads machine-generated bundle surfaces instead of parsing markdown prose,
   - prove the committed report includes package quality, replay blockers, export posture, and explicit comparison against the jamming baseline,
   - prove blocker statements do not overclaim green readiness when flags remain.

5. **Manual bounded-slice review**
   - a reviewer should still inspect the final report and confirm that the computational-mechanics slice stayed inside the frozen Phase 9 boundary,
   - the report must explicitly distinguish packet-specific limitations from generic compiler regressions,
   - the blocker queue must point cleanly to the next cycle owner (`L2`, packet construction, or `L4` aggregation).

**Expected commands:**
- Quick path: `cd backend; .\.venv\Scripts\python.exe -m pytest tests\\test_phase10_multi_paper_validation.py tests\\test_replay_io.py -q`
- Full path: `cd backend; .\.venv\Scripts\python.exe -m pytest tests\\test_phase10_multi_paper_validation.py tests\\test_phase10_multi_paper_validation_cli.py tests\\test_bounded_packet_audit.py tests\\test_route_state_package.py tests\\test_historical_replay_compiler.py tests\\test_decision_prior_builder.py tests\\test_decision_episode_builder.py tests\\test_decision_episode_export.py tests\\test_decision_episode_export_cli.py tests\\test_replay_io.py -q`

The phase does not need to prove broad corpus-scale generalization. It only needs to prove that one bounded corpus packet can traverse the existing `L3/L4` pipeline reproducibly enough to surface where the next optimization cycle belongs.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| packet-role bridge | a new permanent packet schema | runtime adapter + existing `RouteStatePackageManifest` contract | Preserves the Phase 9 packet as canonical |
| replay runner | ad hoc JSON writing inside one script | `compile_historical_replay(...)` + `write_replay_bundle(...)` | Keeps quality and inspection outputs aligned with prior phases |
| prior-review path | manual accepted-id bookkeeping in docs | `build_prior_candidate_registry_from_package(...)` + `write_prior_candidate_review_bundle(...)` | Reuses the review boundary the export code already understands |
| audited export | copying replay episode JSON directly | `build_decision_episode_audit_export(...)` + `write_decision_episode_export_bundle(...)` | Preserves review truth and visibility buckets |
| baseline comparison | report-text diffing | bundle summaries / inspections + explicit comparison summary JSON | Makes Phase 11 evidence computable and repeatable |

## Common Pitfalls

### Pitfall 1: The Phase 10 runner mutates the Phase 9 boundary
**What goes wrong:** The implementation silently adds or drops packet members while “preparing” runtime inputs.  
**Why it happens:** The package compiler expects a different shape than the committed packet docs.  
**How to avoid:** Treat the committed Phase 9 packet and assembly manifest as immutable inputs and generate runtime adapter artifacts beside them.  
**Warning signs:** Runtime bundle membership no longer matches the committed packet docs.

### Pitfall 2: Placeholder `L1` remains in the loop
**What goes wrong:** Replay outputs exist, but they do not represent a meaningful multi-paper `L3` run because the packet still lacks a real historical environment.  
**Why it happens:** Package/replay code can run structurally without a strong snapshot.  
**How to avoid:** Add an explicit preflight that blocks replay-quality claims until a real snapshot is attached.  
**Warning signs:** Reports talk about replay quality while still referencing a placeholder snapshot id.

### Pitfall 3: Review truth gets bypassed
**What goes wrong:** The export carries replay-time priors or anti-patterns directly instead of the reviewed allowlist.  
**Why it happens:** Replay output looks “complete enough” and the review layer gets treated as optional.  
**How to avoid:** Keep the export step wired to the review bundle manifest and accepted ids every time.  
**Warning signs:** Exported `selected_prior_ids` do not line up with `accepted_prior_ids`.

### Pitfall 4: Comparison is too qualitative
**What goes wrong:** The final report says the packet is “worse than jamming” or “still blocked” without identifying which stage or signal regressed.  
**Why it happens:** The comparison is written from memory or manual reading.  
**How to avoid:** Compute comparisons from structured package, replay, review, and export summaries first, then write the report.  
**Warning signs:** The markdown report contains conclusions that cannot be traced back to JSON bundle fields.

### Pitfall 5: Thin support and held-out depth get softened into optimism
**What goes wrong:** The report sounds healthy even though the packet still has one-paper alternative and held-out groups.  
**Why it happens:** The phase is judged on whether artifacts exist, not whether the blocker story is honest.  
**How to avoid:** Keep the Phase 9 gap notes visible in the Phase 10 report and explicitly state whether they manifested as package, replay, prior, or export blockers.  
**Warning signs:** Remaining gaps disappear from the report without corresponding artifact evidence.

## Open Questions

1. **Should the Phase 10 bridge generate runtime subset packet files, or should it generate only a `RouteStatePackageManifest` pointing back to the canonical packet plus trace refs?**  
   Recommendation: prefer the lightest adapter that leaves the committed packet untouched and minimizes duplicate runtime artifacts.

2. **Where should the real computational-mechanics `L1` snapshot come from?**  
   Recommendation: reuse the existing historical snapshot contract and add one packet-matched runtime artifact rather than inventing a new L1 shape.

3. **Should the final Phase 10 report include one consolidated blocker queue or separate per-stage blocker sections?**  
   Recommendation: do both, but compute the consolidated queue from per-stage sections so Phase 11 can reuse it directly.

## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`
- `.planning/STATE.md`
- `.planning/phases/10-multi-paper-l3-and-l4-validation/10-CONTEXT.md`
- `.planning/phases/09-bounded-packet-construction-from-corpus/09-CONTEXT.md`
- `.planning/phases/09-bounded-packet-construction-from-corpus/09-VERIFICATION.md`
- `docs/replay/pilot_packets/phase9-route-packet.json`
- `docs/replay/pilot_packets/phase9-assembly-manifest.json`
- `docs/replay/pilot_packets/phase9-selection-notes.md`
- `docs/replay/reports/phase9-bounded-packet-audit.md`
- `docs/replay/reports/route-state-package-pilot-2026-04-02.md`
- `docs/replay/reports/phase6-decision-episode-export-pilot.md`
- `backend/app/research_logic/route_state_package.py`
- `backend/app/research_logic/historical_replay_compiler.py`
- `backend/app/research_logic/prior_induction.py`
- `backend/app/research_logic/decision_episode_export.py`
- `backend/app/research_logic/replay_io.py`
- `backend/scripts/run_route_state_package.py`
- `backend/scripts/run_replay_pilot.py`
- `backend/scripts/export_decision_episode_pilot.py`
- `backend/tests/test_route_state_package.py`
- `backend/tests/test_historical_replay_compiler.py`
- `backend/tests/test_decision_prior_builder.py`
- `backend/tests/test_decision_episode_builder.py`
- `backend/tests/test_decision_episode_export.py`
- `backend/tests/test_decision_episode_export_cli.py`
- `backend/tests/test_replay_io.py`

### Observed Runtime Evidence (HIGH confidence)
- Phase 9 already produced a structural handoff with `ready_for_phase10 = true`, but the report kept `L1` placeholder and role-depth weaknesses explicit.
- The repo already contains runnable `tmp/phase3_route_state_package/`, `tmp/phase5_multi_route_prior_induction/`, and `tmp/phase6_decision_episode_audit_export/` bundles that show the expected artifact layering.
- No dedicated Phase 10 orchestration script or Phase 10 tests exist yet under `backend/scripts/` or `backend/tests/`.

### Secondary (MEDIUM confidence)
- `docs/superpowers/specs/2026-04-01-logickg-l2-to-l3-l4-compiler-contract.md`
- `docs/superpowers/specs/2026-04-01-logickg-route-packet-schema.md`

## Recommended Plan Split

### Plan 01
Build the Phase 10 runtime bridge: resolve real `L1`, derive route-state-package inputs from the committed Phase 9 packet boundary, compile the package, and run replay with focused tests around the package/replay contract.

### Plan 02
Extend the pipeline through prior-review and audited export, then add structured comparison/blocker-summary helpers plus tests that preserve review truth and baseline-comparison signals.

### Plan 03
Run the real computational-mechanics packet through the new Phase 10 workflow, commit the human-readable report and verification evidence, and document exactly which stage-level blockers remain relative to the jamming baseline.

## Metadata

- Core technology: route-state package orchestration, historical replay compilation, prior review, audited export, and structured comparison reporting
- Ecosystem: `research_logic`, backend CLIs, typed bundle writers, committed packet docs, and pytest-based backend verification
- Patterns: immutable packet boundary, layered artifact families, review-allowlist truth, machine-readable summaries before markdown conclusions
- Pitfalls: packet drift, placeholder `L1`, bypassed review truth, prose-only comparison, softened blocker reporting
- Confidence: HIGH for reuse points and artifact contracts, MEDIUM-HIGH for the exact Phase 10 runner shape, MEDIUM for final report layout choices
