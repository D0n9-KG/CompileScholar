# L2 Content Quality Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise L2 toward a trustworthy single-paper representation layer by tightening metadata quality, restoring missing method content in Chinese papers, and improving bilingual summary coherence without overfitting.

**Architecture:** Keep the work localized to ingestion metadata repair, direct extraction role stabilization, and derived summary selection. Each phase starts from real sampled papers, adds the narrowest failing regression test first, implements the smallest general fix, reruns focused plus full verification, and then commits the phase.

**Tech Stack:** Python, FastAPI backend modules, pytest, git, real-paper sampling scripts against the shared markdown corpus.

---

## Chunk 1: Baseline L2 Quality Hardening

### Task 1: Land verified metadata and Chinese method-role fixes

**Files:**
- Modify: `backend/app/ingest/paper_metadata_enrichment.py`
- Modify: `backend/app/paper_logic_trace/direct_extraction.py`
- Modify: `backend/app/paper_logic_trace/derived_views.py`
- Modify: `backend/app/paper_logic_trace/gates.py`
- Modify: `backend/app/paper_logic_trace/compiler.py`
- Modify: `backend/app/paper_logic_trace/models.py`
- Modify: `backend/app/ingest/parse_md.py`
- Modify: `backend/app/ingest/pipeline.py`
- Modify: `backend/app/ingest/models.py`
- Modify: `backend/app/ingest/paper_identity.py`
- Modify: `backend/app/ingest/scan_upload.py`
- Modify: `backend/app/graph/neo4j_client.py`
- Test: `backend/tests/test_paper_metadata_enrichment.py`
- Test: `backend/tests/test_paper_logic_trace_direct_extraction.py`
- Test: `backend/tests/test_paper_logic_trace_derived_views.py`
- Test: `backend/tests/test_paper_logic_trace_gates.py`
- Test: `backend/tests/test_paper_logic_trace_compiler.py`
- Test: `backend/tests/test_ingest_pipeline_reingest_idempotency.py`
- Test: `backend/tests/test_parse_md_sections.py`
- Test: `backend/tests/test_scan_upload_doi_strategy.py`
- Test: `backend/tests/test_scan_upload_metadata_enrichment.py`
- Test: `backend/tests/test_paper_identity.py`

- [x] Verify the current branch is `codex/l2-content-quality`.
- [x] Confirm the existing full-suite evidence is fresh or rerun it if needed.
- [x] Stage only the L2-quality files listed above.
- [x] Commit the staged baseline with a message describing metadata cleanup plus Chinese method-role recovery.

## Chunk 2: Bilingual Summary Coherence

### Task 2: Add a failing regression for mixed-language one-paragraph summaries

**Files:**
- Modify: `backend/tests/test_paper_logic_trace_derived_views.py`

- [x] Write a regression test where bilingual paper moves contain both Chinese and English summary candidates.
- [x] Run the targeted pytest selection and confirm the new test fails for the current summary selector.

### Task 3: Implement minimal language-coherent summary selection

**Files:**
- Modify: `backend/app/paper_logic_trace/derived_views.py`
- Test: `backend/tests/test_paper_logic_trace_derived_views.py`

- [x] Add lightweight language-signal helpers for summary sentences and selected move groups.
- [x] Prefer same-language sentence bundles when building `one_paragraph_summary`, while preserving current role/quality priorities.
- [x] Keep the fix conservative: do not rewrite move text, only change summary move selection/tie-break behavior.
- [x] Run focused derived-view tests and confirm the new regression passes.
- [x] Re-run fresh real-paper checks on bilingual samples and inspect the actual summary text.
- [x] Commit the phase with a message describing bilingual summary coherence.

## Chunk 3: Fresh Audit Refresh

### Task 4: Re-audit random sampled papers after the new fixes

**Files:**
- Create or update: `backend/.codex_tmp/manual_random_content_audit/20260329_*`
- Modify: `docs/superpowers/plans/2026-03-29-l2-content-quality.md`

- [ ] Re-run a fresh content audit on a small random bilingual/mixed-language sample.
- [ ] Summarize repeated remaining issues from actual extracted content, not only quality flags.
- [ ] Mark completed plan steps in this document as the work lands.
- [ ] Commit the refreshed audit artifacts or summary only if they are intended to stay in-repo; otherwise leave them untracked and document results in the final report.

## Chunk 4: Final Verification

### Task 5: Run the full backend verification gate and prepare the next phase

**Files:**
- Modify only if required by fixes from earlier chunks.

- [x] Run `cd backend; .\.venv\Scripts\python.exe -m pytest -q`.
- [x] Review the exact output and only then claim completion for the current execution window.
- [x] If tests pass, record the next highest-value unresolved L2 gap and continue with another phase in the same branch.

Progress note (2026-03-29):
- Verified a new findings-quality phase on real paper `1243_Data-Driven Computational Plasticity`.
- Added a regression for conclusion-section achievement claims that were being stabilized as `method` instead of `result`, then confirmed the real sample now surfaces result-backed `key_findings`.
- Tightened `paper_content_profile.key_findings` ordering so explicit `result` summaries are listed before earlier `interpretation` summaries.
- Next highest-value unresolved L2 gap: theory-heavy papers still show thin route-state evidence because comparator/effect/measurement signals remain sparse even after findings are recovered.

Progress note (2026-03-29, later phase):
- Added regression coverage for theory-heavy `research_objects` recovery, including conclusion-scope objects like `internal variables` and problem-scope objects like `constitutive models`.
- Tightened research-object cleanup so generic singleton noise such as `parameters`, long `introduced into the weak form ...` clause fragments, and leading `establish ...` verb phrases do not leak into L2 topic objects.
- Added a conservative `route_state_seed` fallback that uses inferred topic objects only when no trusted topic-object entries are available, improving downstream coverage without changing the normal trusted-first path.
- Real-sample check: `1243_Data-Driven Computational Plasticity` now has non-empty `topic_scope_candidates`; anti-overfit check on `1607_Shear jamming and fragility in dense suspensions` kept strong topic and method candidates intact.
- Next highest-value unresolved L2 gap: theory-heavy papers still admit some broad context objects in topic scope ordering, so the next phase should improve topic-object ranking/filtering rather than only increasing recall.

Progress note (2026-03-29, trusted topic-signal phase):
- Added regression coverage for promoting scope-explicit research objects into trusted `normalized + strong` signals, including cases where move role is only stabilized to `result` after the initial slot-augmentation pass.
- Kept broad context phrases conservative: method/background-style objects such as `engineered materials` still remain `inferred + weak` unless a stronger scope relation is present in the source sentence.
- Added a preferred-merge path so post-stabilization trusted scope objects can replace same-surface inferred objects without overwriting already-direct evidence.
- Real-sample check: `1243_Data-Driven Computational Plasticity` now produces non-empty trusted `route_compiler_contract.topic_signals.objects`, with `topic_scope_candidates` narrowing to `nonlinear elasticity`, `internal variables`, and `constitutive model`.
- Anti-overfit check: `1607_Shear jamming and fragility in dense suspensions` kept healthy topic candidates centered on `fragile shear-jammed state`, `shear-jammed states`, and `rigid-particle suspensions`.
- Next highest-value unresolved L2 gap: trusted topic signals are materially better, but L2 still undersupplies measurement/resource/toolchain evidence on many theory-heavy papers, limiting how fully L3/L4 can reconstruct readiness and constraints from a single paper trace.

Progress note (2026-03-29, prediction-target topic phase):
- Added regression coverage for prediction-target sentences such as `predict mechanical field distributions in composite microstructures` and `predicting mechanical fields in composites`, promoting these explicit task/object phrases into trusted `normalized + strong` research objects.
- Expanded conservative research-object head hints for common scientific object heads such as `fields`, `microstructures`, `composites`, `laminates`, and `distributions`, but kept the promotion path limited to explicit prediction-target patterns so broad method/background phrases are not upgraded wholesale.
- Extended preferred-merge behavior so later trusted prediction-target objects can replace same-surface inferred placeholders produced earlier in the extraction pipeline.
- Random-sample audit after the change:
- `s1338_Integrated convolutional and graph neural networks for predicting mechanical fields in composite microstructures` improved from empty trusted topic signals plus noisy fallback candidates to trusted objects centered on `mechanical field distributions in composite microstructures`, `mechanical fields in composites`, `stress fields`, and `fiber-reinforced composites`.
- `s870_Pb-activated amine-assisted photocatalytic hydrogen evolution reaction...` remained healthy, with topic candidates still centered on `photocatalytic hydrogen evolution reaction`, `mapbi3 surface`, and the `PbAAA` mechanism.
- `s93_Numerical investigation of twin-liquid film on spoked rotating disk reactor...` remained healthy, with topic candidates still centered on `twin-liquid film`, `film flow hydrodynamics and mass transfer`, and `vertical plate with openings`.
- Next highest-value unresolved L2 gap: topic objects are improving, but many single-paper traces still lack strong measurement/resource/toolchain signals and challenging evidence, which blocks fuller L3 readiness reconstruction even when the paper’s topic and method content are now represented well.

Progress note (2026-03-29, route-state seed audit calibration phase):
- Added regression coverage for two complementary non-overfit cases where a single-paper `route_state_seed` is substantively usable for L3 follow-on work even though it does not mention benchmarks or data resources:
- a theory-like bottleneck/challenge case with strong topic, method, supporting evidence, challenging evidence, and bottleneck signals but no benchmark/toolchain coverage
- a measurement/enabling case with strong topic, method, supporting evidence, measurement protocol signals, and enabling conditions but no benchmark/data-resource coverage
- Reworked `route_state_seed_audit` so readiness is no longer a blanket requirement that every benchmark, protocol, toolchain, infrastructure, and data-resource slot be non-empty at once.
- The new audit keeps a strict core requirement on topic, method, evidence, and source coverage, then requires at least two distinct context groups (challenge, bottleneck, measurement, resource, infrastructure, enabling) before marking the seed ready for route compilation.
- The audit still records all missing components, but now separates true route-compilation blockers from non-blocking omissions so honest single-paper traces are not mislabeled as thin just because the paper never claimed a benchmark or toolchain.
- Real-sample recheck after the change:
- `1243_Data-Driven Computational Plasticity` no longer trips `route_state_seed_thin`; its route seed now compiles on the strength of explicit challenge and bottleneck coverage while still honestly reporting missing benchmark/protocol/toolchain fields.
- `s1338_Integrated convolutional and graph neural networks for predicting mechanical fields in composite microstructures`, `s870_Pb-activated amine-assisted photocatalytic hydrogen evolution reaction...`, and `s93_Numerical investigation of twin-liquid film...` likewise stop failing solely for absent benchmark/data-resource slots; each now passes route-seed readiness through the context signals it actually contains.
- Next highest-value unresolved L2 gap: several real papers still remain yellow because their paper summary selection drifts away from the paper title/topic center or because title metadata is mis-extracted, which weakens single-paper readability even when the underlying topic/method slots are now materially better.

Progress note (2026-03-29, title-aligned summary selection phase):
- Added regression coverage for two summary-selection failure modes seen in real-paper audits:
- opening/problem sentences that stay in broad domain context even when a later sentence is much closer to the paper title and topic center
- generic method-context sentences that outrank later method sentences carrying the paper’s actual target object
- Updated `build_paper_summaries` to use title/title-alt lexical alignment as a conservative tie-break inside the existing role-based summary selector, while preserving current language-coherence and role-priority behavior.
- The change only reorders comparable summary candidates; it does not rewrite text or override stronger language/role constraints with a title match.
- Real-sample recomputation after the change:
- `s1338_Integrated convolutional and graph neural networks for predicting mechanical fields in composite microstructures` now produces a summary centered on predicting mechanical field distributions in composite microstructures and turns green on recomputed quality.
- `s93_Numerical investigation of twin-liquid film on spiked rotating disk reactor with highly viscous fluid` now selects a much more title-aligned opening/result bundle and likewise turns green on recomputed quality.
- `1243_Data-Driven Computational Plasticity` remains yellow, but now for a clearly narrower reason: the canonical title metadata itself is still wrong (`2.1. Non-isothermal elasto-visco-plastic behavior`), so the remaining gap has shifted from summary drift to metadata title recovery.
- Next highest-value unresolved L2 gap: recover or correct mis-extracted canonical titles when `title` is section-heading-like and `title_alt` carries the true paper title, because this still harms single-paper readability and triggers honest metadata-summary mismatch flags on otherwise improved traces.

Progress note (2026-03-29, local title-repair handoff phase):
- Added regression coverage at both the metadata-enrichment layer and the direct `paper_logic_trace` build path for the case where `title` is a section-heading-like fragment but `title_alt` carries the real paper title.
- Tightened local metadata repair so a clean `title_alt` is preferred over a generic `paper_source` slug when promoting a fallback title, keeping the repair aligned with actual paper content rather than directory naming.
- Wired a no-network local metadata repair step into `build_paper_logic_trace_inputs` for direct trace construction when no upstream `metadata_enrichment` record is present, so manual audits and direct extraction runs do not silently bypass the existing title-fix logic.
- Real-sample recheck: rerunning `1243_Data-Driven_Computational_Plasticity.md` through the direct trace builder now restores the canonical title to `Data-Driven Computational Plasticity` and removes the prior title-level mismatch failure mode from the single-paper output.
- Next highest-value unresolved L2 gap: the remaining yellow cases are now more concentrated in substantive slot coverage differences between theory-style papers and the current empirical-default expectations, rather than in obvious summary or title-surface defects.

Progress note (2026-03-29, theory-modeling completeness calibration phase):
- Added regression coverage for a theory/modeling evidence profile that should be considered L3/L4-ready even without empirical `metrics` / `comparators` / `effects`, as long as the trace has grounded `problem -> method -> interpretation` structure plus explicit constraint/context evidence.
- Added a paired negative control showing that a theory-like trace with only topic and method content, but no grounded interpretation/constraint move, must remain yellow and not unlock L4.
- Reworked `l2_completeness_audit` to keep the existing empirical-default L4 path unchanged, while adding a narrow theory/modeling readiness branch that requires trusted object + method support and at least one grounded constraint/context move in `interpretation` / `limitation` / `future_work`.
- This calibration is profile-based rather than a blanket per-paper-type relaxation: empirical traces still need effect/comparator-style evidence, while theory/modeling traces can qualify through explanatory completeness and explicit constraint coverage.
- Verification:
- Focused regression: `backend/tests/test_paper_logic_trace_gates.py -k theoretical`
- Focused file: `backend/tests/test_paper_logic_trace_gates.py`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `520 passed, 1 warning`
- Real-sample check after the gate change shows no false green on existing audited traces: `1243_Data-Driven Computational Plasticity` remains yellow because its current extracted content still lacks grounded constraint moves and strong topic-object support in the saved audit artifact, not because theory papers are still being forced through an empirical comparator/effect gate.
- Next highest-value unresolved L2 gap: improve actual extraction/stabilization of theory-style constraint and limitation content, and reduce `paper_type` / role drift on real papers such as `1243`, so the richer theory/modeling branch can be reached by real traces instead of only synthetic regressions.

Progress note (2026-03-29, explicit drawback bottleneck extraction phase):
- Added regression coverage for a real `1243_Data-Driven Computational Plasticity` failure mode where an introduction/problem move explicitly states a bottleneck (`the main drawback ... is the huge amount of data required for running simulations`) but previously exported no `limitation_types` and therefore contributed no trusted bottleneck signal to `route_state_seed`.
- Added a narrow explicit-limitation extraction path for phrases such as `main drawback is ...` / `main limitation is ...`, promoting those rows as `normalized + strong` rather than weak heuristic noise.
- Kept the broader heuristic limitation path unchanged for normal cases, but allowed the new explicit-limitation path to run in `problem` / `background` moves as well, so explicit bottleneck sentences in introductions are no longer dropped just because the move is not already labeled `limitation`.
- Verification:
- Focused regression: `backend/tests/test_paper_logic_trace_direct_extraction.py -k explicit_main_drawback`
- Focused file: `backend/tests/test_paper_logic_trace_direct_extraction.py` -> `95 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `521 passed, 1 warning`
- Real-sample recheck with current direct local rebuild of `1243_Data-Driven Computational Plasticity`:
- `route_state_seed.known_bottleneck_candidates` now includes `huge amount of data required` and `huge amount of data required for running simulations`
- `route_state_seed.challenging_evidence_ids` is no longer empty
- `topic_scope_candidates` stay centered on `nonlinear elasticity`, `internal variables`, and `mathematical constitutive model`
- Anti-overfit recheck on `1607_Shear jamming and fragility in dense suspensions` remained reasonable: bottlenecks stayed centered on content-grounded constraints such as `computational cost`, `finite size effect`, `particles too soft`, and `narrow range of area fractions`, rather than introducing unrelated generic limitation noise.
- Broader macro audit on real direct rebuilds stayed directionally healthy rather than overfit to `1243`:
- `s870_Pb-activated amine-assisted photocatalytic hydrogen evolution reaction...` stayed clean, with no new generic limitation noise.
- `s93_Numerical investigation of twin-liquid film...` still shows some older heuristic limitation phrases such as `indicating limitation` / `difficult`, but those were not introduced by the new explicit-drawback path.
- Random paper `155_Characterization of force chains in granular material` surfaced content-grounded limitations around `restrictive and cumbersome visualization`, `qualitative methods`, and `lack of directional information`, rather than generic drawback spam.
- Additional stability audit showed the more macro remaining gap: direct local rebuilds still depend heavily on `_extract_window_moves_llm`, so the same paper can preserve or lose explicit bottleneck/constraint content depending on whether the upstream window extractor anchors the right intro chunks.
- Next highest-value unresolved L2 gap: reduce direct-extraction role/anchor instability for theory-style intro and constraint-heavy windows, so explicit bottleneck / limitation content survives even when the upstream move extractor is sparse or summary-compressive, instead of widening more slot heuristics.

Progress note (2026-03-30, window-support limitation recovery phase):
- Added regression coverage for a stability failure mode where a `problem` move is correctly extracted from an introduction window, but the upstream move extractor anchors only the earlier chunk while the explicit `main drawback ...` sentence lives in a later chunk of the same window.
- Root-cause audit showed the local slot augmenter was already capable of extracting the bottleneck phrase, but `_move_support_text` only saw the anchored chunk(s), so explicit limitation content could disappear before local normalization whenever the upstream move extractor produced sparse anchors.
- Added a narrow window-level recovery path for explicit limitation phrases only: when a `problem` / `background` / `interpretation` / `limitation` / `future_work` / `result` move shares a window with later chunks containing `main drawback ...` / `main limitation ...`, those chunks are added back into support and their normalized limitation rows are merged conservatively.
- Verification:
- Red-green regression: `backend/tests/test_paper_logic_trace_direct_extraction.py -k later_intro_chunk`
- Related drawback regressions: `backend/tests/test_paper_logic_trace_direct_extraction.py -k "explicit_main_drawback or later_intro_chunk"`
- Focused file: `backend/tests/test_paper_logic_trace_direct_extraction.py`
- Real-sample stability recheck with forced local fallback extraction:
- `1243_Data-Driven Computational Plasticity` now recovers `huge amount of data required for running simulations` as a bottleneck candidate even when the move is built from a sparse intro/problem window.
- `1607_Shear jamming and fragility in dense suspensions`, `s870_Pb-activated amine-assisted photocatalytic hydrogen evolution reaction...`, `s93_Numerical investigation of twin-liquid film...`, and random paper `155_Characterization of force chains in granular material` did not pick up new explicit-drawback noise under the same forced-fallback audit.
- Direct local rebuild recheck on `1243` also still surfaces the bottleneck after the change, indicating the new recovery path helps the intended real path while reducing one source of upstream-anchor fragility.
- Next highest-value unresolved L2 gap: theory-style papers still remain materially under-typed and under-interpreted after bottlenecks are recovered, so the next phase should focus on stabilizing `interpretation` / `limitation` moves or theory-style `paper_type` inference rather than adding more limitation extractors.

Progress note (2026-03-30, grounded limitation-move stabilization phase):
- Added regression coverage for two downstream-facing failures where a `problem` move already carries an explicit drawback/limitation signal but still fails to produce a dedicated grounded constraint move:
- the normal sparse-anchor case, where the explicit drawback sentence is present and recovered into `limitation_types`
- the long-window case, where the drawback sentence exists in anchored chunks but falls outside truncated `move_support_text`
- Root-cause audit showed that L2 could already recover the bottleneck phrase itself, but downstream completeness still stayed thin because:
- `_stabilize_move_role_and_act_type` only promoted limitation roles when the cue was visible in the move summary, not support text
- the pipeline emitted no companion `limitation` move for `problem` windows carrying explicit drawback sentences
- companion-move generation originally looked only at truncated support text, so long intro windows could still lose the explicit drawback sentence even after limitation recovery
- The fix keeps the original `problem` move intact, but now emits a narrow companion `limitation` move when an anchored chunk contains an explicit `main drawback ...` / `main limitation ...` sentence, and it allows support-text limitation cues to stabilize non-problem roles more reliably.
- Verification:
- Red-green regressions:
- `backend/tests/test_paper_logic_trace_direct_extraction.py -k grounded_limitation_move`
- `backend/tests/test_paper_logic_trace_direct_extraction.py -k truncates`
- Related drawback regressions:
- `backend/tests/test_paper_logic_trace_direct_extraction.py -k "explicit_main_drawback or later_intro_chunk or grounded_limitation_move or truncates"`
- Focused file: `backend/tests/test_paper_logic_trace_direct_extraction.py`
- Real-sample rechecks:
- Forced local fallback rebuild of `1243_Data-Driven Computational Plasticity` now produces an explicit `limitation` move with summary `Its main drawback is the huge amount of data required for running simulations.` and raises `grounded_constraint_move_count` from `0` to `1`.
- Direct local rebuild of `1243_Data-Driven Computational Plasticity` now yields grounded limitation coverage as well (`grounded_constraint_move_count=2` in the latest check), while preserving the original `problem` narrative and the recovered bottleneck candidate.
- Anti-overfit fallback recheck on `1607_Shear jamming and fragility in dense suspensions` still produces no synthetic limitation moves, so the new companion-move path is not firing on ordinary non-explicit constraint language.
- Next highest-value unresolved L2 gap: `1243` is now materially closer to theory/modeling completeness, but it still remains `paper_type=unknown`, so the next phase should target theory-style paper-type inference or profile selection rather than more limitation-slot logic.

Progress note (2026-03-30, content-based theory-profile selection phase):
- Added regression coverage for a theory-like trace that remains `paper_type='unknown'` but already contains the structure L2 actually needs downstream: grounded `problem + method + result + limitation`, trusted object/method support, and no empirical benchmark/comparator evidence.
- Root-cause audit on the current `1243` direct rebuild showed that after the earlier limitation phases it still failed L4 for a purely profile-selection reason:
- the trace already had grounded limitation moves and `grounded_constraint_move_count > 0`
- but `_uses_theory_modeling_evidence_profile` still forced `paper_type='unknown'` through the empirical-default branch
- and the theoretical expectation logic still treated `interpretation` / `conditions` as strictly required even when the same theory-style constraint content was already captured as trusted `limitation` / `limitation_types`
- Reworked the theory-profile gate narrowly rather than globally:
- for `paper_type='theoretical'`, trusted `limitation` can now satisfy the theory-side explanatory role that was previously hard-coded as `interpretation`
- for `paper_type='theoretical'`, trusted `limitation_types` can satisfy the theory-side constraint/context expectation that was previously hard-coded as `conditions`
- for `paper_type='unknown'`, the gate can now select `theory_modeling` only when the trace has a narrow theory-like signature: supported `problem + method + limitation`, no `experiment` role, trusted limitation support, and no trusted `metrics` / `comparators`
- Verification:
- Red-green regression: `backend/tests/test_paper_logic_trace_gates.py -k theory_like_limitation_profile`
- Related theoretical gate checks: `backend/tests/test_paper_logic_trace_gates.py -k theoretical`
- Focused file: `backend/tests/test_paper_logic_trace_gates.py`
- Real-sample rechecks:
- `1243_Data-Driven Computational Plasticity` now turns `green` on a direct local rebuild with `l4_evidence_profile='theory_modeling'`, `ready_for_l4=True`, and `grounded_constraint_move_count=2`, even though the metadata `paper_type` still remains `unknown`.
- Anti-overfit direct rebuild checks:
- `s870_Pb-activated amine-assisted photocatalytic hydrogen evolution reaction...` stays on `empirical_default` and does not get promoted into the theory/modeling branch.
- `s93_Numerical investigation of twin-liquid film...` likewise stays on `empirical_default` while preserving its existing green empirical path.
- Next highest-value unresolved L2 gap: now that theory-style limitation content can actually unlock the right downstream profile, the remaining macro issue is repeated/overlapping move clutter in some direct rebuilds (for example duplicated problem/limitation content within long introductions), so the next phase should likely target move deduplication / compression rather than more gate widening.

Progress note (2026-03-30, companion-limitation dedup phase):
- Added a regression for the concrete direct-extraction clutter case where two raw moves from the same introduction window both recover the same explicit `main drawback ...` sentence and previously emitted two identical companion `limitation` moves.
- Root-cause audit showed the duplicate was not a gate or compile artifact: `_move_rows_from_windows` generated companion limitation moves independently per raw move, so the same explicit limitation sentence could be materialized multiple times inside one semantic window whenever upstream move extraction split the surrounding introduction into overlapping `problem` summaries.
- Reworked companion limitation emission conservatively:
- dedup is limited to the current semantic window
- the dedup key is the normalized explicit limitation sentence itself, so it only collapses repeated restatements of the same recovered drawback
- when a duplicate is detected, richer slot content from the later raw move is merged back into the already-created limitation move instead of silently discarding it
- Verification:
- Red-green regression: `backend/tests/test_paper_logic_trace_direct_extraction.py -k one_companion_limitation_move`
- Related drawback regressions: `backend/tests/test_paper_logic_trace_direct_extraction.py -k "explicit_main_drawback or later_intro_chunk or grounded_limitation_move or truncates or one_companion_limitation_move"`
- Focused file: `backend/tests/test_paper_logic_trace_direct_extraction.py` -> `99 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `527 passed, 1 warning`
- Real-sample rechecks:
- `1243_Data-Driven Computational Plasticity` now keeps a single grounded companion limitation move on direct local rebuild: summary `Its main drawback is the huge amount of data required for running simulations.`, one `limitation` move instead of repeated same-sentence clones, and `l4_evidence_profile='theory_modeling'` with `ready_for_l4=True` remains intact.
- The surviving `1243` limitation move still carries grounded content rather than a compressed generic phrase, with `limitation_types` centered on `huge amount of data required for running simulations`.
- Anti-overfit direct rebuild checks stayed stable:
- `s870_Pb-activated amine-assisted photocatalytic hydrogen evolution reaction...` still stays on `empirical_default`, remains `yellow`, and emits no limitation moves.
- `s93_Numerical investigation of twin-liquid film...` still stays on the existing green empirical path; its remaining limitation content is the older heuristic-style noise (`indicating limitation`, `difficult`) rather than a new artifact introduced by the companion-move dedup.
- Next highest-value unresolved L2 gap: the new duplication bug is closed, but L2 still has a broader compression/cleanup problem rather than a pure recall problem. In practical terms, single-paper traces like `1243` are now structurally sufficient for downstream L3/L4, yet some empirical papers still retain legacy heuristic limitation clutter and overlapping move phrasing, so the next phase should target macro move compression / noise cleanup without narrowing recall to a few hand-tuned papers.

Progress note (2026-03-30, generic limitation-noise cleanup phase):
- Added red-green helper regressions for two corpus-level low-information limitation patterns that were still leaking through `s93` and similar papers:
- summary-derived phrases like `indicating limitation` that mention the existence of a limitation but not its actual scope
- cue-word fallbacks that collapse a real clause such as `it is difficult for the free film to stabilize ...` into the near-empty singleton `difficult`
- Reworked `_refine_limitation_rows` conservatively rather than widening extraction:
- low-information phrases such as `difficult`, `difficulty`, and `indicating limitation` are now treated as placeholders that must be rewritten from the supporting text into a more specific limitation phrase when possible
- new rewrite paths recover patterns such as `limitation of <scope>`, `difficult for <object> to <action>`, `difficult to <action>`, and `difficulty of <scope>`
- if no specific rewrite can be recovered, these placeholder phrases are dropped instead of being exported as misleading L2 content
- Verification:
- Red-green regressions: `backend/tests/test_paper_logic_trace_direct_extraction.py -k "refine_limitation_rows_rewrites"`
- Focused file: `backend/tests/test_paper_logic_trace_direct_extraction.py` -> `101 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `529 passed, 1 warning`
- Real-sample rechecks:
- `1243_Data-Driven Computational Plasticity` remains stable after the cleanup: still `green`, still `theory_modeling`, still `ready_for_l4=True`, and still keeps the single grounded drawback limitation `huge amount of data required for running simulations`.
- `s93_Numerical investigation of twin-liquid film...` improves materially at the content level rather than only numerically:
- the older limitation bundle `jammed open windows`, `thick dragged film`, `could not accommodate highly viscous fluid`, `indicating limitation` is reduced to the more specific grounded pair `jammed open windows` and `could not accommodate highly viscous fluid`
- the separate singleton `difficult` limitation is rewritten to the more content-grounded phrase `due to non-slip boundary condition`
- `s870_Pb-activated amine-assisted photocatalytic hydrogen evolution reaction...` remains untouched by the cleanup: still `empirical_default`, still `yellow`, and still emits no limitation moves
- Next highest-value unresolved L2 gap: limitation noise is cleaner now, but the broader macro gap remains overlapping move phrasing and under-compressed narrative redundancy across roles. In other words, L2 is getting closer to being both faithful and downstream-usable, yet some papers still spread one idea across several adjacent moves or keep weak summary phrasing when the underlying evidence is good. The next phase should therefore focus on move-level compression / redundancy cleanup, not on adding more slot-specific heuristics.

Progress note (2026-03-30, nearby method-dedup phase):
- Added regression coverage for two method-clutter cases that were still leaking through after the companion-limitation cleanup:
- nearby `method -> result -> method` duplication where the later method restates the same proposal with richer slot content
- bilingual nearby duplication where the Chinese abstract and English abstract each emit the same `propose_method` move but lexical summary overlap is low across languages
- Reworked `_compress_adjacent_redundant_method_moves` conservatively rather than broadening generic compression:
- widened the merge scan only to a tiny `max_lookahead = 2`, so the earlier move can absorb one nearby duplicate even when a single interleaved move sits between them
- kept the existing summary-overlap and method-signature thresholds for normal same-language merges
- added a narrow bilingual exception only for `propose_method` pairs with one CJK summary and one Latin summary plus at least two grounded method mentions on both sides, so common Chinese/English abstract restatements can merge without opening the door to generic English method collapse
- Verification:
- Targeted regressions:
- `backend/tests/test_paper_logic_trace_direct_extraction.py -k "adjacent_redundant_method_moves_merge_into_one_richer_move or adjacent_method_moves_with_distinct_method_signatures_do_not_merge or nearby_redundant_method_moves_merge_across_interleaved_result_move or bilingual_nearby_redundant_method_moves_merge_when_method_signature_matches"`
- Focused file: `backend/tests/test_paper_logic_trace_direct_extraction.py` -> `105 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `533 passed, 1 warning`
- Real-sample rechecks after the change:
- `1267` now rebuilds to `move_count=7` with `method_count=4`, keeping the main neural-network proposal once while preserving the distinct training-data method move and later GRNN/BP method elaborations.
- `1243_Data-Driven Computational Plasticity` remains structurally healthy, with `move_count=11`, `method_count=5`, and the grounded drawback limitation still preserved.
- `1607_Shear jamming and fragility in dense suspensions` keeps its distinct stress-controlled / Stokesian / contact-force method chain (`move_count=19`, `method_count=10`) rather than collapsing into one over-merged method blob.
- Next highest-value unresolved L2 gap: L2 single-paper traces are now less cluttered by repeated limitation/method restatements, but several papers still show over-split method ladders or thin experiment/result anchoring where the extracted moves are distinct enough to avoid dedup yet still not ideal for downstream L3/L4 route reconstruction. The next phase should inspect those remaining high-frequency content patterns from fresh corpus samples before changing any more compression rules.

Progress note (2026-03-30, method-backfill noise cleanup phase):
- Added red-green regression coverage for three concrete method-slot noise patterns surfaced by the new sparse-method backfill path:
- generic head phrases such as `efficient method`
- action-led sentence fragments such as `focus on simple model`
- cross-sentence token-backtrack pollution such as `smaller particles stokesian dynamics`
- Root-cause audit showed `_method_mentions_from_text` had two independent failure modes:
- broad method-head matching could preserve short generic `modifier + head` phrases and verb-led fragments when a summary clearly mentioned a real method elsewhere in the same sentence
- the token-backtrack fallback ignored sentence boundaries, so context words from the preceding sentence could leak into the extracted method phrase
- Reworked `_method_mentions_from_text` conservatively rather than broadening dedup or slot merging:
- strip only a narrow set of leading method-intro verbs such as `employ`, `use`, and `focus on` before normalizing the candidate phrase
- reject short phrases that are only generic modifiers attached to a generic method head
- reject remaining action-led method candidates such as `reproduce ...` / `represent ...` / `simulate ...`
- constrain the token-backtrack fallback to operate within sentence/clause segments instead of across the whole summary text
- Verification:
- Red-green regressions: `backend/tests/test_paper_logic_trace_direct_extraction.py -k "generic_efficient_method_fragment or focus_on_simple_model_fragment"`
- Focused method/object regressions: `backend/tests/test_paper_logic_trace_direct_extraction.py -k "backfills_particle_dynamics_simulation or backfills_finite_element_method or incidental_algorithm_reference_as_method or promoted_method_move_backfills_method_mentions_after_role_stabilization or descriptive_clause_phrases or scheme_reporting_and_description_fragments or generic_efficient_method_fragment or focus_on_simple_model_fragment"` -> `8 passed`
- Focused file: `backend/tests/test_paper_logic_trace_direct_extraction.py` -> `111 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `539 passed, 1 warning`
- Real-sample forced-local-fallback recheck after the change:
- `1607_Shear jamming and fragility in dense suspensions` now keeps method candidates centered on `stokesian dynamics`, `sd approach`, and `soft-constraint approach`, while dropping prior noise such as `efficient method`, `focus on simple model`, `simple model`, `reproduce particle dynamics`, and `smaller particles stokesian dynamics`
- the cleanup stayed narrow: earlier sparse-method wins such as `particle dynamics simulations` and `finite element method` remain covered by regression tests, and the research-object preservation regressions introduced in the previous subphase remain green
- Macro gap observed after the cleanup:
- the sparse local-fallback path is now less noisy on empirical method-heavy papers, but it still undersupplies role structure on harder papers such as `1243_Data-Driven Computational Plasticity` and Chinese paper `282_叶片间隙对潜水搅拌器流场特性的影响`, where fallback extraction remains problem-dominant and can miss method/result moves entirely
- Next highest-value unresolved L2 gap: move from slot-level cleanup to fallback window-role / summary stabilization for theory-heavy and Chinese papers, because L2 still falls short of the “single paper can stand on its own for L3/L4” target whenever the upstream move extractor is sparse or unavailable.

Progress note (2026-03-30, fallback role/summary stabilization phase):
- Added red-green regression coverage for two fallback-structure failures that were blocking single-paper usability when upstream move extraction is sparse:
- a long `background` window whose first two sentences are problem/context but whose later support sentence clearly states the method
- a Chinese `4 结论` section that should surface as `result` under fallback rather than collapsing into the default `background -> problem` path
- Root-cause audit showed two different structural causes:
- fallback `background` moves were immediately assigned `identify_gap`, which promotes them to `problem` before stabilization; if the first-two-sentence fallback summary did not itself contain a method cue, later method sentences in the same window were ignored
- Chinese conclusion sections lacked section-level result cues, so fallback could not recover a `result` role when the conclusion text did not happen to contain the existing English result phrases
- Reworked fallback stabilization conservatively rather than widening extraction globally:
- let method-role stabilization consider method mentions recovered from full `support_text`, not only explicit summary-level method-statement patterns
- require explicit problem-signal summaries to stay `problem` even if method-like terms appear in the same sentence/window, preventing challenge statements such as `can simulation proceed ... without a constitutive model` from being retyped as `method`
- add Chinese conclusion section cues (`结论`, `结语`) to the section-role/result hints
- when a move originates from local fallback, reselect its summary after role stabilization using a role-aware sentence picker, so fallback `method` / `result` moves are not forced to keep a stale problem-context opening sentence
- Verification:
- Red-green regressions: `backend/tests/test_paper_logic_trace_direct_extraction.py -k "later_support_sentence or chinese_conclusion_section_promotes_to_result_role or explicit_challenge_sentence_stays_problem_despite_method_like_terms"` -> `3 passed`
- Focused file: `backend/tests/test_paper_logic_trace_direct_extraction.py` -> `114 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `542 passed, 1 warning`
- Real-sample forced-local-fallback rechecks after the change:
- `282_叶片间隙对潜水搅拌器流场特性的影响` improves structurally from an almost-all-`problem` fallback trace to `problem + result + method + result`, which is materially closer to an L2 trace that can feed later layers
- `1243_Data-Driven Computational Plasticity` no longer collapses entirely into `problem`; fallback now retains `problem + limitation + method ...` structure, while the explicit challenge sentence about the constitutive model correctly remains `problem`
- Remaining macro gap after this phase:
- the structural fallback path is better, but theory-heavy fallback `method` summaries are still content-noisy (`intrusive`, `another approach`, `hardening law`) and some mixed Chinese windows still keep front-matter/keyword clutter in their summaries
- Next highest-value unresolved L2 gap: clean fallback summary/method content quality on theory-heavy and mixed-language papers without undoing the structural gains, especially for cases like `1243` and `282` where the roles are closer to correct but the extracted single-paper content is still not yet precise enough for downstream L3/L4 use.

Progress note (2026-03-30, fallback summary cleanup and method precision phase):
- Added red-green regression coverage for the next layer of fallback content-quality failures after role stabilization:
- summary cleaning now strips embedded chunk markers, `Abstract:` / `摘要：`, `Original Paper ... 2019 ...` prefixes, and leading author / affiliation / keyword noise before summary selection
- method summary selection now prefers explicit method-statement sentences over generic context sentences, including later fallback-window sentences such as `we use CFX software ...`
- method mention backfill is narrower and more content-faithful: it now rejects generic descriptor fragments (`another approach`, `simple model`), equation / law pseudo-methods, pronoun-led model fragments, and reporting/count wrappers such as `ten simulations`
- Verification:
- Focused regression file: `backend/tests/test_paper_logic_trace_direct_extraction.py` -> `128 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `556 passed, 1 warning`
- Real-sample forced-fallback rechecks:
- `282` improved materially at the content level, not just structurally:
- the abstract summary now starts with the actual Chinese content instead of author/keyword clutter
- the fallback method move now centers on the CFX simulation sentence and keeps method mentions narrowed to `cfx`
- `1607_Shear jamming and fragility in dense suspensions` now drops the earlier worst method-slot noise (`efficient method`, `simple model`, `smaller particles stokesian dynamics`) and keeps stronger method candidates such as `stokesian dynamics`, `sd approach`, and `soft-constraint approach`
- `1243_Data-Driven Computational Plasticity` likewise no longer exports the earlier worst fallback method junk (`intrusive`, `it is simply model`, `simple model`), but it still has thin formula-led / clause-fragment fallback windows
- Fresh anti-overfit fallback audit on additional papers after the change surfaced the next macro gaps more clearly:
- `1505_Velocity Profiles in Slowly Sheared Bubble Rafts` still over-promotes model-discussion clauses like `It is consistent with ...`, `solid lines are fits ...`, and `These simulations did not include ...` into `method`
- `1534_Hidden structure in liquids` still admits an equation-fragment pseudo-method (`uses Φ to decline.`) and formula-led problem clutter
- `1732_改性双基推进剂松弛模量的确定方法` still leaks front-matter article-number text into the leading problem summary and can choose broad context as `key_method_summary`
- `781` reduces to an almost-empty BOM-only fallback trace, which confirms there is still a small class of source/cleanup failures that should be dropped or neutralized rather than exported as a fake `problem`
- Next highest-value unresolved L2 gap: keep the new fallback recall/structure gains, but now target clause-fragment and front-matter contamination more directly, especially false `method` moves created from model-description / equation / reporting sentences and boilerplate Chinese header text that still pollutes single-paper readability.

Progress note (2026-03-30, weak method-clause and front-matter noise phase):
- Added a new red-green regression cluster for the next macro error class revealed by the fresh audit rather than by the earlier familiar papers alone:
- front-matter cleanup now strips article-number prefixes before summary selection, including repeated `article number` / `文章编号` style prefixes in the leading sentence
- fallback method-role heuristics are now narrower for weak clause fragments:
- sentences starting with weak method-like clauses such as `uses $...`, `it is consistent with ...`, `these simulations ...`, and `did not include ...` are no longer treated as method statements by default
- method mention backfill now drops the corresponding low-information phrase fragments instead of exporting them as pseudo-method names
- method summary scoring now gives extra preference to explicit proposal / naming cues, which helps real fallback windows choose actual method-introduction sentences over nearby method-context sentences that mainly state a problem with an existing method
- Verification:
- Targeted red-green selection for the new regressions -> `5 passed`
- Focused file: `backend/tests/test_paper_logic_trace_direct_extraction.py` -> `133 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `561 passed, 1 warning`
- Fresh forced-fallback audit recheck after the change:
- `1607_Shear jamming and fragility in dense suspensions` improved materially at the paper-summary level: `key_method_summary` now returns `we employ an algorithm to mimic stress-controlled rheology [24].` instead of the earlier weak fragment `uses a similar stress dependence;`
- `1505_Velocity Profiles in Slowly Sheared Bubble Rafts` no longer exports the earlier worst false method clauses `It is consistent with ...` and `These simulations did not include ...`; the trace regains a `problem` move and the remaining method clutter is now concentrated in later clause-fragment summaries rather than in broad reporting wrappers
- The single-sentence false positives from the new regression set now stay out of `method` completely, which confirms the fix is catching a real corpus pattern and not only a synthetic test case
- Remaining macro gap after this phase:
- some method noise is now clearly a second-order trimming issue rather than a first-order role trigger issue, for example `using a rate-controlled setup.`, `uses $\Phi$ to decline.`, `power-law model for viscosity ...`, and `highly nonlinear and not consistent ...`
- the `1732` Chinese sample still shows a broader mojibake/front-matter readability problem in real output, even though explicit article-number stripping and summary-selection regressions are now covered
- `781` still shows that a tiny class of almost-empty / BOM-dominated markdowns should probably be dropped earlier instead of becoming a fake one-move `problem` trace
- Next highest-value unresolved L2 gap: continue from weak-clause filtering into cue-trim cleanup, so fallback `method` summaries are not allowed to collapse from a longer matched sentence into a leftover clause fragment when the surrounding sentence does not actually contain a downstream-usable method statement.
