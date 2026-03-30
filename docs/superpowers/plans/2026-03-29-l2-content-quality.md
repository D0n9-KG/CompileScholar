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

Progress note (2026-03-30, fallback method-summary calibration phase):
- Added regression coverage for three related fallback failures in `direct_extraction`:
- chunk-marker pollution such as `[c-1]` or long hashed anchor ids leaking into summary/method scoring
- prior-work author sentences like `Gadala-Maria and Acrivos [31] performed ...` outranking current-work method sentences
- background/result windows mentioning generic `simulations` or prior models being incorrectly stabilized to `method`
- Root-cause audit showed the remaining bad `1607` method summaries were not caused by `_summary_from_role_text` alone:
- one path came from chunk-marker noise creating fake method mentions (`73f44 ca 346 df simulations`)
- another path came from prior-work reporting sentences scoring as stronger methods than current-work `Here, we simulate ...`
- a third path came from fallback role stabilization treating any support-text method noun phrase as enough to promote `background/problem -> method`
- The fix stayed narrow and behavior-based:
- strip inline chunk markers before method-statement / method-mention detection
- suppress prior-work author-reporting sentences even when they appear after a lead-in clause and citation
- use full-window support only for fallback role stabilization, so later true method sentences can still rescue a sparse fallback anchor
- require stronger support-text method mentions for role promotion, so `finite element method` can still promote a current-work window while generic `simulations` / `models` in captions and background reviews do not
- Real-sample recheck after the change:
- `1607_Shear jamming and fragility in dense suspensions` no longer emits the bad method summaries `using a rate-controlled setup.` or `These SJ states can be con simulations ...`; surviving method summaries are centered on `we perform dynamic simulations ...`, `we employ an algorithm to mimic stress-controlled rheology [24].`, and `we employ a soft-constraint approach.`
- `1505_Velocity Profiles in Slowly Sheared Bubble Rafts` no longer emits the earlier fake method summary `using continuous functions of shear rate ...`; however, the direct fallback trace still remains thin on explicit method recovery, so this paper now reads as cleaner but still incomplete rather than falsely method-rich.
- `781.md` remains a separate residual class: near-empty / encoding-damaged markdown can still collapse into a fake `problem` trace (`ERROR UnicodeEncodeError ...`), so the next phase should target BOM / mojibake / empty-content guards rather than more method heuristics.
- Next highest-value unresolved L2 gap: keep improving single-paper completeness on degraded markdowns and method-thin fallback cases without reintroducing background/caption method clutter, with special attention to encoding-damaged inputs like `781` and readability-heavy Chinese papers like `1732`.

Progress note (2026-03-30, BOM-only empty-markdown guard phase):
- Added a regression for the concrete degraded-markdown case where a document contains only a UTF-8 BOM (`\ufeff`) and previously still emitted a fake fallback `problem` move.
- Root-cause check showed this was not paper content at all: `781.md` is literally a 3-byte BOM file, but `direct_extraction._normalize_space` preserved `\ufeff`, so `_semantic_windows` still treated it as non-empty text.
- Applied the narrowest fix at the L2 layer: treat BOM characters as ignorable whitespace inside `direct_extraction._normalize_space`, so BOM-only chunks collapse to empty and never form fallback windows or summaries.
- Real-sample recheck: `781.md` now yields `evidence_rows = 0` under forced local fallback extraction instead of a fake `problem` trace.
- Anti-regression check: `1607_Shear jamming and fragility in dense suspensions` kept the improved method/profile behavior from the previous phase after the BOM cleanup.
- Next highest-value unresolved L2 gap: broader mojibake / readability-heavy markdowns, especially Chinese papers like `1732`, still need a more general degraded-text handling pass beyond the BOM-only empty-file guard.

Progress note (2026-03-30, readable-Chinese front-matter and bilingual-title phase):
- Root-cause recheck on `1732_改性双基推进剂松弛模量的确定方法` showed the source markdown itself is readable, not fundamentally mojibake:
- Chinese abstract/body text are intact
- the concrete L2 issues were instead a leaked front-matter chunk (`文章编号...`) and a bad `title_alt` choice (`1.3 Sorvari法`) caused by heading ranking
- Added regression coverage for two narrow failures:
- a leading article-number metadata line with `section=None` should be filtered before fallback move construction
- a bilingual title page should prefer the real cross-language title as `title_alt` instead of a later numbered section heading
- Applied two conservative fixes:
- in `direct_extraction`, sectionless lines matching explicit front-matter metadata cues now count as noise even outside the title block
- in `parse_md`, `title_alt` now prefers a non-numbered bilingual counterpart heading before falling back to the next raw heading-score candidate
- Real-sample recheck after the change:
- `1732` no longer emits the fake `problem` summary based on `文章编号：1000-4750(2012)09-0359-04`
- `1732` now keeps `title='DETERMINATION WAY OF RELAXATION MODULUS OF MODIFIED DB PROPELLANT'` and `title_alt='改性双基推进剂松弛模量的确定方法'`, which is materially more faithful than the previous `title_alt='1.3 Sorvari法'`
- Residual gap after this phase: `1732` is cleaner and metadata-aligned, but its fallback role structure still skews `method/result` with a thin explicit `problem` layer, so the next phase should target problem recovery/compression for method-heavy bilingual abstracts rather than more front-matter cleanup.
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

Progress note (2026-03-30, method cue-trim boundary phase):
- Root-cause audit on the remaining real-sample method clutter found a deeper failure mode than the previous phase:
- `_trim_to_first_method_cue(...)` and method-statement detection were still using bare substring hits for literal cues such as `uses `, which meant ordinary words like `causes` could falsely trigger method trimming and produce clause fragments such as `uses a similar stress dependence` or `uses $\Phi$ to decline`
- in mixed prior-work/current-work sentences, earlier literature context like `performed ... using a rate-controlled setup` could still outrank the actual current-work sentence because the method-cue logic recognized the `using ...` clause but did not recognize nearby current-work sentences like `we simulate ...`
- Reworked the method-cue logic conservatively rather than broadening generic extraction:
- English literal method cues now use word-boundary-aware matching instead of raw substring search, so `causes` no longer contains a fake `uses` cue
- explicit current-work cues such as `we simulate`, `we perform`, and `we apply` are now recognized as method statements, helping the selector prefer the paper's own method sentence over prior-work `using ...` references
- prior-work author-name sentences such as `X and Y performed ...` are now treated as weak method statements for fallback selection instead of being promoted as the paper's own method
- Verification:
- Targeted red-green selection for the new boundary/trim regressions -> `3 passed`
- Focused file: `backend/tests/test_paper_logic_trace_direct_extraction.py` -> `136 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `564 passed, 1 warning`
- Fresh forced-fallback audit recheck after the change:
- `1534_Hidden structure in liquids` no longer exports the earlier fake method move `uses $\Phi$ to decline.`; its trace now stays `problem + limitation` instead of inventing a method from the word `causes`
- `1607_Shear jamming and fragility in dense suspensions` keeps the improved `key_method_summary` and closes the earlier `causes ... stress dependence` cue-trim failure mode, though it still has a smaller residual clause-fragment tail later in the trace
- `1505_Velocity Profiles in Slowly Sheared Bubble Rafts` remains yellow but the residual clutter is now narrower and more interpretable; the next cleanup no longer needs to fight raw `causes/uses` false positives
- Remaining macro gap after this phase:
- there is still a smaller cluster of residual method clutter that comes from long matched sentences whose surviving clause is not quite wrong enough to be caught by the current weak-statement filters, for example `using a rate-controlled setup.`, `power-law model for viscosity ...`, and `highly nonlinear and not consistent ...`
- the `1732` Chinese sample still points to a broader mojibake/front-matter readability issue, and `781` still points to nearly-empty markdown handling
- Next highest-value unresolved L2 gap: tighten second-order clause-fragment cleanup and figure/caption-style method pollution without undoing the recall gains from the last several fallback phases.

Progress note (2026-03-30, paper-content profile prioritization phase):
- Shifted this phase from raw move extraction into `derived_views.paper_content_profile`, because the next repeated corpus-level gap was no longer missing slots but single-paper story drift: the exported `problem_statements` / `method_statements` lists could still surface prior-work method review or broad background/setup sentences ahead of the paper's own main content.
- Added red-green regression coverage for two concrete downstream-facing failures:
- `method_statements` should prefer the paper's own method/experiment narrative over prior-work review when enough current-paper method content is already present
- `problem_statements` should prefer genuine `problem` moves over earlier `background` setup context when the three-statement budget is already filled by actual problem/task content
- Reworked `build_paper_content_profile(...)` conservatively rather than changing canonical move extraction:
- `_role_summaries(...)` now ranks candidate summaries by role-aware content score instead of pure sequence order
- method-summary ranking now penalizes explicit prior-work cues (`previous work`, `prior work`, `reported by`, `et al`, etc.) and prefers current-paper method signals
- when at least two non-prior-work method summaries are already available, prior-work method-review summaries are no longer allowed to occupy the limited `method_statements` budget
- author-fragment detection and summary-content checks are now CJK-safe, so short but complete Chinese method summaries are not discarded just because they lack English-style whitespace tokenization
- Verification:
- Red-green regressions: `backend/tests/test_paper_logic_trace_derived_views.py -k "prefers_current_work_method_statements_over_prior_work_review or prefers_problem_statements_over_background_context_when_limit_reached"` -> `2 passed`
- Focused file: `backend/tests/test_paper_logic_trace_derived_views.py` -> `25 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `574 passed, 1 warning`
- Real-sample rechecks on stored current traces after the change:
- `1732_改性双基推进剂松弛模量的确定方法` now exports only current-paper method content in `paper_content_profile.method_statements`: the improved Sorvari method description, the Chinese method-introduction sentence, and the Chinese tensile-relaxation experiment sentence. The earlier prior-work summary (`Previous work by Zapas ...`) no longer displaces the paper's own method story.
- `1243_Data-Driven Computational Plasticity` keeps the meaningful challenge/problem sentence in `problem_statements` while dropping the earlier generic finite-element background sentence from the top-three content profile.
- `1505_Velocity Profiles in Slowly Sheared Bubble Rafts` now prioritizes the two actual problem/gap statements before any broader setup context, which is a better single-paper summary surface even though one background/setup sentence still remains as the third fallback item.
- Remaining macro gap after this phase:
- some canonical `method` moves in real papers are still semantically broader than ideal, for example `1607` still keeps general setup/overview method summaries such as `The authors describe their simulation system ...` alongside stronger current-work method sentences
- `problem_statements` can still retain one broad setup/background sentence as the final fallback item when a paper genuinely has only one or two explicit problem/task moves
- Next highest-value unresolved L2 gap: continue from content-profile prioritization into canonical move-level cleanup for broad setup/prior-work method sentences, so L2 itself becomes cleaner rather than relying on derived-view ranking to hide the remaining noise.

Progress note (2026-03-30, prior-work method-signal suppression phase):
- Followed the next gap from the previous phase into the L3 bridge itself: even after `paper_content_profile.method_statements` was cleaned up, prior-work review moves could still leak trusted method mentions into `route_compiler_contract.topic_signals.methods` and then into `route_state_seed.dominant_method_candidates`.
- Added a red-green regression for the concrete bilingual-method case where current-paper method evidence is already sufficient (`improved Sorvari method`, `tensile relaxation test`, `numerical iteration`), but a prior-work review move still exports `sorvari method` into L3 method candidates.
- Reworked method-signal compilation conservatively rather than changing canonical moves:
- `build_route_compiler_contract(...)` now filters prior-work method entries only when there are already at least two non-prior-work method labels available in the same paper
- the filter uses the same summary-level current-work / prior-work cue family added in the previous phase, so it removes review-derived method entries without dropping sparse papers that only have literature-context method mentions
- Verification:
- Red-green regression: `backend/tests/test_paper_logic_trace_derived_views.py -k "excludes_prior_work_method_signal_when_current_method_evidence_is_sufficient"` -> `1 passed`
- Focused file: `backend/tests/test_paper_logic_trace_derived_views.py` -> `26 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `575 passed, 1 warning`
- Real-sample rechecks on stored current traces after the change:
- `1732_改性双基推进剂松弛模量的确定方法` no longer exports `sorvari method` from the prior-work review move into `route_compiler_contract.topic_signals.methods` or `route_state_seed.dominant_method_candidates`; the surviving method candidates are now centered on `改进型sorvari法`, `拉伸松弛试验`, `improved sorvari method`, and `numerical iteration`
- `1607_Shear jamming and fragility in dense suspensions` remains effectively unchanged under the same filter, confirming the phase is narrow: its L3 method candidates still center on stress-controlled rheology, Stokesian dynamics, and the soft-constraint / governing-equation mechanics of the actual paper rather than losing legitimate method content
- Remaining macro gap after this phase:
- some current-paper method signals are still too broad or low-yield at the canonical/contract level, for example `1607` still exports generic mechanics/setup labels such as `force and torque balance equations`, `linear resistance`, and `brownian simulations` alongside stronger method tokens
- some theory-style papers such as `1243` still surface framework/mechanics labels that are individually grounded but not yet compressed into the most downstream-useful method vocabulary
- Next highest-value unresolved L2 gap: tighten method-signal ranking or filtering for broad setup/framework labels that are technically correct but still less useful than the paper's core operative method, so L3 receives a shorter and more decision-useful method maturity picture per paper.

Progress note (2026-03-30, operative method ranking phase):
- Continued the cleanup one layer deeper into `route_state_seed`: after prior-work method review was suppressed, the next repeated gap was that `dominant_method_candidates` still used near-insertion order, so broad framework/setup labels could crowd out more operative method signals in papers with dense method sections.
- Added a paired regression set for this next macro issue:
- when operative current-paper methods are available, `dominant_method_candidates` should prioritize them above broad framework labels such as `force and torque balance equations` / `linear resistance`
- when a theory-style paper only has a framework-level grounded method signal, that framework label should still be preserved rather than dropped wholesale
- Implemented the fix narrowly inside `build_route_state_seed(...)`:
- kept `route_compiler_contract.topic_signals.methods` as the fuller trusted method-entry pool
- added a dedicated `_rank_method_signal_entries(...)` path only for route-seed method candidate selection
- the new ranking prefers operative summaries and operative method labels (`algorithm`, `approach`, `test`, `simulation`, `iteration`, etc.), penalizes broad summary contexts (`governing equations`, `simulation model considers`, `parameters are selected`, etc.), and lightly penalizes generic framework labels such as `equations`, `laws`, `resistance`, and `regularization`
- Verification:
- Red-green regressions: `backend/tests/test_paper_logic_trace_derived_views.py -k "prioritizes_operational_methods_over_broad_framework_labels or keeps_framework_method_when_it_is_the_only_grounded_method_signal"` -> `2 passed`
- Focused file: `backend/tests/test_paper_logic_trace_derived_views.py` -> `28 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `577 passed, 1 warning`
- Real-sample rechecks after the change:
- `1607_Shear jamming and fragility in dense suspensions` no longer lets framework labels such as `force and torque balance equations` and `linear resistance` crowd out its route-seed top methods; the top-five candidate list now centers on operational method content (`determine shear rate from given shear stress`, `determine particle velocities from shear rate`, `stress-controlled rheology algorithm`, `stokesian dynamics`, `soft-constraint approach`)
- `1732_改性双基推进剂松弛模量的确定方法` stays healthy under the same ranking and is further compressed toward operative method content: `numerical iteration`, `improved sorvari method`, `改进型sorvari法`, `拉伸松弛试验`
- `1243_Data-Driven Computational Plasticity` still retains `continuum-thermodynamics framework`, confirming the phase is not deleting framework methods when they are genuinely part of the paper's main method story
- Additional non-overfit spot checks on previously stored random audit traces stayed directionally healthy rather than collapsing method diversity:
- CFD / propulsion / DEM / neural-network samples continued to surface domain-appropriate operative methods such as `mrf method`, `gabp neural network model`, and `discrete element method`
- Remaining macro gap after this phase:
- some route-seed method candidates are now better ranked but still too clause-like or solver-step-like, for example `determine shear rate from given shear stress` and `determine particle velocities from shear rate` can outrank shorter canonical method names in `1607`
- Next highest-value unresolved L2 gap: compress or down-rank verbose clause-style method labels when shorter canonical method names from the same paper already exist, so route-state method maturity reflects a concise core method vocabulary rather than implementation-step phrasing.

Progress note (2026-03-30, clause-style method label deprioritization phase):
- Continued the route-seed method cleanup into a more semantic ranking issue: after framework/setup labels were pushed down, some verbose clause-style method labels (`determine ... from ...`) could still outrank shorter canonical method names even when the same paper already had cleaner method names such as `stress-controlled rheology algorithm`, `stokesian dynamics`, or `soft-constraint approach`.
- Added a paired regression set for this next gap:
- clause-style method labels should rank below shorter named methods when those named methods already exist in the same paper
- clause-style labels should still be preserved when they are the only grounded method signal available
- Reworked `_rank_method_signal_entries(...)` narrowly:
- added a label-level penalty for verb-led clause-style method labels such as `determine ...`, `predict ...`, `calculate ...`, `derive ...`, and similar implementation-step phrasing
- kept the penalty local to route-seed method ranking, so the underlying canonical/contract method evidence remains available rather than being deleted
- Verification:
- Red-green regressions: `backend/tests/test_paper_logic_trace_derived_views.py -k "deprioritizes_clause_style_method_labels_when_named_methods_exist or keeps_clause_style_method_label_when_it_is_the_only_grounded_method_signal"` -> `2 passed`
- Focused file: `backend/tests/test_paper_logic_trace_derived_views.py` -> `30 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `579 passed, 1 warning`
- Real-sample rechecks after the change:
- `1607_Shear jamming and fragility in dense suspensions` now ranks concise core method names first in `dominant_method_candidates`: `stress-controlled rheology algorithm`, `stokesian dynamics`, `soft-constraint approach`, `harmonic penalty function`, with the more verbose clause label `determine shear rate from given shear stress` pushed later
- `1732_改性双基推进剂松弛模量的确定方法` remains healthy under the same change, still centered on `numerical iteration`, `improved sorvari method`, `改进型sorvari法`, and `拉伸松弛试验`
- `1243_Data-Driven Computational Plasticity` remains stable, keeping framework-level method vocabulary because that is still a meaningful part of the paper's core method story rather than a clause-label artifact
- Additional random spot checks on stored audit traces stayed directionally healthy and did not collapse useful domain methods in CFD / DEM / neural-network samples
- Remaining macro gap after this phase:
- some route-seed method candidates are now cleaner but still somewhat over-complete, for example `1607` still includes secondary method internals such as `harmonic penalty function`, and some other random traces still keep mixed method-step phrases like `cross-experiment comparison within geometry` or `controlled flow rate via aperture or moving wall`
- Next highest-value unresolved L2 gap: refine method candidate compression one level further, so route-state method maturity emphasizes the paper's shortest stable core method vocabulary and demotes secondary implementation details and analysis-step phrases when stronger method names are already present.

Progress note (2026-03-30, move-aware method candidate compression phase):
- Root-cause audit on the next residual `route_state_seed` gap showed the remaining issue was no longer only label ranking. Even after clause-style penalties, same-move secondary labels could still crowd out other moves' core methods because route-seed selection ranked the full method pool globally and then truncated.
- Added a stronger regression for this selection-layer failure mode:
- when a paper already has named methods from multiple moves, route-seed method selection should surface one core non-secondary label per move before it spends slots on same-move detail labels
- clause-style labels should still remain available when they are the only grounded method evidence rather than being deleted entirely
- Implemented the fix narrowly inside `build_route_state_seed(...)`:
- kept the fuller ranked method-entry pool unchanged
- added `_is_secondary_method_detail_entry(...)` to conservatively identify clause-style / analysis-step / implementation-detail method labels for route-seed compression only
- added `_method_candidate_labels(...)` so route-state selection now runs in three passes: first non-secondary label per move, then remaining non-secondary labels, then secondary/detail labels only if there is still room
- reused this compressed label set for both `dominant_method_candidates` and `readiness_feature_inputs.method_maturity_signals`, keeping downstream readiness aligned with the cleaned route-seed view
- Verification:
- Targeted regressions: `backend/tests/test_paper_logic_trace_derived_views.py -k "deprioritizes_clause_style_method_labels_when_named_methods_exist or keeps_clause_style_method_label_when_it_is_the_only_grounded_method_signal"` -> `2 passed`
- Focused file: `backend/tests/test_paper_logic_trace_derived_views.py` -> `30 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `579 passed, 1 warning`
- Real-sample rechecks after the change:
- `1607_Shear jamming and fragility in dense suspensions` now surfaces cross-move core methods earlier in `dominant_method_candidates`, with `stress-controlled shear reversal test` promoted ahead of the old clause label tail and the top list centered on `stress-controlled rheology algorithm`, `soft-constraint approach`, `stress-controlled shear reversal test`, and `stokesian dynamics`
- `1732` stays healthy and concise under the same compression, still centered on `numerical iteration`, `改进型sorvari法`, `拉伸松弛试验`, and `improved sorvari method`
- `1243_Data-Driven Computational Plasticity` remains stable and still preserves framework-level method vocabulary when that is genuinely the paper's main method story rather than detail noise
- Additional non-overfit spot checks on stored random audit traces stayed directionally healthy: the phase did not collapse domain-specific method diversity in neural-network / CFD / DEM / photocatalysis samples, though some older traces still expose broader context-heavy method labels
- Remaining macro gap after this phase:
- route-seed method candidates are now better distributed across moves, but some papers still keep context or implementation-detail phrases such as `harmonic penalty function`, `controlled flow rate via aperture or moving wall`, `silo flow`, or `gravity-driven flow` in the compressed top list when the underlying method extraction itself is broad
- Next highest-value unresolved L2 gap: continue compressing route-seed methods toward the shortest stable core method vocabulary without overfitting, especially by demoting context/setup labels that are still treated as non-secondary in mixed empirical papers.

Progress note (2026-03-30, setup-context method filler compression phase):
- Follow-up audit on the new route-seed outputs showed a second residual selection-layer problem after the cross-move compression landed:
- papers with enough core methods could still spend top-method slots on setup/context filler labels such as `controlled flow rate via aperture or moving wall`, `silo flow`, `gravity-driven flow`, or `simple shear flow assumption`
- sparse papers with only contextual method evidence still needed to keep one grounded label rather than collapsing to an empty method list
- Added a paired regression set for this next gap:
- when core methods already exist, setup/context filler labels should not back-fill `dominant_method_candidates`
- when a setup/context label is the only grounded method signal, the best grounded label should still be preserved
- Reworked route-seed method compression narrowly inside `derived_views.py`:
- introduced `_is_contextual_method_filler_entry(...)` to distinguish contextual/setup method labels from the more method-relevant secondary detail labels already handled in the previous phase
- treated contextual fillers as secondary during route-seed selection, so they no longer enter the first-pass / second-pass primary method pool
- capped secondary-detail backfill at a shorter core-method target (`<= 4` labels) and only fall back to a single contextual label when no stronger method candidate is available
- kept all changes local to route-state method candidate selection and readiness inputs; canonical move evidence and the broader method-entry pool remain intact
- Verification:
- Targeted regressions: `backend/tests/test_paper_logic_trace_derived_views.py -k "deprioritizes_clause_style_method_labels_when_named_methods_exist or omits_setup_context_method_fillers_when_core_methods_exist or keeps_setup_context_method_label_when_it_is_the_only_grounded_signal"` -> `3 passed`
- Focused file: `backend/tests/test_paper_logic_trace_derived_views.py` -> `32 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `581 passed, 1 warning`
- Real-sample rechecks after the change:
- `1607_Shear jamming and fragility in dense suspensions` now compresses to four core route-seed methods: `stress-controlled rheology algorithm`, `soft-constraint approach`, `stress-controlled shear reversal test`, and `stokesian dynamics`, dropping the earlier secondary tail
- `1732` remains healthy and concise with `numerical iteration`, `改进型sorvari法`, `拉伸松弛试验`, and `improved sorvari method`
- `1243_Data-Driven Computational Plasticity` remains stable and still preserves framework-level methods because those are part of the paper's actual method story
- `04_1901_On dense granular flows` now compresses to `prandtl mixing length approach` plus `local rheology described by a friction μ(i)` instead of back-filling setup/context phrases
- Additional non-overfit spot checks stayed directionally healthy:
- `02_1416_Young-Dupre Revisited` still keeps `adsorption isotherm analysis`
- neural-network and photocatalysis samples still keep domain-specific methods such as `genetic algorithm backpropagation (gabp) neural network model` and `quantum mechanics in explicit solvent`
- Remaining macro gap after this phase:
- some route-seed method labels are now shorter and cleaner, but old or degraded traces can still surface renamed clause-style labels such as `stress-controlled flow determination`, and paper-level `key_method_summary` selection can still drift toward governing-equation/setup prose even when route-seed methods are already good
- Next highest-value unresolved L2 gap: continue tightening canonical method-label compression and paper-level method-summary selection so L2 not only picks the right method slots, but also phrases the single-paper method story in the shortest accurate form for downstream L3/L4 use.

Progress note (2026-03-30, operative method-summary preference phase):
- The next L2 readability gap was no longer mainly in route-state candidates; it had shifted into `paper_summaries.key_method_summary`.
- Root-cause audit on real `1607` showed that the summary selector could still overvalue a broad governing-equation method sentence simply because it carried many trusted method labels, even when a neighboring move contained a much clearer operative current-work method sentence (`employ an algorithm to mimic stress-controlled rheology ...`).
- Added a regression for this summary-selection failure mode:
- when a paper contains both a broad governing-equation/context sentence and a more operative current-work method sentence, `key_method_summary` should choose the operative method sentence
- Reworked `_method_focus_score(...)` narrowly:
- replaced the old raw `len(trusted_methods)` reward with a quality-aware count that distinguishes primary method mentions from secondary detail labels and contextual fillers
- added a penalty for broad method-summary cues already used in route-seed cleanup (`governing equations`, `simulation model considers`, `initial configuration generation method`, etc.)
- lightly rewarded explicit current-work phrasing and penalized prior-work-only method summaries, keeping key-method selection aligned with the single-paper method story rather than generic context
- Verification:
- Targeted regressions:
- `backend/tests/test_paper_logic_trace_derived_views.py -k "prefers_operative_method_over_governing_equation_context or prefers_method_focused_sentence_over_outcome_colored_sentence or prefers_content_moves_over_author_line_noise"` -> `3 passed`
- Focused file: `backend/tests/test_paper_logic_trace_derived_views.py` -> `33 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `583 passed, 1 warning`
- Real-sample rechecks after the change:
- `1607_Shear jamming and fragility in dense suspensions` no longer uses the old governing-equation sentence as `key_method_summary`; it now surfaces the paper's actual operative method story centered on the soft-constraint / stress-controlled simulation pipeline
- `1732` stays strong and still summarizes the tensile relaxation experiment plus improved Sorvari method coherently
- `1243_Data-Driven Computational Plasticity` still uses the `LaTIn` method sentence as its key method summary
- `s870` remains healthy, still centered on `Quantum Mechanics (QM) in explicit solvent`
- Remaining macro gap after this phase:
- some setup-heavy empirical traces, such as `04_1901_On dense granular flows`, can still let a high-priority setup/stress-distribution sentence outrank the cleaner grounded method sentence (`Prandtl mixing length approach`) because the selector still considers method-role sentences that carry no grounded method mentions when their section score is high
- Next highest-value unresolved L2 gap: continue tightening `key_method_summary` so method-role setup/condition sentences without grounded method evidence no longer outrank real method sentences in setup-heavy papers, while still preserving a fallback summary when extraction is sparse.

Progress note (2026-03-30, grounded key-method summary gating phase):
- Root-cause audit on `04_1901_On dense granular flows` showed the remaining `key_method_summary` failure was narrower than general summary scoring:
- once route-seed methods were already clean, the summary selector could still choose a setup/condition sentence like `Describes the stress distribution in the experimental geometry ...` simply because it lived in a method-heavy section and carried grounded `conditions` / `research_objects`, even though another move in the same paper contained the actual grounded method sentence `The paper proposes a Prandtl mixing length approach ...`
- Added a regression for this exact failure mode:
- when a trace contains at least one method/experiment move with grounded trusted `methods`, `key_method_summary` should be selected from those grounded method moves rather than from setup/condition sentences that lack any grounded method mention
- Implemented the fix narrowly in `_select_key_method_move(...)`:
- preserved the existing summary-ranking logic
- added a conservative gating step so, if any contentful method/experiment move carries trusted `methods`, the selector first narrows candidates to that grounded subset
- kept the old behavior as a fallback when extraction is sparse and no grounded method mentions are available anywhere in the trace
- Verification:
- Targeted regressions:
- `backend/tests/test_paper_logic_trace_derived_views.py -k "prefers_grounded_method_move_over_setup_condition_sentence or prefers_operative_method_over_governing_equation_context or prefers_method_focused_sentence_over_outcome_colored_sentence"` -> `3 passed`
- Focused file: `backend/tests/test_paper_logic_trace_derived_views.py` -> `34 passed`
- Full backend suite: `cd backend; .\.venv\Scripts\python.exe -m pytest -q` -> `583 passed, 1 warning`
- Real-sample rechecks after the change:
- `04_1901_On dense granular flows` now uses the grounded method sentence about the `Prandtl mixing length approach` as `key_method_summary` instead of the earlier setup/stress-distribution sentence
- `1607`, `1732`, `1243`, and `s870` remain stable under the same change, keeping operative or framework-level method summaries that are still faithful to their paper-level method story
- Remaining macro gap after this phase:
- route-state method candidates and key-method summaries are now materially cleaner, but some traces still keep method labels that are accurate yet not maximally canonical, such as renamed clause-style variants (`stress-controlled flow determination`) or parallel near-duplicates (`latin` / `latin method`)
- Next highest-value unresolved L2 gap: continue tightening canonical method-label normalization and de-dup compression so L2 presents the shortest stable per-paper method vocabulary while staying faithful enough to support downstream L3/L4 synthesis.
