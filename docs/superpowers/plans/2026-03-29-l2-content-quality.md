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
