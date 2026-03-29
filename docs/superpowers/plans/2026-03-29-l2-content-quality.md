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
