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

- [ ] Verify the current branch is `codex/l2-content-quality`.
- [ ] Confirm the existing full-suite evidence is fresh or rerun it if needed.
- [ ] Stage only the L2-quality files listed above.
- [ ] Commit the staged baseline with a message describing metadata cleanup plus Chinese method-role recovery.

## Chunk 2: Bilingual Summary Coherence

### Task 2: Add a failing regression for mixed-language one-paragraph summaries

**Files:**
- Modify: `backend/tests/test_paper_logic_trace_derived_views.py`

- [ ] Write a regression test where bilingual paper moves contain both Chinese and English summary candidates.
- [ ] Run the targeted pytest selection and confirm the new test fails for the current summary selector.

### Task 3: Implement minimal language-coherent summary selection

**Files:**
- Modify: `backend/app/paper_logic_trace/derived_views.py`
- Test: `backend/tests/test_paper_logic_trace_derived_views.py`

- [ ] Add lightweight language-signal helpers for summary sentences and selected move groups.
- [ ] Prefer same-language sentence bundles when building `one_paragraph_summary`, while preserving current role/quality priorities.
- [ ] Keep the fix conservative: do not rewrite move text, only change summary move selection/tie-break behavior.
- [ ] Run focused derived-view tests and confirm the new regression passes.
- [ ] Re-run fresh real-paper checks on bilingual samples and inspect the actual summary text.
- [ ] Commit the phase with a message describing bilingual summary coherence.

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

- [ ] Run `cd backend; .\.venv\Scripts\python.exe -m pytest -q`.
- [ ] Review the exact output and only then claim completion for the current execution window.
- [ ] If tests pass, record the next highest-value unresolved L2 gap and continue with another phase in the same branch.
