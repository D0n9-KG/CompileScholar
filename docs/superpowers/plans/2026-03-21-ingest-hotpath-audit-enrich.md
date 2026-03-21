# Ingest Hot Path / Audit-Enrich Split Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce ingest time-to-first-usable-paper by counting a paper as complete after core logic/claim extraction and deferring slower enrichment work.

**Architecture:** Split the current per-paper ingest flow into a synchronous `FastIngest` core and a non-blocking `AuditEnrich` tail. Keep core outputs and Neo4j writes compatible while moving citation-purpose classification and completion accounting off the hot path.

**Tech Stack:** FastAPI, Python 3.11, existing ingest pipeline, pytest

---

## File Structure

### Modify

- `backend/app/ingest/pipeline.py`
- `backend/app/llm/paper_type_classifier.py`
- `backend/tests/test_ingest_llm_progress.py`
- `backend/tests/test_pipeline_gate.py`
- `backend/tests/test_citation_purpose.py`

### Optional Modify

- `backend/app/extraction/orchestrator.py`

---

## Chunk 1: Progress and Completion Boundary

### Task 1: Add failing test for core completion semantics

**Files:**
- Modify: `backend/tests/test_ingest_llm_progress.py`
- Modify: `backend/app/ingest/pipeline.py`

- [ ] **Step 1: Write the failing test**

Add a test that models a paper whose core extraction succeeds but whose citation-purpose enrichment is deferred, and assert that ingest progress counts that paper as completed.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q backend/tests/test_ingest_llm_progress.py`

- [ ] **Step 3: Implement minimal progress boundary change**

Refactor the per-paper ingest result so `_llm_extract_one()` reports a core-complete result before enrichment completion is required for progress accounting.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q backend/tests/test_ingest_llm_progress.py`

- [ ] **Step 5: Commit**

Commit message: `refactor: count papers complete after core extraction`

## Chunk 2: Move Citation Purpose out of the Hot Path

### Task 2: Add failing test for deferred citation-purpose enrichment

**Files:**
- Modify: `backend/tests/test_citation_purpose.py`
- Modify: `backend/app/ingest/pipeline.py`

- [ ] **Step 1: Write the failing test**

Add a test that verifies citation-purpose classification can run as a separate enrichment phase without blocking the core paper result.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q backend/tests/test_citation_purpose.py`

- [ ] **Step 3: Implement minimal extraction split**

Extract the citation-purpose call path into a helper used after core extraction. Ensure failures in this phase do not prevent the paper from being counted as extracted.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q backend/tests/test_citation_purpose.py`

- [ ] **Step 5: Commit**

Commit message: `refactor: defer citation purpose enrichment`

## Chunk 3: Reduce Avoidable LLM Work Before Core Extraction

### Task 3: Add failing test for cheaper paper-type resolution

**Files:**
- Modify: `backend/tests/test_pipeline_gate.py`
- Modify: `backend/app/llm/paper_type_classifier.py`
- Modify: `backend/app/ingest/pipeline.py`

- [ ] **Step 1: Write the failing test**

Add a test that verifies obvious or metadata-backed paper types resolve without unnecessary LLM classification.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q backend/tests/test_pipeline_gate.py`

- [ ] **Step 3: Implement minimal resolver shortcut**

Update paper-type resolution so valid metadata and clear rule-based outcomes avoid the LLM call.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q backend/tests/test_pipeline_gate.py`

- [ ] **Step 5: Commit**

Commit message: `perf: avoid unnecessary paper type llm calls`

## Chunk 4: Final Verification

### Task 4: Run focused regression suite

**Files:**
- No code changes expected

- [ ] **Step 1: Run focused ingest regressions**

Run: `python -m pytest -q backend/tests/test_ingest_llm_progress.py backend/tests/test_pipeline_gate.py backend/tests/test_citation_purpose.py backend/tests/test_citation_purpose_rules.py backend/tests/test_ingest_pipeline_reingest_idempotency.py`

- [ ] **Step 2: Inspect git diff and confirm scope**

Run: `git diff --stat`

- [ ] **Step 3: Commit any final adjustments**

Commit message: `test: cover ingest hot path split`
