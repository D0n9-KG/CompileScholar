# Stage 4 Conflict Detection - Review Findings

**Review Date:** 2026-02-15
**Reviewer:** Codex
**Status:** ✅ Functionally Complete, ⚠️ Not Fully Production-Ready

---

## Summary

Stage 4 (Conflict Detection Enhancements) is functionally complete with all planned features implemented and tested. However, two architectural issues should be addressed before production deployment.

**Verdict**: "Ready for final regression/review tasks, not ready for deployment sign-off."

---

## Implementation Completed ✅

### Task 4.1-4.3: Batch Processing + JSON Repair
- ✅ Small-batch processing (15 pairs, configurable 5-50)
- ✅ Retry mechanism (3 attempts, exponential backoff)
- ✅ Graceful degradation (mark as 'insufficient' on failure)
- ✅ json-repair dependency added

### Task 4.4: Metrics Logging
- ✅ Comprehensive semantic metrics (6 new fields)
- ✅ Defensive handling (malformed labels/scores, unknown pair IDs)
- ✅ Structured logging for observability
- ✅ Fallback path metrics

### Task 4.5: Testing
- ✅ 6 robustness tests (all passed in 145.81s)
- ✅ Coverage: basic, large batch, empty, configuration, limits

---

## Strengths

1. **Batch Processing**: Correctly implemented in `conflict_judge.py:166-220`
2. **Metrics Expansion**: Solid implementation in `orchestrator.py` with all 6 new metrics
3. **Defensive Coding**: Proper filtering/normalization for unknown pair IDs, malformed labels/scores
4. **Structured Logging**: Comprehensive observability with `logger.info()` at line 1184

---

## Production Blockers (Before Deployment)

### Blocker 1: JSON Repair Not Effectively Used

**Severity:** High (Production Blocker)
**Status:** 🔴 Open

**Description:**
`json-repair` dependency is imported but never called in runtime path. `_repair_and_parse()` exists (line 36) but is not invoked. JSON errors fall back to marking batch as 'insufficient' (line 137) without attempting repair.

**Impact:**
- JSON repair capability advertised but not functional
- Lost opportunity to recover from malformed LLM responses
- Higher 'insufficient' rate than necessary

**Location:**
- `backend/app/llm/conflict_judge.py:36-46` (_repair_and_parse unused)
- `backend/app/llm/conflict_judge.py:137-145` (fallback without repair)

**Root Cause:**
`call_json()` in `llm/client.py` already parses JSON internally and doesn't expose raw malformed text. To use repair, we'd need to modify `call_json()` to return raw response on parse failure, or catch the error earlier.

**Fix Required:**
- Option A: Modify `call_json()` to return raw response dict on JSONDecodeError
- Option B: Add separate `call_json_raw()` function that returns unparsed text
- Option C: Catch JSONDecodeError in client and attempt repair before re-raising

**Owner:** TBD
**Target:** Before production deployment

---

### Blocker 2: Retry Amplification

**Severity:** High (Production Blocker)
**Status:** 🔴 Open

**Description:**
Multiple retry layers compound latency and cost:
- Outer retries in `_judge_single_batch()`: 3 attempts (line 105)
- Inner retries in `call_json()`: tenacity decorator (client.py:46)
- Model-level retries: model client (client.py:42)

**Impact:**
- Worst-case: 3 * 3 * N = 9N+ attempts for persistent failures
- Extended latency under error conditions
- Multiplied API costs for failing requests
- SLO/timeout risk

**Location:**
- `backend/app/llm/conflict_judge.py:105` (outer retry loop)
- `backend/app/llm/client.py:46` (tenacity retry)
- Model client retries (varies by provider)

**Root Cause:**
No coordination between retry layers. Each layer independently retries without awareness of outer/inner retry state.

**Fix Required:**
- Option A: Remove outer retry in judge, rely only on `call_json()` retry
- Option B: Make `call_json()` configurable (pass `retry=False` from judge)
- Option C: Add retry budget tracking (shared counter across layers)

**Owner:** TBD
**Target:** Before production deployment

---

## Non-Blocking Concerns

### NC-1: Test Determinism

**Severity:** Medium
**Status:** 🟡 Tracked

**Description:**
Robustness tests are integration-style with unmocked LLM calls, making them slow (145s) and potentially flaky for CI.

**Impact:** Slow CI feedback, potential test flakiness

**Fix:** Mock LLM responses in tests for determinism and speed

---

### NC-2: Schema Exposure Gap

**Severity:** Low
**Status:** 🟡 Tracked

**Description:**
`phase2_conflict_batch_size` is used in judge but not surfaced in schema defaults/validation/presets.

**Impact:** Users cannot easily configure batch size without reading code

**Fix:** Add to `schema_store.py` and `schema_presets.py`

---

### NC-3: Fallback Metric Consistency

**Severity:** Low
**Status:** 🟡 Tracked

**Description:**
Lexical fallback `comparable_pairs` may differ from semantic candidate count due to different filtering logic.

**Impact:** Minor metric inconsistency in fallback scenarios

**Fix:** Align lexical and semantic candidate generation logic

---

## Codex Recommendation

**Proceed to Final Tasks:**
- ✅ Final.1: Run Regression Test
- ✅ Final.2: Final Codex Review
- ⚠️ Final.3: Create Deployment Summary (conditional on Blocker 1 & 2 fixes)

**Do NOT deploy to production until:**
1. Blocker 1 (JSON repair) resolved
2. Blocker 2 (retry amplification) resolved

---

## Next Steps

1. Proceed with Final.1 (regression test) to validate E2E integration
2. Document Blocker 1 & 2 in deployment summary
3. Prioritize blockers for post-P1 follow-up
4. Consider non-blocking concerns for P2 optimization

---

**Review Conducted By:** Codex (MCP Session 019c5f3b-322a-7f23-8bfa-dcd8309a4617)
**Files Reviewed:**
- `backend/app/llm/conflict_judge.py`
- `backend/app/extraction/orchestrator.py`
- `backend/tests/test_conflict_robustness.py`
- `backend/requirements.txt`

**Commits Reviewed:**
- 555bc83: feat(conflict): add comprehensive semantic detection metrics logging
- 4dfba67: test(conflict): add robustness tests for batch processing and metrics
