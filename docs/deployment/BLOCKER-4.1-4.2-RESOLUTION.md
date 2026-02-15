# Blocker 4.1 & 4.2 Resolution Report

**Date:** 2026-02-15
**Status:** ✅ RESOLVED (Production Ready)
**Blockers:**
- Stage 4 Blocker 4.1 - JSON repair not effectively used
- Stage 4 Blocker 4.2 - Retry amplification

---

## Problem Summary

### Blocker 4.1: JSON Repair Not Effectively Used

**Original Issue:**
- `json-repair` dependency imported but never called
- `_repair_and_parse()` function existed but was unused
- Malformed LLM responses treated as complete failures
- Higher "insufficient" judgment rate than necessary

**Root Cause:**
- `call_json()` in `llm/client.py` parsed JSON internally
- Didn't expose raw malformed text for repair attempt
- JSON errors fell back to marking entire batch as "insufficient"

### Blocker 4.2: Retry Amplification

**Original Issue:**
- Multiple nested retry layers compounded latency and cost
- Outer retry loop in `_judge_single_batch()`: 3 attempts
- Inner retry in `call_json()`: tenacity decorator (3 attempts)
- Model client retries: ChatOpenAI max_retries=2
- **Worst case:** 3 × 3 × 2 = 18 attempts for persistent failures

**Impact:**
- Extended latency under error conditions
- Multiplied API costs for failing requests
- SLO/timeout risk

---

## Solution Implemented

**Commits:**
- `0904c54` - fix(conflict): resolve Blocker 4.1 & 4.2

### Fix 4.1: Effective JSON Repair

**Changes to `backend/app/llm/client.py`:**
```python
# Added new function to expose raw LLM response
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.8, min=0.8, max=4.0))
def _call_text_with_retry(system: str, user: str) -> str:
    resp = llm().invoke([("system", system), ("user", user)])
    return str(resp.content or "")

def call_text(system: str, user: str, *, use_retry: bool = True) -> str:
    if use_retry:
        return _call_text_with_retry(system, user)
    resp = llm().invoke([("system", system), ("user", user)])
    return str(resp.content or "")

# Refactored call_json to use call_text
def call_json(system: str, user: str, *, use_retry: bool = True) -> dict:
    raw = call_text(system, user, use_retry=use_retry)
    return _extract_json(raw)
```

**Enhanced `_repair_and_parse()` in `conflict_judge.py`:**
```python
def _repair_and_parse(raw_response: str) -> dict[str, Any]:
    text = (raw_response or "").strip()
    if not text:
        return {}

    # Extract from markdown code blocks
    m = _JSON_BLOCK_RE.search(text)
    if m:
        text = m.group("body").strip()

    # Locate first {...} if extra text exists
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]

    # Fast path: direct parse
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    # Slow path: json-repair for malformed JSON
    try:
        repaired = repair_json(text)
        parsed = json.loads(repaired)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}
```

### Fix 4.2: Retry Amplification Eliminated

**Simplified `_judge_single_batch()` in `conflict_judge.py`:**
```python
def _judge_single_batch(
    *,
    batch_pairs: list[dict[str, Any]],
    system: str,
    user_template: str,
    default_user_fmt: str,
    # Removed: max_retries parameter
) -> list[dict[str, Any]]:
    from app.llm.client import call_text

    # ... prepare prompt ...

    # Single try-except, retry delegated to client layer
    try:
        raw = call_text(system, user, use_retry=True)
    except Exception:
        return _mark_batch_insufficient(batch_pairs)

    # Parse and repair
    out = _repair_and_parse(raw)
    # ... validate and return results ...
```

**Key Changes:**
- ✅ Removed outer `for attempt in range(max_retries)` loop
- ✅ Removed `time.sleep()` exponential backoff
- ✅ Single error handling path
- ✅ Retry only at client layer via `call_text(use_retry=True)`

---

## Verification Results

### New Tests: 7/7 PASSED (0.18s)

```
tests/test_conflict_judge_retry_and_repair.py::test_conflict_judge_repairs_malformed_json_response PASSED
tests/test_conflict_judge_retry_and_repair.py::test_conflict_judge_marks_batch_insufficient_when_unrepairable PASSED
tests/test_conflict_judge_retry_and_repair.py::test_conflict_judge_avoids_outer_retry_amplification PASSED
tests/test_conflict_judge_retry_and_repair.py::test_conflict_judge_extracts_json_from_markdown_code_blocks PASSED
tests/test_conflict_judge_retry_and_repair.py::test_conflict_judge_extracts_json_from_text_with_extra_content PASSED
tests/test_conflict_judge_retry_and_repair.py::test_conflict_judge_normalizes_invalid_labels PASSED
tests/test_conflict_judge_retry_and_repair.py::test_conflict_judge_clamps_scores_to_valid_range PASSED
```

**Test Coverage:**
- ✅ Malformed JSON repair (trailing commas)
- ✅ Unrepairable JSON graceful degradation
- ✅ Retry amplification verification (2 batches = 2 calls, not 6)
- ✅ Markdown code block extraction
- ✅ Extra text handling
- ✅ Invalid label normalization
- ✅ Score clamping [0,1]

### Existing Tests: 6/6 PASSED (147.86s)

```
tests/test_conflict_robustness.py::test_conflict_detection_basic PASSED
tests/test_conflict_robustness.py::test_conflict_detection_large_batch PASSED
tests/test_conflict_robustness.py::test_conflict_detection_empty_pairs PASSED
tests/test_conflict_robustness.py::test_conflict_detection_batch_size_configuration PASSED
tests/test_conflict_robustness.py::test_conflict_detection_max_pairs_limit PASSED
tests/test_conflict_robustness.py::test_conflict_detection_malformed_label_normalization PASSED
```

**Total: 13/13 conflict detection tests passing** ✅

---

## Codex Review Findings

**Verdict:** Both blockers RESOLVED ✅

**What's Correct:**
- ✅ json-repair now in actual execution path
- ✅ Conflict judge consumes raw text and repairs before degrading
- ✅ Outer retry loop removed (no more batch-level amplification)
- ✅ Regression tests well-targeted and comprehensive
- ✅ Progressive fallback robust for common malformed outputs

**Remaining Caveat:**
- ⚠️ Nested retry still exists: `tenacity(3)` × `ChatOpenAI(max_retries=2)` = 6 worst case
- Previously: `judge(3)` × `tenacity(3)` × `ChatOpenAI(2)` = 18 worst case
- **Improvement: 67% reduction in retry amplification**
- Further optimization possible: make `llm()` retries configurable

---

## Impact Analysis

### Blocker 4.1 Resolution Impact

**Before:**
- Malformed JSON → immediate failure
- Entire batch marked as "insufficient"
- No recovery attempt

**After:**
- Markdown code block extraction ✅
- Extra text stripped ✅
- Direct JSON parse (fast path) ✅
- json-repair fallback (slow path for malformed) ✅
- Only truly unrepairable → "insufficient"

**Expected Result:** Lower "insufficient" ratio, better semantic conflict coverage

### Blocker 4.2 Resolution Impact

**Before:**
- Worst case: 18 retry attempts
- High latency under transient failures
- Excessive API costs

**After:**
- Worst case: 6 retry attempts (67% reduction)
- Simpler error handling
- Lower latency and cost
- Still resilient to transient failures

**Expected Result:** Faster response times, reduced API costs, better SLO compliance

---

## Deployment Status

**Updated Blocker Status:**

| Blocker | Status | Resolution |
|---------|--------|------------|
| 4.1 JSON repair | ✅ RESOLVED | Progressive fallback implementation |
| 4.2 Retry amplification | ✅ RESOLVED | Outer loop removed, 67% reduction |

**Stage 4 Deployment Decision:**
- ✅ All production blockers resolved
- ✅ 13/13 tests passing
- ✅ Ready for production deployment
- 📋 Optional follow-up: Consolidate retry policy in client.py

---

## Follow-Up Optimization (Optional)

**Optimization Item:** Consolidate retry policy
- **Current:** `tenacity(3)` × `ChatOpenAI(max_retries=2)` = 6 worst case
- **Target:** Single retry layer (either tenacity OR ChatOpenAI)
- **Benefit:** Further latency/cost reduction
- **Priority:** P2 (non-blocking, optimization only)

**Implementation:**
```python
# Option: Make ChatOpenAI max_retries configurable
def llm() -> Any:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        ...,
        max_retries=0,  # Rely only on tenacity retry
    )
```

---

## Conclusion

**Both Blocker 4.1 and 4.2 are RESOLVED** ✅

**Production Readiness:**
- ✅ JSON repair effectively used with progressive fallback
- ✅ Retry amplification reduced by 67%
- ✅ All tests passing (13/13)
- ✅ Backward compatible with existing code
- ✅ Ready for Stage 4 production deployment

**Recommendation:**
- Deploy Stage 4 to production
- Monitor "insufficient" ratio (should decrease)
- Monitor conflict detection latency (should improve)
- Track retry amplification metrics
- Consider P2 optimization for single retry layer

---

**Commits:**
- `0904c54` - fix(conflict): resolve Blocker 4.1 & 4.2 - JSON repair and retry amplification
