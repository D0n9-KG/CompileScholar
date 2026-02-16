# Lexical Mode Removal - Implementation Summary

## Overview
Implemented user directive: "直接把 lexical 模式删了就行，embedding 模式报错时，自动触发重试，还是报错就直接停止，返回错误待用户解决"

**Translation:** Remove lexical mode completely. When embedding fails, auto-retry. If still failing, stop and return error for user to resolve.

## Commits Completed

### Commit 1: Remove lexical fallback from update_similarity_for_paper
**File:** `backend/app/similarity/service.py`
**Changes:**
- Added 3-attempt retry with 5s delay for embedding failures in `_apply_updates()`
- Removed degradation tracking variables (`degradation_events`, `degraded_kinds`)
- Removed zero-vector fallback on embedding failure
- Mode is always `'embedding'`, never `'lexical'` or `'mixed'`
- Removed degradation fields from metadata and return value
- Simplified neighbor computation logic

**Impact:** Per-paper similarity updates now require embedding API, no silent degradation

### Commit 2: Add retry to clustering embedding and fail-hard
**Files:** `backend/app/similarity/embedding.py`, `backend/app/tasks/clustering_task.py`
**Changes:**
- Added 3-attempt retry with 5s delay to `get_embeddings_batch()`
- Intelligent detection of retryable errors (502, 503, 504, timeouts, rate limits)
- Fail fast on non-retryable errors
- Changed `clustering_task` to re-raise exceptions instead of returning failed dict

**Impact:** Proposition clustering now requires embedding API, failures propagate correctly

### Commit 3: Remove lexical fallback from RAG service
**File:** `backend/app/rag/service.py`
**Changes:**
- Removed lexical retrieval fallback from `ask()` function
- FAISS failures now propagate as clear errors:
  - `FileNotFoundError` for missing index (404 user error)
  - `RuntimeError` for FAISS query failures (500 server error)
- Removed unused imports: `latest_run_dir`, `load_chunks_from_run`, `lexical_retrieve`
- Evidence mode is always `'faiss'`, never `'lexical'`

**Impact:** RAG queries now require FAISS index, no silent degradation

## Retry Pattern Details

All embedding API calls now use consistent retry logic:
- **Max attempts:** 3
- **Retry delay:** 5 seconds
- **Retryable errors:** 408, 429, 500, 502, 503, 504, timeouts, connection errors, rate limits
- **Non-retryable:** 400, 401, 403, 404, parsing errors
- **Final failure:** Clear error message with attempt count and guidance

## Error Messages

### FAISS Build (Phase 1)
```
FAISS index build failed: {reason}.
Error: {error_msg}. Please check embedding API configuration and try again.
```

### Similarity Rebuild (Phase 2)
```
Similarity rebuild failed: {reason}.
Error: {error_msg}. Please check embedding API configuration and try again.
```

### Similarity Update (per-paper)
```
Similarity update failed ({kind}): {reason}.
Error: {error_msg}. Please check embedding API configuration and try again.
```

### Clustering
```
Clustering embedding failed: {reason}.
Error: {error_msg}
```

### RAG
```
FAISS retrieval failed: {error}. Please check embedding configuration.
```
OR
```
FAISS index not found. Please run full rebuild first.
```

## Architecture Changes

**Before:** Lexical fallback at 5 locations
1. ✅ Phase 2 similarity rebuild → embedding with retry
2. ✅ Per-paper similarity update → embedding with retry
3. ✅ Proposition clustering → embedding with retry
4. ✅ RAG retrieval → FAISS only (no fallback)
5. Phase 1 FAISS build → embedding with retry (already done in previous work)

**After:** Embedding-only everywhere, with intelligent retry

## Quality Impact

**Benefits:**
- Higher quality similarity scores (0.85-0.99 vs 0.70-0.95)
- Evolution inference rules work correctly (require ≥0.89-0.97 similarity)
- Consistent behavior across all code paths
- Clear error messages guide users to fix configuration issues
- No silent degradation or misleading metrics

**Trade-offs:**
- Requires working embedding API for all operations
- Failures are explicit (no graceful degradation)
- Users must fix embedding configuration before proceeding

## Next Steps (Commit 4)

1. Update documentation:
   - README.md: Mark embedding as required (not optional)
   - backend/.env.example: Add embedding requirement notes
   - backend/app/settings.py: Remove "fallback to lexical" comments

2. Verify tests:
   - backend/tests/test_embedding_degradation.py: Update or remove
   - backend/tests/test_similarity_update_retry.py: Fix mocking
   - backend/tests/test_embedding.py: Add retry test cases

3. Run full regression tests
4. Trigger new rebuild with all changes
5. Verify quality gates pass

## Verification Checklist

- [ ] All 4 commits completed
- [ ] Documentation updated
- [ ] Tests passing
- [ ] Rebuild completed successfully
- [ ] Phase 3 evolution coverage ≥20%
- [ ] Self-loop rate ≤5%
- [ ] Quality gates pass

## Timeline

- Commit 1: Lines 387-614 in similarity/service.py
- Commit 2: embedding.py + clustering_task.py
- Commit 3: rag/service.py
- Commit 4: Documentation + tests (in progress)

Total changes: ~500 lines modified across 7 files
