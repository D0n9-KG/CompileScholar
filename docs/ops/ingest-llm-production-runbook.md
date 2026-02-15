# Ingest LLM Production Runbook

## Current run stuck at 70% UI

- If `backend/runs/<run_id>/` continues producing new `*.llm_imrad.json` files, let the run finish naturally.
- Cancel/restart only if both task progress and run artifacts are unchanged for >30 minutes.
- Preferred stall check: compare latest artifact mtime instead of only task `progress`.

## Retry strategy (`max_retries=0` vs `1`)

- Default recommendation: `LLM_CLIENT_MAX_RETRIES=0`.
- Reason: there is already outer tenacity retry (`stop_after_attempt(3)`), so keeping client retries at `0` avoids multiplicative retry amplification and long-tail hangs.
- Optional override: set `LLM_CLIENT_MAX_RETRIES=1` if transient provider/network failures are frequent and longer latency is acceptable.

## Schema v7 vs v8

- Keep research schema at v7 for production quality consistency unless you explicitly want a throughput/latency trade-off.
- Treat v8 as an operational fallback for heavy backlog or incident response, not the default quality-optimized profile.

## Recommended deployment order

1. Deploy heartbeat + progress logging (Patch A) and timeout/retry controls (Patch B).
2. Keep the active schema unchanged initially.
3. Monitor one 20-paper run; only then decide whether a schema throughput downgrade is needed.

## Heartbeat Configuration

- Default heartbeat interval: 20 seconds (`INGEST_LLM_HEARTBEAT_SECONDS=20`)
- Heartbeat progress messages include:
  - `running=N` - number of papers currently being processed
  - `slowest=<paper_id>:<seconds>s` - identifies longest-running paper
- This prevents UI freeze even when Neo4j writes block completion callbacks

## Timeout Configuration

### LLM Timeouts
- `LLM_TIMEOUT_SECONDS=60` (default) - per-request timeout
- Range: 10-600 seconds
- Recommended: keep at 60s for DeepSeek

### Neo4j Connection Timeout
- `NEO4J_CONNECTION_TIMEOUT_SECONDS=15` (default)
- Range: 1-120 seconds
- Helps detect stalled database connections early

## Troubleshooting

### Progress frozen but artifacts still updating
- **Diagnosis**: Progress reporting blocked by slow Neo4j write
- **Solution**: Wait for completion, upgrade to heartbeat-enabled version
- **Check**: `ls -lt backend/runs/<run_id>/raw_pool | head -5` to see latest artifact updates

### High retry amplification
- **Symptom**: Single paper takes >10 minutes per LLM call
- **Diagnosis**: `max_retries=2` × `tenacity(3)` = up to 6 attempts per call
- **Solution**: Set `LLM_CLIENT_MAX_RETRIES=0` to reduce to 3 total attempts

### Neo4j connection hangs
- **Symptom**: Progress stops, no new artifacts, no error logs
- **Diagnosis**: Database connection deadlock or network issue
- **Solution**: Lower `NEO4J_CONNECTION_TIMEOUT_SECONDS` to fail faster, check Neo4j server health
