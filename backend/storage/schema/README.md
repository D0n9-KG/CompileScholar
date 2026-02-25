# Schema Evolution Storage

This directory documents the on-disk contract for fusion schema evolution.

Runtime files are generated under:

- `backend/storage/schema/evolution/candidates.jsonl`
- `backend/storage/schema/evolution/versions/*.json`

Conventions:

1. Unknown entity/relation labels are first stored as candidates.
2. Candidate clusters become patch proposals keyed by normalized label.
3. Gate policy:
   - score `>= T_high`: auto-accept and persist as a new schema patch version
   - `T_mid <= score < T_high`: pending review
   - score `< T_mid`: rejected
4. Accepted patches expose an incremental replay scope:
   - only affected `evidence_chunk_ids` are replayed.

Notes:

- Do not commit generated candidate/version artifacts.
- This README is intentionally tracked as the canonical storage contract.
