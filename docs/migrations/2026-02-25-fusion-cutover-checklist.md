# Fusion Cutover Checklist (Release Window)

## Pre-Cutover

- [ ] Confirm baseline tag exists: `pre-fusion-rebuild-2026-02-25`
- [ ] Export current quality baseline report
- [ ] Backup Neo4j data (or snapshot)
- [ ] Confirm frontend build and backend smoke pass on release branch

## Migration Order (One-Shot Window)

1. Deploy backend with fusion modules and legacy route still disabled, not deleted.
2. Run textbook local extraction on validation sample.
3. Run `/fusion/rebuild` and verify graph materialization.
4. Switch `/rag/ask` retrieval to fusion path.
5. Deploy frontend with `/fusion` route and nav replacement.
6. Execute proposition cleanup script only after quality gate passes.
7. Remove legacy evolution modules and publish release candidate tag.

## Smoke Test Order

1. `GET /health`
2. Textbook ingest (local extractor path) task submission and completion.
3. `POST /fusion/rebuild`
4. `GET /fusion/graph`
5. `GET /fusion/paper/{paper_id}/sections`
6. `GET /fusion/paper/{paper_id}/section/{step_type}/basics`
7. `POST /rag/ask` (must include paper + textbook evidence)
8. Frontend:
   - open `/fusion`
   - select paper section
   - inspect matched textbook fundamentals

## Stop Conditions

Stop cutover and rollback if any is true:

- backend cannot boot cleanly after merge,
- fusion APIs fail contract-level smoke tests,
- RAG outputs lose textbook evidence linkage,
- benchmark score regresses beyond gate threshold.

## Rollback Procedure

1. Checkout and redeploy `pre-fusion-rebuild-2026-02-25`.
2. Keep proposition subgraph untouched (skip cleanup execute mode).
3. Re-enable legacy `/evolution` UI/API.
4. Record failure cause and blocked milestone.

## Post-Cutover

- [ ] Tag release milestone
- [ ] Archive migration script logs
- [ ] Archive quality gate report
- [ ] Publish runbook updates
