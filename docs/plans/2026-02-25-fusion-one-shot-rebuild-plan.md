# LogicKG Fusion One-Shot Rebuild Plan (Execution Copy)

## Baseline Snapshot (Recorded)

- Baseline commit: `cf93fb3`
- Baseline tag: `pre-fusion-rebuild-2026-02-25`
- Worktree branch for implementation: `feature/fusion-one-shot-rebuild`

## Goal

Replace the old Proposition/Evolution chain with a fusion graph pipeline:

- keep `LogicStep`/`Claim` paper structure,
- locally extract textbook KG,
- build cross-source links (`EXPLAINS`),
- run fusion-community summarization,
- upgrade `/rag/ask` retrieval using fusion evidence,
- switch UI from `/evolution` to `/fusion`.

## Legacy Surface to Remove After Gate

- API routes:
  - `/evolution/*`
  - `/tasks/rebuild/evolution`
  - `/textbooks/fusion/link` (old proposition bridge)
- Backend modules:
  - `app/evolution/*`
  - proposition clustering tasks (`app/tasks/clustering_task.py`)
- Frontend route and nav:
  - `/evolution`
  - `frontend/src/pages/EvolutionPage.tsx`

## Rollback Criteria

Rollback immediately if any is true:

1. `/rag/ask` accuracy regresses on fixed benchmark set.
2. Section-to-basics linkage fails above threshold.
3. Fusion rebuild introduces startup/runtime breakage.

Rollback action:

1. `git checkout pre-fusion-rebuild-2026-02-25`
2. redeploy backend + frontend from that tag
3. re-enable legacy `/evolution` route
4. postpone proposition deletion script execution

## Milestone Tasks

1. Cutover docs and guardrails.
2. Fusion schema + proposition cleanup migration.
3. Local textbook extractor (default).
4. Schema evolution with confidence gates.
5. Fusion builder + cross-source linking.
6. Community detection + keyword outputs.
7. Fusion APIs + task wiring.
8. Fusion-aware RAG retrieval.
9. Frontend `/fusion` page and nav switch.
10. Remove legacy evolution/proposition chain.
11. Quality gate benchmark.
12. Runbook and release handoff.

## Verification Checklist

1. Backend targeted tests for changed modules pass.
2. Frontend `npm run build` passes.
3. `/fusion/rebuild` then `/rag/ask` returns paper + textbook evidence.
4. No unresolved imports from removed evolution/proposition modules.
5. Migration dry-run and execute logs are archived.
