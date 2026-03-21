# LogicKG Ingest Hot Path / Audit-Enrich Split Design

## 1. Background

The current paper ingest pipeline makes a single paper feel slow because "paper complete" is defined too late. A paper is only counted as finished after it has gone through:

1. optional paper type classification
2. logic extraction
3. chunk-batched claim extraction with retry fallbacks
4. merge and validation packaging
5. citation-purpose classification
6. Neo4j materialization

In practice, this means the user waits for a long composite workflow before seeing the paper marked complete, even though useful `LogicStep` and `Claim` artifacts may already exist on disk and be ready to write back earlier.

This design introduces a clear split between:

1. `FastIngest`: the minimum synchronous path needed to produce stable L2 paper assets
2. `AuditEnrich`: slower quality, citation, and enrichment tasks that can run after the first result exists

## 2. Goals

1. Reduce time-to-first-usable-paper during ingest.
2. Preserve existing extraction quality semantics where possible.
3. Keep the path that produces `LogicStep` and `Claim` assets simple and observable.
4. Make later enrichment optional and movable to background jobs.

## 3. Non-Goals

1. No change to L2.5, overlapping community, or L3/L4 semantics in this pass.
2. No attempt to redesign the extraction schema itself.
3. No change to the meaning of claim validation; only when and where slower checks run.
4. No full async job system in this pass if a lighter synchronous split is sufficient.

## 4. Approaches Considered

### Approach A: Only raise concurrency

Increase worker counts, timeouts, and model parallelism while keeping the pipeline structure unchanged.

Pros:

1. Minimal code change.
2. Fastest to try.

Cons:

1. Does not remove unnecessary hot-path work.
2. Increases pressure on LLM workers and makes failures noisier.
3. Still leaves poor progress observability.

### Approach B: Keep structure, but add more granular progress

Expose sub-stage progress per paper while keeping current synchronous behavior.

Pros:

1. Improves visibility.
2. Low product risk.

Cons:

1. Only fixes perception, not actual latency.
2. Citation-purpose and audit work still block completion.

### Approach C: Split FastIngest from AuditEnrich

Move the minimum extraction path into the synchronous completion definition and defer slower enrichment work.

Pros:

1. Improves both actual latency and perceived latency.
2. Matches the user's stated goal of making LogicKG a better second-layer extraction engine.
3. Creates a cleaner boundary for future scheduling and background execution.

Cons:

1. Requires touching ingest orchestration and tests.
2. Needs careful compatibility handling for outputs expected by downstream code.

### Recommendation

Choose Approach C.

## 5. Target Design

### 5.1 FastIngest

`FastIngest` is the path that must complete before a paper counts as extracted.

It includes:

1. markdown parse
2. schema / paper type resolution
3. logic extraction
4. claim extraction
5. merge into validated paper-level output
6. immediate write of logic and claims to Neo4j
7. immediate write of `*.llm_imrad.json`

It does not wait for:

1. citation-purpose classification
2. citation-target enrichment beyond what is already cheap and local
3. extra audit packaging that is derived from already available data
4. later FAISS / similarity / community stages

### 5.2 AuditEnrich

`AuditEnrich` contains slower or optional work:

1. citation-purpose classification
2. future claim- or citation-level enrichments that do not affect first availability
3. audit-only reporting artifacts that can be derived after the core extraction is materialized

In the first implementation pass, this can remain in-process, but it must no longer decide whether a paper is counted as complete in ingest progress.

### 5.3 Progress Semantics

Per-paper progress should distinguish:

1. `core_done`: logic + claims are extracted and written
2. `enrich_done`: optional citation-purpose and related enrichments are finished

Batch progress should be based on `core_done`, because that is the first moment a paper is useful to the system.

## 6. Concrete Changes

### 6.1 Remove or demote non-essential hot-path work

1. Treat citation-purpose classification as enrichment, not as part of the completion gate.
2. Prefer rule-based paper-type classification when metadata is sufficient, and avoid paying an LLM call for obvious research papers.
3. Keep grounding/audit report generation lightweight if the current mode is already effectively `skip`.

### 6.2 Keep compatibility

1. Existing output files should still be produced where reasonable.
2. `llm_imrad.json` remains the primary synchronous artifact.
3. Citation-purpose artifacts can be written later without blocking the batch.

### 6.3 Improve observability

1. Update progress messages to reflect `core_done` counts.
2. Preserve enough information in outputs so that later enrich steps can be retried independently.

## 7. Testing Strategy

1. Add or update ingest pipeline tests so a paper is marked complete after core extraction even if enrichment is deferred.
2. Verify citation-purpose behavior is preserved as a separate path.
3. Keep upload and existing ingest regression tests passing.

## 8. Expected Outcome

After this change, the ingest pipeline should behave like a true second-layer extractor:

1. It produces useful `LogicStep` and `Claim` results earlier.
2. It stops blocking paper completion on optional enrichment.
3. It becomes easier to optimize further without changing extraction semantics.
