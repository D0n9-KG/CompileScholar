# PaperLogicTrace L2 Hard-Cut Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old `LogicStep / Claim` second-layer schema with a high-throughput `PaperLogicTrace` compiler that exports canonical `ResearchMove`-based traces and leaves clean interfaces for future L1, community, L3, and L4 builders.

**Architecture:** Keep the existing evidence substrate and paper identity plumbing, but rebuild the second-layer around a new canonical `PaperLogicTrace` model. The hot path should compile `ResearchMove`, `MoveRelation`, and evidence-backed L2.5 slots directly into `canonical_core`, then derive a small set of compiler-friendly views without preserving the old `LogicStep / Claim` pipeline shape. Lightweight audit stays optional and narrow.

**Tech Stack:** FastAPI, Python 3.11, Pydantic, Neo4j, existing ingest/extraction pipeline, pytest

---

## File Structure

### Create

- `backend/app/paper_logic_trace/compiler.py`
- `backend/app/paper_logic_trace/derived_views.py`
- `backend/app/paper_logic_trace/gates.py`
- `backend/tests/test_paper_logic_trace_compiler.py`
- `backend/tests/test_paper_logic_trace_derived_views.py`
- `backend/tests/test_paper_logic_trace_gates.py`

### Modify

- `backend/app/paper_logic_trace/models.py`
- `backend/app/paper_logic_trace/exporter.py`
- `backend/app/paper_logic_trace/normalization.py`
- `backend/app/paper_logic_trace/__init__.py`
- `backend/app/extraction/orchestrator.py`
- `backend/app/ingest/pipeline.py`
- `backend/app/ingest/models.py`
- `backend/app/ingest/parse_md.py`
- `backend/app/ingest/figures.py`
- `backend/app/graph/neo4j_client.py`
- `backend/app/api/routers/papers.py`
- `backend/tests/test_paper_logic_trace_models.py`
- `backend/tests/test_paper_logic_trace_export.py`
- `backend/tests/test_phase1_extraction_orchestrator.py`
- `backend/tests/test_ingest_llm_progress.py`
- `backend/tests/test_papers_api.py`

### Optional Modify

- `backend/app/community/candidate_graph.py`
- `backend/tests/test_community_candidate_graph.py`
- `README.md`
- `TECHNICAL_OVERVIEW.zh-CN.md`

## Responsibility Map

- `paper_logic_trace/models.py`: canonical `PaperLogicTrace`, `ResearchMove`, `EvidenceAnchor`, `MoveRelation`, `MentionValue`, `EffectValue`, `SlotProvenance`, and quality/read-model types.
- `paper_logic_trace/compiler.py`: compile framed paper evidence into canonical `PaperLogicTrace`.
- `paper_logic_trace/derived_views.py`: build `l2_5_slot_inventory`, `l1_bridge_hints`, `community_signatures`, `route_feature_candidates`, and human-readable summaries.
- `paper_logic_trace/gates.py`: minimal hot-path gate plus lightweight-audit eligibility logic.
- `paper_logic_trace/exporter.py`: external entrypoint that assembles graph/evidence inputs, calls the compiler, and returns the canonical export.
- `extraction/orchestrator.py`: produce move-oriented extraction artifacts instead of old `LogicStep / Claim`-first packaging.
- `ingest/pipeline.py`: shorten the synchronous path to “compile first valid `PaperLogicTrace`,” not “finish every old extraction stage.”
- `graph/neo4j_client.py`: supply all inputs needed by the compiler/exporter and, if retained, materialize read-model projections instead of old schema assumptions.
- `api/routers/papers.py`: expose the new `PaperLogicTrace` payload cleanly.

## Chunk 1: Canonical L2 Schema

### Task 1: Replace the old `PaperLogicTrace` model with the new canonical schema

**Files:**
- Modify: `backend/app/paper_logic_trace/models.py`
- Modify: `backend/app/paper_logic_trace/__init__.py`
- Test: `backend/tests/test_paper_logic_trace_models.py`

- [ ] **Step 1: Write the failing model tests**

Add tests that require the new schema objects:

```python
from app.paper_logic_trace.models import (
    PaperLogicTrace,
    PaperMetadata,
    CanonicalCore,
    ResearchMove,
    EvidenceAnchor,
)


def test_paper_logic_trace_uses_canonical_core():
    trace = PaperLogicTrace(
        trace_id='trace-1',
        schema_version='v2',
        built_at='2026-03-22T00:00:00Z',
        paper_metadata=PaperMetadata(paper_id='paper-1', title='Demo', source_refs=['chunk:1']),
        canonical_core=CanonicalCore(
            evidence_anchors=[
                EvidenceAnchor(anchor_id='a-1', paper_id='paper-1', source_ref='chunk:1', modality='text', section_path=[], locator={}, quote='demo', citation_ids=[], support_type='direct', weak=False)
            ],
            moves=[
                ResearchMove(move_id='m-1', sequence_no=1, role='method', act_type='propose_method', summary='Uses graph encoding', anchor_ids=['a-1'], confidence=0.8)
            ],
            move_relations=[],
            citation_acts=[],
            figure_refs=[],
            table_refs=[],
        ),
        derived_views={},
        quality={'quality_tier': 'green', 'hot_path_gate_report': {}},
    )

    assert trace.canonical_core.moves[0].role == 'method'
```

- [ ] **Step 2: Run the focused model tests and verify they fail**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_paper_logic_trace_models.py`

Expected: FAIL because the old models only expose `logic_steps` and `claims`.

- [ ] **Step 3: Implement the new canonical model layer**

Add the new Pydantic models for:

1. `PaperMetadata`
2. `EvidenceAnchor`
3. `MentionValue`
4. `EffectValue`
5. `SlotProvenance`
6. `ResearchMove`
7. `MoveRelation`
8. `CanonicalCore`
9. `PaperLogicTrace`

Keep optional fields narrow in the hot path. Do not reintroduce `LogicStepTrace` or `ClaimTrace` in the canonical model.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_paper_logic_trace_models.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/paper_logic_trace/models.py backend/app/paper_logic_trace/__init__.py backend/tests/test_paper_logic_trace_models.py
git commit -m "feat: add canonical paper logic trace schema"
```

### Task 2: Add derived-view model coverage without turning them into canonical truth

**Files:**
- Create: `backend/app/paper_logic_trace/derived_views.py`
- Test: `backend/tests/test_paper_logic_trace_derived_views.py`
- Modify: `backend/app/paper_logic_trace/models.py`

- [ ] **Step 1: Write the failing derived-view tests**

Add tests that require:

1. `community_signatures` to point back to `move_id`
2. `route_feature_candidates` to be derived from canonical moves
3. `l1_bridge_hints` to preserve provenance references

Example:

```python
from app.paper_logic_trace.derived_views import build_community_signatures


def test_build_community_signatures_uses_move_slots_not_raw_summary():
    signatures = build_community_signatures([
        {
            'move_id': 'm-1',
            'paper_id': 'paper-1',
            'role': 'method',
            'act_type': 'propose_method',
            'methods': [{'normalized': 'graph neural network'}],
            'research_objects': [{'normalized': 'entity relation graph'}],
            'metrics': [],
            'conditions': [],
            'comparators': [],
            'effects': [],
            'limitation_types': [],
            'resource_mentions': [],
        }
    ])

    assert signatures[0]['move_id'] == 'm-1'
    assert 'graph neural network' in signatures[0]['method_tokens']
```

- [ ] **Step 2: Run the focused derived-view tests and verify they fail**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_paper_logic_trace_derived_views.py`

Expected: FAIL because the helper module does not exist.

- [ ] **Step 3: Implement the minimal derived-view builders**

Implement helpers for:

1. `build_l2_5_slot_inventory()`
2. `build_l1_bridge_hints()`
3. `build_community_signatures()`
4. `build_route_feature_candidates()`
5. `build_paper_summaries()`

Each output must be traceable to canonical move ids or anchor ids.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_paper_logic_trace_derived_views.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/paper_logic_trace/derived_views.py backend/app/paper_logic_trace/models.py backend/tests/test_paper_logic_trace_derived_views.py
git commit -m "feat: add paper logic trace derived views"
```

## Chunk 2: PaperLogicTrace Compiler and Gates

### Task 3: Build the `ResearchMove` compiler around framed semantic windows

**Files:**
- Create: `backend/app/paper_logic_trace/compiler.py`
- Modify: `backend/app/extraction/orchestrator.py`
- Test: `backend/tests/test_paper_logic_trace_compiler.py`
- Modify: `backend/tests/test_phase1_extraction_orchestrator.py`

- [ ] **Step 1: Write the failing compiler test**

Add a test that feeds framed paper windows into a compiler entrypoint and expects canonical `ResearchMove` output:

```python
from app.paper_logic_trace.compiler import compile_paper_logic_trace


def test_compile_paper_logic_trace_emits_research_moves():
    trace = compile_paper_logic_trace(
        paper_metadata={'paper_id': 'paper-1', 'title': 'Demo', 'source_refs': ['chunk:1']},
        evidence_rows=[
            {
                'anchor_id': 'a-1',
                'source_ref': 'chunk:1',
                'quote': 'We propose a graph encoder for retrieval.',
                'role_hint': 'method',
                'act_hint': 'propose_method',
            }
        ],
        figure_rows=[],
        table_rows=[],
        citation_rows=[],
    )

    assert trace.canonical_core.moves[0].act_type == 'propose_method'
    assert trace.canonical_core.moves[0].anchor_ids == ['a-1']
```

- [ ] **Step 2: Run the focused compiler test and verify it fails**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_paper_logic_trace_compiler.py`

Expected: FAIL because the compiler module does not exist.

- [ ] **Step 3: Implement the minimal compiler path**

Implement a compiler that:

1. builds `EvidenceAnchor` objects
2. compiles move candidates from framed rows
3. merges duplicate move candidates
4. stitches minimal `MoveRelation` links
5. builds `derived_views`
6. returns a valid `PaperLogicTrace`

Do not preserve the old `LogicStep / Claim`-first packaging as an intermediate requirement.

- [ ] **Step 4: Update `orchestrator.py` to produce move-oriented compiler inputs**

Refactor extraction orchestration so the compiler consumes:

1. framed paper metadata
2. semantic windows or equivalent evidence rows
3. figure/table refs
4. citation rows

Remove or bypass old stages that only exist to materialize `LogicStep / Claim`.

- [ ] **Step 5: Run the focused tests and verify they pass**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_paper_logic_trace_compiler.py tests/test_phase1_extraction_orchestrator.py`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/paper_logic_trace/compiler.py backend/app/extraction/orchestrator.py backend/tests/test_paper_logic_trace_compiler.py backend/tests/test_phase1_extraction_orchestrator.py
git commit -m "feat: compile research moves into paper logic traces"
```

### Task 4: Add the minimal hot-path gate and lightweight-audit eligibility logic

**Files:**
- Create: `backend/app/paper_logic_trace/gates.py`
- Test: `backend/tests/test_paper_logic_trace_gates.py`
- Modify: `backend/app/paper_logic_trace/compiler.py`

- [ ] **Step 1: Write the failing gate tests**

Add tests that enforce:

1. a move without `role`, `act_type`, or anchors cannot enter canonical export
2. sparse-but-valid traces can still export with downgraded quality
3. lightweight-audit status is a quality flag, not a blocker for first export

Example:

```python
from app.paper_logic_trace.gates import evaluate_hot_path_gate


def test_hot_path_gate_rejects_anchorless_move():
    result = evaluate_hot_path_gate(
        moves=[{'move_id': 'm-1', 'role': 'method', 'act_type': 'propose_method', 'anchor_ids': []}],
        anchors=[],
    )

    assert result['passed'] is False
```

- [ ] **Step 2: Run the focused gate tests and verify they fail**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_paper_logic_trace_gates.py`

Expected: FAIL because the gate module does not exist.

- [ ] **Step 3: Implement the minimal gate helpers**

Implement:

1. `evaluate_hot_path_gate()`
2. `build_quality_payload()`
3. `needs_lightweight_audit()`

Keep this narrow. Do not turn it into a heavy audit workflow.

- [ ] **Step 4: Wire the compiler to use the new gate**

Make `compile_paper_logic_trace()` attach:

1. `quality_tier`
2. `hot_path_gate_report`
3. `audit_status`

- [ ] **Step 5: Run the focused tests and verify they pass**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_paper_logic_trace_gates.py tests/test_paper_logic_trace_compiler.py`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/paper_logic_trace/gates.py backend/app/paper_logic_trace/compiler.py backend/tests/test_paper_logic_trace_gates.py
git commit -m "feat: add paper logic trace hot-path gates"
```

## Chunk 3: Hot Path Rewire

### Task 5: Rewire ingest so a paper completes when a valid `PaperLogicTrace` exists

**Files:**
- Modify: `backend/app/ingest/pipeline.py`
- Modify: `backend/app/ingest/models.py`
- Modify: `backend/tests/test_ingest_llm_progress.py`

- [ ] **Step 1: Write the failing ingest-progress test**

Add a test that marks a paper complete once a valid `PaperLogicTrace` has been compiled, even if lightweight audit has not yet run.

- [ ] **Step 2: Run the focused ingest-progress test and verify it fails**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_ingest_llm_progress.py`

Expected: FAIL because completion still depends on the old extraction shape.

- [ ] **Step 3: Update ingest result semantics**

Refactor the per-paper ingest path so completion is defined by:

1. valid evidence substrate
2. successful `PaperLogicTrace` compilation
3. hot-path gate result

Do not wait for broad enrichment work.

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_ingest_llm_progress.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingest/pipeline.py backend/app/ingest/models.py backend/tests/test_ingest_llm_progress.py
git commit -m "refactor: complete ingest after paper logic trace compile"
```

### Task 6: Rebuild the exporter entrypoint on top of the new compiler

**Files:**
- Modify: `backend/app/paper_logic_trace/exporter.py`
- Modify: `backend/app/graph/neo4j_client.py`
- Modify: `backend/tests/test_paper_logic_trace_export.py`

- [ ] **Step 1: Write the failing exporter test for canonical core**

Add a test that expects `export_paper_logic_trace()` to return:

1. `canonical_core.moves`
2. `canonical_core.move_relations`
3. `derived_views.community_signatures`
4. `quality.audit_status`

instead of the old `logic_steps / claims` export shape.

- [ ] **Step 2: Run the focused exporter test and verify it fails**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_paper_logic_trace_export.py`

Expected: FAIL because the current exporter still emits the old structure.

- [ ] **Step 3: Implement the new exporter contract**

Make `Neo4jClient.get_paper_logic_trace_inputs()` supply the compiler with:

1. paper metadata
2. substrate evidence rows
3. figure/table refs
4. citation rows

Then make `export_paper_logic_trace()` call the compiler and return the new `PaperLogicTrace`.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_paper_logic_trace_export.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/paper_logic_trace/exporter.py backend/app/graph/neo4j_client.py backend/tests/test_paper_logic_trace_export.py
git commit -m "feat: export canonical paper logic trace payloads"
```

## Chunk 4: Public API and Downstream Interfaces

### Task 7: Update the paper API to expose the new canonical export cleanly

**Files:**
- Modify: `backend/app/api/routers/papers.py`
- Modify: `backend/tests/test_papers_api.py`

- [ ] **Step 1: Write the failing papers API test**

Add a test for `/papers/{paper_id}/logic-trace` that asserts the response exposes:

1. `paper_metadata`
2. `canonical_core`
3. `derived_views`
4. `quality`

and no longer depends on `logic_steps / claims` field names.

- [ ] **Step 2: Run the focused papers API test and verify it fails**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_papers_api.py -k logic_trace`

Expected: FAIL because the route still exposes the old model contract.

- [ ] **Step 3: Update the route to return the new model**

Keep the route path stable if possible, but change the payload to the canonical export.

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_papers_api.py -k logic_trace`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routers/papers.py backend/tests/test_papers_api.py
git commit -m "feat: serve canonical paper logic trace api"
```

### Task 8: Generate the minimal downstream interface views and verify they are usable

**Files:**
- Modify: `backend/app/paper_logic_trace/derived_views.py`
- Optional Modify: `backend/app/community/candidate_graph.py`
- Optional Modify: `backend/tests/test_community_candidate_graph.py`
- Modify: `backend/tests/test_paper_logic_trace_derived_views.py`

- [ ] **Step 1: Write the failing downstream-interface test**

Add a test that proves:

1. `community_signatures` can be generated from canonical moves
2. `route_feature_candidates` can be generated without reparsing raw text
3. `l1_bridge_hints` preserve provenance links

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_paper_logic_trace_derived_views.py`

Expected: FAIL if any builder still relies on the old `LogicStep / Claim` shape.

- [ ] **Step 3: Implement the minimal downstream view contract**

Keep scope tight:

1. no community algorithm rewrite yet
2. no L1 schema implementation yet
3. only the signatures and candidate features needed to unblock future builders

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `cd backend; .\.venv\Scripts\python.exe -m pytest -q tests/test_paper_logic_trace_derived_views.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/paper_logic_trace/derived_views.py backend/tests/test_paper_logic_trace_derived_views.py
git commit -m "feat: expose downstream paper logic trace interfaces"
```

## Chunk 5: Final Verification and Cleanup

### Task 9: Run focused backend regression coverage for the new L2 path

**Files:**
- No code changes expected

- [ ] **Step 1: Run the focused regression suite**

Run:

```bash
cd backend
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_paper_logic_trace_models.py `
  tests/test_paper_logic_trace_derived_views.py `
  tests/test_paper_logic_trace_compiler.py `
  tests/test_paper_logic_trace_gates.py `
  tests/test_paper_logic_trace_export.py `
  tests/test_phase1_extraction_orchestrator.py `
  tests/test_ingest_llm_progress.py `
  tests/test_papers_api.py
```

Expected: PASS

- [ ] **Step 2: Inspect diff shape**

Run:

```bash
git diff --stat HEAD~9..HEAD
```

Expected: changes are concentrated in `paper_logic_trace/`, extraction/ingest wiring, and paper API surfaces.

- [ ] **Step 3: Run broader backend confidence check**

Run:

```bash
cd backend
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: PASS, or only known pre-existing warnings.

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "test: verify paperlogictrace l2 hard cut"
```

### Task 10: Document the new L2 contract for future L1/L3/L4 builders

**Files:**
- Optional Modify: `README.md`
- Optional Modify: `TECHNICAL_OVERVIEW.zh-CN.md`

- [ ] **Step 1: Add concise contract notes**

Document:

1. `PaperLogicTrace` is the canonical L2 export
2. `ResearchMove` replaces `LogicStep / Claim` as the internal core unit
3. `community_signatures`, `route_feature_candidates`, and `l1_bridge_hints` are derived interfaces

- [ ] **Step 2: Verify docs render cleanly**

Run: inspect the changed markdown locally and confirm links/paths are valid.

- [ ] **Step 3: Commit**

```bash
git add README.md TECHNICAL_OVERVIEW.zh-CN.md
git commit -m "docs: describe paperlogictrace l2 contract"
```
