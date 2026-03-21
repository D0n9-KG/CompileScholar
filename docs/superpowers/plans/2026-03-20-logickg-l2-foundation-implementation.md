# LogicKG L2 Foundation and Overlapping Communities Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn LogicKG into a stable second-layer research logic foundation by adding canonical `PaperLogicTrace` exports, L2.5 normalization, and a scalable overlapping `LogicStep` community engine that can feed future L3/L4 builders.

**Architecture:** Keep the raw graph as the source of truth, add a canonical export layer for `PaperLogicTrace`, and build a separate overlapping community pipeline over sparse `LogicStep` candidate graphs. Materialize only read-friendly community outputs and research-logic contracts so later `RouteState` and prior builders consume stable assets instead of Neo4j internals.

**Tech Stack:** FastAPI, Neo4j, Python 3.11+, NumPy/SciPy sparse ops, existing similarity embeddings, pytest

---

## File Structure

### New Files

- `backend/app/paper_logic_trace/__init__.py`
- `backend/app/paper_logic_trace/models.py`
- `backend/app/paper_logic_trace/normalization.py`
- `backend/app/paper_logic_trace/exporter.py`
- `backend/app/community/candidate_graph.py`
- `backend/app/community/overlap_detection.py`
- `backend/app/community/labeling.py`
- `backend/app/community/materializer.py`
- `backend/app/community/service_v2.py`
- `backend/research_logic/__init__.py`
- `backend/research_logic/contracts/paper_logic_trace.schema.json`
- `backend/research_logic/contracts/route_state_inputs.schema.json`
- `backend/research_logic/route_builder/topic_scope_builder.py`
- `backend/tests/test_paper_logic_trace_models.py`
- `backend/tests/test_paper_logic_trace_export.py`
- `backend/tests/test_community_candidate_graph.py`
- `backend/tests/test_community_overlap_detection.py`
- `backend/tests/test_community_labeling.py`
- `backend/tests/test_global_community_service_v2.py`
- `backend/tests/test_topic_scope_builder.py`

### Modified Files

- `backend/app/graph/neo4j_client.py`
- `backend/app/community/service.py`
- `backend/app/community/__init__.py`
- `backend/app/api/routers/community.py`
- `backend/app/api/routers/papers.py`
- `backend/app/tasks/handlers.py`
- `backend/app/tasks/models.py`
- `backend/app/main.py`
- `backend/app/settings.py`
- `README.md`
- `TECHNICAL_OVERVIEW.zh-CN.md`

### Responsibility Map

- `paper_logic_trace/`: convert current graph/internal extraction results into stable, versioned export objects.
- `community/candidate_graph.py`: build sparse `LogicStep` candidate neighbors from `SIMILAR_LOGIC`, shared `EXPLAINS`, and citation boosts.
- `community/overlap_detection.py`: detect overlapping communities from sparse candidate graphs without all-pairs matrices.
- `community/labeling.py`: generate interpretable titles, summaries, and representative evidence.
- `community/materializer.py`: convert detector output into read-model rows for storage and APIs.
- `service_v2.py`: orchestrate the new community pipeline while `service.py` remains a compatibility shell.
- `backend/research_logic/`: hold contracts and thin builders that consume canonical L2 exports instead of Neo4j internals.

## Chunk 1: PaperLogicTrace and L2.5 Foundation

### Task 1: Introduce canonical `PaperLogicTrace` models and schema tests

**Files:**
- Create: `backend/app/paper_logic_trace/__init__.py`
- Create: `backend/app/paper_logic_trace/models.py`
- Create: `backend/research_logic/contracts/paper_logic_trace.schema.json`
- Test: `backend/tests/test_paper_logic_trace_models.py`

- [ ] **Step 1: Write the failing test**

```python
from app.paper_logic_trace.models import PaperLogicTrace, LogicStepTrace, ClaimTrace


def test_paper_logic_trace_requires_canonical_sections():
    trace = PaperLogicTrace(
        schema_version='v1',
        paper_metadata={'paper_id': 'paper-1'},
        logic_steps=[LogicStepTrace(logic_step_id='ls-1', step_type='Method', summary='Uses graph encoding')],
        claims=[ClaimTrace(claim_id='cl-1', text='Graph encoding improves retrieval')],
        claim_evidence_links=[],
        figures=[],
        limitations=[],
        future_work_signals=[],
        citation_acts=[],
        quality_tier='A',
    )

    assert trace.schema_version == 'v1'
    assert trace.logic_steps[0].step_type == 'Method'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_paper_logic_trace_models.py`
Expected: FAIL with `ModuleNotFoundError` or missing model definitions.

- [ ] **Step 3: Write minimal implementation**

```python
from pydantic import BaseModel, Field


class LogicStepTrace(BaseModel):
    logic_step_id: str
    step_type: str
    summary: str


class ClaimTrace(BaseModel):
    claim_id: str
    text: str


class PaperLogicTrace(BaseModel):
    schema_version: str = Field(default='v1')
    paper_metadata: dict
    logic_steps: list[LogicStepTrace]
    claims: list[ClaimTrace]
    claim_evidence_links: list[dict]
    figures: list[dict]
    limitations: list[dict]
    future_work_signals: list[dict]
    citation_acts: list[dict]
    quality_tier: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_paper_logic_trace_models.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/paper_logic_trace backend/research_logic/contracts/paper_logic_trace.schema.json backend/tests/test_paper_logic_trace_models.py
git commit -m "feat: add paper logic trace core models"
```

### Task 2: Build `PaperLogicTrace` exporter and L2.5 normalization

**Files:**
- Create: `backend/app/paper_logic_trace/normalization.py`
- Create: `backend/app/paper_logic_trace/exporter.py`
- Modify: `backend/app/graph/neo4j_client.py`
- Test: `backend/tests/test_paper_logic_trace_export.py`

- [ ] **Step 1: Write the failing exporter test**

```python
from app.paper_logic_trace.exporter import export_paper_logic_trace


class _FakeClient:
    def get_paper_logic_trace_inputs(self, paper_id: str):
        return {
            'paper_metadata': {'paper_id': paper_id, 'title': 'Demo'},
            'logic_steps': [{'logic_step_id': 'paper-1:Method', 'step_type': 'Method', 'summary': 'Uses graph encoding'}],
            'claims': [{'claim_id': 'cl-1', 'text': 'Graph encoding improves retrieval', 'step_type': 'Method'}],
            'claim_evidence_links': [{'claim_id': 'cl-1', 'chunk_id': 'chunk-1'}],
            'citation_acts': [],
            'figures': [],
        }


def test_export_paper_logic_trace_adds_l25_slots():
    trace = export_paper_logic_trace(_FakeClient(), 'paper-1')

    assert trace.logic_steps[0].operation_or_method == ['graph encoding']
    assert trace.claims[0].comparison_target == []
    assert trace.quality_tier
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_paper_logic_trace_export.py::test_export_paper_logic_trace_adds_l25_slots`
Expected: FAIL because exporter and Neo4j helper do not exist.

- [ ] **Step 3: Implement minimal exporter and normalization helpers**

```python
def normalize_logic_step(row: dict) -> dict:
    summary = str(row.get('summary') or '').strip()
    return {
        **row,
        'operation_or_method': [summary.lower()] if summary else [],
        'condition_context': [],
        'resource_mentions': [],
    }


def export_paper_logic_trace(client, paper_id: str) -> PaperLogicTrace:
    payload = client.get_paper_logic_trace_inputs(paper_id)
    normalized_steps = [normalize_logic_step(row) for row in payload['logic_steps']]
    ...
```

- [ ] **Step 4: Add the backing Neo4j query helper**

Run: implement `Neo4jClient.get_paper_logic_trace_inputs(paper_id)` that aggregates existing `Paper`, `LogicStep`, `Claim`, evidence, figure, limitation, and citation-act data into one payload.

- [ ] **Step 5: Run focused tests**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_paper_logic_trace_export.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/paper_logic_trace backend/app/graph/neo4j_client.py backend/tests/test_paper_logic_trace_export.py
git commit -m "feat: export canonical paper logic traces"
```

### Task 3: Add API and rebuild hooks for canonical exports

**Files:**
- Modify: `backend/app/api/routers/papers.py`
- Modify: `backend/app/tasks/models.py`
- Modify: `backend/app/tasks/handlers.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_paper_management_api.py`

- [ ] **Step 1: Write the failing API test**

```python
def test_paper_logic_trace_endpoint_returns_canonical_export(client, monkeypatch):
    monkeypatch.setattr(
        'app.api.routers.papers.export_paper_logic_trace',
        lambda *args, **kwargs: {'schema_version': 'v1', 'paper_metadata': {'paper_id': 'paper-1'}},
    )

    response = client.get('/papers/paper-1/logic-trace')

    assert response.status_code == 200
    assert response.json()['schema_version'] == 'v1'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_paper_management_api.py -k logic_trace`
Expected: FAIL with missing route or import.

- [ ] **Step 3: Add the route and optional export task**

```python
@router.get('/{paper_id}/logic-trace')
def get_paper_logic_trace(paper_id: str):
    ...
```

- [ ] **Step 4: Wire rebuild flow to refresh export artifacts after paper rebuild**

Run: update `handle_rebuild_paper` so successful paper rebuild can optionally refresh/export `PaperLogicTrace` assets.

- [ ] **Step 5: Run focused tests**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_paper_management_api.py -k logic_trace`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/routers/papers.py backend/app/tasks/models.py backend/app/tasks/handlers.py backend/app/main.py backend/tests/test_paper_management_api.py
git commit -m "feat: expose canonical paper logic trace exports"
```

## Chunk 2: Overlapping LogicStep Community Engine

### Task 4: Build sparse `LogicStep` candidate graph inputs

**Files:**
- Create: `backend/app/community/candidate_graph.py`
- Modify: `backend/app/graph/neo4j_client.py`
- Test: `backend/tests/test_community_candidate_graph.py`

- [ ] **Step 1: Write the failing candidate-graph test**

```python
from app.community.candidate_graph import build_logicstep_candidate_graph


def test_candidate_graph_ignores_next_and_keeps_cross_paper_neighbors():
    graph = build_logicstep_candidate_graph(
        logic_steps=[
            {'logic_step_id': 'a', 'paper_id': 'p1', 'summary': 'graph encoding'},
            {'logic_step_id': 'b', 'paper_id': 'p2', 'summary': 'relation-aware graph encoding'},
        ],
        similar_logic_edges=[{'source': 'a', 'target': 'b', 'score': 0.91}],
        shared_entity_edges=[],
        citation_boosts=[],
    )

    assert graph['nodes'] == ['a', 'b']
    assert graph['edges'][0]['weight'] > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_community_candidate_graph.py`
Expected: FAIL because builder is missing.

- [ ] **Step 3: Implement minimal candidate graph builder**

```python
def build_logicstep_candidate_graph(*, logic_steps, similar_logic_edges, shared_entity_edges, citation_boosts):
    ...
    return {'nodes': [...], 'edges': [...]}
```

- [ ] **Step 4: Add Neo4j helpers for shared-entity and citation boosts**

Run: add targeted query helpers to `Neo4jClient` for:
- logic-step rows with text/paper metadata
- cross-paper `SIMILAR_LOGIC`
- shared `EXPLAINS` overlaps aggregated as step-step candidates
- paper citation boosts

- [ ] **Step 5: Run focused tests**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_community_candidate_graph.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/community/candidate_graph.py backend/app/graph/neo4j_client.py backend/tests/test_community_candidate_graph.py
git commit -m "feat: build sparse logicstep candidate graphs"
```

### Task 5: Implement overlapping community detection core

**Files:**
- Create: `backend/app/community/overlap_detection.py`
- Modify: `backend/app/settings.py`
- Test: `backend/tests/test_community_overlap_detection.py`

- [ ] **Step 1: Write the failing detector test**

```python
from app.community.overlap_detection import detect_overlapping_communities


def test_detector_allows_logic_step_to_belong_to_multiple_communities():
    result = detect_overlapping_communities(
        nodes=['a', 'b', 'c'],
        edges=[
            {'source': 'a', 'target': 'b', 'weight': 0.9},
            {'source': 'a', 'target': 'c', 'weight': 0.88},
        ],
        max_memberships_per_node=2,
        min_community_size=2,
    )

    memberships = result['memberships']['a']
    assert len(memberships) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_community_overlap_detection.py`
Expected: FAIL because detector is missing.

- [ ] **Step 3: Implement minimal weighted overlap detector**

```python
def detect_overlapping_communities(...):
    # Start with weighted seed neighborhoods, optimize memberships, trim by threshold.
    return {'communities': [...], 'memberships': {...}}
```

- [ ] **Step 4: Add settings for thresholds and caps**

Run: add community-v2 settings such as:
- `global_community_v2_min_size`
- `global_community_v2_max_memberships_per_node`
- `global_community_v2_neighbor_cap`

- [ ] **Step 5: Run focused tests**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_community_overlap_detection.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/community/overlap_detection.py backend/app/settings.py backend/tests/test_community_overlap_detection.py
git commit -m "feat: add overlapping logicstep community detection"
```

### Task 6: Add labeling, materialization, and v2 orchestration

**Files:**
- Create: `backend/app/community/labeling.py`
- Create: `backend/app/community/materializer.py`
- Create: `backend/app/community/service_v2.py`
- Modify: `backend/app/community/service.py`
- Modify: `backend/app/community/__init__.py`
- Modify: `backend/app/api/routers/community.py`
- Modify: `backend/app/tasks/handlers.py`
- Test: `backend/tests/test_community_labeling.py`
- Test: `backend/tests/test_global_community_service_v2.py`

- [ ] **Step 1: Write the failing labeling test**

```python
from app.community.labeling import label_community


def test_labeler_prefers_distinctive_method_phrase_over_generic_tokens():
    label = label_community(
        core_members=[
            {'summary': 'Uses relation-aware graph encoding for reasoning', 'paper_title': 'A'},
            {'summary': 'Proposes relation-aware graph representations', 'paper_title': 'B'},
        ],
        claim_rows=[],
    )

    assert 'relation-aware graph' in label['title'].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_community_labeling.py`
Expected: FAIL because labeler is missing.

- [ ] **Step 3: Implement minimal labeler and read-model materializer**

```python
def label_community(core_members, claim_rows):
    ...


def materialize_community_rows(...):
    return {'communities': [...], 'memberships': [...], 'keywords': [...]}
```

- [ ] **Step 4: Orchestrate the v2 pipeline**

Run: in `service_v2.py`, compose:
- fetch inputs
- build candidate graph
- detect overlaps
- label communities
- materialize rows
- write read model

Then make `community/service.py` delegate to v2 behind a feature flag so rollback stays easy.

- [ ] **Step 5: Run focused tests**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_community_labeling.py tests/test_global_community_service_v2.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/community backend/app/api/routers/community.py backend/app/tasks/handlers.py backend/tests/test_community_labeling.py backend/tests/test_global_community_service_v2.py
git commit -m "feat: ship overlapping global community pipeline"
```

## Chunk 3: Research-Logic Contracts and L3/L4 Handoff

### Task 7: Add `research_logic` contracts and topic-scope builder

**Files:**
- Create: `backend/research_logic/__init__.py`
- Create: `backend/research_logic/contracts/route_state_inputs.schema.json`
- Create: `backend/research_logic/route_builder/topic_scope_builder.py`
- Test: `backend/tests/test_topic_scope_builder.py`

- [ ] **Step 1: Write the failing topic-scope builder test**

```python
from backend.research_logic.route_builder.topic_scope_builder import build_topic_scope_candidates


def test_topic_scope_builder_uses_cross_paper_communities_as_candidates():
    candidates = build_topic_scope_candidates(
        communities=[
            {'community_id': 'c1', 'title': 'Relation-aware graph reasoning', 'paper_count': 4, 'member_ids': ['a', 'b']},
        ]
    )

    assert candidates[0]['topic_scope'] == 'Relation-aware graph reasoning'
    assert candidates[0]['paper_count'] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_topic_scope_builder.py`
Expected: FAIL because module is missing.

- [ ] **Step 3: Implement minimal contracts and builder**

```python
def build_topic_scope_candidates(communities):
    return [
        {'topic_scope': row['title'], 'community_id': row['community_id'], 'paper_count': row['paper_count']}
        for row in communities
        if row.get('paper_count', 0) >= 2
    ]
```

- [ ] **Step 4: Run focused tests**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_topic_scope_builder.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/research_logic backend/tests/test_topic_scope_builder.py
git commit -m "feat: add research logic contracts and topic scope builder"
```

### Task 8: Document rollout, verification, and migration guardrails

**Files:**
- Modify: `README.md`
- Modify: `TECHNICAL_OVERVIEW.zh-CN.md`
- Modify: `docs/superpowers/specs/2026-03-20-logickg-research-logic-l2-foundation-design.md`

- [ ] **Step 1: Write the failing documentation checklist**

Create a checklist in the PR description or scratch notes that requires:
- canonical export path documented
- v2 community semantics documented
- L3/L4 boundary documented

- [ ] **Step 2: Update docs with exact commands**

Add:
- how to export `PaperLogicTrace`
- how to rebuild overlapping communities
- what outputs are read models vs source-of-truth graph data

- [ ] **Step 3: Run final verification**

Run: `cd backend; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_paper_logic_trace_models.py tests/test_paper_logic_trace_export.py tests/test_community_candidate_graph.py tests/test_community_overlap_detection.py tests/test_community_labeling.py tests/test_global_community_service_v2.py tests/test_topic_scope_builder.py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add README.md TECHNICAL_OVERVIEW.zh-CN.md docs/superpowers/specs/2026-03-20-logickg-research-logic-l2-foundation-design.md
git commit -m "docs: document L2 foundation and community v2 rollout"
```
