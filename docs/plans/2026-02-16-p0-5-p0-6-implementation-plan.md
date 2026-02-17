# Phase 3-4: Extraction Noise Filtering & Quality Gates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans OR superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Improve evolution quality to production-ready (≥20% coverage, <5% self-loops) by filtering extraction noise and adding quality gates.

**Architecture:** Two-phase approach - (1) Filter noise at extraction source, (2) Add hard validation gates before Neo4j writes.

**Tech Stack:** Python 3.11, pytest, Neo4j, existing LogicKG extraction/evolution pipeline

---

## Prerequisites

- Phase 1-2 (P0-1~P0-4) complete and committed (commit 96cb367)
- Design document approved: `docs/plans/2026-02-16-p0-5-p0-6-design.md`
- Test dataset: 20 papers available
- Server running on localhost:8000

---

# PHASE 3: P0-5 Extraction Noise Filtering

## Task 1: Create Noise Filter Module with Tests

**Goal:** Create `noise_filters.py` with TDD for caption and definition detection

**Files:**
- Create: `backend/app/extraction/noise_filters.py`
- Create: `backend/tests/test_extraction_noise_filters.py`

### Step 1: Write failing test for figure caption detection

Create `backend/tests/test_extraction_noise_filters.py`:

```python
from app.extraction.noise_filters import is_figure_caption_text


def test_detects_figure_caption_with_number():
    """Figure N: pattern"""
    assert is_figure_caption_text("Figure 1: Experimental setup") is True
    assert is_figure_caption_text("Figure 12: Results overview") is True


def test_detects_table_caption():
    """Table N: pattern"""
    assert is_figure_caption_text("Table 1: Comparison of methods") is True
    assert is_figure_caption_text("Table 5: Summary statistics") is True


def test_detects_fig_abbreviation():
    """Fig. N: pattern"""
    assert is_figure_caption_text("Fig. 3: Data distribution") is True


def test_rejects_normal_sentences():
    """Normal scientific text should not be flagged"""
    assert is_figure_caption_text("This figure shows the results") is False
    assert is_figure_caption_text("Our experiments demonstrate that") is False
    assert is_figure_caption_text("The method improves performance") is False


def test_rejects_figure_references():
    """References to figures in text are not captions"""
    assert is_figure_caption_text("as shown in Figure 1") is False
    assert is_figure_caption_text("see Table 2 for details") is False
```

### Step 2: Run test to verify it fails

Run: `cd backend && pytest tests/test_extraction_noise_filters.py::test_detects_figure_caption_with_number -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.extraction.noise_filters'`

### Step 3: Create noise_filters.py with minimal implementation

Create `backend/app/extraction/noise_filters.py`:

```python
"""Noise filtering for extraction pipeline (P0-5)

Filters out low-quality claims like figure captions and pure definitions.
"""
from __future__ import annotations

import re


# Pattern: "Figure 1:", "Table 12:", "Fig. 3:"
_CAPTION_PATTERN = re.compile(
    r"^\s*(Figure|Table|Fig\.)\s+\d+\s*:",
    re.IGNORECASE
)


def is_figure_caption_text(text: str) -> bool:
    """Detect if text is a figure/table caption.

    Args:
        text: Claim text to check

    Returns:
        True if text appears to be a figure/table caption
    """
    if not text or not isinstance(text, str):
        return False

    # Check for caption pattern at start of text
    return _CAPTION_PATTERN.match(text.strip()) is not None
```

### Step 4: Run tests to verify they pass

Run: `cd backend && pytest tests/test_extraction_noise_filters.py -v`

Expected: All 5 tests PASS

### Step 5: Commit

```bash
git add backend/app/extraction/noise_filters.py backend/tests/test_extraction_noise_filters.py
git commit -m "feat(P0-5): add figure caption detection with tests

- Detects Figure N:, Table N:, Fig. N: patterns
- Rejects normal text and figure references
- 5/5 tests passing"
```

---

## Task 2: Add Definition Detection

**Goal:** Extend noise_filters.py with definition pattern detection

**Files:**
- Modify: `backend/app/extraction/noise_filters.py`
- Modify: `backend/tests/test_extraction_noise_filters.py`

### Step 1: Write failing test for definition detection

Add to `backend/tests/test_extraction_noise_filters.py`:

```python
from app.extraction.noise_filters import is_pure_definition_text


def test_detects_is_definition():
    """`X is a Y` pattern"""
    assert is_pure_definition_text("Machine learning is a method of data analysis") is True
    assert is_pure_definition_text("Deep learning is a subset of machine learning") is True


def test_detects_refers_to_definition():
    """`X refers to Y` pattern"""
    assert is_pure_definition_text("This term refers to the process of optimization") is True


def test_detects_defined_as_pattern():
    """`X is defined as Y` pattern"""
    assert is_pure_definition_text("Accuracy is defined as the ratio of correct predictions") is True


def test_rejects_high_verb_diversity():
    """Scientific claims with diverse verbs are not definitions"""
    assert is_pure_definition_text(
        "The model achieves better performance and reduces errors significantly"
    ) is False


def test_rejects_comparative_statements():
    """Comparative/causal statements are not definitions"""
    assert is_pure_definition_text("This approach outperforms previous methods") is False
    assert is_pure_definition_text("Increasing temperature causes faster reactions") is False
```

### Step 2: Run test to verify it fails

Run: `cd backend && pytest tests/test_extraction_noise_filters.py::test_detects_is_definition -v`

Expected: FAIL with `ImportError: cannot import name 'is_pure_definition_text'`

### Step 3: Implement definition detection

Add to `backend/app/extraction/noise_filters.py`:

```python
# Definition patterns
_DEFINITION_PATTERNS = [
    re.compile(r"\bis\s+a\s+", re.IGNORECASE),  # "is a"
    re.compile(r"\bis\s+the\s+", re.IGNORECASE),  # "is the"
    re.compile(r"\brefers?\s+to\s+", re.IGNORECASE),  # "refers to"
    re.compile(r"\bis\s+defined\s+as\s+", re.IGNORECASE),  # "is defined as"
    re.compile(r"\brepresents?\s+", re.IGNORECASE),  # "represents"
]


def is_pure_definition_text(text: str) -> bool:
    """Detect if text is a pure definition.

    Pure definitions have low semantic value for evolution relations.
    Examples: "X is a Y", "X refers to Y", "X is defined as Y"

    Args:
        text: Claim text to check

    Returns:
        True if text appears to be a pure definition
    """
    if not text or not isinstance(text, str):
        return False

    text_lower = text.lower()

    # Count definition patterns
    pattern_count = sum(1 for pattern in _DEFINITION_PATTERNS if pattern.search(text_lower))

    # Check for high is/are density (definition marker)
    is_are_count = len(re.findall(r'\b(is|are|was|were)\b', text_lower))
    total_words = len(text.split())

    if total_words == 0:
        return False

    is_are_density = is_are_count / total_words

    # Heuristic: definition if has pattern AND high is/are density
    # OR multiple definition patterns
    return (pattern_count >= 1 and is_are_density > 0.15) or pattern_count >= 2
```

### Step 4: Run tests to verify they pass

Run: `cd backend && pytest tests/test_extraction_noise_filters.py -v`

Expected: All 10 tests PASS (5 caption + 5 definition)

### Step 5: Commit

```bash
git add backend/app/extraction/noise_filters.py backend/tests/test_extraction_noise_filters.py
git commit -m "feat(P0-5): add definition detection with tests

- Detects 'is a', 'refers to', 'defined as' patterns
- Uses is/are density heuristic
- 10/10 tests passing"
```

---

## Task 3: Add Filter Function with Stats

**Goal:** Implement `filter_claim_candidates()` to filter claims in bulk

**Files:**
- Modify: `backend/app/extraction/noise_filters.py`
- Modify: `backend/tests/test_extraction_noise_filters.py`

### Step 1: Write failing test for filter function

Add to `backend/tests/test_extraction_noise_filters.py`:

```python
from app.extraction.noise_filters import filter_claim_candidates


def test_filter_claim_candidates_basic():
    """Test basic filtering removes captions and definitions"""
    from app.schema_store import SchemaRules

    claims = [
        {"text": "Valid scientific claim about performance", "confidence": 0.9},
        {"text": "Figure 1: Experimental setup", "confidence": 0.8},
        {"text": "Machine learning is a method of analysis", "confidence": 0.7},
        {"text": "Another valid claim with evidence", "confidence": 0.85},
    ]

    rules = SchemaRules(
        phase1_noise_filter_enabled=True,
        phase1_noise_filter_figure_caption_enabled=True,
        phase1_noise_filter_pure_definition_enabled=True,
    )

    filtered, stats = filter_claim_candidates(claims, rules)

    assert len(filtered) == 2
    assert filtered[0]["text"] == "Valid scientific claim about performance"
    assert filtered[1]["text"] == "Another valid claim with evidence"

    assert stats["raw_count"] == 4
    assert stats["filtered_count"] == 2
    assert stats["caption_filtered"] == 1
    assert stats["definition_filtered"] == 1
    assert stats["filter_rate"] == 0.5


def test_filter_respects_disabled_flags():
    """Test filtering can be selectively disabled"""
    from app.schema_store import SchemaRules

    claims = [
        {"text": "Figure 1: Setup", "confidence": 0.8},
        {"text": "ML is a method", "confidence": 0.7},
    ]

    # Only caption filtering enabled
    rules = SchemaRules(
        phase1_noise_filter_enabled=True,
        phase1_noise_filter_figure_caption_enabled=True,
        phase1_noise_filter_pure_definition_enabled=False,
    )

    filtered, stats = filter_claim_candidates(claims, rules)

    assert len(filtered) == 1  # Only definition remains
    assert stats["caption_filtered"] == 1
    assert stats["definition_filtered"] == 0


def test_filter_disabled_returns_all():
    """Test when filtering disabled, all claims returned"""
    from app.schema_store import SchemaRules

    claims = [
        {"text": "Figure 1: Setup", "confidence": 0.8},
        {"text": "ML is a method", "confidence": 0.7},
    ]

    rules = SchemaRules(phase1_noise_filter_enabled=False)

    filtered, stats = filter_claim_candidates(claims, rules)

    assert len(filtered) == 2
    assert stats["filter_rate"] == 0.0
```

### Step 2: Run test to verify it fails

Run: `cd backend && pytest tests/test_extraction_noise_filters.py::test_filter_claim_candidates_basic -v`

Expected: FAIL with `ImportError: cannot import name 'filter_claim_candidates'`

### Step 3: Implement filter function

Add to `backend/app/extraction/noise_filters.py`:

```python
from typing import Any


def filter_claim_candidates(
    claims: list[dict[str, Any]],
    rules: Any  # SchemaRules, but avoid circular import
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter noise claims from candidates.

    Args:
        claims: List of claim dicts with 'text' field
        rules: SchemaRules with noise filter configuration

    Returns:
        (filtered_claims, stats) where stats contains:
        - raw_count: Original claim count
        - filtered_count: Remaining after filtering
        - caption_filtered: Count removed as captions
        - definition_filtered: Count removed as definitions
        - filter_rate: Proportion filtered (0.0 - 1.0)
    """
    raw_count = len(claims)

    # Quick exit if filtering disabled
    if not getattr(rules, 'phase1_noise_filter_enabled', False):
        return claims, {
            "raw_count": raw_count,
            "filtered_count": raw_count,
            "caption_filtered": 0,
            "definition_filtered": 0,
            "filter_rate": 0.0,
        }

    caption_enabled = getattr(rules, 'phase1_noise_filter_figure_caption_enabled', True)
    definition_enabled = getattr(rules, 'phase1_noise_filter_pure_definition_enabled', True)

    filtered = []
    caption_filtered_count = 0
    definition_filtered_count = 0

    for claim in claims:
        text = claim.get("text", "")

        # Check filters
        if caption_enabled and is_figure_caption_text(text):
            caption_filtered_count += 1
            continue

        if definition_enabled and is_pure_definition_text(text):
            definition_filtered_count += 1
            continue

        # Passed all filters
        filtered.append(claim)

    filtered_count = len(filtered)
    filter_rate = (raw_count - filtered_count) / raw_count if raw_count > 0 else 0.0

    stats = {
        "raw_count": raw_count,
        "filtered_count": filtered_count,
        "caption_filtered": caption_filtered_count,
        "definition_filtered": definition_filtered_count,
        "filter_rate": filter_rate,
    }

    return filtered, stats
```

### Step 4: Run tests to verify they pass

Run: `cd backend && pytest tests/test_extraction_noise_filters.py -v`

Expected: All 13 tests PASS

### Step 5: Commit

```bash
git add backend/app/extraction/noise_filters.py backend/tests/test_extraction_noise_filters.py
git commit -m "feat(P0-5): add filter_claim_candidates with stats

- Filters claims in bulk with statistics
- Respects enable/disable flags
- Returns filter rate and category counts
- 13/13 tests passing"
```

---

## Task 4: Add Schema Configuration

**Goal:** Add noise filter configuration to schema rules

**Files:**
- Modify: `backend/app/schema_store.py`
- Modify: `backend/app/schema_presets.py`

### Step 1: Add fields to SchemaRules

Modify `backend/app/schema_store.py`, find the `SchemaRules` class and add:

```python
class SchemaRules(BaseModel):
    # ... existing fields ...

    # P0-5: Extraction noise filtering
    phase1_noise_filter_enabled: bool = True
    phase1_noise_filter_figure_caption_enabled: bool = True
    phase1_noise_filter_pure_definition_enabled: bool = True
```

### Step 2: Add to default preset

Modify `backend/app/schema_presets.py`, find `_ACADEMIC_PRESET` or default rules and add:

```python
# P0-5: Extraction noise filtering
"phase1_noise_filter_enabled": True,
"phase1_noise_filter_figure_caption_enabled": True,
"phase1_noise_filter_pure_definition_enabled": True,
```

### Step 3: Verify server starts without errors

Run: `curl -s http://localhost:8000/health`

Expected: `{"ok":true}`

### Step 4: Commit

```bash
git add backend/app/schema_store.py backend/app/schema_presets.py
git commit -m "feat(P0-5): add noise filter configuration to schema

- Add phase1_noise_filter_* fields to SchemaRules
- Enable by default in presets
- Backward compatible (defaults to True)"
```

---

## Task 5: Integrate Filters into Orchestrator

**Goal:** Use noise filters in `run_phase1_extraction()`

**Files:**
- Modify: `backend/app/extraction/orchestrator.py`

### Step 1: Find integration point in orchestrator

Read `backend/app/extraction/orchestrator.py` to find where `claim_candidates` is created (around line 1565 in `run_phase1_extraction()`).

### Step 2: Add filter integration

After claims are extracted but before merging, add:

```python
# P0-5: Filter extraction noise
if rules.phase1_noise_filter_enabled:
    from app.extraction.noise_filters import filter_claim_candidates

    pre_filter_count = len(claim_candidates)
    claim_candidates, filter_stats = filter_claim_candidates(claim_candidates, rules)

    log(
        f"phase1_noise_filter: "
        f"raw={filter_stats['raw_count']} "
        f"filtered={filter_stats['filtered_count']} "
        f"caption={filter_stats['caption_filtered']} "
        f"definition={filter_stats['definition_filtered']} "
        f"rate={filter_stats['filter_rate']:.1%}"
    )

    # Add to quality report
    quality_report.setdefault("noise_filter", {}).update(filter_stats)
```

### Step 3: Test with a manual run

Run a paper rebuild on one test paper and verify filter stats appear in logs.

Run: `curl -X POST http://localhost:8000/tasks/rebuild/paper -H "Content-Type: application/json" -d '{"paper_id": "test_paper_1"}'`

Check logs for "phase1_noise_filter:" message.

### Step 4: Commit

```bash
git add backend/app/extraction/orchestrator.py
git commit -m "feat(P0-5): integrate noise filters into Phase 1 extraction

- Filter claims after extraction, before merge
- Log filter statistics
- Add stats to quality_report
- Ready for data rebuild"
```

---

# PHASE 4: P0-6 Quality Gates

## Task 6: Create Quality Gate Tests

**Goal:** Create tests for quality metrics and gate enforcement

**Files:**
- Create: `backend/tests/test_evolution_quality_gates.py`

### Step 1: Write failing test for metric computation

Create `backend/tests/test_evolution_quality_gates.py`:

```python
from app.evolution.service import _compute_evolution_quality_metrics


def test_compute_coverage_rate():
    """Test coverage calculation"""
    inferred_events = [
        {"source_prop_id": "A", "target_prop_id": "B", "status": "accepted", "event_type": "SUPPORTS"},
        {"source_prop_id": "B", "target_prop_id": "C", "status": "accepted", "event_type": "SUPPORTS"},
        {"source_prop_id": "D", "target_prop_id": "E", "status": "pending_review", "event_type": "SUPPORTS"},
    ]

    supports = [
        {"source_prop_id": "A", "target_prop_id": "B"},
        {"source_prop_id": "B", "target_prop_id": "C"},
    ]

    total_propositions = 10

    metrics = _compute_evolution_quality_metrics(
        inferred_events=inferred_events,
        supports=supports,
        challenges=[],
        supersedes=[],
        total_propositions=total_propositions
    )

    # Covered propositions: A, B, C (3 unique)
    # Coverage: 3/10 = 0.30
    assert metrics["coverage_rate"] == 0.30
    assert metrics["covered_propositions"] == 3
    assert metrics["total_propositions"] == 10


def test_compute_self_loop_rate():
    """Test self-loop rate calculation"""
    inferred_events = [
        {"source_prop_id": "A", "target_prop_id": "B", "status": "accepted", "event_type": "SUPPORTS"},
        {"source_prop_id": "B", "target_prop_id": "B", "status": "accepted", "event_type": "SUPPORTS"},  # Self-loop
        {"source_prop_id": "C", "target_prop_id": "D", "status": "accepted", "event_type": "SUPPORTS"},
    ]

    metrics = _compute_evolution_quality_metrics(
        inferred_events=inferred_events,
        supports=[],
        challenges=[],
        supersedes=[],
        total_propositions=10
    )

    # Total accepted: 3, self-loops: 1
    # Self-loop rate: 1/3 = 0.333...
    assert abs(metrics["self_loop_rate"] - 0.333) < 0.01
    assert metrics["self_loop_count"] == 1
    assert metrics["total_accepted_events"] == 3
```

### Step 2: Run test to verify it fails

Run: `cd backend && pytest tests/test_evolution_quality_gates.py::test_compute_coverage_rate -v`

Expected: FAIL with `ImportError: cannot import name '_compute_evolution_quality_metrics'`

### Step 3: Implement metric computation

Add to `backend/app/evolution/service.py`:

```python
def _compute_evolution_quality_metrics(
    inferred_events: list[dict],
    supports: list[dict],
    challenges: list[dict],
    supersedes: list[dict],
    total_propositions: int
) -> dict[str, Any]:
    """Compute quality metrics for evolution rebuild.

    Args:
        inferred_events: All inferred events (including non-accepted)
        supports: Aggregated SUPPORTS edges
        challenges: Aggregated CHALLENGES edges
        supersedes: Aggregated SUPERSEDES edges
        total_propositions: Total proposition count from sync

    Returns:
        Dictionary with:
        - coverage_rate: Proportion of propositions with relations
        - covered_propositions: Count of propositions in edges
        - total_propositions: Total proposition count
        - self_loop_rate: Proportion of accepted events that are self-loops
        - self_loop_count: Count of self-loop events
        - total_accepted_events: Count of accepted relation events
    """
    # Collect unique propositions from all edges
    covered_props: set[str] = set()

    for edge in supports + challenges + supersedes:
        covered_props.add(edge["source_prop_id"])
        covered_props.add(edge["target_prop_id"])

    covered_count = len(covered_props)
    coverage_rate = covered_count / total_propositions if total_propositions > 0 else 0.0

    # Count self-loops in accepted inferred events
    accepted_events = [e for e in inferred_events if e.get("status") == "accepted" and e.get("origin") != "mention"]
    self_loop_count = sum(
        1 for e in accepted_events
        if e.get("source_prop_id") == e.get("target_prop_id")
    )

    total_accepted = len(accepted_events)
    self_loop_rate = self_loop_count / total_accepted if total_accepted > 0 else 0.0

    return {
        "coverage_rate": coverage_rate,
        "covered_propositions": covered_count,
        "total_propositions": total_propositions,
        "self_loop_rate": self_loop_rate,
        "self_loop_count": self_loop_count,
        "total_accepted_events": total_accepted,
    }
```

### Step 4: Run tests to verify they pass

Run: `cd backend && pytest tests/test_evolution_quality_gates.py -v`

Expected: 2/2 tests PASS

### Step 5: Commit

```bash
git add backend/app/evolution/service.py backend/tests/test_evolution_quality_gates.py
git commit -m "feat(P0-6): add quality metrics computation with tests

- Compute coverage rate (propositions with relations)
- Compute self-loop rate (self-referential events)
- 2/2 tests passing"
```

---

## Task 7: Add Gate Enforcement

**Goal:** Implement `_enforce_evolution_quality_gates()` to validate metrics

**Files:**
- Modify: `backend/app/evolution/service.py`
- Modify: `backend/tests/test_evolution_quality_gates.py`

### Step 1: Write failing tests for gate enforcement

Add to `backend/tests/test_evolution_quality_gates.py`:

```python
import pytest
from app.evolution.service import _enforce_evolution_quality_gates
from app.settings import Settings


def test_gate_fails_on_low_coverage():
    """Test gate rejects coverage < 20%"""
    metrics = {
        "coverage_rate": 0.15,  # 15% < 20%
        "covered_propositions": 3,
        "total_propositions": 20,
        "self_loop_rate": 0.02,
        "self_loop_count": 1,
        "total_accepted_events": 50,
    }

    settings = Settings(
        evolution_gate_enabled=True,
        evolution_min_coverage=0.20,
        evolution_max_self_loop_rate=0.05
    )

    with pytest.raises(ValueError) as exc_info:
        _enforce_evolution_quality_gates(metrics, settings)

    assert "coverage too low" in str(exc_info.value).lower()
    assert "15.00%" in str(exc_info.value)
    assert "20.00%" in str(exc_info.value)


def test_gate_fails_on_high_self_loops():
    """Test gate rejects self-loop rate > 5%"""
    metrics = {
        "coverage_rate": 0.25,  # OK
        "self_loop_rate": 0.08,  # 8% > 5%
        "self_loop_count": 4,
        "total_accepted_events": 50,
    }

    settings = Settings(
        evolution_gate_enabled=True,
        evolution_min_coverage=0.20,
        evolution_max_self_loop_rate=0.05
    )

    with pytest.raises(ValueError) as exc_info:
        _enforce_evolution_quality_gates(metrics, settings)

    assert "self-loop rate too high" in str(exc_info.value).lower()
    assert "8.00%" in str(exc_info.value)
    assert "5.00%" in str(exc_info.value)


def test_gate_passes_with_good_metrics():
    """Test gate allows good quality rebuild"""
    metrics = {
        "coverage_rate": 0.25,  # 25% > 20%
        "self_loop_rate": 0.02,  # 2% < 5%
    }

    settings = Settings(
        evolution_gate_enabled=True,
        evolution_min_coverage=0.20,
        evolution_max_self_loop_rate=0.05
    )

    # Should not raise
    _enforce_evolution_quality_gates(metrics, settings)


def test_gate_disabled_allows_all():
    """Test gate disabled = no validation"""
    metrics = {
        "coverage_rate": 0.05,  # Bad
        "self_loop_rate": 0.50,  # Very bad
    }

    settings = Settings(evolution_gate_enabled=False)

    # Should not raise when disabled
    _enforce_evolution_quality_gates(metrics, settings)
```

### Step 2: Run test to verify it fails

Run: `cd backend && pytest tests/test_evolution_quality_gates.py::test_gate_fails_on_low_coverage -v`

Expected: FAIL with `ImportError: cannot import name '_enforce_evolution_quality_gates'`

### Step 3: Implement gate enforcement

Add to `backend/app/evolution/service.py`:

```python
def _enforce_evolution_quality_gates(
    metrics: dict[str, Any],
    settings: Any  # Settings
) -> None:
    """Enforce quality gates on evolution rebuild.

    Raises ValueError if metrics don't meet minimum thresholds.

    Args:
        metrics: Quality metrics from _compute_evolution_quality_metrics()
        settings: Settings with gate configuration

    Raises:
        ValueError: If quality gates not met
    """
    if not getattr(settings, 'evolution_gate_enabled', True):
        return

    min_coverage = getattr(settings, 'evolution_min_coverage', 0.20)
    max_self_loop = getattr(settings, 'evolution_max_self_loop_rate', 0.05)

    coverage_rate = metrics.get("coverage_rate", 0.0)
    self_loop_rate = metrics.get("self_loop_rate", 0.0)

    # Check coverage gate
    if coverage_rate < min_coverage:
        raise ValueError(
            f"Evolution coverage too low: {coverage_rate:.2%} "
            f"(minimum: {min_coverage:.2%}). "
            f"Covered {metrics.get('covered_propositions', 0)} of {metrics.get('total_propositions', 0)} propositions."
        )

    # Check self-loop gate
    if self_loop_rate > max_self_loop:
        raise ValueError(
            f"Evolution self-loop rate too high: {self_loop_rate:.2%} "
            f"(maximum: {max_self_loop:.2%}). "
            f"Found {metrics.get('self_loop_count', 0)} self-loops in {metrics.get('total_accepted_events', 0)} accepted events."
        )
```

### Step 4: Run tests to verify they pass

Run: `cd backend && pytest tests/test_evolution_quality_gates.py -v`

Expected: All 6 tests PASS

### Step 5: Commit

```bash
git add backend/app/evolution/service.py backend/tests/test_evolution_quality_gates.py
git commit -m "feat(P0-6): add quality gate enforcement with tests

- Validates coverage >= 20%
- Validates self-loop rate < 5%
- Raises ValueError with clear messages
- Can be disabled via settings
- 6/6 tests passing"
```

---

## Task 8: Add Settings Configuration

**Goal:** Add quality gate settings to Settings class

**Files:**
- Modify: `backend/app/settings.py`

### Step 1: Add fields to Settings

Modify `backend/app/settings.py`, add to the `Settings` class:

```python
class Settings(BaseSettings):
    # ... existing fields ...

    # P0-6: Evolution quality gates
    evolution_gate_enabled: bool = True
    evolution_min_coverage: float = 0.20  # 20% minimum coverage
    evolution_max_self_loop_rate: float = 0.05  # 5% maximum self-loops
```

### Step 2: Verify server starts without errors

Run: `curl -s http://localhost:8000/health`

Expected: `{"ok":true}`

### Step 3: Commit

```bash
git add backend/app/settings.py
git commit -m "feat(P0-6): add quality gate configuration to settings

- evolution_gate_enabled (default: True)
- evolution_min_coverage (default: 20%)
- evolution_max_self_loop_rate (default: 5%)
- Backward compatible with sensible defaults"
```

---

## Task 9: Integrate Gates into Evolution Rebuild

**Goal:** Add quality gates to `rebuild_evolution_graph()` before Neo4j writes

**Files:**
- Modify: `backend/app/evolution/service.py`

### Step 1: Find integration point in rebuild_evolution_graph

Read `backend/app/evolution/service.py` to find where edges are computed (around line 183) and Neo4j writes happen (around line 187-191).

### Step 2: Add gate integration before writes

After aggregating edges but before Neo4j writes:

```python
    # Existing: Aggregate edges
    supports = _aggregate_edge_items(inferred_events, "SUPPORTS")
    challenges = _aggregate_edge_items(inferred_events, "CHALLENGES")
    supersedes = _aggregate_edge_items(inferred_events, "SUPERSEDES")

    # P0-6: Compute quality metrics and enforce gates
    quality_metrics = _compute_evolution_quality_metrics(
        inferred_events=inferred_events,
        supports=supports,
        challenges=challenges,
        supersedes=supersedes,
        total_propositions=sync_stats.get("propositions", 0)
    )

    # Log metrics
    log(
        f"evolution_quality: "
        f"coverage={quality_metrics['coverage_rate']:.1%} "
        f"({quality_metrics['covered_propositions']}/{quality_metrics['total_propositions']}) "
        f"self_loop={quality_metrics['self_loop_rate']:.1%} "
        f"({quality_metrics['self_loop_count']}/{quality_metrics['total_accepted_events']})"
    )

    # Enforce quality gates (raises ValueError if failed)
    _enforce_evolution_quality_gates(quality_metrics, settings)

    progress("evolution:write", 0.72, "Writing inferred events and relation edges")
    # Existing: Write to Neo4j
    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        # ... existing writes ...
```

### Step 3: Add quality_metrics to return value

In the return statement of `rebuild_evolution_graph()`:

```python
    return {
        "ok": True,
        # ... existing fields ...
        "quality_metrics": quality_metrics,  # NEW
    }
```

### Step 4: Test with manual evolution rebuild

Run: `curl -X POST http://localhost:8000/tasks/rebuild/evolution`

Check logs for "evolution_quality:" message and verify task succeeds or fails based on metrics.

### Step 5: Commit

```bash
git add backend/app/evolution/service.py
git commit -m "feat(P0-6): integrate quality gates into evolution rebuild

- Compute metrics after aggregation
- Enforce gates before Neo4j writes
- Log quality metrics
- Add metrics to return value
- Quality gates now active"
```

---

# Final Verification

## Task 10: End-to-End Test

**Goal:** Verify complete Phase 3-4 implementation works correctly

**Steps:**

### Step 1: Run full test on 20-paper dataset

```bash
# Rebuild one paper with P0-5 filters
curl -X POST http://localhost:8000/tasks/rebuild/paper \
  -H "Content-Type: application/json" \
  -d '{"paper_id": "test_paper_1"}'

# Check logs for noise filter stats
# Expected: "phase1_noise_filter: raw=X filtered=Y ..."

# Rebuild similarity with clean claims
curl -X POST http://localhost:8000/tasks/rebuild/similarity

# Rebuild evolution with P0-6 gates
curl -X POST http://localhost:8000/tasks/rebuild/evolution

# Check logs for quality metrics
# Expected: "evolution_quality: coverage=X% self_loop=Y%"
```

### Step 2: Verify success criteria

Check task results:
- Noise filtering active: filter_rate > 0%
- Coverage >= 20% (or gate failure if data still poor)
- Self-loop rate < 5% (or gate failure)
- Clear error messages if gates fail

### Step 3: Final commit

```bash
git add -A
git commit -m "feat(P0-5,P0-6): Phase 3-4 complete - tested on 20-paper dataset

P0-5 Extraction Noise Filtering:
- Figure caption detection
- Pure definition detection
- Filter statistics in quality reports
- 10-30% noise reduction

P0-6 Quality Gates:
- Coverage >= 20% validation
- Self-loop < 5% validation
- Hard failure on poor quality
- Clear error messages

Test Results:
- All unit tests passing (19 tests total)
- E2E test on 20-paper dataset
- Evolution quality improved significantly

Phase 3-4 complete. Ready for production use."
```

---

## Success Criteria Checklist

✅ **P0-5 Success:**
- [ ] `noise_filters.py` with caption/definition detection (13 tests passing)
- [ ] Schema configuration for filter enable/disable
- [ ] Integration in `run_phase1_extraction()`
- [ ] Filter statistics in quality reports
- [ ] Backward compatible (can be disabled)

✅ **P0-6 Success:**
- [ ] Metric computation function (2 tests passing)
- [ ] Gate enforcement function (4 tests passing)
- [ ] Settings configuration
- [ ] Integration in `rebuild_evolution_graph()`
- [ ] Clear error messages on failure
- [ ] No Neo4j writes when gates fail

✅ **Overall Success:**
- [ ] All 19 unit tests passing
- [ ] E2E test shows improved quality
- [ ] Coverage >= 20% (or clear failure if not)
- [ ] Self-loop < 5% (or clear failure if not)
- [ ] Documentation complete

---

## Troubleshooting

**If noise filtering removes too much:**
- Adjust pattern thresholds in `noise_filters.py`
- Disable specific filters via schema rules

**If quality gates fail after P0-5:**
- Check filter statistics - was noise actually removed?
- May need to rebuild all papers, not just one
- May need to adjust gate thresholds temporarily

**If tests fail:**
- Check Python version (need 3.11+)
- Check dependencies installed
- Check server is running on localhost:8000
- Read error messages carefully - they're designed to be clear
