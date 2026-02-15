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


def test_mention_origin_excluded_from_self_loops():
    """Test that mention-origin events are excluded from self-loop counting"""
    inferred_events = [
        {"source_prop_id": "A", "target_prop_id": "A", "status": "accepted", "origin": "mention"},  # Mention self-loop (excluded)
        {"source_prop_id": "B", "target_prop_id": "B", "status": "accepted", "origin": "inferred"},  # Inferred self-loop (counted)
        {"source_prop_id": "C", "target_prop_id": "D", "status": "accepted"},  # Normal event (no origin field)
    ]

    metrics = _compute_evolution_quality_metrics(
        inferred_events=inferred_events,
        supports=[],
        challenges=[],
        supersedes=[],
        total_propositions=10
    )

    # Only events without origin="mention" count: 2 accepted (B->B self-loop, C->D normal)
    # Self-loops: 1 (B->B)
    assert metrics["self_loop_count"] == 1
    assert metrics["total_accepted_events"] == 2
    assert metrics["self_loop_rate"] == 0.5


def test_zero_denominators():
    """Test behavior with zero denominators"""
    metrics = _compute_evolution_quality_metrics(
        inferred_events=[],
        supports=[],
        challenges=[],
        supersedes=[],
        total_propositions=0
    )

    assert metrics["coverage_rate"] == 0.0
    assert metrics["self_loop_rate"] == 0.0
    assert metrics["covered_propositions"] == 0
    assert metrics["self_loop_count"] == 0
    assert metrics["total_accepted_events"] == 0
