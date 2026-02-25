import pytest

from scripts.migrate.remove_proposition_layer import (
    CLEANUP_COUNT_QUERIES,
    CLEANUP_DELETE_QUERIES,
    INTEGRITY_CHECK_QUERIES,
    collect_cleanup_counts,
    collect_fusion_integrity,
    ensure_cleanup_allowed,
    execute_cleanup,
)


def _fetch_from_map(mapping: dict[str, int]):
    def _fetch(query: str) -> int:
        if query not in mapping:
            raise AssertionError(f"Unexpected query: {query}")
        return mapping[query]

    return _fetch


def test_dry_run_collects_counts_for_targeted_subgraph() -> None:
    values = {
        CLEANUP_COUNT_QUERIES["propositions"]: 11,
        CLEANUP_COUNT_QUERIES["proposition_groups"]: 3,
        CLEANUP_COUNT_QUERIES["maps_to_edges"]: 25,
        CLEANUP_COUNT_QUERIES["in_group_edges"]: 10,
        CLEANUP_COUNT_QUERIES["proposition_relation_edges"]: 7,
    }
    counts = collect_cleanup_counts(_fetch_from_map(values))

    assert counts == {
        "propositions": 11,
        "proposition_groups": 3,
        "maps_to_edges": 25,
        "in_group_edges": 10,
        "proposition_relation_edges": 7,
    }


def test_execute_cleanup_returns_only_targeted_delete_stats() -> None:
    values = {
        CLEANUP_DELETE_QUERIES["maps_to_edges"]: 25,
        CLEANUP_DELETE_QUERIES["in_group_edges"]: 10,
        CLEANUP_DELETE_QUERIES["proposition_relation_edges"]: 7,
        CLEANUP_DELETE_QUERIES["proposition_groups"]: 3,
        CLEANUP_DELETE_QUERIES["propositions"]: 11,
    }
    deleted = execute_cleanup(_fetch_from_map(values))

    assert deleted == {
        "maps_to_edges": 25,
        "in_group_edges": 10,
        "proposition_relation_edges": 7,
        "proposition_groups": 3,
        "propositions": 11,
    }


def test_cleanup_is_blocked_when_fusion_artifacts_not_ready() -> None:
    values = {
        INTEGRITY_CHECK_QUERIES["fusion_communities"]: 0,
        INTEGRITY_CHECK_QUERIES["explains_edges"]: 0,
        INTEGRITY_CHECK_QUERIES["in_community_edges"]: 0,
    }
    integrity = collect_fusion_integrity(_fetch_from_map(values))

    with pytest.raises(RuntimeError, match="Fusion pre-check failed"):
        ensure_cleanup_allowed(integrity, force=False)


def test_cleanup_can_be_forced_even_if_precheck_fails() -> None:
    values = {
        INTEGRITY_CHECK_QUERIES["fusion_communities"]: 0,
        INTEGRITY_CHECK_QUERIES["explains_edges"]: 0,
        INTEGRITY_CHECK_QUERIES["in_community_edges"]: 0,
    }
    integrity = collect_fusion_integrity(_fetch_from_map(values))

    ensure_cleanup_allowed(integrity, force=True)
