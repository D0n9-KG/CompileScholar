from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

from app.graph.neo4j_client import Neo4jClient
from app.settings import settings


ScalarFetcher = Callable[[str], int]


CLEANUP_COUNT_QUERIES: dict[str, str] = {
    "propositions": "MATCH (n:Proposition) RETURN count(n) AS cnt",
    "proposition_groups": "MATCH (n:PropositionGroup) RETURN count(n) AS cnt",
    "maps_to_edges": "MATCH (:KnowledgeEntity)-[r:MAPS_TO]->(:Proposition) RETURN count(r) AS cnt",
    "in_group_edges": "MATCH (:Proposition)-[r:IN_GROUP]->(:PropositionGroup) RETURN count(r) AS cnt",
    "proposition_relation_edges": "MATCH (:Proposition)-[r:SUPPORTS|CHALLENGES|SUPERSEDES]->(:Proposition) RETURN count(r) AS cnt",
}


INTEGRITY_CHECK_QUERIES: dict[str, str] = {
    "fusion_communities": "MATCH (fc:FusionCommunity) RETURN count(fc) AS cnt",
    "explains_edges": "MATCH ()-[r:EXPLAINS]->() RETURN count(r) AS cnt",
    "in_community_edges": "MATCH ()-[r:IN_COMMUNITY]->(:FusionCommunity) RETURN count(r) AS cnt",
}


CLEANUP_DELETE_QUERIES: dict[str, str] = {
    "maps_to_edges": """
MATCH (:KnowledgeEntity)-[r:MAPS_TO]->(:Proposition)
WITH collect(r) AS rels, count(r) AS cnt
FOREACH (x IN rels | DELETE x)
RETURN cnt AS cnt
""".strip(),
    "in_group_edges": """
MATCH (:Proposition)-[r:IN_GROUP]->(:PropositionGroup)
WITH collect(r) AS rels, count(r) AS cnt
FOREACH (x IN rels | DELETE x)
RETURN cnt AS cnt
""".strip(),
    "proposition_relation_edges": """
MATCH (:Proposition)-[r:SUPPORTS|CHALLENGES|SUPERSEDES]->(:Proposition)
WITH collect(r) AS rels, count(r) AS cnt
FOREACH (x IN rels | DELETE x)
RETURN cnt AS cnt
""".strip(),
    "proposition_groups": """
MATCH (n:PropositionGroup)
WITH collect(n) AS nodes, count(n) AS cnt
FOREACH (x IN nodes | DETACH DELETE x)
RETURN cnt AS cnt
""".strip(),
    "propositions": """
MATCH (n:Proposition)
WITH collect(n) AS nodes, count(n) AS cnt
FOREACH (x IN nodes | DETACH DELETE x)
RETURN cnt AS cnt
""".strip(),
}


def collect_cleanup_counts(fetch_scalar: ScalarFetcher) -> dict[str, int]:
    return {name: int(fetch_scalar(query)) for name, query in CLEANUP_COUNT_QUERIES.items()}


def collect_fusion_integrity(fetch_scalar: ScalarFetcher) -> dict[str, int]:
    return {name: int(fetch_scalar(query)) for name, query in INTEGRITY_CHECK_QUERIES.items()}


def ensure_cleanup_allowed(integrity: dict[str, int], force: bool = False) -> None:
    if force:
        return

    missing = [
        name
        for name, threshold in (
            ("fusion_communities", 1),
            ("explains_edges", 1),
            ("in_community_edges", 1),
        )
        if int(integrity.get(name, 0)) < threshold
    ]
    if missing:
        detail = ", ".join(f"{k}={integrity.get(k, 0)}" for k in missing)
        raise RuntimeError(f"Fusion pre-check failed: {detail}")


def execute_cleanup(fetch_scalar: ScalarFetcher) -> dict[str, int]:
    return {name: int(fetch_scalar(query)) for name, query in CLEANUP_DELETE_QUERIES.items()}


def _session_scalar(session, query: str) -> int:
    row = session.run(query).single()
    if not row:
        return 0
    value = row.get("cnt")
    try:
        return int(value or 0)
    except Exception:
        return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove legacy Proposition layer after Fusion cutover.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Collect counts and pre-check metrics only.")
    mode.add_argument("--execute", action="store_true", help="Execute cleanup deletion for Proposition layer.")
    parser.add_argument("--force", action="store_true", help="Bypass Fusion integrity pre-check.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    with Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password) as client:
        client.ensure_schema()
        with client._driver.session() as session:  # noqa: SLF001 - script-level maintenance use
            fetch = lambda q: _session_scalar(session, q)
            counts_before = collect_cleanup_counts(fetch)
            integrity = collect_fusion_integrity(fetch)

            output: dict[str, object] = {
                "mode": "dry-run" if args.dry_run else "execute",
                "counts_before": counts_before,
                "integrity": integrity,
            }

            if args.dry_run:
                print(json.dumps(output, ensure_ascii=False, indent=2))
                return 0

            ensure_cleanup_allowed(integrity, force=bool(args.force))
            deleted = execute_cleanup(fetch)
            counts_after = collect_cleanup_counts(fetch)
            output["deleted"] = deleted
            output["counts_after"] = counts_after
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
