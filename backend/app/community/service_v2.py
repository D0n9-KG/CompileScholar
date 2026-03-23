from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Callable

from app.community.candidate_graph import build_move_candidate_graph
from app.community.labeling import label_community
from app.community.materializer import materialize_community_rows
from app.community.overlap_detection import detect_overlapping_communities
from app.community.refinement import disambiguate_community_labels, merge_labeled_communities
from app.graph.neo4j_client import Neo4jClient
from app.settings import settings


ProgressFn = Callable[[str, float, str | None], None]
LogFn = Callable[[str], None]

_WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?", re.IGNORECASE)
_GENERIC_TOKENS = {
    'a',
    'an',
    'and',
    'approach',
    'based',
    'for',
    'in',
    'method',
    'methods',
    'model',
    'of',
    'on',
    'paper',
    'problem',
    'research',
    'result',
    'results',
    'study',
    'system',
    'the',
    'this',
    'to',
    'using',
    'with',
}


def _noop_progress(stage: str, p: float, msg: str | None = None) -> None:
    del stage, p, msg


def _noop_log(line: str) -> None:
    del line


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _filter_materialized_communities(
    *,
    communities: list[dict[str, Any]],
    memberships: dict[str, list[dict[str, Any]]],
    min_member_count: int,
) -> dict[str, Any]:
    safe_min = max(2, int(min_member_count))
    kept = [
        dict(row)
        for row in communities
        if len(row.get('member_ids') or []) >= safe_min
    ]
    allowed_ids = {str(row.get('community_id') or '').strip() for row in kept}
    filtered_memberships: dict[str, list[dict[str, Any]]] = {}
    for member_id, rows in memberships.items():
        kept_rows = [
            dict(row)
            for row in rows
            if str(row.get('community_id') or '').strip() in allowed_ids
        ]
        if kept_rows:
            filtered_memberships[member_id] = kept_rows
    return {
        'communities': kept,
        'memberships': filtered_memberships,
    }


def _tokenize(text: object) -> set[str]:
    return {
        token.lower()
        for token in _WORD_RE.findall(str(text or '').lower())
        if token and token.lower() not in _GENERIC_TOKENS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter <= 0:
        return 0.0
    return inter / max(1, len(a | b))


def _signature_tokens(move: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in (
        'method_tokens',
        'object_tokens',
        'metric_tokens',
        'condition_tokens',
        'comparator_tokens',
        'limitation_tokens',
        'resource_tokens',
    ):
        for token in move.get(key) or []:
            normalized = str(token or '').strip().lower()
            if normalized and normalized not in _GENERIC_TOKENS:
                tokens.add(normalized)
    return tokens


def _move_similarity_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    signature_sim = _jaccard(
        set(left.get('_signature_tokens') or set()),
        set(right.get('_signature_tokens') or set()),
    )
    summary_sim = _jaccard(
        set(left.get('_summary_tokens') or set()),
        set(right.get('_summary_tokens') or set()),
    )
    act_bonus = 0.12 if str(left.get('act_type') or '') == str(right.get('act_type') or '') else 0.0
    role_bonus = 0.05 if str(left.get('role') or '') == str(right.get('role') or '') else 0.0
    return round(signature_sim * 0.58 + summary_sim * 0.25 + act_bonus + role_bonus, 6)


def _build_local_similarity_edges(
    *,
    moves: list[dict[str, Any]],
    min_score: float,
    neighbor_cap: int,
    limit_total: int,
) -> list[dict[str, Any]]:
    safe_neighbor_cap = max(1, int(neighbor_cap))
    safe_limit_total = max(1, int(limit_total))
    rows_by_source: dict[str, list[dict[str, Any]]] = {}

    for left in moves:
        left_id = str(left.get('move_id') or '').strip()
        left_paper = str(left.get('paper_id') or '').strip()
        if not left_id or not left_paper:
            continue
        neighbors: list[dict[str, Any]] = []
        for right in moves:
            right_id = str(right.get('move_id') or '').strip()
            right_paper = str(right.get('paper_id') or '').strip()
            if not right_id or not right_paper or right_id == left_id or right_paper == left_paper:
                continue
            score = _move_similarity_score(left, right)
            if score < float(min_score):
                continue
            neighbors.append({'source': left_id, 'target': right_id, 'score': score})
        neighbors.sort(key=lambda item: (-float(item.get('score') or 0.0), str(item.get('target') or '')))
        rows_by_source[left_id] = neighbors[:safe_neighbor_cap]

    edges: list[dict[str, Any]] = []
    for source_id in sorted(rows_by_source):
        for row in rows_by_source[source_id]:
            edges.append(row)
            if len(edges) >= safe_limit_total:
                return edges
    return edges


def _build_shared_signal_edges(
    *,
    moves: list[dict[str, Any]],
    limit_total: int,
) -> list[dict[str, Any]]:
    safe_limit_total = max(1, int(limit_total))
    edges: list[dict[str, Any]] = []
    for left_index, left in enumerate(moves):
        left_id = str(left.get('move_id') or '').strip()
        left_paper = str(left.get('paper_id') or '').strip()
        left_tokens = set(left.get('_signature_tokens') or set())
        if not left_id or not left_paper or not left_tokens:
            continue
        for right in moves[left_index + 1 :]:
            right_id = str(right.get('move_id') or '').strip()
            right_paper = str(right.get('paper_id') or '').strip()
            right_tokens = set(right.get('_signature_tokens') or set())
            if not right_id or not right_paper or left_paper == right_paper or not right_tokens:
                continue
            overlap = sorted(left_tokens & right_tokens)
            if not overlap:
                continue
            score = round(min(0.92, 0.24 + 0.09 * len(overlap)), 6)
            edges.append(
                {
                    'source': left_id,
                    'target': right_id,
                    'score': score,
                    'shared_signals': overlap[:8],
                }
            )
            if len(edges) >= safe_limit_total:
                return edges
    return edges


def _build_citation_boost_edges(
    *,
    moves: list[dict[str, Any]],
    citation_pairs: list[dict[str, Any]],
    similar_move_edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not similar_move_edges or not citation_pairs:
        return []

    paper_by_move = {
        str(row.get('move_id') or '').strip(): str(row.get('paper_id') or '').strip()
        for row in moves
        if str(row.get('move_id') or '').strip() and str(row.get('paper_id') or '').strip()
    }
    citation_pair_set = {
        (
            str(row.get('source_paper_id') or '').strip(),
            str(row.get('target_paper_id') or '').strip(),
        )
        for row in citation_pairs
        if str(row.get('source_paper_id') or '').strip() and str(row.get('target_paper_id') or '').strip()
    }

    boosts: list[dict[str, Any]] = []
    for row in similar_move_edges:
        source = str(row.get('source') or '').strip()
        target = str(row.get('target') or '').strip()
        pair = (paper_by_move.get(source, ''), paper_by_move.get(target, ''))
        if not pair[0] or not pair[1]:
            continue
        if pair in citation_pair_set or (pair[1], pair[0]) in citation_pair_set:
            boosts.append(
                {
                    'source': source,
                    'target': target,
                    'weight': settings.global_community_v2_citation_boost,
                }
            )
    return boosts


def rebuild_global_communities_v2(
    *,
    client: Neo4jClient | Any | None = None,
    progress: ProgressFn | None = None,
    log: LogFn | None = None,
) -> dict[str, Any]:
    progress = progress or _noop_progress
    log = log or _noop_log

    own_client = client is None
    if own_client:
        client = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        client.__enter__()

    try:
        client.ensure_schema()
        progress('community:init', 0.05, 'Preparing global community rebuild v2')

        moves = client.list_research_moves(limit=settings.global_community_max_nodes)
        for move in moves:
            move['_summary_tokens'] = _tokenize(move.get('summary'))
            move['_signature_tokens'] = _signature_tokens(move) or set(move.get('_summary_tokens') or set())

        paper_ids = sorted(
            {
                str(row.get('paper_id') or '').strip()
                for row in moves
                if str(row.get('paper_id') or '').strip()
            }
        )

        progress('community:projection', 0.25, 'Building sparse ResearchMove candidate graph')
        similar_move_edges = _build_local_similarity_edges(
            moves=moves,
            min_score=settings.global_community_v2_similarity_min_score,
            neighbor_cap=settings.global_community_v2_neighbor_cap,
            limit_total=settings.global_community_max_edges,
        )
        shared_signal_edges = _build_shared_signal_edges(
            moves=moves,
            limit_total=settings.global_community_max_edges,
        )
        citation_pairs = client.list_paper_citation_pairs(
            paper_ids,
            limit=settings.global_community_max_edges,
        )
        citation_boosts = _build_citation_boost_edges(
            moves=moves,
            citation_pairs=citation_pairs,
            similar_move_edges=similar_move_edges,
        )
        graph = build_move_candidate_graph(
            moves=moves,
            similar_move_edges=similar_move_edges,
            shared_signal_edges=shared_signal_edges,
            citation_boosts=citation_boosts,
            neighbor_cap=settings.global_community_v2_neighbor_cap,
        )
        projection_nodes = len(graph['nodes'])
        projection_edges = len(graph['edges'])
        log(f'global community v2 projection: nodes={projection_nodes}, edges={projection_edges}')

        progress('community:cluster', 0.6, 'Running overlapping ResearchMove community detection')
        detection = detect_overlapping_communities(
            nodes=list(graph['nodes']),
            edges=list(graph['edges']),
            max_memberships_per_node=settings.global_community_v2_max_memberships_per_node,
            min_community_size=settings.global_community_v2_min_size,
        )

        moves_by_id = {
            str(row.get('move_id') or '').strip(): row
            for row in moves
            if str(row.get('move_id') or '').strip()
        }
        labels: dict[str, dict[str, Any]] = {}
        for community in detection['communities']:
            community_id = str(community.get('community_id') or '').strip()
            core_members = [
                moves_by_id[member_id]
                for member_id in (community.get('core_member_ids') or [])
                if member_id in moves_by_id
            ]
            labels[community_id] = label_community(core_members=core_members, evidence_rows=[])

        refined = merge_labeled_communities(
            communities=detection['communities'],
            labels=labels,
            moves_by_id=moves_by_id,
            max_memberships_per_node=settings.global_community_v2_max_memberships_per_node,
            min_community_size=settings.global_community_v2_min_size,
        )
        publishable = _filter_materialized_communities(
            communities=refined['communities'],
            memberships=refined['memberships'],
            min_member_count=settings.global_community_v2_publish_min_size,
        )

        labels = {}
        for community in publishable['communities']:
            community_id = str(community.get('community_id') or '').strip()
            core_members = [
                moves_by_id[member_id]
                for member_id in (community.get('core_member_ids') or [])
                if member_id in moves_by_id
            ]
            labels[community_id] = label_community(core_members=core_members, evidence_rows=[])
        labels = disambiguate_community_labels(
            communities=publishable['communities'],
            labels=labels,
            moves_by_id=moves_by_id,
        )

        progress('community:write', 0.85, 'Writing global communities to Neo4j')
        materialized = materialize_community_rows(
            communities=publishable['communities'],
            memberships=publishable['memberships'],
            labels=labels,
            moves=moves,
            version=settings.global_community_version,
            built_at=_utc_now_iso(),
        )
        cleared = client.clear_global_communities()
        communities_written = client.upsert_global_communities(materialized['communities'])
        keywords_written = client.upsert_global_keywords(materialized['keywords'])
        memberships_written = client.replace_global_memberships(materialized['memberships'])

        progress('community:done', 1.0, 'Global community rebuild v2 complete')
        return {
            'ok': True,
            'projection_nodes': projection_nodes,
            'projection_edges': projection_edges,
            'communities': len(materialized['communities']),
            'keywords': len(materialized['keywords']),
            'communities_written': int(communities_written),
            'keywords_written': int(keywords_written),
            'memberships_written': int(memberships_written),
            'cleared': cleared,
            'version': settings.global_community_version,
        }
    finally:
        if own_client and client is not None:
            client.__exit__(None, None, None)
