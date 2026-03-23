from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any


def _community_score(member_ids: list[str], edge_weight: dict[frozenset[str], float]) -> float:
    if len(member_ids) < 2:
        return 0.0
    scores = [edge_weight.get(frozenset((left, right)), 0.0) for left, right in combinations(member_ids, 2)]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _expand_seed_members(
    *,
    seed_members: list[str],
    adjacency: dict[str, dict[str, float]],
    edge_weight: dict[frozenset[str], float],
) -> list[str]:
    members = list(dict.fromkeys(seed_members))
    candidates = sorted(
        set(node for member in members for node in adjacency.get(member, {})) - set(members)
    )
    while candidates:
        best_node = ''
        best_score = 0.0
        for node in candidates:
            weights = [
                edge_weight.get(frozenset((node, member)), 0.0)
                for member in members
            ]
            if any(weight <= 0.0 for weight in weights):
                continue
            candidate_score = sum(weights) / len(weights)
            if candidate_score > best_score:
                best_score = candidate_score
                best_node = node
        if not best_node or best_score < 0.55:
            break
        members.append(best_node)
        candidates = sorted(
            set(node for member in members for node in adjacency.get(member, {})) - set(members)
        )
    return sorted(members)


def merge_high_overlap_communities(
    *,
    communities: list[dict[str, Any]],
    edge_weight: dict[frozenset[str], float],
    min_community_size: int,
    overlap_threshold: float = 0.72,
    score_retention_floor: float = 0.72,
) -> list[dict[str, Any]]:
    merged = [dict(row) for row in communities]
    safe_overlap = max(0.5, min(0.95, float(overlap_threshold)))
    safe_floor = max(0.4, min(0.98, float(score_retention_floor)))

    changed = True
    while changed:
        changed = False
        next_round: list[dict[str, Any]] = []
        consumed: set[int] = set()
        for left_index, left in enumerate(merged):
            if left_index in consumed:
                continue
            current_members = set(str(item).strip() for item in (left.get('member_ids') or []) if str(item).strip())
            current_core = set(str(item).strip() for item in (left.get('core_member_ids') or []) if str(item).strip())
            current_confidence = float(left.get('confidence') or _community_score(sorted(current_members), edge_weight))
            base_community = dict(left)

            for right_index in range(left_index + 1, len(merged)):
                if right_index in consumed:
                    continue
                right = merged[right_index]
                right_members = set(str(item).strip() for item in (right.get('member_ids') or []) if str(item).strip())
                if len(current_members) < min_community_size or len(right_members) < min_community_size:
                    continue
                overlap = len(current_members & right_members)
                if overlap <= 0:
                    continue
                union = current_members | right_members
                jaccard = overlap / max(1, len(union))
                if jaccard < safe_overlap:
                    continue
                merged_score = _community_score(sorted(union), edge_weight)
                right_confidence = float(right.get('confidence') or _community_score(sorted(right_members), edge_weight))
                if merged_score < min(current_confidence, right_confidence) * safe_floor:
                    continue
                current_members = union
                current_core |= set(str(item).strip() for item in (right.get('core_member_ids') or []) if str(item).strip())
                current_confidence = max(current_confidence, right_confidence, merged_score)
                consumed.add(right_index)
                changed = True

            base_community['member_ids'] = sorted(current_members)
            base_community['core_member_ids'] = sorted(current_core & current_members)
            base_community['confidence'] = round(current_confidence, 6)
            next_round.append(base_community)
        merged = next_round
    return merged


def detect_overlapping_communities(
    *,
    nodes: list[str],
    edges: list[dict[str, Any]],
    max_memberships_per_node: int = 2,
    min_community_size: int = 2,
) -> dict[str, Any]:
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    edge_weight: dict[frozenset[str], float] = {}

    for row in edges:
        source = str(row.get('source') or '').strip()
        target = str(row.get('target') or '').strip()
        if not source or not target or source == target:
            continue
        weight = float(row.get('weight') or 0.0)
        if weight <= 0.0:
            continue
        adjacency[source][target] = max(weight, adjacency[source].get(target, 0.0))
        adjacency[target][source] = max(weight, adjacency[target].get(source, 0.0))
        edge_weight[frozenset((source, target))] = max(weight, edge_weight.get(frozenset((source, target)), 0.0))

    raw_communities: list[dict[str, Any]] = []
    seen_member_sets: set[tuple[str, ...]] = set()
    for source, neighbors in adjacency.items():
        for target, weight in neighbors.items():
            if source >= target:
                continue
            member_ids = tuple(
                _expand_seed_members(
                    seed_members=[source, target],
                    adjacency=adjacency,
                    edge_weight=edge_weight,
                )
            )
            if len(member_ids) < min_community_size or member_ids in seen_member_sets:
                continue
            seen_member_sets.add(member_ids)
            raw_communities.append(
                {
                    'community_id': f'gc-v2-{len(raw_communities) + 1}',
                    'member_ids': list(member_ids),
                    'core_member_ids': list(member_ids),
                    'confidence': round(weight, 6),
                }
            )

    raw_communities.sort(key=lambda row: (-float(row['confidence']), row['community_id']))
    raw_communities = merge_high_overlap_communities(
        communities=raw_communities,
        edge_weight=edge_weight,
        min_community_size=min_community_size,
    )

    memberships: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for community in raw_communities:
        member_ids = list(community['member_ids'])
        community_score = max(float(community.get('confidence') or 0.0), _community_score(member_ids, edge_weight))
        community['confidence'] = round(community_score, 6)
        for member_id in member_ids:
            memberships[member_id].append(
                {
                    'community_id': community['community_id'],
                    'score': round(community_score, 6),
                    'is_core': member_id in set(community.get('core_member_ids') or []),
                }
            )

    trimmed_memberships: dict[str, list[dict[str, Any]]] = {}
    allowed_by_node: dict[str, set[str]] = {}
    safe_cap = max(1, int(max_memberships_per_node))
    for node in nodes:
        node_memberships = sorted(
            memberships.get(node, []),
            key=lambda row: (-float(row['score']), row['community_id']),
        )[:safe_cap]
        for index, row in enumerate(node_memberships, start=1):
            row['rank'] = index
        trimmed_memberships[node] = node_memberships
        allowed_by_node[node] = {str(row['community_id']) for row in node_memberships}

    kept_communities: list[dict[str, Any]] = []
    for community in raw_communities:
        community_id = str(community['community_id'])
        member_ids = [
            member_id
            for member_id in community['member_ids']
            if community_id in allowed_by_node.get(member_id, set())
        ]
        if len(member_ids) < min_community_size:
            continue
        community['member_ids'] = member_ids
        community['member_count'] = len(member_ids)
        community['core_member_ids'] = [
            member_id
            for member_id in community.get('core_member_ids') or []
            if member_id in member_ids
        ]
        kept_communities.append(community)

    surviving_community_ids = {str(row['community_id']) for row in kept_communities}
    for node in list(trimmed_memberships.keys()):
        trimmed_memberships[node] = [
            row
            for row in trimmed_memberships[node]
            if str(row['community_id']) in surviving_community_ids
        ]

    return {
        'communities': kept_communities,
        'memberships': trimmed_memberships,
    }
