from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.community.labeling import _is_distinctive_phrase, _normalize_phrase


_WEAK_TITLES = {
    'analysis',
    'discussion',
    'experimental data',
    'experimental setup',
    'future work',
    'good agreement',
    'method',
    'methods',
    'result',
    'results',
    'simulation data',
    'statistical analysis',
}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap <= 0:
        return 0.0
    return overlap / max(1, len(left | right))


def _community_member_set(community: dict[str, Any]) -> set[str]:
    return {
        str(member_id).strip()
        for member_id in (community.get('member_ids') or [])
        if str(member_id).strip()
    }


def _community_core_set(community: dict[str, Any]) -> set[str]:
    return {
        str(member_id).strip()
        for member_id in (community.get('core_member_ids') or [])
        if str(member_id).strip()
    }


def _paper_ids(member_ids: set[str], moves_by_id: dict[str, dict[str, Any]]) -> set[str]:
    return {
        str((moves_by_id.get(member_id) or {}).get('paper_id') or '').strip()
        for member_id in member_ids
        if str((moves_by_id.get(member_id) or {}).get('paper_id') or '').strip()
    }


def _role_ids(member_ids: set[str], moves_by_id: dict[str, dict[str, Any]]) -> set[str]:
    return {
        str((moves_by_id.get(member_id) or {}).get('role') or '').strip()
        for member_id in member_ids
        if str((moves_by_id.get(member_id) or {}).get('role') or '').strip()
    }


def _dominant_role(member_ids: set[str], moves_by_id: dict[str, dict[str, Any]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for member_id in member_ids:
        role = str((moves_by_id.get(member_id) or {}).get('role') or '').strip()
        if role:
            counts[role] += 1
    if not counts:
        return ''
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _title_tokens(title: object) -> set[str]:
    normalized = _normalize_phrase(title)
    return {token for token in normalized.split() if token}


def _should_merge_pair(
    *,
    left: dict[str, Any],
    right: dict[str, Any],
    labels: dict[str, dict[str, Any]],
    moves_by_id: dict[str, dict[str, Any]],
) -> bool:
    left_members = _community_member_set(left)
    right_members = _community_member_set(right)
    if not left_members or not right_members:
        return False

    member_jaccard = _jaccard(left_members, right_members)
    if member_jaccard >= 0.72:
        return True

    left_title_tokens = _title_tokens((labels.get(str(left.get('community_id') or '')) or {}).get('title'))
    right_title_tokens = _title_tokens((labels.get(str(right.get('community_id') or '')) or {}).get('title'))
    title_jaccard = _jaccard(left_title_tokens, right_title_tokens)
    if title_jaccard < 0.85:
        return False

    left_papers = _paper_ids(left_members, moves_by_id)
    right_papers = _paper_ids(right_members, moves_by_id)
    paper_jaccard = _jaccard(left_papers, right_papers)

    left_roles = _role_ids(left_members, moves_by_id)
    right_roles = _role_ids(right_members, moves_by_id)
    role_jaccard = _jaccard(left_roles, right_roles) if left_roles and right_roles else 1.0
    if role_jaccard < 0.5:
        return False

    if paper_jaccard >= 0.8:
        return True
    if member_jaccard >= 0.55 and paper_jaccard >= 0.5:
        return True
    return False


def _rebuild_memberships(
    *,
    communities: list[dict[str, Any]],
    max_memberships_per_node: int,
    min_community_size: int,
) -> dict[str, Any]:
    safe_cap = max(1, int(max_memberships_per_node))
    memberships_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for community in communities:
        member_ids = sorted(_community_member_set(community))
        community['member_ids'] = member_ids
        community['core_member_ids'] = sorted(_community_core_set(community) & set(member_ids))
        community['member_count'] = len(member_ids)
        for member_id in member_ids:
            memberships_by_node[member_id].append(
                {
                    'community_id': str(community.get('community_id') or '').strip(),
                    'score': round(float(community.get('confidence') or 0.0), 6),
                    'is_core': member_id in set(community.get('core_member_ids') or []),
                }
            )

    trimmed_memberships: dict[str, list[dict[str, Any]]] = {}
    allowed_by_node: dict[str, set[str]] = {}
    for member_id, rows in memberships_by_node.items():
        kept = sorted(rows, key=lambda row: (-float(row['score']), str(row['community_id'])))[:safe_cap]
        for rank, row in enumerate(kept, start=1):
            row['rank'] = rank
        trimmed_memberships[member_id] = kept
        allowed_by_node[member_id] = {str(row['community_id']) for row in kept}

    kept_communities: list[dict[str, Any]] = []
    for community in communities:
        community_id = str(community.get('community_id') or '').strip()
        member_ids = [
            member_id
            for member_id in (community.get('member_ids') or [])
            if community_id in allowed_by_node.get(member_id, set())
        ]
        if len(member_ids) < int(min_community_size):
            continue
        community['member_ids'] = member_ids
        community['core_member_ids'] = [
            member_id
            for member_id in (community.get('core_member_ids') or [])
            if member_id in member_ids
        ]
        community['member_count'] = len(member_ids)
        kept_communities.append(community)

    surviving_ids = {str(community.get('community_id') or '').strip() for community in kept_communities}
    for member_id in list(trimmed_memberships.keys()):
        trimmed_memberships[member_id] = [
            row
            for row in trimmed_memberships[member_id]
            if str(row.get('community_id') or '').strip() in surviving_ids
        ]

    return {
        'communities': kept_communities,
        'memberships': trimmed_memberships,
    }


def merge_labeled_communities(
    *,
    communities: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    moves_by_id: dict[str, dict[str, Any]],
    max_memberships_per_node: int,
    min_community_size: int,
) -> dict[str, Any]:
    merged = [dict(row) for row in communities]

    changed = True
    while changed:
        changed = False
        next_round: list[dict[str, Any]] = []
        consumed: set[int] = set()
        for left_index, left in enumerate(merged):
            if left_index in consumed:
                continue
            current = dict(left)
            current_members = _community_member_set(current)
            current_core = _community_core_set(current)
            current_confidence = float(current.get('confidence') or 0.0)

            for right_index in range(left_index + 1, len(merged)):
                if right_index in consumed:
                    continue
                right = merged[right_index]
                if not _should_merge_pair(
                    left=current,
                    right=right,
                    labels=labels,
                    moves_by_id=moves_by_id,
                ):
                    continue
                current_members |= _community_member_set(right)
                current_core |= _community_core_set(right)
                current_confidence = max(current_confidence, float(right.get('confidence') or 0.0))
                consumed.add(right_index)
                changed = True

            current['member_ids'] = sorted(current_members)
            current['core_member_ids'] = sorted(current_core & current_members)
            current['confidence'] = round(current_confidence, 6)
            next_round.append(current)
        merged = next_round

    return _rebuild_memberships(
        communities=merged,
        max_memberships_per_node=max_memberships_per_node,
        min_community_size=min_community_size,
    )


def _candidate_titles(label: dict[str, Any], *, include_base: bool = False) -> list[str]:
    base_title = str(label.get('title') or '').strip()
    base_normalized = _normalize_phrase(base_title)
    candidates: list[str] = []
    for raw in label.get('keywords') or []:
        candidate = str(raw or '').strip()
        normalized = _normalize_phrase(candidate)
        if not normalized or not _is_distinctive_phrase(normalized):
            continue
        if normalized == base_normalized and not include_base:
            continue
        if normalized in _WEAK_TITLES:
            continue
        if normalized in {_normalize_phrase(item) for item in candidates}:
            continue
        candidates.append(candidate)
    return candidates


def disambiguate_community_labels(
    *,
    communities: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    moves_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    updated = {community_id: dict(row) for community_id, row in labels.items()}
    communities_by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for community in communities:
        community_id = str(community.get('community_id') or '').strip()
        title = str((updated.get(community_id) or {}).get('title') or '').strip()
        communities_by_title[_normalize_phrase(title)].append(community)

    for normalized_title, rows in communities_by_title.items():
        used_titles: set[str] = set()
        duplicate_group = len(rows) > 1
        ranked_rows = sorted(
            rows,
            key=lambda row: (-int(row.get('member_count') or len(row.get('member_ids') or [])), -float(row.get('confidence') or 0.0), str(row.get('community_id') or '')),
        )
        for index, community in enumerate(ranked_rows):
            community_id = str(community.get('community_id') or '').strip()
            label = updated.get(community_id) or {'title': community_id, 'summary': '', 'keywords': []}
            base_title = str(label.get('title') or community_id).strip()
            base_normalized = _normalize_phrase(base_title)
            weak_title = base_normalized in _WEAK_TITLES
            new_title = base_title

            if (duplicate_group and index > 0) or weak_title:
                for candidate in _candidate_titles(label):
                    normalized_candidate = _normalize_phrase(candidate)
                    if normalized_candidate in used_titles:
                        continue
                    new_title = candidate
                    break
                else:
                    if duplicate_group and index > 0:
                        dominant_role = _dominant_role(_community_member_set(community), moves_by_id)
                        if dominant_role and dominant_role not in base_normalized.split():
                            new_title = f'{base_title} {dominant_role}'

            normalized_new_title = _normalize_phrase(new_title)
            if duplicate_group and normalized_new_title in used_titles:
                dominant_role = _dominant_role(_community_member_set(community), moves_by_id)
                if dominant_role and dominant_role not in normalized_new_title.split():
                    new_title = f'{new_title} {dominant_role}'
                    normalized_new_title = _normalize_phrase(new_title)

            used_titles.add(normalized_new_title or base_normalized)
            deduped_keywords = [new_title]
            for keyword in label.get('keywords') or []:
                candidate = str(keyword or '').strip()
                if not candidate:
                    continue
                if _normalize_phrase(candidate) == _normalize_phrase(new_title):
                    continue
                if candidate in deduped_keywords:
                    continue
                deduped_keywords.append(candidate)

            label['title'] = new_title
            label['keywords'] = deduped_keywords[:5]
            updated[community_id] = label

    return updated
