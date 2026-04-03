from __future__ import annotations

import re
from pathlib import Path

from app.research_logic import load_route_packet


def test_phase9_route_packet_manifest_contract() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / 'docs' / 'replay' / 'pilot_packets' / 'phase9-route-packet.json'
    notes_path = repo_root / 'docs' / 'replay' / 'pilot_packets' / 'phase9-selection-notes.md'

    raw = manifest_path.read_text(encoding='utf-8')
    notes = notes_path.read_text(encoding='utf-8').lower()
    packet = load_route_packet(manifest_path)

    assert packet.packet_id == 'phase9_comp_mech_2021_packet_01'
    assert packet.topic_scope_candidate == 'data-driven constitutive and multiscale computational mechanics'
    assert packet.cutoff_year == 2021
    assert packet.packet_status == 'reviewed'

    assert packet.packet_composition.coverage_ok is True
    assert packet.packet_composition.actual_size == len(packet.included_items) == 7
    assert packet.packet_composition.missing_roles == []
    assert packet.packet_composition.role_counts.core_method == 3
    assert packet.packet_composition.role_counts.resource_or_benchmark == 1
    assert packet.packet_composition.role_counts.limitation_or_critique == 1
    assert packet.packet_composition.role_counts.survey_or_review == 1
    assert packet.packet_composition.role_counts.alternative_route == 1

    role_by_id = {item.paper_id: item.item_role for item in packet.included_items}
    assert role_by_id == {
        '1000': 'core_method',
        '1001': 'core_method',
        '1002': 'resource_or_benchmark',
        '1005': 'survey_or_review',
        '1007': 'alternative_route',
        '1017': 'core_method',
        '1023': 'limitation_or_critique',
    }
    assert all(item.inclusion_reason.strip() for item in packet.included_items)
    assert all(item.trace_id and item.trace_id.strip() for item in packet.included_items)

    excluded_by_id = {item.paper_id: item.exclusion_reason for item in packet.excluded_items}
    assert excluded_by_id == {
        '1004': 'after_cutoff',
        '1010': 'off_topic',
        '1012': 'duplicate_signal',
        '1107': 'off_topic',
    }

    leakage_policy = packet.inclusion_rules.leakage_policy.lower()
    assert '2021 cutoff' in leakage_policy
    assert 'audit only' in leakage_policy or 'audit-only' in leakage_policy
    assert 'hindsight' in leakage_policy

    assert packet.packet_quality.quality_tier == 'yellow'
    assert packet.packet_quality.ready_for_route_state is False
    assert packet.packet_quality.manual_review_status == 'completed'
    assert 'l1_snapshot_placeholder' in packet.packet_quality.quality_flags
    assert 'support_cluster_still_small' in packet.packet_quality.quality_flags
    assert 'txt_fallback_sources_present' in packet.packet_quality.quality_flags

    assert '1004' in notes
    assert 'post-cutoff' in notes
    assert '1107' in notes
    assert 'out-of-slice' in notes
    assert '2021' in notes
    assert 'does not pretend' in notes

    assert not re.search(r'[A-Za-z]:\\\\', raw)
    assert not re.search(r'\\\\[A-Za-z0-9._-]+\\', raw)
