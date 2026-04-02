from __future__ import annotations

import re
from pathlib import Path

from app.research_logic import load_route_packet


def test_phase1_route_packet_manifest_contract() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / 'docs' / 'replay' / 'pilot_packets' / 'phase1-route-packet.json'
    raw = manifest_path.read_text(encoding='utf-8')
    packet = load_route_packet(manifest_path)

    assert packet.packet_status in {'reviewed', 'frozen'}
    assert packet.packet_composition.coverage_ok is True
    assert packet.packet_composition.missing_roles == []
    assert packet.packet_composition.actual_size == len(packet.included_items)
    assert packet.packet_composition.role_counts.core_method >= 2
    assert packet.packet_composition.role_counts.resource_or_benchmark >= 1
    assert packet.packet_composition.role_counts.limitation_or_critique >= 1
    assert packet.packet_composition.role_counts.survey_or_review >= 1
    assert packet.packet_composition.role_counts.alternative_route >= 1

    assert all(item.item_role for item in packet.included_items)
    assert all(item.inclusion_reason.strip() for item in packet.included_items)
    assert all(item.trace_id and item.trace_id.strip() for item in packet.included_items)

    leakage_policy = packet.inclusion_rules.leakage_policy.lower()
    assert 'after the cutoff' in leakage_policy or 'after cutoff' in leakage_policy
    assert 'audit-only' in leakage_policy or 'audit only' in leakage_policy
    assert 'hindsight' in leakage_policy

    assert packet.packet_quality.manual_review_status == 'completed'
    assert 'small_pilot_packet' in packet.packet_quality.quality_flags

    assert not re.search(r'[A-Za-z]:\\\\', raw)
    assert not re.search(r'\\\\[A-Za-z0-9._-]+\\', raw)
