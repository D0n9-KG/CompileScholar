from __future__ import annotations

import json
import re
from pathlib import Path

from app.research_logic import load_bounded_packet_assembly_manifest, load_route_packet, validate_bounded_packet_assembly


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


def test_phase9_assembly_manifest_and_audit_report_align_with_packet() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    packet_path = repo_root / 'docs' / 'replay' / 'pilot_packets' / 'phase9-route-packet.json'
    assembly_manifest_path = repo_root / 'docs' / 'replay' / 'pilot_packets' / 'phase9-assembly-manifest.json'
    audit_summary_path = repo_root / 'tmp' / 'phase9_bounded_packet_audit' / 'baseline' / 'audit_summary.json'
    report_path = repo_root / 'docs' / 'replay' / 'reports' / 'phase9-bounded-packet-audit.md'

    packet = load_route_packet(packet_path)
    manifest = load_bounded_packet_assembly_manifest(assembly_manifest_path)
    audit = validate_bounded_packet_assembly(packet, manifest)
    audit_summary = json.loads(audit_summary_path.read_text(encoding='utf-8'))
    report = report_path.read_text(encoding='utf-8').lower()

    assert manifest.packet_id == packet.packet_id
    assert manifest.topic_scope == packet.topic_scope_candidate
    assert manifest.cutoff_year == packet.cutoff_year
    assert manifest.packet_manifest_ref == 'docs/replay/pilot_packets/phase9-route-packet.json'

    support_ids = [member.paper_id for member in manifest.support.members]
    alternative_ids = [member.paper_id for member in manifest.alternative.members]
    held_out_ids = [member.paper_id for member in manifest.held_out.members]
    assert support_ids == ['1000', '1001', '1002', '1005', '1017']
    assert alternative_ids == ['1007']
    assert held_out_ids == ['1023']
    assert not (set(support_ids) & set(held_out_ids))
    assert all(member.trace_ref and not Path(member.trace_ref).is_absolute() for member in manifest.support.members)
    assert all(member.trace_ref and not Path(member.trace_ref).is_absolute() for member in manifest.alternative.members)
    assert all(member.trace_ref and not Path(member.trace_ref).is_absolute() for member in manifest.held_out.members)

    assert audit.quality_tier == 'green'
    assert audit.ready_for_phase10 is True
    assert audit.role_counts.support == 5
    assert audit.role_counts.alternative == 1
    assert audit.role_counts.held_out == 1
    assert audit.quality_flags == []
    assert audit.structural_errors == []
    assert audit.missing_trace_ref_paper_ids == []
    assert audit.support_paper_ids == support_ids
    assert audit.alternative_paper_ids == alternative_ids
    assert audit.held_out_paper_ids == held_out_ids
    assert len(audit.known_gap_notes) == 5

    assert audit_summary == {
        'packet_id': 'phase9_comp_mech_2021_packet_01',
        'topic_scope_candidate': 'data-driven constitutive and multiscale computational mechanics',
        'cutoff_year': 2021,
        'packet_included_item_count': 7,
        'packet_excluded_item_count': 4,
        'support_count': 5,
        'alternative_count': 1,
        'held_out_count': 1,
        'quality_tier': 'green',
        'ready_for_phase10': True,
        'quality_flags': [],
        'structural_errors': [],
        'missing_trace_ref_paper_ids': [],
        'packet_items_missing_trace_id': [],
        'indistinct_alternative_paper_ids': [],
        'support_held_out_overlap_paper_ids': [],
        'missing_exclusion_note_paper_ids': [],
        'known_gap_note_count': 5,
        'exclusion_note_count': 4,
    }

    assert 'derived from runtime bundle' in report
    assert 'support members: `5`' in report
    assert 'alternative members: `1`' in report
    assert 'held-out members: `1`' in report
    assert 'quality tier: `green`' in report
    assert 'ready for phase 10 handoff: `true`' in report
    assert 'structural handoff is ready' in report
    assert 'support density is still thin' in report
    assert 'alternative coverage is only one paper deep' in report
    assert 'held-out coverage is only one paper deep' in report
    assert 'placeholder l1 snapshot ref' in report
    assert 'portable but not fully normalized' in report
