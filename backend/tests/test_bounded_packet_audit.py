from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.research_logic import (
    BoundedPacketAssemblyManifest,
    RoutePacket,
    audit_bounded_packet_assembly,
    build_bounded_packet_audit_summary,
    load_bounded_packet_assembly_manifest,
    validate_bounded_packet_assembly,
    write_bounded_packet_audit_bundle,
)


def _member(paper_id: str, *, trace_ref: str | None = 'traces/default.json', reason: str | None = None) -> dict[str, object]:
    return {
        'paper_id': paper_id,
        'reason': reason or f'{paper_id} supports the bounded packet role assignment.',
        'trace_ref': trace_ref,
    }


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return path


def _packet(
    *,
    alternative_item_role: str = 'alternative_route',
    missing_trace_id_paper_ids: set[str] | None = None,
) -> RoutePacket:
    missing_trace_id_paper_ids = missing_trace_id_paper_ids or set()
    return RoutePacket.model_validate(
        {
            'packet_id': 'phase9-demo-packet',
            'built_at': '2026-04-03T09:00:00Z',
            'topic_scope_candidate': 'data-driven constitutive and multiscale computational mechanics',
            'cutoff_year': 2021,
            'packet_status': 'reviewed',
            'packet_composition': {
                'target_size': 4,
                'actual_size': 4,
                'role_counts': {
                    'core_method': 2,
                    'resource_or_benchmark': 0,
                    'limitation_or_critique': 1,
                    'survey_or_review': 0,
                    'alternative_route': 1 if alternative_item_role == 'alternative_route' else 0,
                },
                'coverage_ok': True,
                'missing_roles': [],
            },
            'inclusion_rules': {
                'scope_definition': 'Audit one bounded packet before Phase 10 compilation.',
                'scope_aliases': ['computational mechanics'],
                'accepted_year_range': {'min_year': 2017, 'max_year': 2021},
                'hard_exclusion_rules': ['exclude papers after cutoff'],
                'role_assignment_rules': ['record support, alternative, and held-out roles in a companion manifest'],
                'leakage_policy': 'Papers after cutoff remain excluded and later retrospectives are audit only.',
            },
            'included_items': [
                {
                    'paper_id': 'paper-1',
                    'trace_id': None if 'paper-1' in missing_trace_id_paper_ids else 'paper-1:trace',
                    'paper_year': 2017,
                    'item_role': 'core_method',
                    'inclusion_reason': 'Anchor data-driven constitutive paper.',
                    'source_selector': 'manual',
                    'evidence_for_inclusion': ['packet-anchor'],
                    'title': 'Data based derivation of material response',
                },
                {
                    'paper_id': 'paper-2',
                    'trace_id': None if 'paper-2' in missing_trace_id_paper_ids else 'paper-2:trace',
                    'paper_year': 2019,
                    'item_role': 'core_method',
                    'inclusion_reason': 'Second support candidate.',
                    'source_selector': 'manual',
                    'evidence_for_inclusion': ['support-cluster'],
                    'title': 'Measuring stress field without constitutive equation',
                },
                {
                    'paper_id': 'paper-3',
                    'trace_id': None if 'paper-3' in missing_trace_id_paper_ids else 'paper-3:trace',
                    'paper_year': 2020,
                    'item_role': alternative_item_role,
                    'inclusion_reason': 'Alternative or contrastive packet member.',
                    'source_selector': 'manual',
                    'evidence_for_inclusion': ['alternative-signal'],
                    'title': 'Data-Driven multiscale modeling in mechanics',
                },
                {
                    'paper_id': 'paper-4',
                    'trace_id': None if 'paper-4' in missing_trace_id_paper_ids else 'paper-4:trace',
                    'paper_year': 2021,
                    'item_role': 'limitation_or_critique',
                    'inclusion_reason': 'Held-out consistency check candidate.',
                    'source_selector': 'manual',
                    'evidence_for_inclusion': ['held-out-check'],
                    'title': 'Adaptive selection of reference stiffness in virtual clustering analysis',
                },
            ],
            'excluded_items': [
                {
                    'paper_id': 'paper-9',
                    'paper_year': 2023,
                    'exclusion_reason': 'after_cutoff',
                    'notes': 'Relevant but outside the 2021 cutoff.',
                }
            ],
            'packet_quality': {
                'quality_tier': 'yellow',
                'ready_for_route_state': False,
                'quality_flags': ['phase9_manual_review_pending'],
                'topic_boundary_confidence': 0.79,
                'leakage_risk': 'low',
                'manual_review_status': 'partial',
            },
            'compiler_hints': {
                'preferred_scope_label': 'data-driven constitutive and multiscale computational mechanics',
                'preferred_method_labels': ['data-driven constitutive modeling'],
                'preferred_benchmark_labels': [],
                'preferred_bottleneck_labels': [],
                'expected_alternative_routes': ['virtual clustering analysis'],
                'notes_for_route_state_compiler': 'Treat held-out members as later consistency checks.',
            },
        }
    )


def _manifest_payload(
    *,
    support: list[dict[str, object]] | None = None,
    alternative: list[dict[str, object]] | None = None,
    held_out: list[dict[str, object]] | None = None,
    alternative_distinctness_rationale: str | None = 'Alternative route differs in clustering strategy.',
    exclusion_reason: str = 'after_cutoff',
) -> dict[str, object]:
    return {
        'packet_id': 'phase9-demo-packet',
        'packet_manifest_ref': 'docs/replay/pilot_packets/phase9-route-packet.json',
        'built_at': '2026-04-03T09:30:00Z',
        'topic_scope': 'data-driven constitutive and multiscale computational mechanics',
        'cutoff_year': 2021,
        'support': {
            'group_reason': 'Support captures the main route family under the cutoff.',
            'members': support or [_member('paper-1'), _member('paper-2')],
        },
        'alternative': {
            'group_reason': 'Alternative preserves a nearby but distinct route family.',
            'distinctness_rationale': alternative_distinctness_rationale,
            'members': alternative or [_member('paper-3')],
        },
        'held_out': {
            'group_reason': 'Held-out papers remain separate for later consistency checks.',
            'members': held_out or [_member('paper-4')],
        },
        'exclusion_notes': [
            {
                'paper_id': 'paper-9',
                'exclusion_reason': exclusion_reason,
                'note': 'Relevant to the same slice, but post-cutoff.',
                'paper_year': 2023,
            }
        ],
        'known_gap_notes': ['Trace reuse remains filesystem-first until Phase 10.'],
    }


def test_validate_bounded_packet_assembly_rejects_missing_packet_members() -> None:
    packet = _packet()
    manifest = _manifest_payload(
        support=[_member('paper-1'), _member('paper-404')],
    )

    with pytest.raises(ValueError, match='support members missing from RoutePacket.included_items: paper-404'):
        validate_bounded_packet_assembly(packet, manifest)


def test_validate_bounded_packet_assembly_rejects_support_held_out_overlap() -> None:
    packet = _packet()
    manifest = _manifest_payload(
        held_out=[_member('paper-2')],
    )

    with pytest.raises(ValueError, match='support and held_out groups cannot reuse paper ids: paper-2'):
        validate_bounded_packet_assembly(packet, manifest)


def test_load_bounded_packet_assembly_manifest_rejects_blank_exclusion_reason(tmp_path: Path) -> None:
    manifest_path = tmp_path / 'bounded_packet_assembly_manifest.json'
    manifest_path.write_text(
        json.dumps(_manifest_payload(exclusion_reason=''), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='exclusion_reason'):
        load_bounded_packet_assembly_manifest(manifest_path)


def test_audit_bounded_packet_assembly_flags_missing_trace_refs_and_indistinct_alternatives() -> None:
    packet = _packet(
        alternative_item_role='core_method',
        missing_trace_id_paper_ids={'paper-4'},
    )
    manifest = _manifest_payload(
        alternative=[_member('paper-3', trace_ref=None)],
        alternative_distinctness_rationale=None,
    )

    audit = audit_bounded_packet_assembly(packet, manifest)

    assert audit.structural_errors == []
    assert audit.quality_tier == 'red'
    assert audit.ready_for_phase10 is False
    assert 'missing_trace_refs' in audit.quality_flags
    assert 'alternative_scope_not_distinct' in audit.quality_flags
    assert audit.missing_trace_ref_paper_ids == ['paper-3']
    assert audit.packet_items_missing_trace_id == ['paper-4']
    assert audit.indistinct_alternative_paper_ids == ['paper-3']


def test_audit_bounded_packet_assembly_preserves_distinctness_rationale_for_overlapping_alternative_scope() -> None:
    packet = _packet(alternative_item_role='core_method')
    manifest = _manifest_payload(
        alternative_distinctness_rationale=(
            'Alternative route stays explicit because it preserves a nearby but still review-relevant synthesis path.'
        ),
    )

    audit = audit_bounded_packet_assembly(packet, manifest)

    assert 'alternative_scope_not_distinct' not in audit.quality_flags
    assert audit.indistinct_alternative_paper_ids == []


def test_write_bounded_packet_audit_bundle_uses_expected_filenames(tmp_path: Path) -> None:
    packet = _packet(
        alternative_item_role='core_method',
        missing_trace_id_paper_ids={'paper-4'},
    )
    manifest = BoundedPacketAssemblyManifest.model_validate(
        _manifest_payload(
            alternative=[_member('paper-3', trace_ref=None)],
            alternative_distinctness_rationale=None,
        )
    )
    audit = audit_bounded_packet_assembly(packet, manifest)
    output_dir = tmp_path / 'phase9_bounded_packet_audit' / 'bundle-01'

    written_files = write_bounded_packet_audit_bundle(
        output_dir,
        route_packet=packet,
        assembly_manifest=manifest,
        audit=audit,
    )

    assert written_files['audit_summary'].name == 'audit_summary.json'
    assert written_files['audit_inspection'].name == 'audit_inspection.json'
    assert (output_dir / 'audit_summary.json').is_file()
    assert (output_dir / 'audit_inspection.json').is_file()
    assert (output_dir / 'inputs' / 'route_packet.json').is_file()
    assert (output_dir / 'inputs' / 'assembly_manifest.json').is_file()
    assert (output_dir / 'bundle_manifest.json').is_file()


def test_build_bounded_packet_audit_summary_keeps_quality_flags_explicit() -> None:
    packet = _packet(
        alternative_item_role='core_method',
        missing_trace_id_paper_ids={'paper-4'},
    )
    manifest = BoundedPacketAssemblyManifest.model_validate(
        _manifest_payload(
            alternative=[_member('paper-3', trace_ref=None)],
            alternative_distinctness_rationale=None,
        )
    )
    audit = audit_bounded_packet_assembly(packet, manifest)

    summary = build_bounded_packet_audit_summary(
        route_packet=packet,
        assembly_manifest=manifest,
        audit=audit,
    )

    assert summary['quality_tier'] == 'red'
    assert 'missing_trace_refs' in summary['quality_flags']
    assert 'alternative_scope_not_distinct' in summary['quality_flags']
    assert summary['missing_trace_ref_paper_ids'] == ['paper-3']
    assert summary['packet_items_missing_trace_id'] == ['paper-4']


def test_run_bounded_packet_audit_cli_surfaces_quality_flags_in_summary(tmp_path: Path) -> None:
    packet_path = _write_json(tmp_path / 'packet.json', _packet(
        alternative_item_role='core_method',
        missing_trace_id_paper_ids={'paper-4'},
    ).model_dump(mode='json', exclude_none=True))
    manifest_path = _write_json(
        tmp_path / 'bounded_packet_assembly_manifest.json',
        _manifest_payload(
            alternative=[_member('paper-3', trace_ref=None)],
            alternative_distinctness_rationale=None,
        ),
    )
    output_dir = tmp_path / 'phase9_bounded_packet_audit' / 'cli-run'
    script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'run_bounded_packet_audit.py'

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            '--packet',
            str(packet_path),
            '--assembly-manifest',
            str(manifest_path),
            '--output-dir',
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)

    assert 'missing_trace_refs' in summary['quality_flags']
    assert 'alternative_scope_not_distinct' in summary['quality_flags']
    assert summary['structural_errors'] == []
    assert summary['audit_summary_file'].endswith('audit_summary.json')
    assert summary['audit_inspection_file'].endswith('audit_inspection.json')


def test_run_bounded_packet_audit_cli_exits_nonzero_on_invalid_alignment(tmp_path: Path) -> None:
    packet_path = _write_json(
        tmp_path / 'packet.json',
        _packet().model_dump(mode='json', exclude_none=True),
    )
    manifest_path = _write_json(
        tmp_path / 'bounded_packet_assembly_manifest.json',
        _manifest_payload(
            support=[_member('paper-1'), _member('paper-404')],
        ),
    )
    output_dir = tmp_path / 'phase9_bounded_packet_audit' / 'invalid-run'
    script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'run_bounded_packet_audit.py'

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            '--packet',
            str(packet_path),
            '--assembly-manifest',
            str(manifest_path),
            '--output-dir',
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert result.returncode == 1
    error_payload = json.loads(result.stderr)
    assert 'support members missing from RoutePacket.included_items: paper-404' in error_payload['error']
