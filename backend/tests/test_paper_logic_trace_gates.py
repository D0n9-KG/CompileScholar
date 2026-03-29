from __future__ import annotations

from app.paper_logic_trace.gates import (
    build_quality_payload,
    evaluate_hot_path_gate,
    needs_lightweight_audit,
)
from app.paper_logic_trace.models import EvidenceAnchor, MoveRelation, ResearchMove


def test_hot_path_gate_rejects_anchorless_move() -> None:
    result = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='method',
                act_type='propose_method',
                summary='Uses graph encoding',
                anchor_ids=[],
            )
        ],
        anchors=[],
    )

    assert result['passed'] is False
    assert result['invalid_move_ids'] == ['m-1']
    assert result['hard_fail_reasons'] == ['no_anchors', 'all_moves_invalid']


def test_sparse_valid_trace_can_export_with_downgraded_quality() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='method',
                act_type='propose_method',
                summary='Uses graph encoding',
                anchor_ids=['a-1'],
            )
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id='a-1',
                paper_id='paper-1',
                source_ref='chunk:1',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
        ],
    )

    quality = build_quality_payload(gate_report)

    assert gate_report['passed'] is True
    assert quality['quality_tier'] == 'yellow'
    assert quality['audit_status'] == 'eligible'
    assert quality['quality_tier_score'] > 0
    assert 'sparse_trace' in quality['quality_flags']


def test_lightweight_audit_is_quality_flag_not_export_blocker() -> None:
    gate_report = {
        'passed': True,
        'invalid_move_ids': [],
        'move_count': 1,
        'anchor_count': 1,
        'sparse_trace': True,
        'quality_tier_score': 0.61,
    }

    assert needs_lightweight_audit(gate_report) is True


def test_mixed_trace_passes_with_yellow_quality_instead_of_hard_failure() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-valid',
                sequence_no=1,
                role='method',
                act_type='propose_method',
                summary='We propose a graph encoder for retrieval tasks.',
                methods=[{'surface': 'graph encoder'}],
                research_objects=[{'surface': 'retrieval tasks'}],
                anchor_ids=['a-1'],
                slot_provenance=[
                    {'field': 'methods', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
            ResearchMove(
                move_id='m-invalid',
                sequence_no=2,
                role='result',
                act_type='report_effect',
                summary='Improves accuracy.',
                anchor_ids=[],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id='a-1',
                paper_id='paper-1',
                source_ref='chunk:1',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
        ],
    )

    quality = build_quality_payload(gate_report)

    assert gate_report['passed'] is True
    assert gate_report['invalid_move_ids'] == ['m-invalid']
    assert quality['quality_tier'] == 'yellow'
    assert quality['audit_status'] == 'eligible'
    assert 'invalid_moves_present' in quality['quality_flags']


def test_richer_trace_without_context_signal_is_yellow() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We study particle crushing prediction under high stress conditions.',
                research_objects=[{'surface': 'particle crushing prediction'}],
                anchor_ids=['a-1'],
                slot_provenance=[{'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We propose a discrete element simulation workflow for crushing behavior.',
                methods=[{'surface': 'discrete element simulation'}],
                anchor_ids=['a-2'],
                slot_provenance=[{'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='result',
                act_type='report_effect',
                summary='The proposed workflow improves prediction accuracy against the baseline.',
                metrics=[{'surface': 'prediction accuracy'}],
                effects=[{'direction': 'improve'}],
                comparators=[{'surface': 'baseline'}],
                anchor_ids=['a-3'],
                slot_provenance=[
                    {'field': 'metrics', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'effects', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3'], start=1)
        ],
    )

    quality = build_quality_payload(gate_report)

    assert gate_report['passed'] is True
    assert quality['quality_tier'] == 'yellow'
    assert quality['audit_status'] == 'eligible'
    assert quality['quality_tier_score'] >= 0.78


def test_empirical_trace_with_missing_result_and_thin_slots_is_downgraded_with_completeness_audit() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We study segregation in ribbon mixing.',
                research_objects=[{'surface': 'segregation in ribbon mixing'}],
                anchor_ids=['a-1'],
                slot_provenance=[{'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We propose a DEM simulation workflow for the mixer.',
                methods=[{'surface': 'DEM simulation workflow'}],
                anchor_ids=['a-2'],
                slot_provenance=[{'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2'], start=1)
        ],
        move_relations=[],
        paper_type='empirical',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert quality['quality_tier'] == 'yellow'
    assert quality['audit_status'] == 'eligible'
    assert audit['missing_expected_roles'] == ['result']
    assert 'metrics' in audit['missing_expected_slot_fields']
    assert 'comparators' in audit['missing_expected_slot_fields']
    assert 'effects' in audit['missing_expected_slot_fields']
    assert audit['ready_for_l3'] is False
    assert audit['ready_for_l4'] is False
    assert 'missing_expected_roles' in quality['quality_flags']
    assert 'missing_expected_slots' in quality['quality_flags']


def test_residual_noise_move_is_flagged_without_hard_blocking_export() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-noise',
                sequence_no=1,
                role='background',
                act_type='identify_gap',
                summary='# Abstract',
                anchor_ids=['a-1'],
            ),
            ResearchMove(
                move_id='m-real',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We propose a graph-based retrieval workflow.',
                methods=[{'surface': 'graph-based retrieval workflow'}],
                anchor_ids=['a-2'],
                slot_provenance=[{'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2'], start=1)
        ],
        move_relations=[],
        paper_type='unknown',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert gate_report['passed'] is True
    assert quality['quality_tier'] == 'yellow'
    assert 'residual_noise_moves' in quality['quality_flags']
    assert audit['noise_move_ids'] == ['m-noise']


def test_rich_empirical_trace_is_marked_ready_for_l3_and_l4() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We study particle crushing prediction under high stress conditions.',
                research_objects=[{'surface': 'particle crushing prediction'}],
                conditions=[{'surface': 'high stress conditions'}],
                anchor_ids=['a-1'],
                slot_provenance=[
                    {'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'conditions', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We propose a discrete element simulation workflow using x-ray microtomography observations.',
                methods=[{'surface': 'discrete element simulation workflow'}],
                resource_mentions=[{'surface': 'x-ray microtomography'}],
                anchor_ids=['a-2'],
                slot_provenance=[
                    {'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'resource_mentions', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='result',
                act_type='report_effect',
                summary='The proposed workflow improves prediction accuracy over the baseline and reduces error.',
                metrics=[{'surface': 'prediction accuracy'}, {'surface': 'error'}],
                comparators=[{'surface': 'baseline'}],
                effects=[{'direction': 'improve'}],
                limitation_types=[{'surface': 'computational cost'}],
                anchor_ids=['a-3'],
                slot_provenance=[
                    {'field': 'metrics', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'comparators', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'effects', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'limitation_types', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3'], start=1)
        ],
        move_relations=[
            {
                'relation_id': 'r-1',
                'source_move_id': 'm-1',
                'target_move_id': 'm-2',
                'relation_type': 'addresses',
                'anchor_ids': ['a-1'],
            },
            {
                'relation_id': 'r-2',
                'source_move_id': 'm-2',
                'target_move_id': 'm-3',
                'relation_type': 'yields',
                'anchor_ids': ['a-2'],
            },
        ],
        paper_type='empirical',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert quality['quality_tier'] == 'green'
    assert quality['audit_status'] == 'not_needed'
    assert audit['missing_expected_roles'] == []
    assert audit['missing_expected_slot_fields'] == []
    assert audit['ready_for_community'] is True
    assert audit['ready_for_l3'] is True
    assert audit['ready_for_l4'] is True


def test_unknown_trace_without_result_signal_is_not_ready_for_upper_layers() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We investigate erosion in a mixed pump.',
                research_objects=[{'surface': 'erosion in a mixed pump'}],
                anchor_ids=['a-1'],
                slot_provenance=[{'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We couple CFD and DEM for the pump simulation.',
                methods=[{'surface': 'CFD-DEM coupled method'}],
                anchor_ids=['a-2'],
                slot_provenance=[{'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='experiment',
                act_type='run_experiment',
                summary='The simulation is carried out under several operating conditions.',
                conditions=[{'surface': 'several operating conditions'}],
                metrics=[{'surface': 'wear rate'}],
                comparators=[{'surface': 'different blade regions'}],
                effects=[{'direction': 'increase'}],
                anchor_ids=['a-3'],
                slot_provenance=[
                    {'field': 'conditions', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'metrics', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'comparators', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'effects', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3'], start=1)
        ],
        move_relations=[
            {
                'relation_id': 'r-1',
                'source_move_id': 'm-1',
                'target_move_id': 'm-2',
                'relation_type': 'addresses',
                'anchor_ids': ['a-1'],
            },
            {
                'relation_id': 'r-2',
                'source_move_id': 'm-2',
                'target_move_id': 'm-3',
                'relation_type': 'evaluates',
                'anchor_ids': ['a-2'],
            },
        ],
        paper_type='unknown',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert quality['quality_tier'] == 'yellow'
    assert audit['ready_for_community'] is True
    assert audit['ready_for_l3'] is False
    assert audit['ready_for_l4'] is False


def test_empirical_trace_without_research_objects_is_not_ready_for_l4() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We investigate segregation in shallow granular flows.',
                anchor_ids=['a-1'],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We extend a continuum model with asymmetric flux functions.',
                methods=[{'surface': 'continuum model'}, {'surface': 'asymmetric flux functions'}],
                anchor_ids=['a-2'],
                slot_provenance=[{'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='result',
                act_type='report_effect',
                summary='The asymmetric model improves agreement with experiments over the symmetric baseline.',
                metrics=[{'surface': 'agreement'}],
                comparators=[{'surface': 'symmetric baseline'}],
                effects=[{'direction': 'improve'}],
                conditions=[{'surface': 'shallow flow conditions'}],
                anchor_ids=['a-3'],
                slot_provenance=[
                    {'field': 'metrics', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'comparators', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'effects', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'conditions', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3'], start=1)
        ],
        move_relations=[
            {
                'relation_id': 'r-1',
                'source_move_id': 'm-1',
                'target_move_id': 'm-2',
                'relation_type': 'addresses',
                'anchor_ids': ['a-1'],
            },
            {
                'relation_id': 'r-2',
                'source_move_id': 'm-2',
                'target_move_id': 'm-3',
                'relation_type': 'yields',
                'anchor_ids': ['a-2'],
            },
        ],
        paper_type='empirical',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert 'research_objects' in audit['missing_expected_slot_fields']
    assert audit['ready_for_l3'] is True
    assert audit['ready_for_l4'] is False


def test_anchorless_relations_do_not_count_toward_l3_readiness() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We study particle crushing prediction under high stress conditions.',
                research_objects=[{'surface': 'particle crushing prediction'}],
                anchor_ids=['a-1'],
                slot_provenance=[{'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We propose a discrete element simulation workflow.',
                methods=[{'surface': 'discrete element simulation workflow'}],
                anchor_ids=['a-2'],
                slot_provenance=[{'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='result',
                act_type='report_effect',
                summary='The workflow improves prediction accuracy over the baseline under high stress conditions.',
                metrics=[{'surface': 'prediction accuracy'}],
                comparators=[{'surface': 'baseline'}],
                conditions=[{'surface': 'high stress conditions'}],
                effects=[{'direction': 'improve'}],
                anchor_ids=['a-3'],
                slot_provenance=[
                    {'field': 'metrics', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'comparators', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'conditions', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'effects', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3'], start=1)
        ],
        move_relations=[
            MoveRelation(
                relation_id='r-1',
                source_move_id='m-1',
                target_move_id='m-2',
                relation_type='addresses',
                anchor_ids=[],
                confidence=0.65,
            ),
            MoveRelation(
                relation_id='r-2',
                source_move_id='m-2',
                target_move_id='m-3',
                relation_type='yields',
                anchor_ids=[],
                confidence=0.65,
            ),
        ],
        paper_type='empirical',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert audit['relation_coverage_ratio'] == 0.0
    assert audit['ready_for_l3'] is False
    assert audit['ready_for_l4'] is False
    assert quality['quality_tier'] == 'yellow'


def test_unknown_effect_does_not_unlock_l4_readiness() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We study particle crushing prediction under high stress conditions.',
                research_objects=[{'surface': 'particle crushing prediction'}],
                anchor_ids=['a-1'],
                slot_provenance=[{'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We propose a discrete element simulation workflow with x-ray microtomography.',
                methods=[{'surface': 'discrete element simulation workflow'}],
                resource_mentions=[{'surface': 'x-ray microtomography'}],
                anchor_ids=['a-2'],
                slot_provenance=[
                    {'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'resource_mentions', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='result',
                act_type='report_effect',
                summary='The workflow changes prediction accuracy relative to the baseline under high stress conditions.',
                metrics=[{'surface': 'prediction accuracy'}],
                comparators=[{'surface': 'baseline'}],
                conditions=[{'surface': 'high stress conditions'}],
                effects=[{'direction': 'unknown'}],
                anchor_ids=['a-3'],
                slot_provenance=[
                    {'field': 'metrics', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'comparators', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'conditions', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'effects', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3'], start=1)
        ],
        move_relations=[
            MoveRelation(
                relation_id='r-1',
                source_move_id='m-1',
                target_move_id='m-2',
                relation_type='addresses',
                anchor_ids=['a-1'],
                confidence=0.86,
            ),
            MoveRelation(
                relation_id='r-2',
                source_move_id='m-2',
                target_move_id='m-3',
                relation_type='yields',
                anchor_ids=['a-2'],
                confidence=0.86,
            ),
        ],
        paper_type='empirical',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert audit['ready_for_l3'] is True
    assert audit['ready_for_l4'] is False
    assert quality['quality_tier'] == 'yellow'
    assert audit['completeness_score'] < 1.0


def test_empirical_trace_without_supported_comparator_is_not_ready_for_l4() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We study particle anti-rotation effects in granular materials.',
                research_objects=[{'surface': 'particle anti-rotation effects'}],
                anchor_ids=['a-1'],
                slot_provenance=[{'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We compare DEM assemblies with rolling resistance and irregular particle shapes.',
                methods=[{'surface': 'DEM comparison workflow'}],
                anchor_ids=['a-2'],
                slot_provenance=[{'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='result',
                act_type='report_effect',
                summary='The irregular-shape workflow improves strain localization fidelity under triaxial shear.',
                metrics=[{'surface': 'strain localization fidelity'}],
                comparators=[{'surface': 'rolling resistance baseline'}],
                conditions=[{'surface': 'triaxial shear'}],
                effects=[{'direction': 'improve'}],
                anchor_ids=['a-3'],
                slot_provenance=[
                    {'field': 'metrics', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'comparators', 'anchor_ids': ['a-3'], 'extraction_mode': 'inferred', 'support_strength': 'weak'},
                    {'field': 'conditions', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'effects', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3'], start=1)
        ],
        move_relations=[
            MoveRelation(
                relation_id='r-1',
                source_move_id='m-1',
                target_move_id='m-2',
                relation_type='addresses',
                anchor_ids=['a-1', 'a-2'],
                confidence=0.82,
            ),
            MoveRelation(
                relation_id='r-2',
                source_move_id='m-2',
                target_move_id='m-3',
                relation_type='yields',
                anchor_ids=['a-2', 'a-3'],
                confidence=0.82,
            ),
        ],
        paper_type='empirical',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert audit['ready_for_l3'] is True
    assert audit['ready_for_l4'] is False
    assert quality['quality_tier'] == 'yellow'


def test_limitation_only_comparator_does_not_unlock_l4_readiness() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We study anti-rotation effects in granular materials.',
                research_objects=[{'surface': 'anti-rotation effects'}],
                anchor_ids=['a-1'],
                slot_provenance=[{'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We use DEM to compare irregular and spherical assemblies.',
                methods=[{'surface': 'DEM comparison workflow'}],
                anchor_ids=['a-2'],
                slot_provenance=[{'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='result',
                act_type='report_effect',
                summary='Irregular assemblies improve strain localization fidelity under shear.',
                metrics=[{'surface': 'strain localization fidelity'}],
                conditions=[{'surface': 'triaxial shear'}],
                effects=[{'direction': 'improve'}],
                anchor_ids=['a-3'],
                slot_provenance=[
                    {'field': 'metrics', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'conditions', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'effects', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
            ResearchMove(
                move_id='m-4',
                sequence_no=4,
                role='limitation',
                act_type='state_limitation',
                summary='Rolling resistance cannot replace particle shape effects.',
                comparators=[{'surface': 'particle shape effects'}],
                limitation_types=[{'surface': 'mechanistic mismatch'}],
                anchor_ids=['a-4'],
                slot_provenance=[
                    {'field': 'comparators', 'anchor_ids': ['a-4'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'limitation_types', 'anchor_ids': ['a-4'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3', 'a-4'], start=1)
        ],
        move_relations=[
            MoveRelation(
                relation_id='r-1',
                source_move_id='m-1',
                target_move_id='m-2',
                relation_type='addresses',
                anchor_ids=['a-1', 'a-2'],
                confidence=0.82,
            ),
            MoveRelation(
                relation_id='r-2',
                source_move_id='m-2',
                target_move_id='m-3',
                relation_type='yields',
                anchor_ids=['a-2', 'a-3'],
                confidence=0.82,
            ),
            MoveRelation(
                relation_id='r-3',
                source_move_id='m-3',
                target_move_id='m-4',
                relation_type='limits',
                anchor_ids=['a-3', 'a-4'],
                confidence=0.82,
            ),
        ],
        paper_type='empirical',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert audit['ready_for_l3'] is True
    assert audit['ready_for_l4'] is False
    assert quality['quality_tier'] == 'yellow'


def test_single_supported_comparator_without_comparison_signal_does_not_unlock_l4() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We study anti-rotation effects in granular materials.',
                research_objects=[{'surface': 'anti-rotation effects'}],
                anchor_ids=['a-1'],
                slot_provenance=[{'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We compare DEM assemblies with rolling resistance and irregular particle shapes.',
                methods=[{'surface': 'DEM comparison workflow'}],
                anchor_ids=['a-2'],
                slot_provenance=[{'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='result',
                act_type='report_effect',
                summary='The irregular-shape workflow improves strain localization fidelity under triaxial shear.',
                metrics=[{'surface': 'strain localization fidelity'}],
                comparators=[{'surface': 'rolling resistance baseline'}],
                conditions=[{'surface': 'triaxial shear'}],
                effects=[{'direction': 'improve'}],
                anchor_ids=['a-3'],
                slot_provenance=[
                    {'field': 'metrics', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'comparators', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'conditions', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'effects', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3'], start=1)
        ],
        move_relations=[
            MoveRelation(
                relation_id='r-1',
                source_move_id='m-1',
                target_move_id='m-2',
                relation_type='addresses',
                anchor_ids=['a-1'],
                confidence=0.86,
            ),
            MoveRelation(
                relation_id='r-2',
                source_move_id='m-2',
                target_move_id='m-3',
                relation_type='yields',
                anchor_ids=['a-2'],
                confidence=0.86,
            ),
        ],
        paper_type='empirical',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert audit['ready_for_l3'] is True
    assert audit['ready_for_l4'] is False
    assert 'comparators' in audit['missing_supported_expected_slot_fields']


def test_compare_baseline_move_with_supported_comparator_unlocks_l4() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We study segregation during die filling.',
                research_objects=[{'surface': 'segregation during die filling'}],
                anchor_ids=['a-1'],
                slot_provenance=[{'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We use DEM and experiments to compare filling behavior.',
                methods=[{'surface': 'DEM and experimental workflow'}],
                resource_mentions=[{'surface': 'high-speed camera'}],
                anchor_ids=['a-2'],
                slot_provenance=[
                    {'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'resource_mentions', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='result',
                act_type='compare_baseline',
                summary='The DEM predictions show good agreement with experimental results under high die velocity.',
                metrics=[{'surface': 'filling ratio'}],
                comparators=[{'surface': 'experimental results'}],
                conditions=[{'surface': 'high die velocity'}],
                effects=[{'direction': 'improve'}],
                anchor_ids=['a-3'],
                slot_provenance=[
                    {'field': 'metrics', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'comparators', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'conditions', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'effects', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3'], start=1)
        ],
        move_relations=[
            MoveRelation(
                relation_id='r-1',
                source_move_id='m-1',
                target_move_id='m-2',
                relation_type='addresses',
                anchor_ids=['a-1'],
                confidence=0.86,
            ),
            MoveRelation(
                relation_id='r-2',
                source_move_id='m-2',
                target_move_id='m-3',
                relation_type='yields',
                anchor_ids=['a-2'],
                confidence=0.86,
            ),
        ],
        paper_type='empirical',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert audit['ready_for_l3'] is True
    assert audit['ready_for_l4'] is True


def test_software_trace_does_not_require_result_role_for_green_quality() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We introduce an open-source DEM platform for granular simulation workflows.',
                research_objects=[{'surface': 'granular simulation workflows'}],
                anchor_ids=['a-1'],
                slot_provenance=[{'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='build_resource',
                summary='The software provides a discrete element engine and reusable scripting interfaces.',
                methods=[{'surface': 'discrete element engine'}],
                resource_mentions=[{'surface': 'open-source software'}],
                anchor_ids=['a-2'],
                slot_provenance=[
                    {'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'resource_mentions', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2'], start=1)
        ],
        move_relations=[
            {
                'relation_id': 'r-1',
                'source_move_id': 'm-1',
                'target_move_id': 'm-2',
                'relation_type': 'addresses',
                'anchor_ids': ['a-1'],
            },
        ],
        paper_type='software',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert quality['quality_tier'] == 'green'
    assert audit['missing_expected_roles'] == []
    assert audit['ready_for_l3'] is True


def test_theoretical_trace_with_grounded_constraints_can_be_ready_for_l3_and_l4() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We study data-driven computational plasticity for nonlinear elasticity under non-isothermal loading.',
                research_objects=[{'surface': 'data-driven computational plasticity'}, {'surface': 'nonlinear elasticity'}],
                conditions=[{'surface': 'non-isothermal loading'}],
                anchor_ids=['a-1'],
                slot_provenance=[
                    {'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'conditions', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We derive an incremental variational formulation with constitutive updates driven by material data instead of fixed yield laws.',
                methods=[{'surface': 'incremental variational formulation'}],
                research_objects=[{'surface': 'constitutive updates'}],
                anchor_ids=['a-2'],
                slot_provenance=[
                    {'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'research_objects', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='interpretation',
                act_type='explain_mechanism',
                summary='The formulation remains thermodynamically consistent under path-dependent loading but is limited by rate-independent assumptions.',
                conditions=[{'surface': 'path-dependent loading'}],
                limitation_types=[{'surface': 'rate-independent assumptions'}],
                anchor_ids=['a-3'],
                slot_provenance=[
                    {'field': 'conditions', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'limitation_types', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3'], start=1)
        ],
        move_relations=[
            MoveRelation(
                relation_id='r-1',
                source_move_id='m-1',
                target_move_id='m-2',
                relation_type='addresses',
                anchor_ids=['a-1'],
                confidence=0.84,
            ),
            MoveRelation(
                relation_id='r-2',
                source_move_id='m-2',
                target_move_id='m-3',
                relation_type='explains',
                anchor_ids=['a-2'],
                confidence=0.84,
            ),
        ],
        paper_type='theoretical',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert audit['missing_expected_roles'] == []
    assert audit['missing_expected_slot_fields'] == []
    assert audit['ready_for_l3'] is True
    assert audit['ready_for_l4'] is True
    assert quality['quality_tier'] == 'green'


def test_theoretical_trace_with_grounded_limitation_can_be_ready_without_interpretation_or_conditions() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We study data-driven constitutive updates for nonlinear plasticity.',
                research_objects=[{'surface': 'data-driven constitutive updates'}],
                anchor_ids=['a-1'],
                slot_provenance=[{'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We derive a variational data-driven solver for path-dependent constitutive response.',
                methods=[{'surface': 'variational data-driven solver'}],
                research_objects=[{'surface': 'path-dependent constitutive response'}],
                anchor_ids=['a-2'],
                slot_provenance=[
                    {'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'research_objects', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='result',
                act_type='report_effect',
                summary='The formulation extends data-driven nonlinear elasticity to internal-variable plasticity.',
                effects=[{'direction': 'improve'}],
                anchor_ids=['a-3'],
                slot_provenance=[{'field': 'effects', 'anchor_ids': ['a-3'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-4',
                sequence_no=4,
                role='limitation',
                act_type='state_limitation',
                summary='Its main drawback is the huge amount of data required for running simulations.',
                limitation_types=[{'surface': 'huge amount of data required for running simulations'}],
                anchor_ids=['a-4'],
                slot_provenance=[{'field': 'limitation_types', 'anchor_ids': ['a-4'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3', 'a-4'], start=1)
        ],
        move_relations=[
            MoveRelation(
                relation_id='r-1',
                source_move_id='m-1',
                target_move_id='m-2',
                relation_type='addresses',
                anchor_ids=['a-1'],
                confidence=0.84,
            ),
            MoveRelation(
                relation_id='r-2',
                source_move_id='m-2',
                target_move_id='m-3',
                relation_type='yields',
                anchor_ids=['a-2'],
                confidence=0.84,
            ),
            MoveRelation(
                relation_id='r-3',
                source_move_id='m-3',
                target_move_id='m-4',
                relation_type='limits',
                anchor_ids=['a-3'],
                confidence=0.84,
            ),
        ],
        paper_type='theoretical',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert audit['missing_expected_roles'] == []
    assert audit['missing_expected_slot_fields'] == []
    assert audit['ready_for_l3'] is True
    assert audit['ready_for_l4'] is True
    assert audit['l4_evidence_profile'] == 'theory_modeling'
    assert quality['quality_tier'] == 'green'


def test_theoretical_trace_without_grounded_constraint_signal_is_not_ready_for_l4() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We study constitutive-model selection in nonlinear solids.',
                research_objects=[{'surface': 'constitutive-model selection'}],
                anchor_ids=['a-1'],
                slot_provenance=[{'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We derive a variational update rule for the constitutive model family.',
                methods=[{'surface': 'variational update rule'}],
                conditions=[{'surface': 'incremental loading'}],
                anchor_ids=['a-2'],
                slot_provenance=[
                    {'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'conditions', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='interpretation',
                act_type='explain_mechanism',
                summary='The framework provides a conceptual interpretation of constitutive updates.',
                anchor_ids=['a-3'],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3'], start=1)
        ],
        move_relations=[
            MoveRelation(
                relation_id='r-1',
                source_move_id='m-1',
                target_move_id='m-2',
                relation_type='addresses',
                anchor_ids=['a-1'],
                confidence=0.84,
            ),
            MoveRelation(
                relation_id='r-2',
                source_move_id='m-2',
                target_move_id='m-3',
                relation_type='explains',
                anchor_ids=['a-2'],
                confidence=0.84,
            ),
        ],
        paper_type='theoretical',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert audit['ready_for_l3'] is True
    assert audit['ready_for_l4'] is False
    assert quality['quality_tier'] == 'yellow'


def test_unknown_trace_with_theory_like_limitation_profile_can_use_theory_modeling_l4_path() -> None:
    gate_report = evaluate_hot_path_gate(
        moves=[
            ResearchMove(
                move_id='m-1',
                sequence_no=1,
                role='problem',
                act_type='define_task',
                summary='We study data-driven computational plasticity for nonlinear constitutive updates.',
                research_objects=[{'surface': 'data-driven computational plasticity'}],
                anchor_ids=['a-1'],
                slot_provenance=[{'field': 'research_objects', 'anchor_ids': ['a-1'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
            ResearchMove(
                move_id='m-2',
                sequence_no=2,
                role='method',
                act_type='propose_method',
                summary='We derive a variational data-driven solver for path-dependent constitutive response.',
                methods=[{'surface': 'variational data-driven solver'}],
                research_objects=[{'surface': 'path-dependent constitutive response'}],
                anchor_ids=['a-2'],
                slot_provenance=[
                    {'field': 'methods', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                    {'field': 'research_objects', 'anchor_ids': ['a-2'], 'extraction_mode': 'direct', 'support_strength': 'strong'},
                ],
            ),
            ResearchMove(
                move_id='m-3',
                sequence_no=3,
                role='result',
                act_type='report_effect',
                summary='The formulation extends data-driven nonlinear elasticity to internal-variable plasticity.',
                anchor_ids=['a-3'],
            ),
            ResearchMove(
                move_id='m-4',
                sequence_no=4,
                role='limitation',
                act_type='state_limitation',
                summary='Its main drawback is the huge amount of data required for running simulations.',
                limitation_types=[{'surface': 'huge amount of data required for running simulations'}],
                anchor_ids=['a-4'],
                slot_provenance=[{'field': 'limitation_types', 'anchor_ids': ['a-4'], 'extraction_mode': 'direct', 'support_strength': 'strong'}],
            ),
        ],
        anchors=[
            EvidenceAnchor(
                anchor_id=anchor_id,
                paper_id='paper-1',
                source_ref=f'chunk:{idx}',
                modality='text',
                section_path=[],
                locator={},
                quote='demo',
                citation_ids=[],
                support_type='direct',
                weak=False,
            )
            for idx, anchor_id in enumerate(['a-1', 'a-2', 'a-3', 'a-4'], start=1)
        ],
        move_relations=[
            MoveRelation(
                relation_id='r-1',
                source_move_id='m-1',
                target_move_id='m-2',
                relation_type='addresses',
                anchor_ids=['a-1'],
                confidence=0.84,
            ),
            MoveRelation(
                relation_id='r-2',
                source_move_id='m-2',
                target_move_id='m-3',
                relation_type='yields',
                anchor_ids=['a-2'],
                confidence=0.84,
            ),
            MoveRelation(
                relation_id='r-3',
                source_move_id='m-3',
                target_move_id='m-4',
                relation_type='limits',
                anchor_ids=['a-3'],
                confidence=0.84,
            ),
        ],
        paper_type='unknown',
    )

    quality = build_quality_payload(gate_report)
    audit = quality['l2_completeness_audit']

    assert audit['l4_evidence_profile'] == 'theory_modeling'
    assert audit['ready_for_l3'] is True
    assert audit['ready_for_l4'] is True
    assert quality['quality_tier'] == 'green'


def test_quality_payload_downgrades_when_title_and_summary_are_semantically_misaligned() -> None:
    gate_report = {
        'passed': True,
        'invalid_move_ids': [],
        'move_count': 4,
        'anchor_count': 4,
        'sparse_trace': False,
        'quality_tier_score': 0.91,
        'soft_flags': [],
        'l2_completeness_audit': {
            'paper_type': 'empirical',
            'missing_expected_roles': [],
            'missing_expected_slot_fields': [],
            'noise_move_ids': [],
            'critical_role_coverage_ratio': 1.0,
            'ready_for_l3': True,
            'ready_for_l4': True,
        },
    }

    quality = build_quality_payload(
        gate_report,
        paper_metadata={
            'title': '高功率光纤激光热光效应及模式不稳定阈值特性研究',
            'authors': ['李学文', '于春雷'],
        },
        derived_views={
            'paper_summaries': {
                'one_paragraph_summary': '本文对卧式双轴圆盘反应器的功率特性进行了比较详细的研究，并给出了统一的功率关联式。',
            }
        },
    )

    assert quality['quality_tier'] == 'yellow'
    assert quality['audit_status'] == 'eligible'
    assert 'metadata_summary_mismatch' in quality['quality_flags']


def test_quality_payload_flags_thin_route_state_seed_for_l3_follow_on_work() -> None:
    gate_report = {
        'passed': True,
        'invalid_move_ids': [],
        'move_count': 4,
        'anchor_count': 4,
        'sparse_trace': False,
        'quality_tier_score': 0.9,
        'soft_flags': [],
        'l2_completeness_audit': {
            'paper_type': 'empirical',
            'missing_expected_roles': [],
            'missing_expected_slot_fields': [],
            'noise_move_ids': [],
            'critical_role_coverage_ratio': 1.0,
            'ready_for_l3': True,
            'ready_for_l4': True,
        },
    }

    quality = build_quality_payload(
        gate_report,
        derived_views={
            'route_state_seed': {
                'paper_id': 'paper-1',
                'paper_type': 'empirical',
                'source_trace_id': 'paper-1:paper_logic_trace',
                'cutoff_year_hint': 2024,
                'topic_scope_candidates': ['entity relation graph'],
                'dominant_method_candidates': ['graph neural network'],
                'active_benchmark_candidates': [],
                'known_bottleneck_candidates': [],
                'enabling_condition_candidates': [],
                'measurement_protocol_candidates': [],
                'toolchain_candidates': [],
                'supporting_evidence_ids': ['a-1'],
                'challenging_evidence_ids': [],
                'source_move_ids': ['m-1', 'm-2', 'm-3'],
                'readiness_feature_inputs': {
                    'method_maturity_signals': ['graph neural network'],
                    'measurement_maturity_signals': [],
                    'data_resource_signals': [],
                    'infrastructure_signals': [],
                    'bottleneck_signals': [],
                },
            }
        },
    )

    assert quality['quality_tier'] == 'yellow'
    assert quality['audit_status'] == 'eligible'
    assert 'route_state_seed_thin' in quality['quality_flags']
    assert quality['route_state_seed_audit']['available'] is True
    assert quality['route_state_seed_audit']['ready_for_route_compilation'] is False
    assert quality['route_state_seed_audit']['missing_seed_components'] == [
        'active_benchmark_candidates',
        'known_bottleneck_candidates',
        'enabling_condition_candidates',
        'alternative_route_candidates',
        'measurement_protocol_candidates',
        'toolchain_candidates',
        'challenging_evidence_ids',
        'readiness_feature_inputs.measurement_maturity_signals',
        'readiness_feature_inputs.data_resource_signals',
        'readiness_feature_inputs.infrastructure_signals',
        'readiness_feature_inputs.bottleneck_signals',
    ]
    assert quality['route_state_seed_audit']['missing_required_seed_components'] == []
    assert quality['route_state_seed_audit']['route_compilation_blockers'] == ['context_groups<2']


def test_quality_payload_accepts_route_seed_with_challenge_and_bottleneck_signals_even_without_benchmark_or_toolchain() -> None:
    gate_report = {
        'passed': True,
        'invalid_move_ids': [],
        'move_count': 6,
        'anchor_count': 6,
        'sparse_trace': False,
        'quality_tier_score': 0.9,
        'soft_flags': [],
        'l2_completeness_audit': {
            'paper_type': 'empirical',
            'missing_expected_roles': [],
            'missing_expected_slot_fields': [],
            'noise_move_ids': [],
            'critical_role_coverage_ratio': 1.0,
            'ready_for_l3': True,
            'ready_for_l4': True,
        },
    }

    quality = build_quality_payload(
        gate_report,
        derived_views={
            'route_state_seed': {
                'paper_id': 'paper-1',
                'paper_type': 'empirical',
                'source_trace_id': 'paper-1:paper_logic_trace',
                'cutoff_year_hint': 2024,
                'topic_scope_candidates': ['nonlinear elasticity', 'internal variables'],
                'dominant_method_candidates': ['data-driven constitutive modeling'],
                'active_benchmark_candidates': [],
                'known_bottleneck_candidates': ['huge amount of data required'],
                'enabling_condition_candidates': [],
                'alternative_route_candidates': [],
                'measurement_protocol_candidates': [],
                'toolchain_candidates': [],
                'supporting_evidence_ids': ['a-1', 'a-2'],
                'challenging_evidence_ids': ['a-3'],
                'source_move_ids': ['m-1', 'm-2', 'm-3', 'm-4'],
                'readiness_feature_inputs': {
                    'method_maturity_signals': ['data-driven constitutive modeling'],
                    'measurement_maturity_signals': [],
                    'data_resource_signals': [],
                    'infrastructure_signals': [],
                    'bottleneck_signals': ['huge amount of data required'],
                },
            }
        },
    )

    assert quality['quality_tier'] == 'green'
    assert quality['audit_status'] == 'not_needed'
    assert 'route_state_seed_thin' not in quality['quality_flags']
    assert quality['route_state_seed_audit']['available'] is True
    assert quality['route_state_seed_audit']['ready_for_route_compilation'] is True


def test_quality_payload_accepts_route_seed_with_measurement_and_enabling_signals_without_benchmark_data_resources() -> None:
    gate_report = {
        'passed': True,
        'invalid_move_ids': [],
        'move_count': 6,
        'anchor_count': 6,
        'sparse_trace': False,
        'quality_tier_score': 0.9,
        'soft_flags': [],
        'l2_completeness_audit': {
            'paper_type': 'empirical',
            'missing_expected_roles': [],
            'missing_expected_slot_fields': [],
            'noise_move_ids': [],
            'critical_role_coverage_ratio': 1.0,
            'ready_for_l3': True,
            'ready_for_l4': True,
        },
    }

    quality = build_quality_payload(
        gate_report,
        derived_views={
            'route_state_seed': {
                'paper_id': 'paper-1',
                'paper_type': 'empirical',
                'source_trace_id': 'paper-1:paper_logic_trace',
                'cutoff_year_hint': 2024,
                'topic_scope_candidates': ['photocatalytic hydrogen evolution reaction'],
                'dominant_method_candidates': ['quantum mechanics in explicit solvent'],
                'active_benchmark_candidates': [],
                'known_bottleneck_candidates': [],
                'enabling_condition_candidates': ['acidic solvent'],
                'alternative_route_candidates': [],
                'measurement_protocol_candidates': [
                    {
                        'move_id': 'm-3',
                        'role': 'result',
                        'act_type': 'report_effect',
                        'summary': 'Reports calculated energy barriers for the proposed mechanism.',
                        'metric_tokens': ['energy barrier'],
                        'comparator_tokens': ['dark reaction'],
                        'condition_tokens': [],
                        'method_tokens': ['quantum mechanical calculations in explicit solvent'],
                        'resource_tokens': [],
                        'anchor_ids': ['a-3'],
                        'confidence': 0.91,
                    }
                ],
                'toolchain_candidates': [],
                'supporting_evidence_ids': ['a-1', 'a-2', 'a-3'],
                'challenging_evidence_ids': [],
                'source_move_ids': ['m-1', 'm-2', 'm-3'],
                'readiness_feature_inputs': {
                    'method_maturity_signals': ['quantum mechanics in explicit solvent'],
                    'measurement_maturity_signals': ['energy barrier'],
                    'data_resource_signals': [],
                    'infrastructure_signals': [],
                    'bottleneck_signals': [],
                },
            }
        },
    )

    assert quality['quality_tier'] == 'green'
    assert quality['audit_status'] == 'not_needed'
    assert 'route_state_seed_thin' not in quality['quality_flags']
    assert quality['route_state_seed_audit']['available'] is True
    assert quality['route_state_seed_audit']['ready_for_route_compilation'] is True
