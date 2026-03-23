from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from .derived_views import build_derived_views
from .gates import build_quality_payload, evaluate_hot_path_gate
from .models import (
    CanonicalCore,
    CitationAct,
    EffectValue,
    EvidenceAnchor,
    FigureRef,
    MentionValue,
    MoveRelation,
    PaperLogicTrace,
    PaperMetadata,
    ResearchMove,
    SlotProvenance,
    TableRef,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _mention_values(rows: list[dict[str, Any]] | None) -> list[MentionValue]:
    values: list[MentionValue] = []
    for row in rows or []:
        values.append(
            MentionValue(
                surface=str(row.get('surface') or '').strip(),
                normalized=str(row.get('normalized') or '').strip() or None,
                type=str(row.get('type') or '').strip() or None,
                anchor_ids=[str(item).strip() for item in (row.get('anchor_ids') or []) if str(item).strip()],
                confidence=row.get('confidence'),
                inferred=bool(row.get('inferred') or False),
            )
        )
    return values


def _effect_values(rows: list[dict[str, Any]] | None) -> list[EffectValue]:
    values: list[EffectValue] = []
    for row in rows or []:
        values.append(
            EffectValue(
                direction=str(row.get('direction') or 'unknown'),
                magnitude_text=str(row.get('magnitude_text') or '').strip() or None,
                magnitude_numeric=row.get('magnitude_numeric'),
                unit=str(row.get('unit') or '').strip() or None,
                comparator_surface=str(row.get('comparator_surface') or '').strip() or None,
                anchor_ids=[str(item).strip() for item in (row.get('anchor_ids') or []) if str(item).strip()],
                confidence=row.get('confidence'),
            )
        )
    return values


def _slot_provenance(rows: list[dict[str, Any]] | None) -> list[SlotProvenance]:
    values: list[SlotProvenance] = []
    for row in rows or []:
        values.append(
            SlotProvenance(
                field=str(row.get('field') or '').strip(),
                value_index=row.get('value_index'),
                anchor_ids=[str(item).strip() for item in (row.get('anchor_ids') or []) if str(item).strip()],
                extraction_mode=str(row.get('extraction_mode') or 'direct'),
                support_strength=str(row.get('support_strength') or 'strong'),
                confidence=row.get('confidence'),
                notes=str(row.get('notes') or '').strip() or None,
            )
        )
    return values


def _compile_anchors(evidence_rows: list[dict[str, Any]]) -> list[EvidenceAnchor]:
    anchors: list[EvidenceAnchor] = []
    for row in evidence_rows:
        anchors.append(
            EvidenceAnchor(
                anchor_id=str(row.get('anchor_id') or '').strip(),
                paper_id=str(row.get('paper_id') or '').strip(),
                source_ref=str(row.get('source_ref') or '').strip(),
                modality=str(row.get('modality') or 'text'),
                section_path=[str(item).strip() for item in (row.get('section_path') or []) if str(item).strip()],
                locator=dict(row.get('locator') or {}),
                quote=str(row.get('quote') or '').strip(),
                citation_ids=[str(item).strip() for item in (row.get('citation_ids') or []) if str(item).strip()],
                support_type=str(row.get('support_type') or 'direct'),
                weak=bool(row.get('weak') or False),
            )
        )
    return anchors


def _compile_moves(paper_id: str, evidence_rows: list[dict[str, Any]]) -> list[ResearchMove]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(evidence_rows, start=1):
        move_id = str(row.get('move_id') or f'{paper_id}:move:{index}').strip()
        grouped[move_id].append(row)

    moves: list[ResearchMove] = []
    for default_index, (move_id, rows) in enumerate(grouped.items(), start=1):
        first = rows[0]
        anchor_ids = [
            str(item.get('anchor_id') or '').strip()
            for item in rows
            if str(item.get('anchor_id') or '').strip()
        ]
        move = ResearchMove(
            move_id=move_id,
            sequence_no=int(first.get('sequence_no') or default_index),
            role=str(first.get('role_hint') or 'background'),
            act_type=str(first.get('act_hint') or 'define_task'),
            summary=str(first.get('summary') or first.get('quote') or '').strip(),
            research_objects=_mention_values(first.get('research_objects')),
            methods=_mention_values(first.get('methods')),
            observed_variables=_mention_values(first.get('observed_variables')),
            metrics=_mention_values(first.get('metrics')),
            comparators=_mention_values(first.get('comparators')),
            conditions=_mention_values(first.get('conditions')),
            effects=_effect_values(first.get('effects')),
            limitation_types=_mention_values(first.get('limitation_types')),
            resource_mentions=_mention_values(first.get('resource_mentions')),
            anchor_ids=anchor_ids,
            slot_provenance=_slot_provenance(first.get('slot_provenance')),
            confidence=first.get('confidence'),
        )
        moves.append(move)

    return sorted(moves, key=lambda item: (item.sequence_no, item.move_id))


def _compile_citation_acts(citation_rows: list[dict[str, Any]]) -> list[CitationAct]:
    acts: list[CitationAct] = []
    for row in citation_rows:
        acts.append(
            CitationAct(
                citation_act_id=str(row.get('citation_act_id') or '').strip(),
                source_move_id=str(row.get('source_move_id') or '').strip() or None,
                target_paper_id=str(row.get('target_paper_id') or '').strip() or None,
                purpose=str(row.get('purpose') or '').strip() or None,
                polarity=str(row.get('polarity') or '').strip() or None,
                semantic_signal=str(row.get('semantic_signal') or '').strip() or None,
                target_scope=str(row.get('target_scope') or '').strip() or None,
                anchor_ids=[str(item).strip() for item in (row.get('anchor_ids') or []) if str(item).strip()],
                confidence=row.get('confidence'),
            )
        )
    return acts


def _compile_figure_refs(rows: list[dict[str, Any]]) -> list[FigureRef]:
    return [
        FigureRef(
            figure_id=str(row.get('figure_id') or '').strip(),
            caption=str(row.get('caption') or '').strip() or None,
            anchor_ids=[str(item).strip() for item in (row.get('anchor_ids') or []) if str(item).strip()],
        )
        for row in rows
    ]


def _compile_table_refs(rows: list[dict[str, Any]]) -> list[TableRef]:
    return [
        TableRef(
            table_id=str(row.get('table_id') or '').strip(),
            caption=str(row.get('caption') or '').strip() or None,
            anchor_ids=[str(item).strip() for item in (row.get('anchor_ids') or []) if str(item).strip()],
        )
        for row in rows
    ]


def _compile_move_relations(rows: list[dict[str, Any]] | None) -> list[MoveRelation]:
    relations: list[MoveRelation] = []
    for row in rows or []:
        relations.append(
            MoveRelation(
                relation_id=str(row.get('relation_id') or '').strip(),
                source_move_id=str(row.get('source_move_id') or '').strip(),
                target_move_id=str(row.get('target_move_id') or '').strip(),
                relation_type=str(row.get('relation_type') or 'motivates'),
                anchor_ids=[str(item).strip() for item in (row.get('anchor_ids') or []) if str(item).strip()],
                confidence=row.get('confidence'),
            )
        )
    return relations


def compile_paper_logic_trace(
    *,
    paper_metadata: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    figure_rows: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    citation_rows: list[dict[str, Any]],
    move_relation_rows: list[dict[str, Any]] | None = None,
    built_at: str | None = None,
) -> PaperLogicTrace:
    metadata = PaperMetadata.model_validate(paper_metadata)
    canonical_core = CanonicalCore(
        evidence_anchors=_compile_anchors(evidence_rows),
        moves=_compile_moves(metadata.paper_id, evidence_rows),
        move_relations=_compile_move_relations(move_relation_rows),
        citation_acts=_compile_citation_acts(citation_rows),
        figure_refs=_compile_figure_refs(figure_rows),
        table_refs=_compile_table_refs(table_rows),
    )
    gate_report = evaluate_hot_path_gate(
        moves=canonical_core.moves,
        anchors=canonical_core.evidence_anchors,
        move_relations=canonical_core.move_relations,
    )
    trace = PaperLogicTrace(
        trace_id=f'{metadata.paper_id}:paper_logic_trace',
        schema_version='v2',
        built_at=built_at or _utc_now_iso(),
        paper_metadata=metadata,
        canonical_core=canonical_core,
        quality=build_quality_payload(gate_report),
    )
    trace.derived_views = build_derived_views(trace)
    return trace
