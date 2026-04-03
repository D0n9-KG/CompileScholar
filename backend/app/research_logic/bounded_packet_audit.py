from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .models import ExclusionReason, RoutePacket


AssemblyRole = Literal['support', 'alternative', 'held_out']
QualityTier = Literal['red', 'yellow', 'green']


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _read_json(path_like: str | Path) -> Any:
    path = _as_path(path_like)
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'bounded packet audit artifact not found: {path}') from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid JSON in bounded packet audit artifact: {path}') from exc


def _normalize_text(value: str | None) -> str:
    return str(value or '').strip()


def _normalize_optional_text(value: str | None) -> str | None:
    normalized = _normalize_text(value)
    return normalized or None


class ContractModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class BoundedPacketRoleMember(ContractModel):
    paper_id: str
    reason: str
    trace_ref: str | None = None
    note: str | None = None

    @field_validator('paper_id', 'reason')
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = _normalize_text(value)
        if not normalized:
            raise ValueError('value must be non-empty')
        return normalized

    @field_validator('trace_ref', 'note')
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value)


class BoundedPacketRoleGroup(ContractModel):
    group_reason: str
    distinctness_rationale: str | None = None
    members: list[BoundedPacketRoleMember] = Field(default_factory=list)

    @field_validator('group_reason')
    @classmethod
    def validate_group_reason(cls, value: str) -> str:
        normalized = _normalize_text(value)
        if not normalized:
            raise ValueError('group_reason must be non-empty')
        return normalized

    @field_validator('distinctness_rationale')
    @classmethod
    def validate_distinctness_rationale(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value)

    @model_validator(mode='after')
    def validate_unique_members(self) -> 'BoundedPacketRoleGroup':
        seen: set[str] = set()
        duplicates: set[str] = set()
        for member in self.members:
            if member.paper_id in seen:
                duplicates.add(member.paper_id)
            seen.add(member.paper_id)
        if duplicates:
            duplicate_text = ', '.join(sorted(duplicates))
            raise ValueError(f'role group contains duplicate paper ids: {duplicate_text}')
        return self

    def paper_ids(self) -> list[str]:
        return [member.paper_id for member in self.members]


class BoundedPacketExclusionNote(ContractModel):
    paper_id: str
    exclusion_reason: ExclusionReason
    note: str
    paper_year: int | None = None
    relative_ref: str | None = None

    @field_validator('paper_id', 'note')
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = _normalize_text(value)
        if not normalized:
            raise ValueError('value must be non-empty')
        return normalized

    @field_validator('relative_ref')
    @classmethod
    def validate_relative_ref(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value)


class BoundedPacketAssemblyManifest(ContractModel):
    packet_id: str
    packet_manifest_ref: str | None = None
    schema_version: str = 'v1'
    built_at: str
    topic_scope: str
    cutoff_year: int
    support: BoundedPacketRoleGroup
    alternative: BoundedPacketRoleGroup
    held_out: BoundedPacketRoleGroup
    exclusion_notes: list[BoundedPacketExclusionNote] = Field(default_factory=list)
    known_gap_notes: list[str] = Field(default_factory=list)

    @field_validator('packet_id', 'built_at', 'topic_scope')
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = _normalize_text(value)
        if not normalized:
            raise ValueError('value must be non-empty')
        return normalized

    @field_validator('packet_manifest_ref')
    @classmethod
    def validate_packet_manifest_ref(cls, value: str | None) -> str | None:
        return _normalize_optional_text(value)

    @field_validator('known_gap_notes', mode='before')
    @classmethod
    def normalize_gap_notes(cls, value: Any) -> Any:
        if value is None:
            return []
        return value

    @field_validator('known_gap_notes')
    @classmethod
    def validate_gap_notes(cls, values: list[str]) -> list[str]:
        notes = [_normalize_text(value) for value in values if _normalize_text(value)]
        return notes

    @model_validator(mode='after')
    def validate_exclusion_notes(self) -> 'BoundedPacketAssemblyManifest':
        if not self.exclusion_notes:
            raise ValueError('BoundedPacketAssemblyManifest requires at least one exclusion note')
        return self

    def role_groups(self) -> dict[AssemblyRole, BoundedPacketRoleGroup]:
        return {
            'support': self.support,
            'alternative': self.alternative,
            'held_out': self.held_out,
        }


class BoundedPacketAuditRoleCounts(ContractModel):
    support: int = 0
    alternative: int = 0
    held_out: int = 0


class BoundedPacketAuditResult(ContractModel):
    packet_id: str
    topic_scope: str
    cutoff_year: int
    packet_included_item_count: int
    packet_excluded_item_count: int
    role_counts: BoundedPacketAuditRoleCounts = Field(default_factory=BoundedPacketAuditRoleCounts)
    quality_tier: QualityTier = 'red'
    ready_for_phase10: bool = False
    quality_flags: list[str] = Field(default_factory=list)
    structural_errors: list[str] = Field(default_factory=list)
    support_paper_ids: list[str] = Field(default_factory=list)
    alternative_paper_ids: list[str] = Field(default_factory=list)
    held_out_paper_ids: list[str] = Field(default_factory=list)
    packet_member_missing_by_role: dict[str, list[str]] = Field(default_factory=dict)
    support_held_out_overlap_paper_ids: list[str] = Field(default_factory=list)
    missing_trace_ref_paper_ids: list[str] = Field(default_factory=list)
    packet_items_missing_trace_id: list[str] = Field(default_factory=list)
    indistinct_alternative_paper_ids: list[str] = Field(default_factory=list)
    missing_exclusion_note_paper_ids: list[str] = Field(default_factory=list)
    exclusion_note_count: int = 0
    known_gap_notes: list[str] = Field(default_factory=list)


class BoundedPacketAuditBundleManifest(ContractModel):
    built_at: str
    packet_id: str
    topic_scope: str
    cutoff_year: int
    route_packet_file: str
    assembly_manifest_file: str
    audit_summary_file: str
    audit_inspection_file: str
    audit_report_file: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _add_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def _packet_item_lookup(packet: RoutePacket) -> dict[str, Any]:
    return {item.paper_id: item for item in packet.included_items}


def load_bounded_packet_assembly_manifest(path_like: str | Path) -> BoundedPacketAssemblyManifest:
    try:
        return BoundedPacketAssemblyManifest.model_validate(_read_json(path_like))
    except ValidationError as exc:
        raise ValueError(f'invalid bounded packet assembly manifest: {path_like}: {exc}') from exc


def audit_bounded_packet_assembly(
    packet: RoutePacket | dict[str, Any],
    manifest: BoundedPacketAssemblyManifest | dict[str, Any],
) -> BoundedPacketAuditResult:
    packet_model = packet if isinstance(packet, RoutePacket) else RoutePacket.model_validate(packet)
    manifest_model = (
        manifest if isinstance(manifest, BoundedPacketAssemblyManifest) else BoundedPacketAssemblyManifest.model_validate(manifest)
    )

    included_by_id = _packet_item_lookup(packet_model)
    role_groups = manifest_model.role_groups()

    support_paper_ids = manifest_model.support.paper_ids()
    alternative_paper_ids = manifest_model.alternative.paper_ids()
    held_out_paper_ids = manifest_model.held_out.paper_ids()

    packet_member_missing_by_role: dict[str, list[str]] = {}
    structural_errors: list[str] = []

    if packet_model.packet_id != manifest_model.packet_id:
        structural_errors.append(
            'packet_id mismatch between RoutePacket and BoundedPacketAssemblyManifest: '
            f'{packet_model.packet_id} != {manifest_model.packet_id}'
        )

    if packet_model.cutoff_year != manifest_model.cutoff_year:
        structural_errors.append(
            'cutoff_year mismatch between RoutePacket and BoundedPacketAssemblyManifest: '
            f'{packet_model.cutoff_year} != {manifest_model.cutoff_year}'
        )

    if _normalize_text(packet_model.topic_scope_candidate).lower() != _normalize_text(manifest_model.topic_scope).lower():
        structural_errors.append('topic_scope mismatch between RoutePacket and BoundedPacketAssemblyManifest')

    for role, group in role_groups.items():
        missing = sorted(member.paper_id for member in group.members if member.paper_id not in included_by_id)
        if missing:
            packet_member_missing_by_role[role] = missing
            structural_errors.append(
                f'{role} members missing from RoutePacket.included_items: {", ".join(missing)}'
            )

    support_overlap = sorted(set(support_paper_ids) & set(held_out_paper_ids))
    if support_overlap:
        structural_errors.append(
            'support and held_out groups cannot reuse paper ids: ' + ', '.join(support_overlap)
        )

    missing_trace_ref_paper_ids = sorted(
        {
            member.paper_id
            for group in role_groups.values()
            for member in group.members
            if not _normalize_text(member.trace_ref)
        }
    )
    packet_items_missing_trace_id = sorted(
        {
            member.paper_id
            for group in role_groups.values()
            for member in group.members
            if member.paper_id in included_by_id and not _normalize_text(included_by_id[member.paper_id].trace_id)
        }
    )

    alternative_has_distinct_signal = bool(_normalize_text(manifest_model.alternative.distinctness_rationale))
    alternative_roles = [
        included_by_id[paper_id].item_role
        for paper_id in alternative_paper_ids
        if paper_id in included_by_id
    ]
    indistinct_alternative_paper_ids = []
    if alternative_paper_ids and not alternative_has_distinct_signal and all(
        role != 'alternative_route' for role in alternative_roles
    ):
        indistinct_alternative_paper_ids = sorted(dict.fromkeys(alternative_paper_ids))

    missing_exclusion_note_paper_ids = sorted(
        {item.paper_id for item in packet_model.excluded_items}
        - {note.paper_id for note in manifest_model.exclusion_notes}
    )

    role_counts = BoundedPacketAuditRoleCounts(
        support=len(support_paper_ids),
        alternative=len(alternative_paper_ids),
        held_out=len(held_out_paper_ids),
    )

    quality_flags: list[str] = []
    if structural_errors:
        _add_flag(quality_flags, 'packet_manifest_alignment_error')
    if packet_member_missing_by_role:
        _add_flag(quality_flags, 'packet_member_refs_missing')
    if role_counts.support < 2:
        _add_flag(quality_flags, 'support_group_too_small')
    if role_counts.alternative < 1:
        _add_flag(quality_flags, 'alternative_group_missing')
    if role_counts.held_out < 1:
        _add_flag(quality_flags, 'held_out_group_missing')
    if support_overlap:
        _add_flag(quality_flags, 'held_out_leakage_detected')
    if missing_trace_ref_paper_ids or packet_items_missing_trace_id:
        _add_flag(quality_flags, 'missing_trace_refs')
    if indistinct_alternative_paper_ids:
        _add_flag(quality_flags, 'alternative_scope_not_distinct')
    if missing_exclusion_note_paper_ids:
        _add_flag(quality_flags, 'exclusion_notes_incomplete')

    blocking_flags = {
        'packet_manifest_alignment_error',
        'packet_member_refs_missing',
        'support_group_too_small',
        'alternative_group_missing',
        'held_out_group_missing',
        'held_out_leakage_detected',
        'missing_trace_refs',
    }
    if structural_errors or any(flag in blocking_flags for flag in quality_flags):
        quality_tier: QualityTier = 'red'
    elif quality_flags:
        quality_tier = 'yellow'
    else:
        quality_tier = 'green'

    return BoundedPacketAuditResult(
        packet_id=packet_model.packet_id,
        topic_scope=packet_model.topic_scope_candidate,
        cutoff_year=packet_model.cutoff_year,
        packet_included_item_count=len(packet_model.included_items),
        packet_excluded_item_count=len(packet_model.excluded_items),
        role_counts=role_counts,
        quality_tier=quality_tier,
        ready_for_phase10=quality_tier == 'green',
        quality_flags=quality_flags,
        structural_errors=structural_errors,
        support_paper_ids=support_paper_ids,
        alternative_paper_ids=alternative_paper_ids,
        held_out_paper_ids=held_out_paper_ids,
        packet_member_missing_by_role=packet_member_missing_by_role,
        support_held_out_overlap_paper_ids=support_overlap,
        missing_trace_ref_paper_ids=missing_trace_ref_paper_ids,
        packet_items_missing_trace_id=packet_items_missing_trace_id,
        indistinct_alternative_paper_ids=indistinct_alternative_paper_ids,
        missing_exclusion_note_paper_ids=missing_exclusion_note_paper_ids,
        exclusion_note_count=len(manifest_model.exclusion_notes),
        known_gap_notes=list(manifest_model.known_gap_notes),
    )


def validate_bounded_packet_assembly(
    packet: RoutePacket | dict[str, Any],
    manifest: BoundedPacketAssemblyManifest | dict[str, Any],
) -> BoundedPacketAuditResult:
    audit = audit_bounded_packet_assembly(packet, manifest)
    if audit.structural_errors:
        raise ValueError('; '.join(audit.structural_errors))
    return audit


def render_bounded_packet_audit_report(
    packet: RoutePacket | dict[str, Any],
    manifest: BoundedPacketAssemblyManifest | dict[str, Any],
    audit: BoundedPacketAuditResult | None = None,
) -> str:
    packet_model = packet if isinstance(packet, RoutePacket) else RoutePacket.model_validate(packet)
    manifest_model = (
        manifest if isinstance(manifest, BoundedPacketAssemblyManifest) else BoundedPacketAssemblyManifest.model_validate(manifest)
    )
    audit_result = audit or audit_bounded_packet_assembly(packet_model, manifest_model)

    lines = [
        f'# Bounded Packet Audit: {packet_model.packet_id}',
        '',
        f'- Topic scope: `{packet_model.topic_scope_candidate}`',
        f'- Cutoff year: `{packet_model.cutoff_year}`',
        f'- Quality tier: `{audit_result.quality_tier}`',
        f'- Ready for Phase 10: `{str(audit_result.ready_for_phase10).lower()}`',
        f'- Role counts: support=`{audit_result.role_counts.support}`, alternative=`{audit_result.role_counts.alternative}`, held_out=`{audit_result.role_counts.held_out}`',
        '',
        '## Quality Flags',
    ]

    if audit_result.quality_flags:
        lines.extend(f'- `{flag}`' for flag in audit_result.quality_flags)
    else:
        lines.append('- None')

    lines.extend(
        [
            '',
            '## Role Mapping',
            f'- Support: {", ".join(audit_result.support_paper_ids) if audit_result.support_paper_ids else "None"}',
            f'- Alternative: {", ".join(audit_result.alternative_paper_ids) if audit_result.alternative_paper_ids else "None"}',
            f'- Held-out: {", ".join(audit_result.held_out_paper_ids) if audit_result.held_out_paper_ids else "None"}',
            '',
            '## Exclusion Notes',
        ]
    )

    for note in manifest_model.exclusion_notes:
        note_text = f'- `{note.paper_id}`: `{note.exclusion_reason}` - {note.note}'
        if note.paper_year is not None:
            note_text += f' (`{note.paper_year}`)'
        lines.append(note_text)

    lines.extend(
        [
            '',
            '## Known Gaps',
        ]
    )
    if audit_result.known_gap_notes:
        lines.extend(f'- {note}' for note in audit_result.known_gap_notes)
    else:
        lines.append('- None')

    return '\n'.join(lines) + '\n'


__all__ = [
    'AssemblyRole',
    'BoundedPacketAssemblyManifest',
    'BoundedPacketAuditBundleManifest',
    'BoundedPacketAuditResult',
    'BoundedPacketAuditRoleCounts',
    'BoundedPacketExclusionNote',
    'BoundedPacketRoleGroup',
    'BoundedPacketRoleMember',
    'QualityTier',
    'audit_bounded_packet_assembly',
    'load_bounded_packet_assembly_manifest',
    'render_bounded_packet_audit_report',
    'validate_bounded_packet_assembly',
]
