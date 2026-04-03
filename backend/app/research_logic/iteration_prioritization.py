from __future__ import annotations

from ast import literal_eval
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from pydantic import Field

from .models import ContractModel


class IterationPriorityPreflightError(RuntimeError):
    """Raised when required upstream prioritization evidence is unavailable."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _string_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == 'none':
        return []
    return [item.strip() for item in text.split(',') if item.strip()]


def _int_map(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, raw_value in value.items():
        try:
            result[str(key)] = int(raw_value)
        except (TypeError, ValueError):
            continue
    return result


def _json_object(value: object, *, label: str) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    raise ValueError(f'{label} must be a JSON object')


def _load_json(path_like: str | Path, *, label: str) -> dict[str, object]:
    path = _as_path(path_like)
    try:
        return _json_object(json.loads(path.read_text(encoding='utf-8')), label=label)
    except FileNotFoundError as exc:
        raise IterationPriorityPreflightError(f'{label} not found: {path}') from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid JSON in {label}: {path}') from exc


def _read_text(path_like: str | Path, *, label: str) -> str:
    path = _as_path(path_like)
    try:
        return path.read_text(encoding='utf-8')
    except FileNotFoundError as exc:
        raise IterationPriorityPreflightError(f'{label} not found: {path}') from exc


def _maybe_int(value: object) -> int | None:
    if value is None or value == '':
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_scalar(raw_value: str | None) -> object:
    if raw_value is None:
        return None
    text = str(raw_value).strip().strip('`').strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    if lowered == 'null':
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return literal_eval(text)
    except (SyntaxError, ValueError):
        pass
    if re.fullmatch(r'-?\d+', text):
        return int(text)
    return text


def _extract_backticks(line: str) -> list[str]:
    return re.findall(r'`([^`]+)`', line)


def _extract_report_section(report_text: str, heading: str) -> str:
    match = re.search(
        rf'^## {re.escape(heading)}\r?\n(?P<body>.*?)(?=^## |\Z)',
        report_text,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        return ''
    return match.group('body').strip()


def _extract_markdown_subsections(section_text: str) -> dict[str, str]:
    subsections: dict[str, str] = {}
    for match in re.finditer(r'^### (?P<title>.+?)\r?\n(?P<body>.*?)(?=^### |\Z)', section_text, re.MULTILINE | re.DOTALL):
        title = match.group('title').strip().lower().replace(' ', '_')
        subsections[title] = match.group('body').strip()
    return subsections


def _extract_line_value(section_text: str, label: str) -> str | None:
    prefix = f'- {label}:'
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def _parse_current_vs_baseline(value: str | None) -> tuple[object | None, object | None]:
    if value is None:
        return None, None
    backticks = _extract_backticks(value)
    if len(backticks) >= 2:
        return _parse_scalar(backticks[0]), _parse_scalar(backticks[1])
    parts = value.split(' vs baseline ', maxsplit=1)
    if len(parts) == 2:
        return _parse_scalar(parts[0]), _parse_scalar(parts[1])
    parsed = _parse_scalar(value)
    return parsed, None


def _parse_quality_flags(value: str | None) -> list[str]:
    if value is None:
        return []
    backticks = _extract_backticks(value)
    if backticks:
        return _string_list(backticks[0])
    return _string_list(value)


def _parse_backtick_literal(value: str | None) -> object:
    if value is None:
        return None
    backticks = _extract_backticks(value)
    if backticks:
        return _parse_scalar(backticks[0])
    return _parse_scalar(value)


def _extract_assignment(text: str, key: str) -> object:
    match = re.search(rf'- `{re.escape(key)} = (?P<value>.+?)`', text)
    return _parse_scalar(match.group('value')) if match else None


def _normalize_recommendation_id(value: object) -> str | None:
    text = str(value or '').strip().lower()
    if not text:
        return None
    aliases = {
        'packet construction': 'packet_construction',
        'packet_construction': 'packet_construction',
        'l4': 'l4_aggregation',
        'l4 aggregation': 'l4_aggregation',
        'l4_aggregation': 'l4_aggregation',
        'l2': 'l2_extraction',
        'l2 extraction': 'l2_extraction',
        'l2_extraction': 'l2_extraction',
    }
    if text in aliases:
        return aliases[text]
    return re.sub(r'[^a-z0-9]+', '_', text).strip('_') or None


def _extract_current_recommendation(verification_text: str) -> str | None:
    strongest_match = re.search(
        r'strongest next-cycle blockers point to `(?P<choice>[^`]+)`',
        verification_text,
        re.IGNORECASE,
    )
    if strongest_match:
        return _normalize_recommendation_id(strongest_match.group('choice'))
    recommended_match = re.search(
        r'prioritize .*?`?(?P<choice>packet construction|L4|L2)`?',
        verification_text,
        re.IGNORECASE,
    )
    if recommended_match:
        return _normalize_recommendation_id(recommended_match.group('choice'))
    return None


def _parse_blocker_line(line: str) -> dict[str, object] | None:
    match = re.match(
        r'^- `(?P<code>[^`]+)`: (?P<message>.*?)(?: \(current=`(?P<current>.*?)`, baseline=`(?P<baseline>.*?)`, '
        r'vs_baseline=`(?P<vs_baseline>.*?)`\))?$',
        line.strip(),
    )
    if not match:
        return None
    payload: dict[str, object] = {
        'code': match.group('code'),
        'message': match.group('message').strip(),
    }
    current = _parse_scalar(match.group('current'))
    baseline = _parse_scalar(match.group('baseline'))
    vs_baseline = match.group('vs_baseline')
    if current is not None:
        payload['current'] = current
    if baseline is not None:
        payload['baseline'] = baseline
    if vs_baseline:
        payload['vs_baseline'] = vs_baseline
    return payload


def _stage_key(title: str) -> str:
    return title.strip().lower().replace(' ', '_')


class IterationPrioritySourceRefs(ContractModel):
    phase8_summary_path: str | None = None
    phase8_inspection_path: str | None = None
    phase10_summary_path: str | None = None
    phase10_verification_path: str | None = None
    phase10_report_path: str | None = None
    phase10_mode: str = 'json'
    fallback_used: bool = False


class IterationPriorityOwnerBucket(ContractModel):
    bucket: str
    priority: int = 0
    fixed_count: int = 0
    random_count: int = 0
    total_count: int = 0
    exemplar_corpus_paper_ids: list[str] = Field(default_factory=list)
    verdict_counts: dict[str, int] = Field(default_factory=dict)


class IterationPriorityBlocker(ContractModel):
    code: str
    message: str
    current: bool | int | str | None = None
    baseline: bool | int | str | None = None
    vs_baseline: str | None = None


class LoadedPhase8ComparisonSummary(ContractModel):
    schema_version: str = 'v1'
    built_at: str | None = None
    iteration_label: str
    previous_iteration_label: str | None = None
    baseline_only: bool = False
    fixed_verdict_counts: dict[str, int] = Field(default_factory=dict)
    random_verdict_counts: dict[str, int] = Field(default_factory=dict)
    owner_buckets: list[IterationPriorityOwnerBucket] = Field(default_factory=list)
    source_ref: str


class LoadedPhase8ComparisonInspection(ContractModel):
    schema_version: str = 'v1'
    built_at: str | None = None
    iteration_label: str
    previous_iteration_label: str | None = None
    baseline_only: bool = False
    fixed_comparisons: list[dict[str, Any]] = Field(default_factory=list)
    random_comparisons: list[dict[str, Any]] = Field(default_factory=list)
    owner_buckets: list[IterationPriorityOwnerBucket] = Field(default_factory=list)
    source_ref: str


class LoadedPhase10ComparisonSurface(ContractModel):
    packet_id: str | None = None
    cutoff_year: int | None = None
    current_recommendation: str | None = None
    source_refs: IterationPrioritySourceRefs
    package: dict[str, Any] = Field(default_factory=dict)
    replay: dict[str, Any] = Field(default_factory=dict)
    prior_review: dict[str, Any] = Field(default_factory=dict)
    export: dict[str, Any] = Field(default_factory=dict)
    blocker_queue: dict[str, list[IterationPriorityBlocker]] = Field(default_factory=dict)
    source_artifacts: dict[str, str] = Field(default_factory=dict)
    notes: dict[str, Any] = Field(default_factory=dict)


class IterationPriorityRecommendation(ContractModel):
    id: str
    rank: int = 0
    title: str
    why_now: str
    score: int = 0
    supporting_owner_buckets: list[str] = Field(default_factory=list)
    supporting_blocker_stages: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class IterationPrioritySummary(ContractModel):
    schema_version: str = 'v1'
    built_at: str = Field(default_factory=_utc_now_iso)
    source_refs: IterationPrioritySourceRefs
    phase8_iteration_label: str | None = None
    previous_iteration_label: str | None = None
    baseline_only: bool = False
    packet_id: str | None = None
    cutoff_year: int | None = None
    current_recommendation: str | None = None
    primary_recommendation_id: str | None = None
    phase8_fixed_verdict_counts: dict[str, int] = Field(default_factory=dict)
    phase8_random_verdict_counts: dict[str, int] = Field(default_factory=dict)
    phase8_owner_buckets: list[IterationPriorityOwnerBucket] = Field(default_factory=list)
    supporting_l2_evidence: list[IterationPriorityOwnerBucket] = Field(default_factory=list)
    phase10_stage_surfaces: dict[str, dict[str, Any]] = Field(default_factory=dict)
    phase10_blocker_queue: dict[str, list[IterationPriorityBlocker]] = Field(default_factory=dict)
    recommendations: list[IterationPriorityRecommendation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class IterationPriorityInspection(ContractModel):
    schema_version: str = 'v1'
    built_at: str = Field(default_factory=_utc_now_iso)
    source_refs: IterationPrioritySourceRefs
    phase8_summary: dict[str, Any] = Field(default_factory=dict)
    phase8_inspection: dict[str, Any] = Field(default_factory=dict)
    phase10_surface: dict[str, Any] = Field(default_factory=dict)
    ranking_signals: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[IterationPriorityRecommendation] = Field(default_factory=list)


def _summary_model(value: IterationPrioritySummary | dict[str, Any]) -> IterationPrioritySummary:
    if isinstance(value, IterationPrioritySummary):
        return value
    return IterationPrioritySummary.model_validate(value)


def _inspection_model(value: IterationPriorityInspection | dict[str, Any]) -> IterationPriorityInspection:
    if isinstance(value, IterationPriorityInspection):
        return value
    return IterationPriorityInspection.model_validate(value)


def _load_owner_buckets(raw_buckets: object) -> list[IterationPriorityOwnerBucket]:
    if not isinstance(raw_buckets, list):
        return []
    buckets: list[IterationPriorityOwnerBucket] = []
    for item in raw_buckets:
        bucket_payload = _json_object(item, label='Phase 8 owner bucket')
        buckets.append(
            IterationPriorityOwnerBucket(
                bucket=str(bucket_payload.get('bucket') or ''),
                priority=int(bucket_payload.get('priority') or 0),
                fixed_count=int(bucket_payload.get('fixed_count') or 0),
                random_count=int(bucket_payload.get('random_count') or 0),
                total_count=int(bucket_payload.get('total_count') or 0),
                exemplar_corpus_paper_ids=_string_list(bucket_payload.get('exemplar_corpus_paper_ids')),
                verdict_counts=_int_map(bucket_payload.get('verdict_counts')),
            )
        )
    return buckets


def _load_blocker_queue(raw_blocker_queue: object) -> dict[str, list[IterationPriorityBlocker]]:
    blocker_queue = _json_object(raw_blocker_queue or {}, label='Phase 10 blocker queue')
    loaded: dict[str, list[IterationPriorityBlocker]] = {}
    for stage_name, blockers in blocker_queue.items():
        if not isinstance(blockers, list):
            loaded[str(stage_name)] = []
            continue
        loaded[str(stage_name)] = [
            IterationPriorityBlocker(**_json_object(blocker, label=f'Phase 10 blocker for {stage_name}'))
            for blocker in blockers
        ]
    return loaded


def load_phase8_comparison_summary(path: str | Path) -> LoadedPhase8ComparisonSummary:
    resolved_path = _as_path(path).resolve()
    payload = _load_json(resolved_path, label='Phase 8 comparison summary')
    return LoadedPhase8ComparisonSummary(
        schema_version=str(payload.get('schema_version') or 'v1'),
        built_at=str(payload.get('built_at')) if payload.get('built_at') else None,
        iteration_label=str(payload.get('iteration_label') or ''),
        previous_iteration_label=str(payload.get('previous_iteration_label')) if payload.get('previous_iteration_label') else None,
        baseline_only=bool(payload.get('baseline_only')),
        fixed_verdict_counts=_int_map(payload.get('fixed_verdict_counts')),
        random_verdict_counts=_int_map(payload.get('random_verdict_counts')),
        owner_buckets=_load_owner_buckets(payload.get('owner_buckets')),
        source_ref=str(resolved_path),
    )


def load_phase8_comparison_inspection(path: str | Path) -> LoadedPhase8ComparisonInspection:
    resolved_path = _as_path(path).resolve()
    payload = _load_json(resolved_path, label='Phase 8 comparison inspection')
    return LoadedPhase8ComparisonInspection(
        schema_version=str(payload.get('schema_version') or 'v1'),
        built_at=str(payload.get('built_at')) if payload.get('built_at') else None,
        iteration_label=str(payload.get('iteration_label') or ''),
        previous_iteration_label=str(payload.get('previous_iteration_label')) if payload.get('previous_iteration_label') else None,
        baseline_only=bool(payload.get('baseline_only')),
        fixed_comparisons=list(payload.get('fixed_comparisons') or []),
        random_comparisons=list(payload.get('random_comparisons') or []),
        owner_buckets=_load_owner_buckets(payload.get('owner_buckets')),
        source_ref=str(resolved_path),
    )


def load_phase10_comparison_summary(path: str | Path) -> LoadedPhase10ComparisonSurface:
    resolved_path = _as_path(path).resolve()
    payload = _load_json(resolved_path, label='Phase 10 comparison summary')
    return LoadedPhase10ComparisonSurface(
        packet_id=str(payload.get('packet_id')) if payload.get('packet_id') else None,
        cutoff_year=_maybe_int(payload.get('cutoff_year')),
        current_recommendation=_normalize_recommendation_id(payload.get('current_recommendation')),
        source_refs=IterationPrioritySourceRefs(
            phase10_summary_path=str(resolved_path),
            phase10_mode='json',
            fallback_used=False,
        ),
        package=_json_object(payload.get('package') or {}, label='Phase 10 package'),
        replay=_json_object(payload.get('replay') or {}, label='Phase 10 replay'),
        prior_review=_json_object(payload.get('prior_review') or {}, label='Phase 10 prior review'),
        export=_json_object(payload.get('export') or {}, label='Phase 10 export'),
        blocker_queue=_load_blocker_queue(payload.get('blocker_queue')),
        source_artifacts={str(key): str(value) for key, value in _json_object(payload.get('source_artifacts') or {}, label='Phase 10 source artifacts').items()},
        notes=_json_object(payload.get('notes') or {}, label='Phase 10 notes'),
    )


def build_phase10_fallback_surface(
    verification_path: str | Path,
    report_path: str | Path,
) -> LoadedPhase10ComparisonSurface:
    verification_file = _as_path(verification_path).resolve()
    report_file = _as_path(report_path).resolve()
    verification_text = _read_text(verification_file, label='Phase 10 verification note')
    report_text = _read_text(report_file, label='Phase 10 validation report')

    report_header = {
        'packet_id': _parse_backtick_literal(_extract_line_value(report_text, 'Packet id')),
        'cutoff_year': _parse_backtick_literal(_extract_line_value(report_text, 'Cutoff year')),
        'baseline_replay_bundle': _parse_backtick_literal(_extract_line_value(report_text, 'Baseline replay bundle')),
        'baseline_export_bundle': _parse_backtick_literal(_extract_line_value(report_text, 'Baseline export bundle')),
    }

    stage_sections = _extract_markdown_subsections(_extract_report_section(report_text, 'Stage Comparison'))
    blocker_sections = _extract_markdown_subsections(_extract_report_section(report_text, 'Blocker Queue'))

    package_section = stage_sections.get('package', '')
    replay_section = stage_sections.get('replay', '')
    prior_section = stage_sections.get('prior_review', '')
    export_section = stage_sections.get('export', '')

    package_quality, package_baseline_quality = _parse_current_vs_baseline(
        _extract_line_value(package_section, 'Current quality')
    )
    package_ready, package_baseline_ready = _parse_current_vs_baseline(
        _extract_line_value(package_section, 'Ready for replay')
    )
    replay_quality, replay_baseline_quality = _parse_current_vs_baseline(
        _extract_line_value(replay_section, 'Current quality')
    )
    replay_ready, replay_baseline_ready = _parse_current_vs_baseline(
        _extract_line_value(replay_section, 'Ready for pilot')
    )
    export_quality, export_baseline_quality = _parse_current_vs_baseline(
        _extract_line_value(export_section, 'Current quality')
    )
    export_training_ready, export_training_baseline_ready = _parse_current_vs_baseline(
        _extract_line_value(export_section, 'Ready for training')
    )
    export_eval_ready, export_eval_baseline_ready = _parse_current_vs_baseline(
        _extract_line_value(export_section, 'Ready for eval')
    )

    package = {
        'current': {
            'quality_tier': package_quality,
            'ready_for_replay': package_ready,
            'quality_flags': _parse_quality_flags(_extract_line_value(package_section, 'Quality flags')),
            'role_counts': _int_map(_parse_backtick_literal(_extract_line_value(package_section, 'Role counts'))),
        },
        'baseline': {
            'quality_tier': package_baseline_quality,
            'ready_for_replay': package_baseline_ready,
        },
    }
    replay = {
        'current': {
            'quality_tier': replay_quality,
            'ready_for_pilot': replay_ready,
            'quality_flags': _parse_quality_flags(_extract_line_value(replay_section, 'Quality flags')),
            'failure_record_count': _parse_backtick_literal(_extract_line_value(replay_section, 'Failure record count')),
            'failure_counts_by_stage': _int_map(_parse_backtick_literal(_extract_line_value(replay_section, 'Failure counts by stage'))),
            'failure_counts_by_layer': _int_map(_extract_assignment(verification_text, 'replay.current.failure_counts_by_layer')),
        },
        'baseline': {
            'quality_tier': replay_baseline_quality,
            'ready_for_pilot': replay_baseline_ready,
        },
        'delta': {
            'failure_counts_by_layer_delta': _int_map(
                _extract_assignment(verification_text, 'replay.delta.failure_counts_by_layer_delta')
            ),
        },
    }
    prior_review = {
        'current': {
            'prior_candidate_count': _extract_assignment(verification_text, 'prior_review.current.prior_candidate_count'),
            'anti_pattern_candidate_count': _extract_assignment(verification_text, 'prior_review.current.anti_pattern_candidate_count'),
            'accepted_prior_ids': _extract_assignment(verification_text, 'prior_review.current.accepted_prior_ids') or [],
            'accepted_anti_pattern_ids': _extract_assignment(verification_text, 'prior_review.current.accepted_anti_pattern_ids') or [],
            'quality_flag_counts': _int_map(_parse_backtick_literal(_extract_line_value(prior_section, 'Quality flag counts'))),
        },
    }
    export = {
        'current': {
            'quality_tier': export_quality,
            'ready_for_training': export_training_ready,
            'ready_for_eval': export_eval_ready,
            'quality_flags': _parse_quality_flags(_extract_line_value(export_section, 'Quality flags')),
            'selected_antipattern_count': _extract_assignment(verification_text, 'export.current.selected_antipattern_count'),
            'visibility_bucket_counts': _int_map(_parse_backtick_literal(_extract_line_value(export_section, 'Visibility buckets'))),
        },
        'baseline': {
            'quality_tier': export_baseline_quality,
            'ready_for_training': export_training_baseline_ready,
            'ready_for_eval': export_eval_baseline_ready,
            'selected_antipattern_count': _extract_assignment(verification_text, 'export.baseline.selected_antipattern_count'),
        },
    }

    blocker_queue: dict[str, list[IterationPriorityBlocker]] = {}
    for title, body in blocker_sections.items():
        stage_name = _stage_key(title)
        blockers: list[IterationPriorityBlocker] = []
        for raw_line in body.splitlines():
            blocker_payload = _parse_blocker_line(raw_line.strip())
            if blocker_payload is None:
                continue
            blockers.append(IterationPriorityBlocker(**blocker_payload))
        blocker_queue[stage_name] = blockers

    current_recommendation = _extract_current_recommendation(verification_text)
    return LoadedPhase10ComparisonSurface(
        packet_id=str(report_header['packet_id']) if report_header['packet_id'] else None,
        cutoff_year=_maybe_int(report_header['cutoff_year']),
        current_recommendation=current_recommendation,
        source_refs=IterationPrioritySourceRefs(
            phase10_verification_path=str(verification_file),
            phase10_report_path=str(report_file),
            phase10_mode='fallback',
            fallback_used=True,
        ),
        package=package,
        replay=replay,
        prior_review=prior_review,
        export=export,
        blocker_queue=blocker_queue,
        source_artifacts={
            'phase10_verification': str(verification_file),
            'phase10_report': str(report_file),
            'baseline_replay_bundle': str(report_header['baseline_replay_bundle'] or ''),
            'baseline_export_bundle': str(report_header['baseline_export_bundle'] or ''),
        },
        notes={
            'fallback_reason': 'Phase 10 comparison summary JSON unavailable; parsed verification and report markdown instead.',
            'source_of_truth': 'Phase 10 verification note and report',
        },
    )


def load_phase10_evidence(
    *,
    summary_path: str | Path | None = None,
    verification_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> LoadedPhase10ComparisonSurface:
    if summary_path is not None and _as_path(summary_path).exists():
        return load_phase10_comparison_summary(summary_path)
    if verification_path is not None and report_path is not None:
        verification_file = _as_path(verification_path)
        report_file = _as_path(report_path)
        if verification_file.exists() and report_file.exists():
            return build_phase10_fallback_surface(verification_file, report_file)

    missing_inputs: list[str] = []
    if summary_path is None:
        missing_inputs.append('phase10 comparison_summary.json path')
    elif not _as_path(summary_path).exists():
        missing_inputs.append(f'missing Phase 10 comparison summary: {_as_path(summary_path)}')

    if verification_path is None:
        missing_inputs.append('phase10 verification markdown path')
    elif not _as_path(verification_path).exists():
        missing_inputs.append(f'missing Phase 10 verification note: {_as_path(verification_path)}')

    if report_path is None:
        missing_inputs.append('phase10 report markdown path')
    elif not _as_path(report_path).exists():
        missing_inputs.append(f'missing Phase 10 validation report: {_as_path(report_path)}')

    raise IterationPriorityPreflightError(
        'Phase 10 evidence unavailable; provide an explicit comparison_summary.json path or both '
        f'fallback markdown files. Missing inputs: {", ".join(missing_inputs)}'
    )


def _recommendation_title(recommendation_id: str) -> str:
    titles = {
        'packet_construction': 'Deepen bounded packet construction',
        'l4_aggregation': 'Tune downstream L4 aggregation',
        'l2_extraction': 'Run a targeted L2 extraction pass',
    }
    return titles.get(recommendation_id, recommendation_id.replace('_', ' ').title())


def _nested_int(mapping: dict[str, Any], *keys: str) -> int:
    current: object = mapping
    for key in keys:
        if not isinstance(current, dict):
            return 0
        current = current.get(key)
    try:
        return int(current)
    except (TypeError, ValueError):
        return 0


def _new_or_higher_blockers(blockers: list[IterationPriorityBlocker]) -> list[IterationPriorityBlocker]:
    return [blocker for blocker in blockers if blocker.vs_baseline in {'new', 'higher', 'regressed'}]


def _blocker_codes(blockers: list[IterationPriorityBlocker], *, limit: int = 3) -> list[str]:
    return [blocker.code for blocker in blockers[:limit]]


def _supporting_owner_buckets(phase8_summary: LoadedPhase8ComparisonSummary) -> list[IterationPriorityOwnerBucket]:
    buckets = [bucket for bucket in phase8_summary.owner_buckets if bucket.total_count > 0]
    return buckets[:2]


def _recommendation(
    *,
    recommendation_id: str,
    score: int,
    why_now: str,
    supporting_owner_buckets: list[str],
    supporting_blocker_stages: list[str],
    evidence: list[str],
) -> IterationPriorityRecommendation:
    return IterationPriorityRecommendation(
        id=recommendation_id,
        title=_recommendation_title(recommendation_id),
        why_now=why_now,
        score=score,
        supporting_owner_buckets=supporting_owner_buckets,
        supporting_blocker_stages=supporting_blocker_stages,
        evidence=evidence,
    )


def rank_iteration_recommendations(
    *,
    phase8_summary: LoadedPhase8ComparisonSummary,
    phase10_surface: LoadedPhase10ComparisonSurface,
) -> list[IterationPriorityRecommendation]:
    supporting_owner_buckets = _supporting_owner_buckets(phase8_summary)
    supporting_owner_bucket_names = [bucket.bucket for bucket in supporting_owner_buckets]

    package_blockers = phase10_surface.blocker_queue.get('package_validation', [])
    replay_blockers = phase10_surface.blocker_queue.get('replay', [])
    prior_blockers = phase10_surface.blocker_queue.get('prior_induction', [])

    replay_l2_delta = _nested_int(phase10_surface.replay, 'delta', 'failure_counts_by_layer_delta', 'l2')
    replay_l3_l4_delta = _nested_int(phase10_surface.replay, 'delta', 'failure_counts_by_layer_delta', 'l3_l4')
    decision_prior_regressions = [
        blocker for blocker in replay_blockers
        if blocker.code == 'failure_stage:decision_prior_card' and blocker.vs_baseline in {'higher', 'new'}
    ]

    new_package_blockers = _new_or_higher_blockers(package_blockers)
    packet_score = 100
    if new_package_blockers:
        packet_score += 50
    if replay_l2_delta == 0:
        packet_score += 20
    packet_score += len(package_blockers) * 3
    if phase10_surface.current_recommendation == 'packet_construction':
        packet_score += 10

    l4_score = 70
    if replay_l3_l4_delta > 0:
        l4_score += 25
    if decision_prior_regressions:
        l4_score += 20
    if new_package_blockers:
        l4_score -= 10
    if phase10_surface.current_recommendation == 'l4_aggregation':
        l4_score += 10

    l2_score = 60 + sum(bucket.total_count for bucket in supporting_owner_buckets)
    if replay_l2_delta > 0:
        l2_score += 25
    if replay_l2_delta == 0:
        l2_score -= 10
    if phase10_surface.current_recommendation == 'l2_extraction':
        l2_score += 10

    recommendations = [
        _recommendation(
            recommendation_id='packet_construction',
            score=packet_score,
            why_now='New package-validation blockers are the first fresh regressions while replay L2 delta stays flat.',
            supporting_owner_buckets=supporting_owner_bucket_names,
            supporting_blocker_stages=['package_validation', 'replay'],
            evidence=[
                f'package blockers: {", ".join(_blocker_codes(package_blockers)) or "none"}',
                f'replay l2 delta: {replay_l2_delta}',
                f'phase10 recommendation: {phase10_surface.current_recommendation or "not stated"}',
            ],
        ),
        _recommendation(
            recommendation_id='l4_aggregation',
            score=l4_score,
            why_now='L4 follow-up stays next because replay-level decision_prior_card failures and L3/L4 delta still rise downstream of packet issues.',
            supporting_owner_buckets=supporting_owner_bucket_names,
            supporting_blocker_stages=['replay', 'prior_induction'],
            evidence=[
                f'replay l3_l4 delta: {replay_l3_l4_delta}',
                f'decision_prior_card regressions: {len(decision_prior_regressions)}',
                f'prior blockers: {", ".join(_blocker_codes(prior_blockers)) or "none"}',
            ],
        ),
        _recommendation(
            recommendation_id='l2_extraction',
            score=l2_score,
            why_now='Phase 8 still shows recurring L2 work, but it is supporting evidence behind the newer packet-first regression surface.',
            supporting_owner_buckets=supporting_owner_bucket_names,
            supporting_blocker_stages=['phase8_owner_queue'],
            evidence=[
                f'phase8 lead owners: {", ".join(supporting_owner_bucket_names) or "none"}',
                f'fixed recurring failures: {phase8_summary.fixed_verdict_counts.get("recurring_failure", 0)}',
                f'replay l2 delta: {replay_l2_delta}',
            ],
        ),
    ]

    ranked = sorted(recommendations, key=lambda recommendation: recommendation.score, reverse=True)
    for index, recommendation in enumerate(ranked, start=1):
        recommendation.rank = index
    return ranked


def _combine_source_refs(
    phase8_summary: LoadedPhase8ComparisonSummary,
    phase8_inspection: LoadedPhase8ComparisonInspection,
    phase10_surface: LoadedPhase10ComparisonSurface,
) -> IterationPrioritySourceRefs:
    return IterationPrioritySourceRefs(
        phase8_summary_path=phase8_summary.source_ref,
        phase8_inspection_path=phase8_inspection.source_ref,
        phase10_summary_path=phase10_surface.source_refs.phase10_summary_path,
        phase10_verification_path=phase10_surface.source_refs.phase10_verification_path,
        phase10_report_path=phase10_surface.source_refs.phase10_report_path,
        phase10_mode=phase10_surface.source_refs.phase10_mode,
        fallback_used=phase10_surface.source_refs.fallback_used,
    )


def _ranking_signals(
    *,
    phase8_summary: LoadedPhase8ComparisonSummary,
    phase10_surface: LoadedPhase10ComparisonSurface,
) -> dict[str, Any]:
    supporting_owner_buckets = [bucket.bucket for bucket in _supporting_owner_buckets(phase8_summary)]
    return {
        'phase10_current_recommendation': phase10_surface.current_recommendation,
        'package_validation_new_blockers': _blocker_codes(
            _new_or_higher_blockers(phase10_surface.blocker_queue.get('package_validation', []))
        ),
        'replay_l2_delta': _nested_int(phase10_surface.replay, 'delta', 'failure_counts_by_layer_delta', 'l2'),
        'replay_l3_l4_delta': _nested_int(phase10_surface.replay, 'delta', 'failure_counts_by_layer_delta', 'l3_l4'),
        'supporting_owner_buckets': supporting_owner_buckets,
    }


def build_iteration_priority_summary(
    *,
    phase8_summary: LoadedPhase8ComparisonSummary,
    phase8_inspection: LoadedPhase8ComparisonInspection,
    phase10_surface: LoadedPhase10ComparisonSurface,
) -> IterationPrioritySummary:
    recommendations = rank_iteration_recommendations(
        phase8_summary=phase8_summary,
        phase10_surface=phase10_surface,
    )
    supporting_l2 = _supporting_owner_buckets(phase8_summary) or phase8_inspection.owner_buckets[:2]
    notes: list[str] = []
    if phase10_surface.source_refs.fallback_used:
        notes.append('Phase 10 comparison summary JSON was unavailable, so prioritization used the verification and report markdown fallback.')
    if phase10_surface.current_recommendation:
        notes.append(f'Phase 10 already pointed to {phase10_surface.current_recommendation} as the current next-cycle lead.')

    return IterationPrioritySummary(
        source_refs=_combine_source_refs(phase8_summary, phase8_inspection, phase10_surface),
        phase8_iteration_label=phase8_summary.iteration_label,
        previous_iteration_label=phase8_summary.previous_iteration_label,
        baseline_only=phase8_summary.baseline_only,
        packet_id=phase10_surface.packet_id,
        cutoff_year=phase10_surface.cutoff_year,
        current_recommendation=phase10_surface.current_recommendation,
        primary_recommendation_id=recommendations[0].id if recommendations else None,
        phase8_fixed_verdict_counts=phase8_summary.fixed_verdict_counts,
        phase8_random_verdict_counts=phase8_summary.random_verdict_counts,
        phase8_owner_buckets=phase8_summary.owner_buckets,
        supporting_l2_evidence=supporting_l2,
        phase10_stage_surfaces={
            'package': phase10_surface.package,
            'replay': phase10_surface.replay,
            'prior_review': phase10_surface.prior_review,
            'export': phase10_surface.export,
        },
        phase10_blocker_queue=phase10_surface.blocker_queue,
        recommendations=recommendations,
        notes=notes,
    )


def build_iteration_priority_inspection(
    *,
    phase8_summary: LoadedPhase8ComparisonSummary,
    phase8_inspection: LoadedPhase8ComparisonInspection,
    phase10_surface: LoadedPhase10ComparisonSurface,
) -> IterationPriorityInspection:
    recommendations = rank_iteration_recommendations(
        phase8_summary=phase8_summary,
        phase10_surface=phase10_surface,
    )
    return IterationPriorityInspection(
        source_refs=_combine_source_refs(phase8_summary, phase8_inspection, phase10_surface),
        phase8_summary=phase8_summary.model_dump(mode='json', exclude_none=True),
        phase8_inspection=phase8_inspection.model_dump(mode='json', exclude_none=True),
        phase10_surface=phase10_surface.model_dump(mode='json', exclude_none=True),
        ranking_signals=_ranking_signals(phase8_summary=phase8_summary, phase10_surface=phase10_surface),
        recommendations=recommendations,
    )


def render_iteration_priority_report(
    summary: IterationPrioritySummary | dict[str, Any],
    inspection: IterationPriorityInspection | dict[str, Any],
) -> str:
    summary_model = _summary_model(summary)
    inspection_model = _inspection_model(inspection)
    source_refs = summary_model.source_refs
    recommendations = sorted(summary_model.recommendations, key=lambda item: (item.rank or 999, item.id))

    supporting_buckets = summary_model.supporting_l2_evidence or summary_model.phase8_owner_buckets[:2]
    supporting_bucket_lines = [
        (
            f"- `{bucket.bucket}`: fixed={bucket.fixed_count}, random={bucket.random_count}, "
            f"total={bucket.total_count or bucket.fixed_count + bucket.random_count}"
        )
        for bucket in supporting_buckets
    ]
    if not supporting_bucket_lines:
        supporting_bucket_lines = ['- No Phase 8 owner buckets were preserved in the summary payload.']

    recommendation_lines: list[str] = []
    for recommendation in recommendations:
        evidence_text = '; '.join(recommendation.evidence) if recommendation.evidence else 'No supporting evidence recorded.'
        blocker_text = ', '.join(recommendation.supporting_blocker_stages) or 'none'
        owner_text = ', '.join(recommendation.supporting_owner_buckets) or 'none'
        recommendation_lines.extend(
            [
                (
                    f"{recommendation.rank}. `{recommendation.id}` - {recommendation.title} "
                    f"(score={recommendation.score})"
                ),
                f"   Why now: {recommendation.why_now}",
                f"   Supporting blocker stages: {blocker_text}",
                f"   Supporting owner buckets: {owner_text}",
                f"   Evidence: {evidence_text}",
            ]
        )
    if not recommendation_lines:
        recommendation_lines = ['1. No recommendations were preserved in the Phase 11 summary payload.']

    ranking_signals = inspection_model.ranking_signals
    fallback_lines = [f"- Phase 10 JSON unavailable: `{'yes' if source_refs.fallback_used else 'no'}`"]
    if source_refs.fallback_used:
        fallback_lines.extend(
            [
                f"- Phase 10 verification fallback: `{source_refs.phase10_verification_path or 'missing'}`",
                f"- Phase 10 report fallback: `{source_refs.phase10_report_path or 'missing'}`",
            ]
        )
    else:
        fallback_lines.append('- Fallback file paths: not used.')
        if source_refs.phase10_verification_path:
            fallback_lines.append(
                f"- Supplemental Phase 10 verification provenance: `{source_refs.phase10_verification_path}`"
            )
        if source_refs.phase10_report_path:
            fallback_lines.append(f"- Supplemental Phase 10 report provenance: `{source_refs.phase10_report_path}`")

    source_of_truth_lines = [
        '- This markdown was rendered from the Phase 11 summary and inspection payloads, not from ad-hoc ranking logic.',
        f"- Primary recommendation in summary payload: `{summary_model.primary_recommendation_id or 'none'}`",
        f"- Phase 8 summary source: `{source_refs.phase8_summary_path or 'missing'}`",
        f"- Phase 8 inspection source: `{source_refs.phase8_inspection_path or 'missing'}`",
        f"- Phase 10 summary JSON source: `{source_refs.phase10_summary_path or 'not provided'}`",
        f"- Phase 10 evidence mode: `{source_refs.phase10_mode}`",
        f"- Inspection replay L2 delta: `{ranking_signals.get('replay_l2_delta', 'n/a')}`",
        f"- Inspection replay L3/L4 delta: `{ranking_signals.get('replay_l3_l4_delta', 'n/a')}`",
    ]

    lines = [
        '# Phase 11 Iteration Prioritization Report',
        '',
        '## Input Evidence',
        '',
        f"- Phase 8 summary JSON: `{source_refs.phase8_summary_path or 'missing'}`",
        f"- Phase 8 inspection JSON: `{source_refs.phase8_inspection_path or 'missing'}`",
        f"- Phase 10 summary JSON: `{source_refs.phase10_summary_path or 'not provided'}`",
        f"- Phase 10 verification note: `{source_refs.phase10_verification_path or 'not provided'}`",
        f"- Phase 10 report markdown: `{source_refs.phase10_report_path or 'not provided'}`",
        f"- Phase 8 iteration label: `{summary_model.phase8_iteration_label or 'unknown'}`",
        f"- Packet id: `{summary_model.packet_id or 'unknown'}`",
        '',
        '## Recommendation Queue',
        '',
        *recommendation_lines,
        '',
        '## Supporting Evidence',
        '',
        *supporting_bucket_lines,
        (
            f"- Ranking signals: replay_l2_delta={ranking_signals.get('replay_l2_delta', 'n/a')}, "
            f"replay_l3_l4_delta={ranking_signals.get('replay_l3_l4_delta', 'n/a')}"
        ),
        (
            f"- Phase 10 current recommendation carried forward: "
            f"`{summary_model.current_recommendation or 'not stated'}`"
        ),
        '',
        '## Missing Or Fallback Evidence',
        '',
        *fallback_lines,
        *(f"- Note: {note}" for note in summary_model.notes),
        '',
        '## Source Of Truth',
        '',
        *source_of_truth_lines,
    ]
    return '\n'.join(lines) + '\n'


__all__ = [
    'build_iteration_priority_inspection',
    'build_iteration_priority_summary',
    'IterationPriorityBlocker',
    'IterationPriorityInspection',
    'IterationPriorityOwnerBucket',
    'IterationPriorityPreflightError',
    'IterationPriorityRecommendation',
    'IterationPrioritySourceRefs',
    'IterationPrioritySummary',
    'LoadedPhase8ComparisonInspection',
    'LoadedPhase8ComparisonSummary',
    'LoadedPhase10ComparisonSurface',
    'build_phase10_fallback_surface',
    'load_phase8_comparison_inspection',
    'load_phase8_comparison_summary',
    'load_phase10_comparison_summary',
    'load_phase10_evidence',
    'render_iteration_priority_report',
    'rank_iteration_recommendations',
]
