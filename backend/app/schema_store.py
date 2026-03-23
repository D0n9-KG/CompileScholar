from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from app.settings import settings


PaperType = Literal['research', 'review', 'software', 'theoretical', 'case_study']
PAPER_TYPES: tuple[str, ...] = ('research', 'review', 'software', 'theoretical', 'case_study')
_PAPER_TYPE_SET: frozenset[str] = frozenset(PAPER_TYPES)

_PROMPT_KEY_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_-]{0,63}$')


@dataclass(frozen=True)
class SchemaVersionInfo:
    paper_type: PaperType
    version: int
    path: Path
    name: str = ''


def coerce_paper_type(value: Any) -> PaperType | None:
    raw = str(value or '').strip().lower()
    if raw in _PAPER_TYPE_SET:
        return cast(PaperType, raw)
    return None


def normalize_paper_type(value: Any, *, default: PaperType = 'research') -> PaperType:
    return coerce_paper_type(value) or default


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _schemas_root() -> Path:
    root = _backend_root() / settings.storage_dir / 'schemas'
    root.mkdir(parents=True, exist_ok=True)
    return root


def _paper_type_dir(paper_type: PaperType) -> Path:
    path = _schemas_root() / paper_type
    path.mkdir(parents=True, exist_ok=True)
    return path


def _active_path(paper_type: PaperType) -> Path:
    return _paper_type_dir(paper_type) / 'active.json'


def _version_path(paper_type: PaperType, version: int) -> Path:
    return _paper_type_dir(paper_type) / f'v{int(version)}.json'


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def _default_prompts() -> dict[str, str]:
    return {
        'research_move_window_system': (
            'You extract canonical ResearchMove records for a PaperLogicTrace.\n'
            'Return strict JSON only.\n'
            'Use only directly supported evidence from the provided source units.\n'
        ),
        'research_move_window_user_template': (
            'Compile ResearchMove items from the supplied paper window.\n'
            'Keep summaries concise and factual, and only emit supported slots.\n'
        ),
        'citation_purpose_batch_system': (
            'Classify citation purpose for each cited paper using the supplied contexts.\n'
            'Return strict JSON only.\n'
        ),
        'citation_purpose_batch_user_template': (
            'For each cited paper, infer purpose labels only when directly supported by the context snippets.\n'
        ),
        'reference_recovery_system': (
            'Recover structured reference entries from the source markdown when parsed references are missing.\n'
            'Return strict JSON only.\n'
        ),
        'reference_recovery_user_template': (
            'Extract the bibliography entries that are actually present in the paper and preserve original author/title/year details.\n'
        ),
    }


def _default_rules() -> dict[str, Any]:
    return {
        'paper_logic_trace_window_chars_max': 5000,
        'paper_logic_trace_moves_per_window_max': 2,
        'paper_logic_trace_gate_min_moves': 2,
        'paper_logic_trace_gate_required_roles': ['problem', 'method', 'result'],
        'paper_logic_trace_gate_min_slot_signals': 1,
        'reference_recovery_enabled': True,
        'reference_recovery_trigger_max_existing_refs': 0,
        'reference_recovery_max_refs': 180,
        'reference_recovery_doc_chars_max': 48000,
        'reference_recovery_agent_timeout_sec': 45.0,
        'citation_event_recovery_enabled': True,
        'citation_event_recovery_trigger_max_existing_events': 0,
        'citation_event_recovery_numeric_bracket_enabled': True,
        'citation_event_recovery_paren_numeric_enabled': False,
        'citation_event_recovery_author_year_enabled': True,
        'citation_event_recovery_max_events_per_chunk': 6,
        'citation_event_recovery_context_chars': 800,
        'crossref_confidence_threshold': 0.55,
    }


def _default_schema(paper_type: PaperType) -> dict[str, Any]:
    return {
        'paper_type': paper_type,
        'version': 1,
        'name': 'PaperLogicTrace Compiler Policy',
        'rules': _default_rules(),
        'prompts': _default_prompts(),
    }


def _normalize_schema_payload(schema: dict[str, Any], *, paper_type: PaperType, version: int) -> dict[str, Any]:
    payload = dict(schema)
    payload['paper_type'] = paper_type
    payload['version'] = int(version)
    if 'name' not in payload or not str(payload.get('name') or '').strip():
        payload['name'] = 'PaperLogicTrace Compiler Policy'
    payload.setdefault('rules', {})
    payload.setdefault('prompts', {})
    return payload


def _validate_rule_int(
    rules: dict[str, Any],
    key: str,
    *,
    default: int,
    lo: int,
    hi: int,
) -> None:
    try:
        value = int(rules.get(key, default))
    except Exception as exc:
        raise ValueError(f'Invalid {key}') from exc
    if value < lo or value > hi:
        raise ValueError(f'Invalid {key}')


def _validate_rule_float(
    rules: dict[str, Any],
    key: str,
    *,
    default: float,
    lo: float,
    hi: float,
) -> None:
    try:
        value = float(rules.get(key, default))
    except Exception as exc:
        raise ValueError(f'Invalid {key}') from exc
    if value < lo or value > hi:
        raise ValueError(f'Invalid {key}')


def _validate_rule_bool(rules: dict[str, Any], key: str, *, default: bool) -> None:
    value = rules.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f'Invalid {key}')


def validate_schema(schema: dict[str, Any]) -> None:
    if not isinstance(schema, dict):
        raise ValueError('schema must be an object')
    paper_type = schema.get('paper_type')
    if paper_type not in _PAPER_TYPE_SET:
        raise ValueError(f"paper_type must be one of: {', '.join(PAPER_TYPES)}")

    name = schema.get('name')
    if name is not None:
        if not isinstance(name, str):
            raise ValueError('name must be a string')
        if len(name.strip()) > 80:
            raise ValueError('name is too long (max 80 chars)')

    rules = schema.get('rules')
    if not isinstance(rules, dict):
        raise ValueError('rules must be an object')

    _validate_rule_int(rules, 'paper_logic_trace_window_chars_max', default=5000, lo=1200, hi=9000)
    _validate_rule_int(rules, 'paper_logic_trace_moves_per_window_max', default=2, lo=1, hi=4)
    _validate_rule_int(rules, 'paper_logic_trace_gate_min_moves', default=2, lo=1, hi=12)
    _validate_rule_int(rules, 'paper_logic_trace_gate_min_slot_signals', default=1, lo=0, hi=24)

    required_roles = rules.get('paper_logic_trace_gate_required_roles', ['problem', 'method', 'result'])
    if not isinstance(required_roles, list):
        raise ValueError('paper_logic_trace_gate_required_roles must be a list')
    allowed_roles = {
        'problem',
        'background',
        'hypothesis',
        'method',
        'experiment',
        'result',
        'interpretation',
        'limitation',
        'future_work',
    }
    for item in required_roles:
        role = str(item or '').strip()
        if role and role not in allowed_roles:
            raise ValueError(f'Unknown paper_logic_trace_gate_required_roles value: {role}')

    _validate_rule_bool(rules, 'reference_recovery_enabled', default=True)
    _validate_rule_int(rules, 'reference_recovery_trigger_max_existing_refs', default=0, lo=0, hi=200)
    _validate_rule_int(rules, 'reference_recovery_max_refs', default=180, lo=1, hi=500)
    _validate_rule_int(rules, 'reference_recovery_doc_chars_max', default=48000, lo=1000, hi=200000)
    _validate_rule_float(rules, 'reference_recovery_agent_timeout_sec', default=45.0, lo=0.5, hi=300.0)

    _validate_rule_bool(rules, 'citation_event_recovery_enabled', default=True)
    _validate_rule_int(rules, 'citation_event_recovery_trigger_max_existing_events', default=0, lo=0, hi=50)
    _validate_rule_bool(rules, 'citation_event_recovery_numeric_bracket_enabled', default=True)
    _validate_rule_bool(rules, 'citation_event_recovery_paren_numeric_enabled', default=False)
    _validate_rule_bool(rules, 'citation_event_recovery_author_year_enabled', default=True)
    _validate_rule_int(rules, 'citation_event_recovery_max_events_per_chunk', default=6, lo=1, hi=40)
    _validate_rule_int(rules, 'citation_event_recovery_context_chars', default=800, lo=120, hi=4000)

    _validate_rule_float(rules, 'crossref_confidence_threshold', default=0.55, lo=0.0, hi=1.0)

    prompts = schema.get('prompts')
    if prompts is not None:
        if not isinstance(prompts, dict):
            raise ValueError('prompts must be an object')
        for key, value in prompts.items():
            prompt_key = str(key or '')
            if not _PROMPT_KEY_RE.match(prompt_key):
                raise ValueError(f'Invalid prompt key: {prompt_key!r}')
            if not isinstance(value, str):
                raise ValueError(f'prompts[{prompt_key!r}] must be a string')
            if len(value) > 20000:
                raise ValueError(f'prompts[{prompt_key!r}] is too long (max 20000 chars)')


def ensure_defaults() -> None:
    for raw_paper_type in PAPER_TYPES:
        paper_type: PaperType = raw_paper_type  # type: ignore[assignment]
        active_path = _active_path(paper_type)
        version_path = _version_path(paper_type, 1)
        if not active_path.exists() or not version_path.exists():
            schema = _default_schema(paper_type)
            _write_json(version_path, schema)
            _write_json(active_path, {'active_version': 1})
            continue
        try:
            active = _read_json(active_path)
            version = int(active.get('active_version') or 1)
            payload = _read_json(_version_path(paper_type, version))
            normalized = _normalize_schema_payload(payload, paper_type=paper_type, version=version)
            validate_schema(normalized)
            if normalized != payload:
                _write_json(_version_path(paper_type, version), normalized)
        except Exception:
            schema = _default_schema(paper_type)
            _write_json(version_path, schema)
            _write_json(active_path, {'active_version': 1})


def list_versions(paper_type: PaperType) -> list[SchemaVersionInfo]:
    directory = _paper_type_dir(paper_type)
    out: list[SchemaVersionInfo] = []
    for path in sorted(directory.glob('v*.json')):
        match = re.match(r'^v(\d+)\.json$', path.name)
        if not match:
            continue
        label = ''
        try:
            payload = _normalize_schema_payload(_read_json(path), paper_type=paper_type, version=int(match.group(1)))
            label = str(payload.get('name') or '').strip()
        except Exception:
            label = ''
        out.append(SchemaVersionInfo(paper_type=paper_type, version=int(match.group(1)), path=path, name=label))
    out.sort(key=lambda item: item.version, reverse=True)
    return out


def load_version(paper_type: PaperType, version: int) -> dict[str, Any]:
    path = _version_path(paper_type, version)
    if not path.exists():
        raise FileNotFoundError(f'Schema version not found: {paper_type} v{version}')
    payload = _normalize_schema_payload(_read_json(path), paper_type=paper_type, version=int(version))
    validate_schema(payload)
    return payload


def load_active(paper_type: PaperType) -> dict[str, Any]:
    ensure_defaults()
    active = _read_json(_active_path(paper_type))
    version = int(active.get('active_version') or 1)
    return load_version(paper_type, version)


def activate_version(paper_type: PaperType, version: int) -> dict[str, Any]:
    schema = load_version(paper_type, version)
    _write_json(_active_path(paper_type), {'active_version': int(version)})
    return schema


def delete_version(paper_type: PaperType, version: int) -> dict[str, Any]:
    ensure_defaults()
    version = int(version)
    path = _version_path(paper_type, version)
    if not path.exists():
        raise FileNotFoundError(f'Schema version not found: {paper_type} v{version}')

    versions = list_versions(paper_type)
    if len(versions) <= 1:
        raise ValueError('Cannot delete the last schema version')

    active_meta = _read_json(_active_path(paper_type))
    old_active = int(active_meta.get('active_version') or 1)
    path.unlink(missing_ok=False)

    remaining = list_versions(paper_type)
    if old_active == version or not _version_path(paper_type, old_active).exists():
        new_active = int(remaining[0].version)
        _write_json(_active_path(paper_type), {'active_version': new_active})
        active_changed = True
    else:
        new_active = old_active
        active_changed = False
    return {
        'paper_type': paper_type,
        'deleted_version': version,
        'active_version': int(new_active),
        'active_changed': bool(active_changed),
    }


def create_new_version(paper_type: PaperType, schema: dict[str, Any], activate: bool = True) -> dict[str, Any]:
    ensure_defaults()
    versions = list_versions(paper_type)
    next_version = (versions[0].version + 1) if versions else 1
    payload = _normalize_schema_payload(dict(schema), paper_type=paper_type, version=next_version)
    validate_schema(payload)
    _write_json(_version_path(paper_type, next_version), payload)
    if activate:
        _write_json(_active_path(paper_type), {'active_version': int(next_version)})
    return payload
