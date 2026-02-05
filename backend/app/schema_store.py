from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.settings import settings


PaperType = Literal["research", "review"]

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,47}$")
_PROMPT_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class SchemaVersionInfo:
    paper_type: PaperType
    version: int
    path: Path


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _schemas_root() -> Path:
    # backend/storage/schemas
    root = _backend_root()
    p = root / settings.storage_dir / "schemas"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _paper_type_dir(paper_type: PaperType) -> Path:
    d = _schemas_root() / paper_type
    d.mkdir(parents=True, exist_ok=True)
    return d


def _active_path(paper_type: PaperType) -> Path:
    return _paper_type_dir(paper_type) / "active.json"


def _version_path(paper_type: PaperType, version: int) -> Path:
    return _paper_type_dir(paper_type) / f"v{int(version)}.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _default_schema(paper_type: PaperType) -> dict[str, Any]:
    # NOTE: IDs are stable ASCII slugs. Labels are editable.
    if paper_type == "research":
        steps = [
            {"id": "Background", "label_zh": "背景", "label_en": "Background", "enabled": True, "order": 0},
            {"id": "Problem", "label_zh": "问题", "label_en": "Problem", "enabled": True, "order": 1},
            {"id": "Method", "label_zh": "方法", "label_en": "Method", "enabled": True, "order": 2},
            {"id": "Experiment", "label_zh": "实验", "label_en": "Experiment", "enabled": True, "order": 3},
            {"id": "Result", "label_zh": "结果", "label_en": "Result", "enabled": True, "order": 4},
            {"id": "Conclusion", "label_zh": "结论", "label_en": "Conclusion", "enabled": True, "order": 5},
        ]
    else:
        # Review papers: default to a "review-ish" chain but keep it editable.
        steps = [
            {"id": "Background", "label_zh": "背景", "label_en": "Background", "enabled": True, "order": 0},
            {"id": "Scope", "label_zh": "范围/定义", "label_en": "Scope", "enabled": True, "order": 1},
            {"id": "Taxonomy", "label_zh": "分类/框架", "label_en": "Taxonomy", "enabled": True, "order": 2},
            {"id": "Comparison", "label_zh": "对比/综述", "label_en": "Comparison", "enabled": True, "order": 3},
            {"id": "Gap", "label_zh": "研究缺口", "label_en": "Gap", "enabled": True, "order": 4},
            {"id": "Conclusion", "label_zh": "总结与展望", "label_en": "Conclusion", "enabled": True, "order": 5},
        ]

    claim_kinds = [
        {"id": "Definition", "label_zh": "定义", "label_en": "Definition", "enabled": True},
        {"id": "Method", "label_zh": "方法", "label_en": "Method", "enabled": True},
        {"id": "Result", "label_zh": "结果", "label_en": "Result", "enabled": True},
        {"id": "Conclusion", "label_zh": "结论", "label_en": "Conclusion", "enabled": True},
        {"id": "Gap", "label_zh": "缺口", "label_en": "Gap", "enabled": True},
        {"id": "Critique", "label_zh": "批判", "label_en": "Critique", "enabled": True},
        {"id": "Limitation", "label_zh": "局限", "label_en": "Limitation", "enabled": True},
        {"id": "FutureWork", "label_zh": "未来工作", "label_en": "FutureWork", "enabled": True},
        {"id": "Comparison", "label_zh": "对比", "label_en": "Comparison", "enabled": True},
        {"id": "Assumption", "label_zh": "假设", "label_en": "Assumption", "enabled": True},
        {"id": "Scope", "label_zh": "范围", "label_en": "Scope", "enabled": True},
        {"id": "Taxonomy", "label_zh": "分类", "label_en": "Taxonomy", "enabled": True},
    ]

    return {
        "paper_type": paper_type,
        "version": 1,
        "steps": steps,
        "claim_kinds": claim_kinds,
        "rules": {
            "claims_per_paper_min": 24,
            "claims_per_paper_max": 48,
            "machine_evidence_min": 1,
            "machine_evidence_max": 2,
            "logic_evidence_min": 1,
            "logic_evidence_max": 2,
            "citation_context_sentence_window": 1,
            "targets_per_claim_max": 3,
            "require_targets_for_kinds": ["Gap", "Critique", "Limitation", "Comparison"],
            "evidence_verification": "llm",
        },
    }


def validate_schema(schema: dict[str, Any]) -> None:
    if not isinstance(schema, dict):
        raise ValueError("schema must be an object")
    paper_type = schema.get("paper_type")
    if paper_type not in {"research", "review"}:
        raise ValueError("paper_type must be 'research' or 'review'")
    if not isinstance(schema.get("steps"), list) or not schema["steps"]:
        raise ValueError("steps must be a non-empty list")
    if not isinstance(schema.get("claim_kinds"), list) or not schema["claim_kinds"]:
        raise ValueError("claim_kinds must be a non-empty list")

    step_ids: set[str] = set()
    for s in schema["steps"]:
        if not isinstance(s, dict):
            raise ValueError("steps[*] must be objects")
        sid = str(s.get("id") or "")
        if not _ID_RE.match(sid):
            raise ValueError(f"Invalid step id: {sid!r}")
        if sid in step_ids:
            raise ValueError(f"Duplicate step id: {sid}")
        step_ids.add(sid)

    kind_ids: set[str] = set()
    for k in schema["claim_kinds"]:
        if not isinstance(k, dict):
            raise ValueError("claim_kinds[*] must be objects")
        kid = str(k.get("id") or "")
        if not _ID_RE.match(kid):
            raise ValueError(f"Invalid claim kind id: {kid!r}")
        if kid in kind_ids:
            raise ValueError(f"Duplicate claim kind id: {kid}")
        kind_ids.add(kid)

    rules = schema.get("rules") or {}
    if not isinstance(rules, dict):
        raise ValueError("rules must be an object")
    cmin = int(rules.get("claims_per_paper_min") or 0)
    cmax = int(rules.get("claims_per_paper_max") or 0)
    if cmin < 1 or cmax < cmin:
        raise ValueError("Invalid claims_per_paper_min/max")
    emin = int(rules.get("machine_evidence_min") or 0)
    emax = int(rules.get("machine_evidence_max") or 0)
    if emin < 0 or emax < emin or emax > 6:
        raise ValueError("Invalid machine_evidence_min/max")
    lmin = int(rules.get("logic_evidence_min") or 0)
    lmax = int(rules.get("logic_evidence_max") or 0)
    if lmin < 0 or lmax < lmin or lmax > 8:
        raise ValueError("Invalid logic_evidence_min/max")
    vw = int(rules.get("citation_context_sentence_window") or 1)
    if vw < 0 or vw > 3:
        raise ValueError("Invalid citation_context_sentence_window")
    tp = int(rules.get("targets_per_claim_max") or 0)
    if tp < 0 or tp > 5:
        raise ValueError("Invalid targets_per_claim_max")
    et = str(rules.get("evidence_verification") or "llm")
    if et not in {"llm", "off"}:
        raise ValueError("rules.evidence_verification must be 'llm' or 'off'")

    req_kinds = rules.get("require_targets_for_kinds") or []
    if not isinstance(req_kinds, list):
        raise ValueError("rules.require_targets_for_kinds must be a list")
    for x in req_kinds:
        if str(x) not in kind_ids:
            # allow referencing disabled kinds but it still must exist
            raise ValueError(f"rules.require_targets_for_kinds contains unknown kind id: {x!r}")

    prompts = schema.get("prompts")
    if prompts is not None:
        if not isinstance(prompts, dict):
            raise ValueError("prompts must be an object")
        for k, v in prompts.items():
            kk = str(k or "")
            if not _PROMPT_KEY_RE.match(kk):
                raise ValueError(f"Invalid prompt key: {kk!r}")
            if not isinstance(v, str):
                raise ValueError(f"prompts[{kk!r}] must be a string")
            if len(v) > 20000:
                raise ValueError(f"prompts[{kk!r}] is too long (max 20000 chars)")


def ensure_defaults() -> None:
    for pt in ("research", "review"):
        paper_type = pt  # type: ignore[assignment]
        d = _paper_type_dir(paper_type)
        if not _active_path(paper_type).exists():
            s = _default_schema(paper_type)
            _write_json(_version_path(paper_type, 1), s)
            _write_json(_active_path(paper_type), {"active_version": 1})
        else:
            # Ensure the active version file exists (avoid recursion into load_active()).
            try:
                active = _read_json(_active_path(paper_type))
                v = int(active.get("active_version") or 1)
                _ = load_version(paper_type, v).get("version")
            except Exception:
                s = _default_schema(paper_type)
                _write_json(_version_path(paper_type, 1), s)
                _write_json(_active_path(paper_type), {"active_version": 1})


def list_versions(paper_type: PaperType) -> list[SchemaVersionInfo]:
    d = _paper_type_dir(paper_type)
    out: list[SchemaVersionInfo] = []
    for p in sorted(d.glob("v*.json")):
        m = re.match(r"^v(\d+)\.json$", p.name)
        if not m:
            continue
        out.append(SchemaVersionInfo(paper_type=paper_type, version=int(m.group(1)), path=p))
    out.sort(key=lambda x: x.version, reverse=True)
    return out


def load_version(paper_type: PaperType, version: int) -> dict[str, Any]:
    p = _version_path(paper_type, version)
    if not p.exists():
        raise FileNotFoundError(f"Schema version not found: {paper_type} v{version}")
    s = _read_json(p)
    validate_schema(s)
    s["paper_type"] = paper_type
    s["version"] = int(version)
    return s


def load_active(paper_type: PaperType) -> dict[str, Any]:
    ensure_defaults()
    ap = _active_path(paper_type)
    active = _read_json(ap)
    v = int(active.get("active_version") or 1)
    return load_version(paper_type, v)


def activate_version(paper_type: PaperType, version: int) -> dict[str, Any]:
    s = load_version(paper_type, version)
    _write_json(_active_path(paper_type), {"active_version": int(version)})
    return s


def create_new_version(paper_type: PaperType, schema: dict[str, Any], activate: bool = True) -> dict[str, Any]:
    ensure_defaults()
    versions = list_versions(paper_type)
    next_v = (versions[0].version + 1) if versions else 1
    schema = dict(schema)
    schema["paper_type"] = paper_type
    schema["version"] = int(next_v)
    validate_schema(schema)
    _write_json(_version_path(paper_type, next_v), schema)
    if activate:
        _write_json(_active_path(paper_type), {"active_version": int(next_v)})
    return schema
