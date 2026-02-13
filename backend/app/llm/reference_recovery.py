from __future__ import annotations

import concurrent.futures
import json
import re
from pathlib import Path
from typing import Any

from app.ingest.models import DocumentIR, ReferenceEntry
from app.llm.client import call_json


_WS_RE = re.compile(r"\s+")
_TPL_RE = re.compile(r"\{\{\s*([A-Za-z][A-Za-z0-9_]*)\s*\}\}")
_REF_PREFIX_RE = re.compile(r"^\s*(?:\[\d{1,4}\]|\(\d{1,4}\)|\d{1,4}[.)])\s*")
_REF_HEADING_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(references|bibliography|reference list|参考文献)\s*$",
    re.IGNORECASE,
)
_SECTION_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _rule_int(rules: dict[str, Any], key: str, default: int, *, lo: int, hi: int) -> int:
    try:
        raw = rules.get(key, default)
        v = int(raw)
    except Exception:
        v = int(default)
    return max(lo, min(hi, v))


def _rule_float(rules: dict[str, Any], key: str, default: float, *, lo: float, hi: float) -> float:
    try:
        raw = rules.get(key, default)
        v = float(raw)
    except Exception:
        v = float(default)
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


def _rule_bool(rules: dict[str, Any], key: str, default: bool) -> bool:
    raw = rules.get(key, default)
    if isinstance(raw, bool):
        return raw
    s = str(raw or "").strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _render_template(template: str, vars: dict[str, Any]) -> str:
    def _sub(m: re.Match[str]) -> str:
        key = m.group(1)
        v = vars.get(key)
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return str(v)

    return _TPL_RE.sub(_sub, template or "")


def _shorten(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "\n[TRUNCATED]"


def _find_reference_heading_index(lines: list[str]) -> int:
    heading_idx = -1
    for i, line in enumerate(lines):
        if _REF_HEADING_RE.match(line.strip()):
            heading_idx = i
    return heading_idx


def _tail_keep(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return "[TRUNCATED_HEAD]\n" + t[-max_chars:]


def _prepare_markdown_for_agent(markdown_text: str, *, max_chars: int) -> str:
    text = markdown_text or ""
    if len(text) <= max_chars:
        return text

    lines = text.splitlines()
    heading_idx = _find_reference_heading_index(lines)
    if heading_idx >= 0:
        tail_from_heading = "\n".join(lines[heading_idx:])
        if len(tail_from_heading) <= max_chars:
            return tail_from_heading
        return _shorten(tail_from_heading, max_chars=max_chars)

    # Fallback: references usually live in the tail.
    return _tail_keep(text, max_chars=max_chars)


def _normalize_ref(text: str) -> str:
    s = _REF_PREFIX_RE.sub("", str(text or "")).strip()
    s = _WS_RE.sub(" ", s)
    return s


def _extract_reference_texts(payload: dict[str, Any], max_refs: int) -> list[str]:
    refs_any = payload.get("references")
    if refs_any is None:
        refs_any = payload.get("refs")
    if not isinstance(refs_any, list):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for item in refs_any:
        raw = None
        if isinstance(item, str):
            raw = item
        elif isinstance(item, dict):
            for key in ("raw", "text", "reference", "entry"):
                v = item.get(key)
                if isinstance(v, str) and v.strip():
                    raw = v
                    break
        if not raw:
            continue
        norm = _normalize_ref(raw)
        if len(norm) < 12:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
        if len(out) >= max_refs:
            break
    return out


def _looks_like_reference_line(line: str) -> bool:
    if len(line) < 12:
        return False
    lower = line.lower()
    norm = _normalize_ref(line)
    if len(norm) < 12:
        return False
    return bool(_YEAR_RE.search(norm) or ("doi" in lower) or ("," in norm and "." in norm))


def _extract_reference_texts_heuristic(markdown_text: str, max_refs: int) -> list[str]:
    lines = (markdown_text or "").splitlines()
    if not lines:
        return []

    heading_idx = _find_reference_heading_index(lines)
    if heading_idx < 0:
        # No explicit heading: use tail-based heuristic.
        out_no_heading: list[str] = []
        seen_no_heading: set[str] = set()
        tail_window = lines[-600:] if len(lines) > 600 else lines
        for raw_line in tail_window:
            line = raw_line.strip()
            if not _looks_like_reference_line(line):
                continue
            norm = _normalize_ref(line)
            if norm in seen_no_heading:
                continue
            seen_no_heading.add(norm)
            out_no_heading.append(norm)
            if len(out_no_heading) >= max_refs:
                break
        return out_no_heading

    out: list[str] = []
    seen: set[str] = set()
    for raw_line in lines[heading_idx + 1 :]:
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if _SECTION_HEADING_RE.match(line) and not _REF_HEADING_RE.match(line):
            break
        if lower.startswith("corresponding author"):
            break
        if not _looks_like_reference_line(line):
            continue
        norm = _normalize_ref(line)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
        if len(out) >= max_refs:
            break
    return out


def recover_references_with_agent(
    doc: DocumentIR,
    *,
    prompt_overrides: dict[str, Any] | None = None,
    rules: dict[str, Any] | None = None,
) -> tuple[DocumentIR, dict[str, Any]]:
    """
    Fallback reference recovery for papers where parser extracted too few references.

    Returns:
    - recovered/new DocumentIR (or original when skipped/failed)
    - report dict for diagnostics/artifacts
    """
    rules = rules or {}
    before_refs = len(doc.references or [])
    trigger_max_existing_refs = _rule_int(rules, "reference_recovery_trigger_max_existing_refs", 0, lo=0, hi=200)
    enabled = _rule_bool(rules, "reference_recovery_enabled", True)
    report: dict[str, Any] = {
        "enabled": enabled,
        "before_refs": before_refs,
        "after_refs": before_refs,
        "trigger_max_existing_refs": trigger_max_existing_refs,
        "agent_called": False,
        "heuristic_used": False,
        "replaced_existing": False,
        "status": "pending",
        "error": None,
    }

    if not enabled:
        report["status"] = "disabled"
        return doc, report

    if before_refs > trigger_max_existing_refs:
        report["status"] = "skipped_existing" if trigger_max_existing_refs == 0 else "skipped_existing_above_trigger"
        return doc, report

    max_refs = _rule_int(rules, "reference_recovery_max_refs", 180, lo=1, hi=500)
    max_chars = _rule_int(rules, "reference_recovery_doc_chars_max", 48000, lo=1000, hi=200000)
    agent_timeout_sec = _rule_float(rules, "reference_recovery_agent_timeout_sec", 45.0, lo=0.5, hi=300.0)
    report["agent_timeout_sec"] = agent_timeout_sec

    md_path = Path(str(doc.paper.md_path or ""))
    if not md_path.exists():
        report["status"] = "error"
        report["error"] = f"Markdown not found: {md_path}"
        return doc, report

    markdown_text = md_path.read_text(encoding="utf-8", errors="ignore")
    markdown_for_agent = _prepare_markdown_for_agent(markdown_text, max_chars=max_chars)
    heuristic_refs = _extract_reference_texts_heuristic(markdown_text, max_refs=max_refs)
    report["markdown_chars_full"] = len(markdown_text)
    report["markdown_chars_agent"] = len(markdown_for_agent)
    report["heuristic_candidates"] = len(heuristic_refs)

    prompts = prompt_overrides or {}
    default_system = (
        "You are a reference-recovery agent for scientific markdown.\n"
        "Task: extract bibliography entries from the provided markdown.\n"
        "Return STRICT JSON only (no prose).\n"
        "Do not fabricate references.\n"
        "If unsure, skip the entry.\n"
    )
    default_user_template = (
        "Recover references for this paper.\n"
        "Title: {{title}}\n"
        "DOI: {{doi}}\n"
        "Max references: {{max_refs}}\n\n"
        "Markdown text:\n"
        "{{markdown_text}}\n\n"
        "Output JSON schema:\n"
        '{ "references": [ {"raw":"..."} ] }\n'
    )
    system = str(prompts.get("reference_recovery_system") or "").strip() or default_system
    user_template = str(prompts.get("reference_recovery_user_template") or "").strip() or default_user_template
    user = _render_template(
        user_template,
        {
            "title": doc.paper.title or doc.paper.title_alt or "",
            "doi": doc.paper.doi or "",
            "max_refs": max_refs,
            "markdown_text": markdown_for_agent,
        },
    )

    report["agent_called"] = True
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(call_json, system, user)
            try:
                out = future.result(timeout=agent_timeout_sec)
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                if heuristic_refs:
                    refs_text = heuristic_refs
                    report["status"] = "recovered_heuristic_after_agent_timeout"
                    report["error"] = f"agent timeout after {agent_timeout_sec:.1f}s"
                    report["heuristic_used"] = True
                else:
                    report["status"] = "agent_timeout"
                    report["error"] = f"agent timeout after {agent_timeout_sec:.1f}s"
                    return doc, report
                out = None
            except Exception:
                raise
    except Exception as exc:  # noqa: BLE001
        if heuristic_refs:
            refs_text = heuristic_refs
            report["status"] = "recovered_heuristic_after_agent_error"
            report["error"] = str(exc)
            report["heuristic_used"] = True
        else:
            report["status"] = "agent_error"
            report["error"] = str(exc)
            return doc, report
    else:
        if report.get("status") == "recovered_heuristic_after_agent_timeout":
            refs_text = heuristic_refs
        else:
            refs_text = _extract_reference_texts(out if isinstance(out, dict) else {}, max_refs=max_refs)
            if not refs_text and heuristic_refs:
                refs_text = heuristic_refs
                report["status"] = "recovered_heuristic_after_empty_agent_result"
                report["heuristic_used"] = True

    if not refs_text:
        if before_refs > 0:
            report["status"] = "empty_result_keep_existing"
            return doc, report
        report["status"] = "empty_result"
        return doc, report

    candidate_refs = [
        ReferenceEntry(
            paper_source=doc.paper.paper_source,
            md_path=doc.paper.md_path,
            ref_num=i + 1,
            raw=text,
        )
        for i, text in enumerate(refs_text)
    ]
    report["candidate_refs"] = len(candidate_refs)

    if before_refs > 0 and len(candidate_refs) <= before_refs:
        report["status"] = "kept_existing_not_improved"
        report["recovered_refs"] = len(candidate_refs)
        return doc, report

    references = candidate_refs

    recovered = DocumentIR(
        paper=doc.paper,
        chunks=doc.chunks,
        references=references,
        citations=doc.citations,
    )
    if before_refs > 0 and len(references) > before_refs:
        report["replaced_existing"] = True
    if not report.get("status", "").startswith("recovered_heuristic"):
        report["status"] = "recovered"
    report["after_refs"] = len(references)
    report["recovered_refs"] = len(references)
    return recovered, report
