from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from app.ingest.models import DocumentIR


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )


def run_phase1_paper_logic_trace(
    *,
    doc: DocumentIR,
    paper_id: str,
    cite_rec: dict[str, Any] | None,
    schema: dict[str, Any],
    artifacts_dir: Path | str,
    move_extractor: Callable[..., dict[str, Any]] | None = None,
    allow_weak: bool = False,
) -> dict[str, Any]:
    from app.paper_logic_trace.compiler import compile_paper_logic_trace
    from app.paper_logic_trace.direct_extraction import (
        build_paper_logic_trace_inputs,
        build_trace_quality_report,
    )

    del allow_weak
    compiler_inputs = build_paper_logic_trace_inputs(
        doc=doc,
        paper_id=paper_id,
        cite_rec=cite_rec,
        schema=schema,
        move_extractor=move_extractor,
    )
    extraction_report = dict(compiler_inputs.pop('extraction_report', {}) or {})
    trace = compile_paper_logic_trace(**compiler_inputs)
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    _json_dump(artifacts / 'paper_logic_trace.json', trace.model_dump(mode='json'))
    quality_report = build_trace_quality_report(trace, extraction_report)
    return {
        'paper_logic_trace': trace,
        'quality_report': quality_report,
    }


__all__ = ['run_phase1_paper_logic_trace']
