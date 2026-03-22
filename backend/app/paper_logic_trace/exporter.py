from __future__ import annotations

from .compiler import compile_paper_logic_trace
from .models import PaperLogicTrace


def export_paper_logic_trace(client, paper_id: str) -> PaperLogicTrace:
    payload = client.get_paper_logic_trace_inputs(paper_id)
    return compile_paper_logic_trace(
        paper_metadata=dict(payload.get('paper_metadata') or {}),
        evidence_rows=list(payload.get('evidence_rows') or []),
        figure_rows=list(payload.get('figure_rows') or []),
        table_rows=list(payload.get('table_rows') or []),
        citation_rows=list(payload.get('citation_rows') or []),
        move_relation_rows=list(payload.get('move_relation_rows') or []),
        built_at=payload.get('built_at'),
    )
