from __future__ import annotations

from .models import PaperLogicTrace


def export_paper_logic_trace(client, paper_id: str) -> PaperLogicTrace:
    payload = client.get_paper_logic_trace(paper_id)
    return PaperLogicTrace.model_validate(payload)
