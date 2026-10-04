# -*- coding: utf-8 -*-
"""W2 P0-3 validation: evidence must be verbatim and a proposal statement; generic names flagged; uses excluded."""
from __future__ import annotations

import json

from compilescholar.compile.skeleton import proposes as P

AB = ("Low-rank adaptation is widely used. In this paper, we propose LoRA-X, a new parameter-efficient fine-tuning "
      "method that factorizes updates. Our method builds upon LoRA and uses Adam. We also release a benchmark called "
      "PEFT-Bench for evaluation.")


def test_validate():
    ok = P.validate({"name": "LoRA-X", "aliases": [], "evidence": "In this paper, we propose LoRA-X, a new parameter-efficient "
                     "fine-tuning method that factorizes updates."}, AB)
    assert ok and ok["method"] == "LoRA-X" and not ok["generic"]
    assert P.validate({"name": "LoRA", "evidence": "Our method builds upon LoRA and uses Adam."}, AB) is None
    assert P.validate({"name": "LoRA-X", "evidence": "We propose something not in the abstract at all here."}, AB) is None
    bench = P.validate({"name": "PEFT-Bench", "evidence": "We also release a benchmark called PEFT-Bench for evaluation."}, AB)
    assert bench and bench["method"] == "PEFT-Bench"
    g = P.validate({"name": "framework", "evidence": "In this paper, we propose LoRA-X, a new parameter-efficient fine-tuning "
                    "method that factorizes updates."}, AB)
    assert g is not None and g["generic"]


def test_extract_with_fake_llm():
    def chat(prompt, **kw):
        assert "Title: T" in prompt
        return json.dumps({"proposes": [
            {"name": "LoRA-X", "aliases": ["LoRA eXtended"], "evidence": "In this paper, we propose LoRA-X, a new "
             "parameter-efficient fine-tuning method that factorizes updates."},
            {"name": "LoRA", "aliases": [], "evidence": "Our method builds upon LoRA and uses Adam."}]})
    out = P.extract("p1", "T", AB, chat=chat)
    assert [o["method"] for o in out] == ["LoRA-X"] and out[0]["paper_id"] == "p1"
    assert P.extract("p2", "T", "", chat=chat) == []
