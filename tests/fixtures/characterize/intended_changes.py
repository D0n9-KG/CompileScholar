# -*- coding: utf-8 -*-
"""Intentional behaviour changes relative to the pre-move goldens. Each entry names the golden key, the W1 item that
changed it, and how the new expected value was obtained. Tests compare these keys against goldens_intended.json
instead of goldens.json; everything else must still match the pre-move goldens byte for byte.

  answer.task_context_cutoff  (W1-12)  cutoff 2023-06: KB evidence from papers published after the cutoff (2023-07+,
                                       2024) is now filtered out of both KB channels. The old code applied the
                                       cutoff only to external retrieval. Regenerate with:
                                       python tests/fixtures/characterize/intended_changes.py

  answer.* prompt-bearing keys  (AQ-1010)  plan.txt/write.txt composition discipline, mechanism-driven from the
                                       dev100 content attribution (dev100_content_analysis.md, four loss patterns):
                                       (A) rubric-critical definitions missing — write rule 1 "delete uncited
                                       sentences" structurally suppressed definitional sentences (official CR judge
                                       skips high-level introductory sentences, so a capped 2-sentence definitional
                                       exception is ~free on CR); (B) taxonomy breadth — plan.txt now requires full
                                       method-category span + definitional grounding of technical terms; (C) quan-
                                       titative specifics dropped — write rule 3 now carries numbers/formulas/dataset
                                       names through; (D) direct answers missing + AP drift (253 irrelevant texts in
                                       72/100 q; "irrelevant" = does not bear on the specific aspect asked) — write
                                       rule 4 sharpened to aspect-level relevance + explicit direct-answer duty.
                                       Generic report-quality rules, no question- or rubric-specific content.
                                       Affected keys: budget1000_cite, nolimit_nocite, noprobe_nostate, noscreen,
                                       task_context_cutoff (rendered prompts embed the changed templates).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _characterize_impl as C  # noqa: E402

INTENDED = {("answer", "task_context_cutoff"): "W1-12",
            ("answer", "budget1000_cite"): "AQ-1010",
            ("answer", "nolimit_nocite"): "AQ-1010",
            ("answer", "noprobe_nostate"): "AQ-1010",
            ("answer", "noscreen"): "AQ-1010"}

if __name__ == "__main__":
    import make_goldens as M
    new = json.loads(json.dumps(M.compute("new"), ensure_ascii=False, default=str))
    out = {}
    for (a, b) in INTENDED:
        v = new[a][b]
        if a == "answer":
            # test_answer_path_matches_golden pops the W1-10/W1-13-added trace fields from the computed
            # goldens before comparing, so intended values must carry the same stripped shape as goldens.json.
            tr = (v.get("result") or {}).get("trace") or {}
            tr.pop("degradation", None)
            (tr.get("assemble") or {}).pop("merged_identities", None)
        out[f"{a}.{b}"] = v
    json.dump(out, open(C.FIX / "goldens_intended.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              sort_keys=True)
    print("intended goldens:", sorted(out))
