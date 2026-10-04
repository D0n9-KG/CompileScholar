# -*- coding: utf-8 -*-
"""Intentional behaviour changes relative to the pre-move goldens. Each entry names the golden key, the W1 item that
changed it, and how the new expected value was obtained. Tests compare these keys against goldens_intended.json
instead of goldens.json; everything else must still match the pre-move goldens byte for byte.

  answer.task_context_cutoff  (W1-12)  cutoff 2023-06: KB evidence from papers published after the cutoff (2023-07+,
                                       2024) is now filtered out of both KB channels. The old code applied the
                                       cutoff only to external retrieval. Regenerate with:
                                       python tests/fixtures/characterize/intended_changes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _characterize_impl as C  # noqa: E402

INTENDED = {("answer", "task_context_cutoff"): "W1-12"}

if __name__ == "__main__":
    import make_goldens as M
    new = json.loads(json.dumps(M.compute("new"), ensure_ascii=False, default=str))
    out = {f"{a}.{b}": new[a][b] for (a, b) in INTENDED}
    json.dump(out, open(C.FIX / "goldens_intended.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              sort_keys=True)
    print("intended goldens:", sorted(out))
