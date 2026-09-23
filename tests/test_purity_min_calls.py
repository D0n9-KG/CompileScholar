"""P2-12 regression: empty/near-empty ledger must not pass purity.

PaperQA case (audit #11): index dead -> zero calls -> assertion read PURE.
min_calls floors the check.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kb_infra.llm import check_arm_purity  # noqa: E402


def _ledger(rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                    encoding="utf-8")
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


ALLOWED = [("local", "Qwen3.8-27B")]


def test_empty_ledger_fails_with_min_calls():
    p = _ledger([])
    r = check_arm_purity(p, allowed_pairs=ALLOWED, min_calls=1)
    assert not r["pure"]
    assert any("insufficient_calls" in v for v in r["violations"])
    Path(p).unlink()


def test_empty_ledger_passes_without_min_calls_backcompat():
    p = _ledger([])
    r = check_arm_purity(p, allowed_pairs=ALLOWED)
    assert r["pure"]  # old behavior preserved for callers without a floor


def test_healthy_ledger_with_floor():
    p = _ledger([{"ok": True, "provider": "local", "model": "Qwen3.8-27B"}] * 20)
    r = check_arm_purity(p, allowed_pairs=ALLOWED, min_calls=10)
    assert r["pure"] and r["total_ok"] == 20
    Path(p).unlink()


def test_few_calls_below_floor_fails():
    p = _ledger([{"ok": True, "provider": "local", "model": "Qwen3.8-27B"}] * 3)
    r = check_arm_purity(p, allowed_pairs=ALLOWED, min_calls=10)
    assert not r["pure"] and r["total_ok"] == 3
    Path(p).unlink()
