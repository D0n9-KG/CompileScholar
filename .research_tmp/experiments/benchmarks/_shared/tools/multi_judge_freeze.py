"""multi_judge_freeze — P2-9: immutable input snapshot before judging.

The 0.5165 voided-score lesson: a mutable answers file + append-only scores
= stitched numbers. check_arm-style data_hash (P0-4) already invalidates
stale rows; this tool adds the belt under it — copy every judge input into
judge/snapshots/<ts>/ with a sha256 MANIFEST before scoring starts, and
judge from the frozen copy (--snapshot).

Usage:
    python multi_judge_freeze.py                # freeze all three arms
    python multi_judge_incremental.py --snapshot judge/snapshots/<ts>
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_MULTI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "scholarqa_multi")

INPUTS = {
    "ours": os.path.join(_MULTI, "baselines", "ours", "answers_ours.json"),
    "lightrag": os.path.join(_MULTI, "baselines", "lightrag",
                             "answers_lightrag.json"),
    "paperqa": os.path.join(_MULTI, "baselines", "paperqa",
                            "answers_paperqa.json"),
    "questions": os.path.join(_MULTI, "data", "scholarqa_multi.json"),
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()[:16]


def freeze(out_dir: str | None = None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap = out_dir or os.path.join(_MULTI, "judge", "snapshots", ts)
    os.makedirs(snap, exist_ok=True)
    manifest = {"frozen_at": ts, "files": {}}
    for name, path in INPUTS.items():
        if not os.path.exists(path):
            print(f"[freeze] MISSING {name}: {path}", file=sys.stderr)
            continue
        dest = os.path.join(snap, os.path.basename(path))
        shutil.copy2(path, dest)
        rows = "n/a"
        try:
            data = json.load(open(dest, encoding="utf-8"))
            rows = len(data) if isinstance(data, list) else len(data.get("questions", data))
        except Exception:
            pass
        manifest["files"][name] = {
            "path": os.path.basename(path), "sha256": _sha256(dest),
            "rows": rows, "source": path,
        }
        print(f"[freeze] {name}: {manifest['files'][name]['sha256']} "
              f"({rows} rows)")
    with open(os.path.join(snap, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[freeze] snapshot ready: {snap}")
    return snap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    freeze(args.out)


if __name__ == "__main__":
    main()
