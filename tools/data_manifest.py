# -*- coding: utf-8 -*-
"""data/ is not in git; its contents are pinned by data/MANIFEST.tsv (tracked): path, size, sha256, origin.

  python tools/data_manifest.py write [subdir ...]   (re)hash the given data subdirectories (default: benchmarks)
  python tools/data_manifest.py check                verify every listed file (missing / size / hash mismatch -> exit 1)
Large or regenerable trees (derived/, library/papers/, external/ parquet) are listed by the stage manifests or their
own release files instead."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
MANIFEST = DATA / "MANIFEST.tsv"
ORIGIN = {"benchmarks/cs2/kb_v2": ".research_tmp/experiments/benchmarks/cs2/base_kb_v2",
          "benchmarks/cs2/arms/memorized": ".research_tmp/experiments/benchmarks/cs2/arm_memorized",
          "benchmarks/cs2/rubrics": ".research_tmp/experiments/benchmarks/scholarqa_multi",
          "benchmarks/cs2/survey_texts": ".research_tmp/experiments/benchmarks/cs2/base_kb/survey_texts",
          "benchmarks/dsb/oracle_inputs.json": ".research_tmp/review_1002/p6/oracle_inputs.json",
          "benchmarks/dsb/arms": ".research_tmp/review_1002/p6"}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _origin(rel: str) -> str:
    best = max((k for k in ORIGIN if rel == k or rel.startswith(k + "/")), key=len, default=None)
    return ORIGIN[best] if best else ""


def read() -> dict[str, tuple[int, str, str]]:
    out = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()[1:]:
            rel, size, sha, origin = line.split("\t")
            out[rel] = (int(size), sha, origin)
    return out


def write(subdirs: list[str]) -> int:
    rows = read()
    for sd in subdirs:
        rows = {k: v for k, v in rows.items() if not (k == sd or k.startswith(sd + "/"))}
        for p in sorted((DATA / sd).rglob("*")):
            if p.is_file():
                rel = p.relative_to(DATA).as_posix()
                rows[rel] = (p.stat().st_size, _sha(p), _origin(rel))
    MANIFEST.write_text("path\tsize\tsha256\torigin\n" + "".join(
        f"{k}\t{s}\t{h}\t{o}\n" for k, (s, h, o) in sorted(rows.items())), encoding="utf-8", newline="\n")
    return len(rows)


def check() -> list[str]:
    bad = []
    for rel, (size, sha, _) in read().items():
        p = DATA / rel
        if not p.exists():
            bad.append(f"missing {rel}")
        elif p.stat().st_size != size or _sha(p) != sha:
            bad.append(f"changed {rel}")
    return bad


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "write":
        print(write(sys.argv[2:] or ["benchmarks"]), "files listed")
    else:
        b = check()
        print("\n".join(b) or "ok")
        sys.exit(1 if b else 0)
