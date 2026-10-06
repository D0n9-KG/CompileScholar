# -*- coding: utf-8 -*-
"""Pre-commit guard (W6 S10). Blocks a commit when a staged file
  1. is larger than 5 MB (except compressed frozen runs under results/),
  2. contains the value of any credential found in .env (exact match; values are never printed),
     or a credential-looking literal assignment (KEY/TOKEN/SECRET/PASSWORD = "<16+ chars>"),
  3. is Python/shell/YAML/TOML under src/, tests/, configs/ or tools/ and contains an absolute user path,
  4. is bytecode (*.pyc / __pycache__/) — `git add -f` bypasses .gitignore, so it is checked here too.
Checks 2 (literal) and 3 skip lines carrying the marker `precommit: allow` (intentional fixtures only).
Install:  python tools/precommit_check.py --install     (writes .git/hooks/pre-commit)
Run:      python tools/precommit_check.py               (checks the staged files)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIMIT = 5 << 20
LITERAL = re.compile(rb"""(?i)\b[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*\b["']?\s*[:=,]\s*["']([A-Za-z0-9_\-\.]{16,})["']""")
ABS_PATH = re.compile(rb"[A-Za-z]:[\\/]+Users[\\/]+[^\s\"']+")
ABS_PATH_SCOPE = ("src/", "tests/", "configs/", "tools/")
PLACEHOLDER = re.compile(rb"^(sk-placeholder|x+|your[_-].*|<.*>)$", re.I)
ENV_NAME = re.compile(rb"^[A-Z][A-Z0-9_]+$")  # an environment-variable NAME, not a value
ALLOW = b"precommit" + b": allow"


def _env_values() -> list[bytes]:
    p = REPO / ".env"
    vals = []
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", line)
            if m and re.search(r"KEY|TOKEN|SECRET|PASSWORD|PASS|AUTH|DATABASE_URL", m.group(1)):
                v = m.group(2).strip().strip('"').strip("'")
                if len(v) >= 12:
                    vals.append(v.encode())
    return vals


def staged() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
                         capture_output=True).stdout
    return [x.decode() for x in out.split(b"\0") if x]


def staged_blob(path: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f":{path}"], capture_output=True).stdout


def _unmarked(data: bytes) -> bytes:
    return b"\n".join(ln for ln in data.split(b"\n") if ALLOW not in ln)


def check(paths: list[str]) -> list[str]:
    secrets = _env_values()
    problems = []
    for p in paths:
        if p.endswith(".pyc") or "__pycache__/" in p:
            problems.append(f"{p}: bytecode must not be committed")
            continue
        data = staged_blob(p)
        if len(data) > LIMIT and not (p.startswith("results/") and p.endswith(".gz")):
            problems.append(f"{p}: {len(data) / 1048576:.1f} MB > 5 MB (put it under data/ and pin it in data/MANIFEST.tsv, not git)")
        if any(v in data for v in secrets):
            problems.append(f"{p}: contains the value of a credential from .env")
        body = _unmarked(data)
        for m in LITERAL.finditer(body):
            if not PLACEHOLDER.match(m.group(1)) and not ENV_NAME.match(m.group(1)):
                problems.append(f"{p}: credential-looking literal assignment")
                break
        if p.startswith(ABS_PATH_SCOPE) and p.endswith((".py", ".sh", ".yaml", ".yml", ".toml")) and ABS_PATH.search(body):
            problems.append(f"{p}: absolute user path (use compilescholar.core.paths)")
    return problems


def install():
    hook = REPO / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexec python tools/precommit_check.py\n", encoding="utf-8", newline="\n")
    print(f"installed {hook}")


if __name__ == "__main__":
    if "--install" in sys.argv:
        install()
        sys.exit(0)
    probs = check(staged())
    for x in probs:
        print("pre-commit:", x, file=sys.stderr)
    sys.exit(1 if probs else 0)
