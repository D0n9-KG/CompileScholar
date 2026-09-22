# -*- coding: utf-8 -*-
"""kb — single CLI entry ([project.scripts] kb = kb_compiler.cli:main).

B6 CLI convergence (2026-09-22): module __main__ blocks remain for
pipeline-internal use (build.py invokes them directly); this entry is the
user-facing surface. Subcommands:

  kb build  [stages...] [--force a,b] [--dry-run] [--exp NAME]   -> build.py
  kb list   [--exp NAME]                                           -> stage table
  kb test                                                          -> pytest
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "build":
        p = subprocess.run([sys.executable, str(REPO / "build.py")] + rest,
                           cwd=str(REPO))
        return p.returncode
    if cmd == "list":
        p = subprocess.run([sys.executable, str(REPO / "build.py"), "--list"] + rest,
                           cwd=str(REPO))
        return p.returncode
    if cmd == "test":
        env = dict(os.environ)
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        env.setdefault("PYTHONIOENCODING", "utf-8")
        p = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                           cwd=str(REPO), env=env)
        return p.returncode
    print(f"unknown subcommand: {cmd}\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
