# -*- coding: utf-8 -*-
"""Remove `sys.path.insert(0, r"C:\\Users\\...\\CompileScholar\\src")` lines from tests (the package is installed)."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PAT = re.compile(r'^\s*sys\.path\.insert\(0,\s*r?["\'][A-Za-z]:[\\/]Users[\\/][^"\']*CompileScholar[\\/]src["\']\)\s*\n', re.M)
for p in sorted((REPO / "tests").glob("*.py")):
    t = p.read_text(encoding="utf-8")
    n = len(PAT.findall(t))
    if n:
        p.write_text(PAT.sub("", t), encoding="utf-8")
        print(p.name, "removed", n)
left = [p.name for p in (REPO / "tests").glob("*.py") if re.search(r"[A-Za-z]:[\\/]Users", p.read_text(encoding="utf-8"))]
print("files still containing an absolute user path:", left)
