# -*- coding: utf-8 -*-
"""Delete the local copy of cold-stored dirs whose NAS copy passed per-file sha256 verification (cold_storage.py)."""
import json
import os
import shutil
import stat
import sys


def _force(func, path, exc):
    """Read-only files (git pack .idx/.pack) block rmtree on Windows: clear the flag and retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
LOGS = r"\\192.168.199.138\Share400T\personal\jhd0n9\CompileScholar-cold-20261004\_logs"

for d in sys.argv[1:]:
    v = json.load(open(os.path.join(LOGS, "verify_" + d.replace("/", "_") + ".json")))
    if not (v["verified"] and not v["missing"] and not v["hash_mismatch"] and not v["size_mismatch"]):
        print(d, "NOT verified, kept")
        continue
    p = "\\\\?\\" + os.path.join(REPO, ".research_tmp", d.replace("/", os.sep))
    if os.path.isdir(p):
        shutil.rmtree(p, onexc=_force)
        print(f"deleted local {d}: {v['bytes'] / 1e9:.1f} GB, {v['files']} files (NAS copy verified)")
        with open(os.path.join(LOGS, "report.txt"), "a") as f:
            f.write(f"{d}: local deleted after verification\n")
