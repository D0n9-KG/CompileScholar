# -*- coding: utf-8 -*-
"""W6 S8 cold storage, long-path safe (\\\\?\\ prefixes): copy retired experiment dirs to the NAS with robocopy, verify
every file by size + sha256 against the source, and only with --delete remove the local copy of a verified dir.

Usage: python cold_storage.py [--delete] [dir ...]   (dirs relative to .research_tmp; default: the retired set)
Writes <NAS>/_logs/verify_<dir>.json and appends to <NAS>/_logs/report.txt.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp")
DST = Path(r"\\192.168.199.138\Share400T\personal\jhd0n9\CompileScholar-cold-20261004")
DEFAULT = ["experiments/archive", "experiments/baselines", "experiments/e2_need_gap", "experiments/stageB",
           "archive_oneoff", "benchmark-audit-0926"]


def lp(p: Path) -> str:
    s = str(p)
    if s.startswith("\\\\?\\"):
        return s
    return "\\\\?\\UNC\\" + s[2:] if s.startswith("\\\\") else "\\\\?\\" + s


def walk(root: Path) -> dict[str, int]:
    out = {}
    base = lp(root)
    for dp, _dns, fns in os.walk(base):
        for fn in fns:
            full = os.path.join(dp, fn)
            out[os.path.relpath(full, base)] = os.path.getsize(full)
    return out


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    delete = "--delete" in sys.argv
    dirs = [a for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT
    (DST / "_logs").mkdir(parents=True, exist_ok=True)
    for d in dirs:
        s, t = SRC / d, DST / d
        if not os.path.isdir(lp(s)):
            print(f"{d}: missing locally (already moved?)")
            continue
        log = DST / "_logs" / (d.replace("/", "_") + ".log")
        rc = subprocess.run(["robocopy", str(s), str(t), "/E", "/COPY:DAT", "/DCOPY:T", "/MT:16", "/R:2", "/W:2",
                             "/NFL", "/NDL", "/NP", f"/LOG:{log}"], capture_output=True).returncode
        sf, tf = walk(s), walk(t)
        missing = [k for k in sf if k not in tf]
        size_bad = [k for k in sf if k in tf and sf[k] != tf[k]]
        hash_bad = [k for k in sf if k in tf and sf[k] == tf[k] and sha(lp(s / k)) != sha(lp(t / k))]
        ok = rc < 8 and not missing and not size_bad and not hash_bad
        rec = {"dir": d, "robocopy_rc": rc, "files": len(sf), "bytes": sum(sf.values()), "missing": missing[:20],
               "size_mismatch": size_bad[:20], "hash_mismatch": hash_bad[:20], "verified": ok}
        json.dump(rec, open(DST / "_logs" / f"verify_{d.replace('/', '_')}.json", "w"), indent=1)
        line = (f"{d}: rc={rc} files={len(sf)} bytes={sum(sf.values())} missing={len(missing)} "
                f"size_mismatch={len(size_bad)} hash_mismatch={len(hash_bad)} -> {'VERIFIED' if ok else 'FAILED'}")
        if ok and delete:
            shutil.rmtree(lp(s))
            line += " ; local deleted"
        print(line, flush=True)
        with open(DST / "_logs" / "report.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
