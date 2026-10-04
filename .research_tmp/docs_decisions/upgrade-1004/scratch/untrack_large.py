# -*- coding: utf-8 -*-
"""W6 S8: stop tracking every file > 5 MB outside results/ and legacy/ (files stay on disk; already in pushed history).
Their sha256 + size + location go to artifacts/MANIFEST.tsv, so the KB snapshot is still pinned by hash. The answer
KB itself (cs2/base_kb_v2/{papers,records,state_merged}.json + record_vecs) is listed there too regardless of size.
Dry run by default; --apply runs `git rm --cached` and writes the manifest."""
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
LIMIT = 5 << 20
KB = ".research_tmp/experiments/benchmarks/cs2/base_kb_v2/"
ALWAYS = [KB + n for n in ("papers.json", "records.json", "state_merged.json", "record_vecs.f32", "record_vecs.meta.json")]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


tree = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "-l", "HEAD"], capture_output=True, text=True).stdout
big = []
for line in tree.splitlines():
    meta, path = line.split("\t", 1)
    size = meta.split()[3]
    if size != "-" and int(size) > LIMIT and not path.startswith(("results/", "legacy/")):
        big.append(path)
rows = ["path\tsha256\tbytes\ttracked_before_20261004\tnote"]
for p in sorted(set(big) | set(ALWAYS)):
    f = REPO / p
    if not f.exists():
        continue
    note = "answer KB (frozen v9b input)" if p in ALWAYS else "intermediate / run output"
    rows.append(f"{p}\t{sha(f)}\t{f.stat().st_size}\t{'yes' if p in big else 'no'}\t{note}")
print(f"untrack {len(big)} files; manifest rows {len(rows) - 1}")
if "--apply" in sys.argv:
    (REPO / "artifacts").mkdir(exist_ok=True)
    (REPO / "artifacts" / "MANIFEST.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for i in range(0, len(big), 50):
        subprocess.run(["git", "-C", str(REPO), "rm", "-q", "--cached", "--", *big[i:i + 50]], check=True)
    print("done")
