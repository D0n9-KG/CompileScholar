# -*- coding: utf-8 -*-
"""密钥精确匹配扫描（只报路径与变量名，永不打印值）。

密钥集合 = .env 中长度 ≥16 的值 + 代码里硬编码在 *_TOKEN / *_KEY 赋值中的长字面量。
扫描范围：
  worktree  —— 工作区全部文件（跳过 .git、venv*、node_modules、>200 MB 的文件）
  unpushed  —— origin/main..main 引入的所有 blob
  pushed    —— origin/main 历史中的所有 blob（已公开）
用法：python secret_scan.py [worktree|unpushed|pushed ...]
"""
import os
import re
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
SKIP_DIRS = {".git", "node_modules", "__pycache__"}
MAX_BYTES = 200 * 1024 * 1024


def secrets() -> dict[str, str]:
    """name -> value。name 只用于报告。"""
    out = {}
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8", errors="replace"):
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", line)
        if m:
            v = m.group(2).strip().strip('"').strip("'")
            # 只收凭据类变量（模型名 / 端点 / 路径不是密钥）
            if (len(v) >= 16 and not v.startswith(("http://", "https://"))
                    and re.search(r"KEY|TOKEN|SECRET|PASSWORD|PASS|AUTH|DATABASE_URL", m.group(1))):
                out.setdefault(v, m.group(1))
    # 已知硬编码位置里的字面量（不在 .env 里的也要收进来）
    pat = re.compile(r"""([A-Z_]*(?:TOKEN|API_KEY|SECRET)[A-Z_]*)["']?\s*[,=:]\s*["']([A-Za-z0-9_\-\.]{16,})["']""")
    for rel in (".research_tmp/experiments/benchmarks/scholarqa_multi/promote_to_deep.py",
                ".research_tmp/experiments/benchmarks/_shared/tools/multi_closedbook_recall.py"):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            for m in pat.finditer(open(p, encoding="utf-8", errors="replace").read()):
                out.setdefault(m.group(2), m.group(1))
    return {name: v for v, name in out.items()}


def scan_bytes(data: bytes, sec: dict[str, bytes]) -> list[str]:
    return [n for n, v in sec.items() if v in data]


def worktree(sec):
    hits = []
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.lower().startswith(("venv", ".venv"))]
        for fn in fns:
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, ROOT).replace("\\", "/")
            if rel == ".env":
                continue
            try:
                if os.path.getsize(p) > MAX_BYTES:
                    continue
                with open(p, "rb") as f:
                    names = scan_bytes(f.read(), sec)
            except OSError:
                continue
            if names:
                hits.append((rel, names))
    return hits


def blobs(rev_range: list[str]) -> list[tuple[str, str]]:
    r = subprocess.run(["git", "-C", ROOT, "rev-list", "--objects", *rev_range], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    pairs = []
    for line in r.stdout.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def history(sec, rev_range):
    pairs = blobs(rev_range)
    sizes = {}
    chk = subprocess.run(["git", "-C", ROOT, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                         input="\n".join(o for o, _ in pairs), capture_output=True, text=True)
    for line in chk.stdout.splitlines():
        o, t, s = line.split()
        if t == "blob":
            sizes[o] = int(s)
    hits = {}
    proc = subprocess.Popen(["git", "-C", ROOT, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    skipped = 0
    for o, path in pairs:
        if o not in sizes:
            continue
        if sizes[o] > MAX_BYTES:
            skipped += 1
            continue
        proc.stdin.write((o + "\n").encode())
        proc.stdin.flush()
        header = proc.stdout.readline()
        n = int(header.split()[2])
        data = proc.stdout.read(n)
        proc.stdout.read(1)
        names = scan_bytes(data, sec)
        if names:
            hits.setdefault(path, set()).update(names)
    proc.stdin.close()
    proc.wait()
    return sorted((p, sorted(n)) for p, n in hits.items()), skipped, len(sizes)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sec = {k: v.encode() for k, v in secrets().items()}
    print(f"secrets loaded: {len(sec)} (names: {', '.join(sorted(sec))})")
    for mode in sys.argv[1:] or ["worktree", "unpushed", "pushed"]:
        if mode == "worktree":
            hits = worktree(sec)
            print(f"[worktree] {len(hits)} files")
        else:
            rng = ["origin/main..main"] if mode == "unpushed" else ["origin/main"]
            hits, skipped, nblob = history(sec, rng)
            print(f"[{mode}] blobs scanned {nblob - skipped}/{nblob} (skipped >200MB: {skipped}); paths with hits: {len(hits)}")
        for rel, names in hits:
            print(f"   {rel}  <- {', '.join(names)}")


if __name__ == "__main__":
    main()
