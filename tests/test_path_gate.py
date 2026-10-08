# -*- coding: utf-8 -*-
"""Path gate (INTEGRATED-SYSTEM-1005 v2 §10.3): the package reads nothing from the scratch area, hardcodes no machine
path, and reads the environment only in core/ and in the registered third-party injection points."""
import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "compilescholar"
SCRATCH = re.compile(r"\.research_tmp")
# a string constant whose value is a drive-letter path (C:\x, C:/x) or a UNC path (\\host\x, //host/x); checked on the
# parsed values (ast), so regex sources such as r"\\times" and code testing for a UNC prefix do not match
MACHINE = re.compile(r"^(?:[A-Za-z]:[\\/]+\w|[\\/]{2}[\w.\-]+[\\/][\w$.\-]+(?:[\\/]|$))")
# environment reads outside core/: credentials handed to third-party libraries that only read os.environ, harness
# subprocess environments, and the documented location overrides
ENV_ALLOWED = {
    "eval/cs2/judge.py", "eval/dsb.py", "eval/field.py", "baselines/harness/runner.py", "baselines/harness/proxy.py",
    "baselines/harness/mcp_server.py", "tools/mcp_server.py", "sources/sciverse.py", "sources/refgraph.py",
    "answer/pipeline.py", "cli.py", "llm/client.py",
    "eval/sc/runner.py",   # passes the environment through to the official evaluate.py subprocess
}


def _files():
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_scratch_area_reads():
    bad = [str(p.relative_to(SRC)) for p in _files() if SCRATCH.search(p.read_text(encoding="utf-8"))]
    assert not bad, bad


def test_no_machine_paths():
    bad = []
    for p in _files():
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and MACHINE.match(node.value):
                bad.append(f"{p.relative_to(SRC)}:{node.lineno}: {node.value[:40]!r}")
    assert not bad, bad


def test_environment_reads_are_registered():
    bad = []
    for p in _files():
        rel = str(p.relative_to(SRC)).replace("\\", "/")
        if rel.startswith("core/") or rel in ENV_ALLOWED:
            continue
        if re.search(r"os\.environ|os\.getenv", p.read_text(encoding="utf-8")):
            bad.append(rel)
    assert not bad, bad


def test_legacy_bench_is_gone():
    from compilescholar.core import paths
    assert not hasattr(paths, "legacy_bench")
