# -*- coding: utf-8 -*-
"""tools/precommit_check.py blocks large files, credential values / literals and absolute user paths."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("precommit_check", REPO / "tools" / "precommit_check.py")
PC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PC)

# fixtures are assembled at runtime so this file itself never matches the guard's patterns
FAKE_SECRET = "abcdefghij" + "klmnopqrstuvwx"
LITERAL_LINE = ("MINERU" + "_TOKEN = \"sk-" + "TrHhrfvplebi4Tgkyn00\"\n").encode()
ABS_LINE = ("sys.path.insert(0, r\"" + "C:" + "\\Users\\someone\\proj\\src\")\n").encode()
ALLOWED_LINE = ("API" + "_KEY = \"" + FAKE_SECRET + "z0\"  # " + "precommit: allow\n").encode()


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".env").write_text(f"SOME_API_KEY={FAKE_SECRET}\nPLAIN=ok\n", encoding="utf-8")
    monkeypatch.setattr(PC, "REPO", tmp_path)
    return tmp_path


def _stage(repo: Path, rel: str, data: bytes):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    subprocess.run(["git", "-C", str(repo), "add", "-f", rel], check=True)


def test_clean_file_passes(repo):
    _stage(repo, "src/pkg/a.py", b"x = 1\n")
    assert PC.check(PC.staged()) == []


def test_blocks_large_secret_literal_and_abs_path(repo):
    _stage(repo, "data/big.json", b"0" * (6 << 20))
    _stage(repo, "notes.md", b"key is " + FAKE_SECRET.encode())
    _stage(repo, "src/pkg/b.py", LITERAL_LINE)
    _stage(repo, "tests/c.py", ABS_LINE)
    msgs = "\n".join(PC.check(PC.staged()))
    assert "data/big.json" in msgs and "> 5 MB" in msgs
    assert "notes.md: contains the value of a credential" in msgs
    assert "src/pkg/b.py: credential-looking literal" in msgs
    assert "tests/c.py: absolute user path" in msgs
    assert FAKE_SECRET not in msgs


def test_allows_compressed_results_placeholders_and_marker(repo):
    """The marker exempts literal / path patterns, never a real credential value from .env."""
    _stage(repo, "results/run/x.json.gz", b"1" * (6 << 20))
    _stage(repo, "src/pkg/d.py", b'os.environ["OPENAI_API_KEY"] = "sk-placeholder"\nkey = secrets.get("LOCAL_API_KEY")\n')
    _stage(repo, "src/pkg/e.py", ALLOWED_LINE)
    assert PC.check(PC.staged()) == []
