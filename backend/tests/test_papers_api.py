"""Tests for papers API endpoints (P2-16)."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.api.routers.papers import (
    _canonical_dir_for_paper_id,
    _doi_sanitized,
    _safe_rel,
)


# ── _doi_sanitized ──

def test_doi_sanitized_basic():
    assert _doi_sanitized("10.1000/abc") == "10.1000_abc"

def test_doi_sanitized_strips_and_lowercases():
    assert _doi_sanitized("  10.ABC/XYZ  ") == "10.abc_xyz"

def test_doi_sanitized_special_chars():
    assert _doi_sanitized("10.1000/a(b)c") == "10.1000_a_b_c"


# ── _safe_rel ──

def test_safe_rel_normal():
    assert _safe_rel("foo/bar.png") == "foo/bar.png"

def test_safe_rel_backslash():
    assert _safe_rel("foo\\bar.png") == "foo/bar.png"

def test_safe_rel_rejects_absolute():
    with pytest.raises(ValueError):
        _safe_rel("/etc/passwd")

def test_safe_rel_rejects_dotdot():
    with pytest.raises(ValueError):
        _safe_rel("../secret")

def test_safe_rel_rejects_empty():
    with pytest.raises(ValueError):
        _safe_rel("")

def test_safe_rel_rejects_drive_letter():
    with pytest.raises(ValueError):
        _safe_rel("C:/Windows")

def test_safe_rel_rejects_special_chars():
    with pytest.raises(ValueError):
        _safe_rel("foo bar.png")


# ── _canonical_dir_for_paper_id ──

def test_canonical_dir_rejects_non_doi():
    with pytest.raises(FileNotFoundError, match="Only DOI"):
        _canonical_dir_for_paper_id("sha256:abc123")

def test_canonical_dir_not_found():
    with pytest.raises(FileNotFoundError, match="not found"):
        _canonical_dir_for_paper_id("doi:10.9999/nonexistent")


# ── get_paper_content endpoint logic ──

def test_get_paper_content_finds_paper_md():
    """Content endpoint should find and return paper.md."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paper_dir = Path(tmpdir) / "papers" / "doi" / "10.1000_test"
        paper_dir.mkdir(parents=True)
        md_file = paper_dir / "paper.md"
        md_file.write_text("# Test Paper\n\nContent here.", encoding="utf-8")

        with patch("app.api.routers.papers._canonical_dir_for_paper_id", return_value=paper_dir):
            from app.api.routers.papers import get_paper_content
            resp = get_paper_content("doi:10.1000/test")
            assert resp.status_code == 200
            assert b"# Test Paper" in resp.body


def test_get_paper_content_fallback_to_source_md():
    """Should fall back to source.md if paper.md doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paper_dir = Path(tmpdir) / "papers" / "doi" / "10.1000_test"
        paper_dir.mkdir(parents=True)
        md_file = paper_dir / "source.md"
        md_file.write_text("# Source", encoding="utf-8")

        with patch("app.api.routers.papers._canonical_dir_for_paper_id", return_value=paper_dir):
            from app.api.routers.papers import get_paper_content
            resp = get_paper_content("doi:10.1000/test")
            assert resp.status_code == 200
            assert b"# Source" in resp.body


def test_get_paper_content_no_md_file():
    """Should raise 404 when no markdown file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        paper_dir = Path(tmpdir) / "papers" / "doi" / "10.1000_test"
        paper_dir.mkdir(parents=True)

        with patch("app.api.routers.papers._canonical_dir_for_paper_id", return_value=paper_dir):
            from fastapi import HTTPException
            from app.api.routers.papers import get_paper_content
            with pytest.raises(HTTPException) as exc_info:
                get_paper_content("doi:10.1000/test")
            assert exc_info.value.status_code == 404
