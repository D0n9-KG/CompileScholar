from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.settings import settings


router = APIRouter(prefix="/papers", tags=["papers"])


_SAFE_RELPATH = re.compile(r"^[A-Za-z0-9_.\-/]+$")


def _doi_sanitized(doi: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", doi.strip().lower())


def _canonical_dir_for_paper_id(paper_id: str) -> Path:
    if not paper_id.startswith("doi:"):
        raise FileNotFoundError("Only DOI papers have canonical image storage")
    doi = paper_id[4:]
    p = Path(__file__).resolve().parents[3] / settings.storage_dir / "papers" / "doi" / _doi_sanitized(doi)
    if not p.exists():
        raise FileNotFoundError(f"Canonical paper directory not found for {paper_id}")
    return p


def _safe_rel(rel: str) -> str:
    s = (rel or "").strip().replace("\\", "/")
    if not s or s.startswith("/") or ":" in s.split("/")[0]:
        raise ValueError("Invalid path")
    if not _SAFE_RELPATH.match(s):
        raise ValueError("Invalid path")
    parts = [p for p in s.split("/") if p not in {"", "."}]
    if any(p == ".." for p in parts):
        raise ValueError("Invalid path")
    return "/".join(parts)


@router.get("/{paper_id:path}/images/{rel_path:path}")
def get_paper_image(paper_id: str, rel_path: str):
    try:
        base = _canonical_dir_for_paper_id(paper_id)
        rel = _safe_rel(rel_path)
        p = (base / "images" / rel).resolve()
        # ensure under base/images
        root = (base / "images").resolve()
        p.relative_to(root)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Image not found: {rel}")
        mt, _ = mimetypes.guess_type(str(p))
        return FileResponse(str(p), media_type=mt or "application/octet-stream")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

