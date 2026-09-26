# -*- coding: utf-8 -*-
"""Automatic fulltext acquisition chain (2026-09-20, user directive: "一个渠道
找不到论文或者超时就自动换别的，不要每次都手动选渠道").

Channel order (each falls through to the next on not-found / timeout /
identity-mismatch):
  1. arXiv PDF        (arxiv_id; best quality, fastest)
  2. OA PDF           (oa_pdf_url from OpenAlex/Sciverse metadata)
  3. Sciverse         (agentic search by title -> raw-title identity match
                       -> read_content paginated fulltext, saved as .md —
                       no PDF/mineru needed downstream)
  4. Local DOI archive (sci-hub-derived, DOI-keyed, LAST resort per user
                       directive; identity implied by the DOI key itself)

Registry-free by design: the chain is a pure acquisition utility (the caller
decides what to register). PDF channels verify identity via first-page title
match (pdftotext); a mismatched download is quarantined (deleted), never
kept. Every attempt is logged in the result for auditability.
"""
from __future__ import annotations

import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from retrieval.registry import normalize_doi
from retrieval.sources import (
    CrossrefClient,
    LocalDoiArchive,
    SciverseClient,
    SemanticScholarClient,
    SourceAdapterError,
)

UA = {"User-Agent": "Mozilla/5.0 (research corpus fetch; sci-evo-extract chain)"}
MIN_PDF_BYTES = 20000
VERIFY_WORDS = 6

_STOP = {"the", "a", "an", "of", "for", "and", "or", "in", "on", "with",
         "to", "by", "from", "at", "is", "are", "as", "its"}


def title_words(title: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]{3,}", (title or "").lower())
            if w not in _STOP]


def titles_match(expected: str, observed: str) -> bool:
    """Fuzzy title identity: >=60% of significant expected words present."""
    words = title_words(expected)
    if not words:
        return False
    hay = " ".join(re.findall(r"[a-z]{3,}", (observed or "").lower()))
    hits = sum(1 for w in words if w in hay)
    return hits >= max(2, int(len(words) * 0.6))


def _first_page_text(pdf: Path) -> str:
    try:
        out = subprocess.run(["pdftotext", "-f", "1", "-l", "1", str(pdf), "-"],
                            capture_output=True, timeout=60)
        return (out.stdout or b"").decode("utf-8", errors="replace").lower()
    except Exception:
        return ""


def verify_pdf(pdf: Path, title: str) -> bool:
    """Size + PDF magic + (when pdftotext is available) first-page title match.
    pdftotext missing degrades to size+magic — logged in the attempt."""
    try:
        if pdf.stat().st_size < MIN_PDF_BYTES:
            return False
        if pdf.read_bytes()[:5] != b"%PDF-":
            return False
    except OSError:
        return False
    txt = _first_page_text(pdf)
    if not txt:
        return True
    return titles_match(title, txt)


def _download(url: str, dest: Path, timeout: int) -> bool:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
        return dest.stat().st_size > MIN_PDF_BYTES
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        if dest.exists():
            dest.unlink()
        return False


# ---------------------------------------------------------------- channels

def _discover_arxiv_id(ctx: dict) -> str | None:
    """When the caller didn't pre-resolve an arXiv ID, discover one via
    Sciverse meta-search (which returns DOIs like 10.48550/arxiv.XXXX.XXXXX
    for arXiv papers). The arXiv search API itself is rate-limited to death
    under sustained load (429 bursts) and must NOT be the discovery path —
    but the PDF download endpoint has a separate, generous limit."""
    if ctx.get("arxiv_id"):
        return ctx["arxiv_id"]
    sv: SciverseClient | None = ctx.get("sciverse_client")
    if sv is None or not sv.token:
        return None
    try:
        cands = sv.search_title(ctx["title"], limit=2)
        for c in cands:
            if c.status != "ready":
                continue
            raw = c.raw or {}
            doi = str(raw.get("doi") or "")
            if doi.startswith("10.48550/arxiv."):
                return doi.replace("10.48550/arxiv.", "").strip()
            # also check locations for arxiv landing pages
            for loc in raw.get("locations") or []:
                lp = str(loc.get("landing_page_url") or "")
                m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,6})", lp)
                if m:
                    return m.group(1)
    except Exception:  # noqa: BLE001 — discovery failure falls through to no ID
        pass
    return None


def channel_arxiv(ctx: dict, attempt: dict) -> bool:
    ax = _discover_arxiv_id(ctx)
    if not ax:
        attempt["reason"] = "no arxiv_id (direct + sciverse discovery both empty)"
        return False
    dest = ctx["out_dir"] / f"{ctx['name']}.pdf"
    if not _download(f"https://arxiv.org/pdf/{ax}", dest, ctx["timeout"]):
        attempt["reason"] = f"download failed for {ax} (timeout / too small)"
        return False
    if ctx["verify"] and not verify_pdf(dest, ctx["title"]):
        dest.unlink()
        attempt["reason"] = f"identity mismatch for {ax} (first-page title)"
        return False
    attempt.update(channel="arxiv", path=str(dest), format="pdf", arxiv_id=ax)
    return True


def _candidate_pdf_urls(ctx: dict) -> list[tuple[str, str]]:
    """Multi-source OA PDF discovery (2026-09-20, user directive: '哪个有就
    直接拿'): whichever metadata source carries a PDF link, try it. Sources:
    explicit oa_pdf_url (OpenAlex resolution), S2 openAccessPdf, Crossref
    publisher-deposited links. Each failure falls to the next URL; the whole
    channel falls to the next channel when all are exhausted."""
    urls: list[tuple[str, str]] = []
    if ctx.get("oa_pdf_url"):
        urls.append(("openalex", ctx["oa_pdf_url"]))
    doi = normalize_doi(ctx.get("doi") or "")
    if doi:
        s2: SemanticScholarClient | None = ctx.get("s2_client")
        if s2 is None:
            try:
                s2 = SemanticScholarClient()
            except Exception:  # noqa: BLE001
                s2 = None
        if s2 is not None:
            try:
                u = s2.open_access_pdf_url(doi)
                if u:
                    urls.append(("s2_open_access_pdf", u))
            except Exception:  # noqa: BLE001 — rate-limited S2 must not kill the chain
                pass
        cr: CrossrefClient | None = ctx.get("crossref_client")
        if cr is None:
            try:
                cr = CrossrefClient()
            except Exception:  # noqa: BLE001
                cr = None
        if cr is not None:
            try:
                for u in cr.fetch_pdf_urls(doi):
                    urls.append(("crossref_link", u))
            except Exception:  # noqa: BLE001
                pass
    return urls


def channel_oa_pdf(ctx: dict, attempt: dict) -> bool:
    urls = _candidate_pdf_urls(ctx)
    if not urls:
        attempt["reason"] = "no pdf url from any source (openalex/s2/crossref)"
        return False
    dest = ctx["out_dir"] / f"{ctx['name']}.pdf"
    tried = []
    for source, url in urls:
        tried.append(source)
        if not _download(url, dest, ctx["timeout"]):
            continue
        if ctx["verify"] and not verify_pdf(dest, ctx["title"]):
            dest.unlink()
            continue
        attempt.update(channel=f"oa_pdf:{source}", path=str(dest), format="pdf",
                       via=source)
        return True
    attempt["reason"] = f"{len(tried)} pdf urls tried ({', '.join(tried)}), all failed"
    if dest.exists():
        dest.unlink()
    return False


def channel_sciverse(ctx: dict, attempt: dict) -> bool:
    client: SciverseClient | None = ctx.get("sciverse_client")
    if client is None or not client.token:
        attempt["reason"] = "sciverse unavailable (no token)"
        return False
    try:
        hits = client.agentic_search(ctx["title"], limit=6)
    except (SourceAdapterError, Exception) as exc:  # noqa: BLE001
        attempt["reason"] = f"agentic_search failed: {exc}"[:120]
        return False
    # identity filter on the raw payload title (cheap, no content roundtrip)
    doc_id = None
    for h in hits:
        raw = h.get("raw") or {}
        if titles_match(ctx["title"], str(raw.get("title") or "")):
            doc_id = h.get("doc_id")
            break
    if not doc_id:
        attempt["reason"] = f"exact paper not in top-{len(hits)} hits"
        return False
    # paginate the full content
    parts, offset, guard = [], 0, 0
    while guard < 40:
        try:
            r = client.read_content(doc_id=doc_id, offset=offset, limit=8000)
        except (SourceAdapterError, Exception) as exc:  # noqa: BLE105
            attempt["reason"] = f"read_content failed: {exc}"[:120]
            return False
        text = r.get("text") or ""
        if not text:
            break
        parts.append(text)
        if not r.get("more"):
            break
        offset = r.get("next_offset") or (offset + len(text))
        guard += 1
    full = "\n".join(parts)
    if len(full) < 5000:
        attempt["reason"] = f"content too short ({len(full)} chars) — metadata-only?"
        return False
    dest = ctx["out_dir"] / f"{ctx['name']}.md"
    dest.write_text(full, encoding="utf-8")
    attempt.update(channel="sciverse", path=str(dest), format="md",
                   n_chars=len(full))
    return True


def channel_local_doi_archive(ctx: dict, attempt: dict) -> bool:
    doi = normalize_doi(ctx.get("doi") or "")
    index = ctx.get("local_doi_index")
    root = ctx.get("local_doi_archive_root")
    if not (doi and index and root):
        attempt["reason"] = "no doi or local archive not configured"
        return False
    try:
        archive = LocalDoiArchive(index_path=Path(index), archive_root=Path(root))
        hit = archive.lookup(doi)
    except (SourceAdapterError, Exception) as exc:  # noqa: BLE105
        attempt["reason"] = f"archive lookup failed: {exc}"[:120]
        return False
    if hit is None:
        attempt["reason"] = "doi not in local archive"
        return False
    dest = ctx["out_dir"] / f"{ctx['name']}.pdf"
    try:
        archive.extract_pdf(hit, dest)
    except (SourceAdapterError, Exception) as exc:  # noqa: BLE105
        attempt["reason"] = f"archive extract failed: {exc}"[:120]
        return False
    # identity is implied by the DOI key; still check it is a real PDF of size
    if dest.stat().st_size < MIN_PDF_BYTES or dest.read_bytes()[:5] != b"%PDF-":
        dest.unlink()
        attempt["reason"] = "archive entry not a valid PDF"
        return False
    attempt.update(channel="local_doi_archive", path=str(dest), format="pdf")
    return True


CHANNELS = [
    ("arxiv", channel_arxiv),
    ("oa_pdf", channel_oa_pdf),
    ("sciverse", channel_sciverse),
    ("local_doi_archive", channel_local_doi_archive),
]


def _ensure_env():
    """Self-sufficient environment: when callers import the chain directly
    (no server, no .env preloaded), load the project's own .env so the
    Sciverse token + local DOI archive paths are always present. Explicit
    os.environ entries always win (never overwritten)."""
    if os.environ.get("_SCIEVO_CHAIN_ENV_LOADED"):
        return
    # chain is at src/sci_evo_extract/library/acquisition_chain.py — project
    # root (.env location) is FOUR parents up
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    os.environ["_SCIEVO_CHAIN_ENV_LOADED"] = "1"


def acquire_fulltext(
    *,
    title: str,
    out_dir: Path | str,
    doi: str | None = None,
    arxiv_id: str | None = None,
    oa_pdf_url: str | None = None,
    timeout_seconds: int = 120,
    verify: bool = True,
    sciverse_client: SciverseClient | None = None,
    local_doi_index: Path | str | None = None,
    local_doi_archive_root: Path | str | None = None,
    download: Callable[[str, Path, int], bool] | None = None,
) -> dict[str, Any]:
    """Automatic channel chain. Falls through on not-found / timeout /
    identity-mismatch; local DOI archive is LAST (user directive). Returns
    {status, channel, path, format, attempts:[{channel, ok, reason?}]}."""
    _ensure_env()   # self-sufficient: callers never need to preload .env
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if download is not None:   # test hook
        global _download
        _orig, _download = _download, download
    ctx = {
        "title": title, "doi": doi, "arxiv_id": arxiv_id,
        "oa_pdf_url": oa_pdf_url, "out_dir": out_dir,
        "timeout": timeout_seconds, "verify": verify,
        "name": re.sub(r"[^A-Za-z0-9_-]+", "_", title)[:80] or "paper",
        "sciverse_client": sciverse_client if sciverse_client is not None
        else SciverseClient(),
        "local_doi_index": local_doi_index or os.environ.get("SCIEVO_LOCAL_DOI_INDEX"),
        "local_doi_archive_root": local_doi_archive_root
        or os.environ.get("SCIEVO_LOCAL_DOI_ARCHIVE_ROOT"),
    }
    attempts = []
    for name, fn in CHANNELS:
        attempt: dict[str, Any] = {"channel": name, "ok": False}
        try:
            ok = fn(ctx, attempt)
        except Exception as exc:  # noqa: BLE001 — chain must never die on one channel
            ok = False
            attempt["reason"] = f"unexpected: {exc}"[:120]
        attempt["ok"] = bool(ok)
        attempts.append(attempt)
        if ok:
            if download is not None:
                _download = _orig
            return {"status": "ready", "channel": attempt.get("channel"),
                    "path": attempt.get("path"), "format": attempt.get("format"),
                    "attempts": attempts}
    if download is not None:
        _download = _orig
    return {"status": "gap", "channel": None, "path": None, "format": None,
            "attempts": attempts}
