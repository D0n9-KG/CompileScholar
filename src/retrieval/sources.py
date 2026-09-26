from __future__ import annotations

import os
import json
import re
import sqlite3
import time
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from retrieval.registry import normalize_doi


USER_AGENT = "sci-evo-extract/0.1"
ENV_TOKEN = object()

# OpenAlex polite pool (2026-08-27): a bare UA without mailto lands in the
# anonymous pool (strictest rate limits — we 429'd a practice run with 8-way
# parallel queries). OPENALEX_MAILTO puts us in the polite pool (~10x budget).
# Retry with exponential backoff on 429/5xx: OpenAlex rate limits are rolling,
# not quota — waiting recovers.
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "sci-evo-extract@example.com")
OPENALEX_MAX_RETRIES = int(os.environ.get("OPENALEX_MAX_RETRIES", "4"))


@dataclass(frozen=True)
class SourceCandidate:
    source_name: str
    query_kind: str
    status: str
    source_record_id: str | None = None
    normalized_doi: str | None = None
    title: str | None = None
    year: int | None = None
    venue: str | None = None
    candidate_score: float | None = None
    pdf_candidates: list[dict[str, Any]] = field(default_factory=list)
    license: str | None = None
    open_access_status: str | None = None
    source_url: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    error_summary: str | None = None


@dataclass(frozen=True)
class LocalDoiArchiveHit:
    normalized_doi: str
    zip_path: Path
    inner_path: str
    source_uri: str


class LocalDoiArchive:
    def __init__(self, *, index_path: Path, archive_root: Path):
        self.index_path = index_path
        self.archive_root = archive_root

    def lookup(self, doi: str) -> LocalDoiArchiveHit | None:
        normalized = normalize_doi(doi)
        if not normalized:
            return None
        if not self.index_path.exists():
            raise SourceAdapterError(f"本地 DOI index 不存在：{self.index_path}")
        with sqlite3.connect(self.index_path) as connection:
            connection.row_factory = sqlite3.Row
            hit = self._lookup_scimag_hit(connection, normalized) or self._lookup_generic_hit(
                connection, normalized
            )
        return hit

    def _build_hit(self, *, normalized: str, zip_value: str, inner_value: str) -> LocalDoiArchiveHit:
        zip_path = Path(zip_value)
        if not zip_path.is_absolute():
            zip_path = self.archive_root / zip_path
        return LocalDoiArchiveHit(
            normalized_doi=normalized,
            zip_path=zip_path,
            inner_path=str(inner_value),
            source_uri=f"local_doi_archive:{zip_path}!{inner_value}",
        )

    def extract_pdf(self, hit: LocalDoiArchiveHit | None, output_path: Path) -> Path:
        if hit is None:
            raise SourceAdapterError("本地 DOI archive 未命中。")
        if not hit.zip_path.exists():
            raise SourceAdapterError(f"本地 DOI archive zip 不存在：{hit.zip_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(hit.zip_path) as archive:
                with archive.open(hit.inner_path) as handle:
                    data = handle.read()
        except (KeyError, zipfile.BadZipFile) as exc:
            raise SourceAdapterError(f"本地 DOI archive 条目不可读：{hit.source_uri}") from exc
        if not data.startswith(b"%PDF-"):
            raise SourceAdapterError(f"本地 DOI archive 条目不是有效 PDF：{hit.source_uri}")
        output_path.write_bytes(data)
        return output_path

    def _lookup_scimag_hit(
        self, connection: sqlite3.Connection, normalized: str
    ) -> LocalDoiArchiveHit | None:
        tables = {
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not {"items", "archives"}.issubset(tables):
            return None
        item_columns = [row["name"] for row in connection.execute("PRAGMA table_info(items)")]
        archive_columns = [row["name"] for row in connection.execute("PRAGMA table_info(archives)")]
        if not {"doi_norm", "archive_id", "inner_path"}.issubset(set(item_columns)):
            return None
        if not {"id", "rel_path"}.issubset(set(archive_columns)):
            return None
        row = connection.execute(
            """
            SELECT archives.rel_path AS zip_path, items.inner_path AS inner_path
            FROM items
            JOIN archives ON archives.id = items.archive_id
            WHERE lower(items.doi_norm) = ?
            LIMIT 1
            """,
            (normalized,),
        ).fetchone()
        if row is None:
            return None
        return self._build_hit(
            normalized=normalized,
            zip_value=str(row["zip_path"]),
            inner_value=str(row["inner_path"]),
        )

    def _lookup_generic_hit(
        self, connection: sqlite3.Connection, normalized: str
    ) -> LocalDoiArchiveHit | None:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for table_row in table_rows:
            table = table_row["name"]
            columns = [row["name"] for row in connection.execute(f"PRAGMA table_info({table})")]
            doi_column = first_existing(columns, ["doi", "normalized_doi"])
            zip_column = first_existing(columns, ["zip_path", "archive_path"])
            inner_column = first_existing(columns, ["inner_path", "file_path"])
            if not doi_column or not zip_column or not inner_column:
                continue
            row = connection.execute(
                f"SELECT * FROM {table} WHERE lower({doi_column}) = ? LIMIT 1",
                (normalized,),
            ).fetchone()
            if row is not None:
                zip_value = row["zip_path"] if "zip_path" in row.keys() else row["archive_path"]
                inner_value = row["inner_path"] if "inner_path" in row.keys() else row["file_path"]
                return self._build_hit(
                    normalized=normalized,
                    zip_value=str(zip_value),
                    inner_value=str(inner_value),
                )
        return None


class OpenAlexClient:
    def __init__(
        self,
        *,
        get_json: Callable[[str, int], object] | None = None,
        timeout_seconds: int = 30,
    ):
        self.get_json = get_json or request_json
        self.timeout_seconds = timeout_seconds

    def fetch_by_doi(self, doi: str) -> SourceCandidate:
        normalized = normalize_doi(doi)
        if not normalized:
            return failed_candidate("openalex", "doi", "invalid DOI")
        url = f"https://api.openalex.org/works/https://doi.org/{quote(normalized, safe='')}"
        try:
            payload = self.get_json(url, self.timeout_seconds)
            if not isinstance(payload, dict):
                raise SourceAdapterError("OpenAlex response is not a JSON object")
            return openalex_candidate(payload, "doi", 1.0)
        except SourceAdapterError as exc:
            return failed_candidate("openalex", "doi", str(exc), normalized)

    def search_title(self, title: str, *, limit: int = 5) -> list[SourceCandidate]:
        query = title.strip()
        if not query:
            return []
        url = "https://api.openalex.org/works?" + urlencode({"search": query, "per-page": str(limit)})
        try:
            payload = self.get_json(url, self.timeout_seconds)
            if not isinstance(payload, dict):
                raise SourceAdapterError("OpenAlex response is not a JSON object")
            results = payload.get("results") or []
            if not isinstance(results, list):
                return []
            return [
                openalex_candidate(item, "title", title_similarity(query, str(item.get("display_name") or "")))
                for item in results
                if isinstance(item, dict)
            ]
        except SourceAdapterError as exc:
            return [failed_candidate("openalex", "title", str(exc))]

    def fetch_work_raw(self, doi: str) -> dict[str, Any] | None:
        """Fetch the full OpenAlex work payload for a DOI (includes
        referenced_works, cited_by_count, cited_by_api_url). Returns the raw
        dict or None on failure."""
        normalized = normalize_doi(doi)
        if not normalized:
            return None
        url = f"https://api.openalex.org/works/https://doi.org/{quote(normalized, safe='')}"
        try:
            payload = self.get_json(url, self.timeout_seconds)
            if isinstance(payload, dict):
                return payload
        except SourceAdapterError:
            return None
        return None

    def fetch_work_references(self, doi: str, *, resolve_metadata: bool = True) -> list[dict[str, Any]]:
        """Fetch this paper's outgoing references (papers it cites).

        Returns items shaped for registry.register_references:
          {ref_doi?, ref_title?, ref_year?, ref_openalex_id, source_ref_id, raw}
        If resolve_metadata=True, batch-resolves each W-id to doi/title/year
        (extra API calls); if False, only returns bare openalex ids."""
        payload = self.fetch_work_raw(doi)
        if not payload:
            return []
        refs = extract_openalex_references(payload)
        if not resolve_metadata or not refs:
            return refs
        meta = fetch_openalex_reference_metadata(
            [r["ref_openalex_id"] for r in refs if r.get("ref_openalex_id")],
            self.get_json, self.timeout_seconds,
        )
        resolved: list[dict[str, Any]] = []
        for r in refs:
            oid = r.get("ref_openalex_id")
            m = meta.get(oid) if oid else None
            if m:
                # merge: keep raw but fill doi/title/year from batch resolve
                merged = dict(r)
                merged.update({k: v for k, v in m.items() if v is not None and k != "raw"})
                merged["raw"] = {**(r.get("raw") or {}), **(m.get("raw") or {})}
                resolved.append(merged)
            else:
                resolved.append(r)
        return resolved


S2_API_KEY_ENV = "S2_API_KEY"
S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_MAX_RETRIES = int(os.environ.get("S2_MAX_RETRIES", "5"))
S2_PAGE_LIMIT = 100
S2_MAX_TOTAL_DEFAULT = 1000
# Graph API fields (2026-09-20): top-level paper detail. externalIds carries
# DOI/ArXiv/MAG; citationCount is global; influentialCitationCount is S2's
# own influence signal (useful as a walk-side prior).
S2_PAPER_FIELDS = (
    "paperId,title,abstract,year,externalIds,venue,publicationDate,"
    "citationCount,referenceCount,influentialCitationCount,publicationTypes,"
    "openAccessPdf"
)
# Nested-paper fields for /references and /citations. contexts = the actual
# citation text spans; intents = S2's classification of the citation's
# purpose (methodology/background/result) — the citation-REASON signal we
# want for the broker's typed-relation walk.
S2_EDGE_FIELDS = (
    "paperId,title,year,externalIds,citationCount,"
    "influentialCitationCount,contexts,intents,isInfluential"
)


def s2_retry_after_seconds(exc: HTTPError) -> float | None:
    """Parse a Retry-After header into seconds (S2 sends plain seconds)."""
    value = exc.headers.get("Retry-After") if exc.headers else None
    if not value:
        return None
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return None


def s2_request_json(url: str, timeout_seconds: int, *, api_key: str | None) -> object:
    """S2 Graph API GET. Unlike OpenAlex (polite-pool mailto), S2 auth is an
    x-api-key header: without it we share the anonymous pool (429s are
    frequent and bursty); with it ~1 req/s dedicated. 429/5xx retries honor
    Retry-After when present, else exponential backoff."""
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key
    for attempt in range(max(0, S2_MAX_RETRIES) + 1):
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < S2_MAX_RETRIES:
                wait = s2_retry_after_seconds(exc) or min(2.0 ** attempt, 30.0)
                time.sleep(wait)
                continue
            raise SourceAdapterError(f"HTTP {exc.code} {exc.reason} for {url}") from exc
        except URLError as exc:
            if attempt < S2_MAX_RETRIES:
                time.sleep(min(2.0 ** attempt, 30.0))
                continue
            raise SourceAdapterError(f"Network error for {url}: {exc.reason}") from exc
        except Exception as exc:  # noqa: BLE001
            raise SourceAdapterError(f"S2 request failed for {url}: {exc}") from exc
    raise SourceAdapterError(f"S2 retries exhausted for {url}")


def s2_paper_id_param(paper_id: str) -> str:
    """Normalize a paper reference into the Graph API path form. Accepts a
    bare DOI, an arXiv id (with or without 'arXiv:' prefix and version
    suffix), a CorpusId, a prefixed form (DOI:/ArXiv:/CorpusId:), or a bare
    S2 paperId (returned as-is)."""
    normalized_doi = normalize_doi(paper_id)
    # normalize_doi only strips URL/prefix forms — it does not validate the
    # 10. directory, so an 'arXiv:...' id would pass through. Only route to
    # DOI: when it actually looks like one.
    if normalized_doi and normalized_doi.startswith("10."):
        return f"DOI:{quote(normalized_doi, safe='')}"
    text = paper_id.strip()
    lower = text.lower()
    if lower.startswith("doi:"):
        return f"DOI:{quote(text[4:].strip(), safe='')}"
    if lower.startswith("arxiv:") or lower.startswith("arxiv/"):
        return f"ArXiv:{text[6:].strip()}"
    if lower.startswith("corpusid:"):
        return f"CorpusId:{text[9:].strip()}"
    if re.match(r"^\d{4}\.\d{4,6}(v\d+)?$", text):
        return f"ArXiv:{text}"
    return quote(text, safe="")


class SemanticScholarClient:
    def __init__(
        self,
        *,
        get_json: Callable[[str, int], object] | None = None,
        api_key: str | None | object = ENV_TOKEN,
        timeout_seconds: int = 30,
    ):
        self.api_key = os.environ.get(S2_API_KEY_ENV) if api_key is ENV_TOKEN else api_key
        if get_json is None:
            bound_key = self.api_key

            def get_json(url: str, timeout: int) -> object:
                return s2_request_json(url, timeout, api_key=bound_key)

        self.get_json = get_json
        self.timeout_seconds = timeout_seconds

    def fetch_by_doi(self, doi: str) -> SourceCandidate:
        normalized = normalize_doi(doi)
        if not normalized:
            return failed_candidate("s2", "doi", "invalid DOI")
        url = f"{S2_BASE_URL}/paper/DOI:{quote(normalized, safe='')}?fields={S2_PAPER_FIELDS}"
        try:
            payload = self.get_json(url, self.timeout_seconds)
            if not isinstance(payload, dict) or not payload.get("paperId"):
                raise SourceAdapterError("S2 response missing paperId")
            return s2_candidate(payload, "doi", 1.0)
        except SourceAdapterError as exc:
            return failed_candidate("s2", "doi", str(exc), normalized)

    def search_title(self, title: str, *, limit: int = 5) -> list[SourceCandidate]:
        query = title.strip()
        if not query:
            return []
        url = f"{S2_BASE_URL}/paper/search?" + urlencode(
            {"query": query, "limit": str(max(1, min(limit, 100))), "fields": S2_PAPER_FIELDS}
        )
        try:
            payload = self.get_json(url, self.timeout_seconds)
            if not isinstance(payload, dict):
                raise SourceAdapterError("S2 search response is not a JSON object")
            data = payload.get("data") or []
            if not isinstance(data, list):
                return []
            return [
                s2_candidate(item, "title", title_similarity(query, str(item.get("title") or "")))
                for item in data
                if isinstance(item, dict)
            ]
        except SourceAdapterError as exc:
            return [failed_candidate("s2", "title", str(exc))]

    def get_paper_raw(self, paper_id: str) -> dict[str, Any] | None:
        url = f"{S2_BASE_URL}/paper/{s2_paper_id_param(paper_id)}?fields={S2_PAPER_FIELDS}"
        try:
            payload = self.get_json(url, self.timeout_seconds)
            if isinstance(payload, dict):
                return payload
        except SourceAdapterError:
            return None
        return None

    def open_access_pdf_url(self, paper_id: str) -> str | None:
        """S2's own open-access PDF link for a paper (openAccessPdf.url) —
        feeds the multi-source OA channel of the acquisition chain."""
        payload = self.get_paper_raw(paper_id)
        oa = payload.get("openAccessPdf") if isinstance(payload, dict) else None
        if isinstance(oa, dict) and oa.get("url"):
            return str(oa["url"])
        return None

    def fetch_references(self, paper_id: str, *, max_total: int = S2_MAX_TOTAL_DEFAULT) -> list[dict[str, Any]]:
        """Outgoing references (papers this paper cites), shaped for
        registry.register_references: {ref_doi?, ref_title?, ref_year?,
        source_ref_id, raw}. Each raw edge carries contexts/intents/isInfluential
        — the citation-reason signal (S2's own intents classification plus the
        literal citation text spans)."""
        edges = self._fetch_edges("references", paper_id, max_total)
        items: list[dict[str, Any]] = []
        for edge in edges:
            paper = edge.get("citedPaper") or {}
            if not paper.get("paperId"):
                continue
            external = paper.get("externalIds") or {}
            items.append(
                {
                    "ref_doi": external.get("DOI"),
                    "ref_title": paper.get("title"),
                    "ref_year": paper.get("year"),
                    "ref_openalex_id": None,
                    "source_ref_id": paper["paperId"],
                    "raw": edge,
                }
            )
        return items

    def fetch_citations(self, paper_id: str, *, max_total: int = S2_MAX_TOTAL_DEFAULT) -> list[dict[str, Any]]:
        """Incoming citations (papers citing this paper), shaped for
        registry.register_citations: {citing_doi?, citing_title?,
        citing_year?, source_ref_id, raw}. This is the direction OpenAlex
        never wired — S2 is the primary provider for the broker's forward
        walk."""
        edges = self._fetch_edges("citations", paper_id, max_total)
        items: list[dict[str, Any]] = []
        for edge in edges:
            paper = edge.get("citingPaper") or {}
            if not paper.get("paperId"):
                continue
            external = paper.get("externalIds") or {}
            items.append(
                {
                    "citing_doi": external.get("DOI"),
                    "citing_title": paper.get("title"),
                    "citing_year": paper.get("year"),
                    "citing_openalex_id": None,
                    "source_ref_id": paper["paperId"],
                    "raw": edge,
                }
            )
        return items

    def _fetch_edges(self, kind: str, paper_id: str, max_total: int) -> list[dict[str, Any]]:
        """Paginate /paper/{id}/{references|citations}. The API caps a page
        at 100 and returns `next` (offset) + `total`; we loop while a next
        offset exists and we are under max_total."""
        path = s2_paper_id_param(paper_id)
        edges: list[dict[str, Any]] = []
        offset = 0
        while offset is not None and len(edges) < max_total:
            url = (
                f"{S2_BASE_URL}/paper/{path}/{kind}?"
                + urlencode(
                    {
                        "fields": S2_EDGE_FIELDS,
                        "limit": str(S2_PAGE_LIMIT),
                        "offset": str(offset),
                    }
                )
            )
            try:
                payload = self.get_json(url, self.timeout_seconds)
            except SourceAdapterError as exc:
                if edges:
                    break  # keep what we have; partial beats nothing
                raise SourceAdapterError(f"S2 {kind} fetch failed: {exc}") from exc
            if not isinstance(payload, dict):
                break
            data = payload.get("data") or []
            if not isinstance(data, list) or not data:
                break
            edges.extend(item for item in data if isinstance(item, dict))
            next_offset = payload.get("next")
            offset = int(next_offset) if next_offset is not None else None
            if payload.get("total") is not None and int(payload.get("total") or 0) <= len(edges):
                break
        return edges[:max_total]


ARXIV_API_BASE = "https://export.arxiv.org/api/query"
# arXiv asks for ~1 request / 3 seconds; a sustained-load test on the
# Multi corpus run burst-429'd exactly here. Single-query scenes are safe
# WITH spacing — enforced below at class level (shared across instances).
ARXIV_MIN_INTERVAL_S = float(os.environ.get("ARXIV_MIN_INTERVAL_S", "3.0"))
_arxiv_last_request_ts = 0.0


def _arxiv_polite_wait() -> None:
    global _arxiv_last_request_ts
    now = time.time()
    wait = _arxiv_last_request_ts + ARXIV_MIN_INTERVAL_S - now
    if wait > 0:
        time.sleep(wait)
    _arxiv_last_request_ts = time.time()


def arxiv_candidate_from_atom_entry(entry: dict[str, Any], query_kind: str, score: float) -> SourceCandidate:
    arxiv_id = str(entry.get("arxiv_id") or "")
    doi = normalize_doi(str(entry.get("doi"))) if entry.get("doi") else None
    return SourceCandidate(
        source_name="arxiv",
        query_kind=query_kind,
        status="ready",
        source_record_id=arxiv_id or None,
        normalized_doi=doi,
        title=(str(entry.get("title")).strip() or None)
        if entry.get("title") else None,
        year=int(entry["year"]) if entry.get("year") else None,
        source_url=(f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None),
        candidate_score=score,
        pdf_candidates=[{"url": f"https://arxiv.org/pdf/{arxiv_id}",
                         "source": "arxiv", "license": None}]
        if arxiv_id else [],
        raw={k: v for k, v in entry.items() if k != "doi"},
    )


def _parse_arxiv_atom(xml_text: str) -> list[dict[str, Any]]:
    """Minimal Atom parse: id/title/published/doi/arxiv_id per entry.
    Namespace-agnostic (tag-localname matching) — avoids pulling in an XML
    dependency for one endpoint."""
    import xml.etree.ElementTree as ET

    entries = []
    for entry in ET.fromstring(xml_text).iter():
        if not entry.tag.endswith("}entry") and entry.tag != "entry":
            continue
        item: dict[str, Any] = {"authors": []}
        for child in entry:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "id":
                raw_id = (child.text or "").strip()
                # http://arxiv.org/abs/2401.12345v2 -> 2401.12345v2
                item["arxiv_id"] = raw_id.rsplit("/abs/", 1)[-1] or None
            elif tag == "title":
                item["title"] = " ".join((child.text or "").split())
            elif tag == "published":
                item["year"] = int((child.text or "")[:4] or 0) or None
            elif tag == "author":
                # <author><name>...</name></author>
                for sub in child:
                    if sub.tag.rsplit("}", 1)[-1] == "name":
                        item["authors"].append((sub.text or "").strip())
            elif tag == "arxiv_doi" or (tag == "doi" and child.text):
                item["doi"] = (child.text or "").strip()
        entries.append(item)
    return entries


class ArxivClient:
    """arXiv API search client (A2, 2026-09-24).

    Deliberately absent during the batch corpus campaign (sustained load
    429'd to death — noted in the sources header). Single-query scenes get
    it back with polite spacing: 1 request / 3s shared across instances,
    exponential backoff on 429/5xx. Returns SourceCandidate rows shaped
    like the other discovery sources, with pdf_candidates pointing at
    arxiv.org/pdf/<id> (the acquisition chain's preferred channel when an
    arxiv id is known)."""

    def __init__(
        self,
        *,
        fetch_text: Callable[[str, int], str] | None = None,
        timeout_seconds: int = 30,
        max_retries: int = 2,
    ):
        self.fetch_text = fetch_text or self._fetch_text_default
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @staticmethod
    def _fetch_text_default(url: str, timeout_seconds: int) -> str:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=timeout_seconds) as resp:
            return resp.read().decode("utf-8", errors="replace")

    # ETREE_UNAVAILABLE is a sentinel for import-time xml.etree failures —
    # stdlib is always present; kept for except-clause symmetry only.
    ETREE_UNAVAILABLE = RuntimeError

    def search(self, query: str, *, limit: int = 10) -> list[SourceCandidate]:
        query_text = (query or "").strip()
        if not query_text:
            return []
        # quote the query for search_query=all:"..." — quotes group terms.
        # safe=":" keeps the field separator literal: arXiv rejects %3A
        # with HTTP 406 (measured 2026-09-24; %22 quotes are fine).
        url = ARXIV_API_BASE + "?" + urlencode({
            "search_query": f'all:"{query_text}"',
            "start": "0",
            "max_results": str(limit),
        }, safe=":")
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                _arxiv_polite_wait()
                xml_text = self.fetch_text(url, self.timeout_seconds)
                entries = _parse_arxiv_atom(xml_text)
                return [
                    arxiv_candidate_from_atom_entry(e, "search",
                                                    title_similarity(query_text, str(e.get("title") or "")))
                    for e in entries
                ]
            except HTTPError as exc:
                last_error = f"HTTP {exc.code}"
                # 406 = arXiv's throttle signal (docs: "excessive use...
                # temporary ban... HTTP 406"), NOT an encoding error —
                # measured 2026-09-24: identical URL 200s, then 406s for
                # minutes after a burst. Longer backoff than 429: the ban
                # window is minutes, we just avoid burning the retries.
                if exc.code in (429, 500, 502, 503) and attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                if exc.code == 406 and attempt < self.max_retries:
                    time.sleep(10 * (attempt + 1))
                    continue
            except URLError as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
            except Exception as exc:  # parse errors etc.
                last_error = f"{type(exc).__name__}: {exc}"
                break
        return [failed_candidate("arxiv", "search", last_error or "request failed")]


class CrossrefClient:
    def __init__(
        self,
        *,
        get_json: Callable[[str, int], object] | None = None,
        timeout_seconds: int = 30,
    ):
        self.get_json = get_json or request_json
        self.timeout_seconds = timeout_seconds

    def fetch_by_doi(self, doi: str) -> SourceCandidate:
        normalized = normalize_doi(doi)
        if not normalized:
            return failed_candidate("crossref", "doi", "invalid DOI")
        url = f"https://api.crossref.org/works/{quote(normalized, safe='')}"
        try:
            payload = self.get_json(url, self.timeout_seconds)
            if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
                raise SourceAdapterError("Crossref response missing message object")
            return crossref_candidate(payload["message"], "doi", 1.0)
        except SourceAdapterError as exc:
            return failed_candidate("crossref", "doi", str(exc), normalized)

    def fetch_pdf_urls(self, doi: str) -> list[str]:
        """Publisher-deposited PDF links from the Crossref record (the
        `link` array) — feeds the multi-source OA channel of the chain."""
        cand = self.fetch_by_doi(doi)
        if cand.status != "ready":
            return []
        return [str(c.get("url")) for c in (cand.pdf_candidates or [])
                if c.get("url")]

    def search_title(self, title: str, *, limit: int = 5) -> list[SourceCandidate]:
        query = title.strip()
        if not query:
            return []
        url = "https://api.crossref.org/works?" + urlencode(
            {"query.bibliographic": query, "rows": str(limit)}
        )
        try:
            payload = self.get_json(url, self.timeout_seconds)
            if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
                raise SourceAdapterError("Crossref response missing message object")
            items = payload["message"].get("items") or []
            if not isinstance(items, list):
                return []
            return [
                crossref_candidate(item, "title", title_similarity(query, first_title(item)))
                for item in items
                if isinstance(item, dict)
            ]
        except SourceAdapterError as exc:
            return [failed_candidate("crossref", "title", str(exc))]


class SciverseClient:
    def __init__(
        self,
        *,
        token: str | None | object = ENV_TOKEN,
        request_json: Callable[..., object] | None = None,
        base_url: str = "https://api.sciverse.space",
        timeout_seconds: int = 30,
    ):
        self.token = os.environ.get("SCIVERSE_API_TOKEN") if token is ENV_TOKEN else token
        self.request_json = request_json or sciverse_request_json
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def fetch_by_doi(self, doi: str) -> SourceCandidate:
        normalized = normalize_doi(doi)
        if not normalized:
            return failed_candidate("sciverse", "doi", "invalid DOI")
        if not self.token:
            return blocked_candidate(
                "sciverse",
                "doi",
                "缺少 SCIVERSE_API_TOKEN，无法调用 Sciverse。",
                normalized,
            )
        payload = {
            "filters": [
                {"field": "doi", "operator": "FILTER_OP_EQ", "value": normalized}
            ],
            "page": 1,
            "page_size": 2,
        }
        try:
            response = self.request_json(
                "POST",
                "/meta-search",
                payload=payload,
                query=None,
                timeout_seconds=self.timeout_seconds,
            )
            results = sciverse_results(response)
            if not results:
                return failed_candidate("sciverse", "doi", "Sciverse 未返回 metadata 结果。", normalized)
            return sciverse_candidate(results[0], "doi", 1.0)
        except SourceAdapterError as exc:
            return failed_candidate("sciverse", "doi", str(exc), normalized)

    def search_title(self, title: str, *, limit: int = 5) -> list[SourceCandidate]:
        query_text = title.strip()
        if not query_text:
            return []
        if not self.token:
            return [
                blocked_candidate(
                    "sciverse",
                    "title",
                    "缺少 SCIVERSE_API_TOKEN，无法调用 Sciverse。",
                )
            ]
        payload = {"query": query_text, "page": 1, "page_size": limit}
        try:
            response = self.request_json(
                "POST",
                "/meta-search",
                payload=payload,
                query=None,
                timeout_seconds=self.timeout_seconds,
            )
            return [
                sciverse_candidate(
                    item,
                    "title",
                    sciverse_score(query_text, item),
                )
                for item in sciverse_results(response)[:limit]
            ]
        except SourceAdapterError as exc:
            return [failed_candidate("sciverse", "title", str(exc))]

    def agentic_search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        query_text = query.strip()
        if not query_text:
            return []
        if not self.token:
            raise SourceAdapterError("缺少 SCIVERSE_API_TOKEN，无法调用 Sciverse。")
        response = self.request_json(
            "POST",
            "/agentic-search",
            payload={"query": query_text, "page_size": limit},
            query=None,
            timeout_seconds=self.timeout_seconds,
        )
        if not isinstance(response, dict):
            raise SourceAdapterError("Sciverse agentic-search response is not a JSON object")
        hits = response.get("hits") or []
        if not isinstance(hits, list):
            return []
        return [
            sciverse_hit(hit, query_text)
            for hit in hits[:limit]
            if isinstance(hit, dict)
        ]

    def semantic_search(self, query: str, *, limit: int = 10) -> list[SourceCandidate]:
        """Semantic full-text channel over /agentic-search, shaped as
        SourceCandidate rows for the tiered discovery path. Unlike
        search_title (keyword metadata match), this reaches papers whose
        vocabulary differs from the query (the measured A6 vocabulary-gap
        case: 'glycosylation' query surfaces the protein-corona gold paper
        whose TITLE matches no query term). No token -> blocked candidate
        (circuit records a failure and the tier degrades), never raises."""
        query_text = (query or "").strip()
        if not query_text:
            return []
        if not self.token:
            return [blocked_candidate(
                "sciverse-semantic", "semantic",
                "缺少 SCIVERSE_API_TOKEN，无法调用 Sciverse 语义检索。")]
        try:
            response = self.request_json(
                "POST", "/agentic-search",
                payload={"query": query_text, "page_size": limit},
                query=None, timeout_seconds=self.timeout_seconds,
            )
            if not isinstance(response, dict):
                raise SourceAdapterError(
                    "Sciverse agentic-search response is not a JSON object")
            hits = response.get("hits") or []
            if not isinstance(hits, list):
                return []
            # doc-level dedup: consecutive chunks of one paper collapse to
            # the first (highest-scored) hit
            seen_docs: set[str] = set()
            out: list[SourceCandidate] = []
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                doc = str(hit.get("doc_id") or "")
                if doc and doc in seen_docs:
                    continue
                if doc:
                    seen_docs.add(doc)
                score = hit.get("score")
                out.append(sciverse_semantic_candidate(
                    hit, float(score) if isinstance(score, (int, float)) else 0.5))
                if len(out) >= limit:
                    break
            return out
        except SourceAdapterError as exc:
            return [failed_candidate("sciverse-semantic", "semantic", str(exc))]

    def read_content(
        self,
        *,
        doc_id: str,
        chunk_id: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        if not self.token:
            raise SourceAdapterError("缺少 SCIVERSE_API_TOKEN，无法调用 Sciverse。")
        if not doc_id:
            raise SourceAdapterError("Sciverse content 读取缺少 doc_id。")
        query: dict[str, object] = {"doc_id": doc_id}
        if chunk_id:
            query["chunk_id"] = chunk_id
        else:
            query["offset"] = offset or 0
            if limit is not None:
                query["limit"] = limit
        response = self.request_json(
            "GET",
            "/content",
            payload=None,
            query=query,
            timeout_seconds=self.timeout_seconds,
        )
        if not isinstance(response, dict):
            raise SourceAdapterError("Sciverse content response is not a JSON object")
        return response

    def resource_status(self) -> dict[str, str]:
        return {
            "status": "unsupported",
            "reason": "Sciverse /resource 需要真实 file_name + type=image/file，当前未确认文件名来源，暂不默认拉取。",
        }


def extract_openalex_references(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract this paper's outgoing references from an OpenAlex work payload.

    OpenAlex `referenced_works` is a list of OpenAlex IDs (W...). We also pull
    any inline metadata (some payloads include a 'referenced_works' of strings
    only). Returns items shaped for register_references:
      {ref_openalex_id, source_ref_id, raw}
    Title/DOI of each reference requires a second fetch (batch filter) — done
    optionally via fetch_openalex_reference_metadata."""
    refs: list[dict[str, Any]] = []
    raw_refs = payload.get("referenced_works") or []
    if not isinstance(raw_refs, list):
        return refs
    for rid in raw_refs:
        if not isinstance(rid, str) or not rid:
            continue
        oa_id = rid.split("/")[-1] if "/" in rid else rid
        refs.append({
            "ref_openalex_id": oa_id,
            "source_ref_id": oa_id,
            "raw": {"openalex_url": rid},
        })
    return refs


def extract_crossref_references(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract outgoing references from a Crossref message payload.

    Crossref 'reference' entries often carry DOI/article-number/unstructured
    title+year. Returns items shaped for register_references:
      {ref_doi?, ref_title?, ref_year?, source_ref_id, raw}"""
    refs: list[dict[str, Any]] = []
    raw_refs = message.get("reference") or []
    if not isinstance(raw_refs, list):
        return refs
    for i, item in enumerate(raw_refs):
        if not isinstance(item, dict):
            continue
        ref_doi = item.get("DOI") or None
        ref_title = item.get("article-title") or item.get("unstructured") or None
        ref_year = item.get("year")
        try:
            ref_year = int(ref_year) if ref_year else None
        except (TypeError, ValueError):
            ref_year = None
        src_ref_id = ref_doi or item.get("key") or f"cr_{i}"
        refs.append({
            "ref_doi": ref_doi,
            "ref_title": ref_title,
            "ref_year": ref_year,
            "source_ref_id": src_ref_id,
            "raw": {k: v for k, v in item.items() if v is not None},
        })
    return refs


def fetch_openalex_reference_metadata(openalex_ids: list[str], get_json: Callable[[str, int], object], timeout_seconds: int = 30, batch: int = 50) -> dict[str, dict[str, Any]]:
    """Batch-resolve OpenAlex W-IDs to {W-id: {doi,title,year,openalex_id}}.

    Uses the filter=openalex:W1|W2|... form (50 per call). Returns a map keyed
    by the bare W-id (no URL prefix). Missing/failed ids are omitted."""
    out: dict[str, dict[str, Any]] = {}
    if not openalex_ids:
        return out
    bare = [(rid.split("/")[-1] if "/" in rid else rid) for rid in openalex_ids if rid]
    for i in range(0, len(bare), batch):
        chunk = bare[i:i + batch]
        flt = "|".join(chunk)
        url = "https://api.openalex.org/works?" + urlencode({"filter": f"openalex:{flt}", "per-page": str(batch)})
        try:
            payload = get_json(url, timeout_seconds)
            if not isinstance(payload, dict):
                continue
            for item in (payload.get("results") or []):
                if not isinstance(item, dict):
                    continue
                oid = item.get("id") or ""
                oid_bare = oid.split("/")[-1] if oid else ""
                if not oid_bare:
                    continue
                doi = item.get("doi") or ""
                doi = normalize_doi(doi) if doi else None
                out[oid_bare] = {
                    "ref_doi": doi,
                    "ref_title": item.get("display_name"),
                    "ref_year": item.get("publication_year"),
                    "ref_openalex_id": oid_bare,
                    "source_ref_id": oid_bare,
                    "raw": {"openalex_url": oid},
                }
        except SourceAdapterError:
            continue
    return out


def openalex_candidate(payload: dict[str, Any], query_kind: str, score: float) -> SourceCandidate:
    primary_location = payload.get("primary_location")
    best_oa_location = payload.get("best_oa_location")
    locations = payload.get("locations")
    open_access = payload.get("open_access") if isinstance(payload.get("open_access"), dict) else {}
    pdf_candidates = openalex_pdf_candidates(payload)
    enriched_raw = dict(payload)
    if isinstance(payload.get("ids"), dict):
        enriched_raw["external_ids"] = payload["ids"]
    return SourceCandidate(
        source_name="openalex",
        query_kind=query_kind,
        status="ready",
        source_record_id=str(payload.get("id")) if payload.get("id") else None,
        normalized_doi=normalize_doi(str(payload.get("doi"))) if payload.get("doi") else None,
        title=str(payload.get("display_name")) if payload.get("display_name") else None,
        year=int(payload["publication_year"]) if payload.get("publication_year") else None,
        venue=venue_from_openalex(payload),
        candidate_score=score,
        pdf_candidates=pdf_candidates,
        license=openalex_best_license(best_oa_location, primary_location, locations),
        open_access_status=str(open_access.get("oa_status")) if open_access.get("oa_status") else None,
        source_url=str(payload.get("id")) if payload.get("id") else None,
        raw=enriched_raw,
    )


def openalex_pdf_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    content_urls = payload.get("content_urls")
    has_content = payload.get("has_content") if isinstance(payload.get("has_content"), dict) else {}
    if isinstance(content_urls, dict):
        for content_key in ("pdf", "grobid_xml"):
            url = content_urls.get(content_key)
            if url:
                candidates.append(
                    {
                        "kind": f"openalex_content_{content_key}",
                        "url": url,
                        "content_type": "application/pdf"
                        if content_key == "pdf"
                        else "application/grobid+xml",
                        "has_content": bool(has_content.get(content_key)),
                        "source": "openalex_content",
                    }
                )
    add_openalex_location_candidate(
        candidates,
        "primary_location",
        payload.get("primary_location"),
    )
    add_openalex_location_candidate(
        candidates,
        "best_oa_location",
        payload.get("best_oa_location"),
    )
    locations = payload.get("locations")
    if isinstance(locations, list):
        for location in locations:
            add_openalex_location_candidate(candidates, "location", location)
    return dedupe_pdf_candidates(candidates)


def add_openalex_location_candidate(
    candidates: list[dict[str, Any]],
    kind: str,
    location: object,
) -> None:
    if not isinstance(location, dict) or not location.get("pdf_url"):
        return
    candidates.append(
        {
            "kind": kind,
            "url": location.get("pdf_url"),
            "landing_page_url": location.get("landing_page_url"),
            "license": location.get("license"),
            "is_oa": location.get("is_oa"),
            "version": location.get("version"),
            "source": "openalex_location",
        }
    )


def dedupe_pdf_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str | None, str | None]] = set()
    deduped: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (
            str(candidate.get("kind")) if candidate.get("kind") else None,
            str(candidate.get("url")) if candidate.get("url") else None,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def openalex_best_license(*locations: object) -> str | None:
    for location in locations:
        if isinstance(location, dict) and location.get("license"):
            return str(location["license"])
    for location_list in locations:
        if not isinstance(location_list, list):
            continue
        for location in location_list:
            if isinstance(location, dict) and location.get("license"):
                return str(location["license"])
    return None


def crossref_candidate(payload: dict[str, Any], query_kind: str, score: float) -> SourceCandidate:
    titles = payload.get("title") if isinstance(payload.get("title"), list) else []
    containers = payload.get("container-title") if isinstance(payload.get("container-title"), list) else []
    licenses = payload.get("license") if isinstance(payload.get("license"), list) else []
    links = payload.get("link") if isinstance(payload.get("link"), list) else []
    pdf_candidates = [
        {"url": link.get("URL"), "content_type": link.get("content-type")}
        for link in links
        if isinstance(link, dict) and link.get("URL")
    ]
    return SourceCandidate(
        source_name="crossref",
        query_kind=query_kind,
        status="ready",
        source_record_id=str(payload.get("DOI")) if payload.get("DOI") else None,
        normalized_doi=normalize_doi(str(payload.get("DOI"))) if payload.get("DOI") else None,
        title=str(titles[0]) if titles else None,
        year=year_from_crossref(payload),
        venue=str(containers[0]) if containers else None,
        candidate_score=score,
        pdf_candidates=pdf_candidates,
        license=str(licenses[0].get("URL")) if licenses and isinstance(licenses[0], dict) else None,
        source_url=str(payload.get("URL")) if payload.get("URL") else None,
        raw=payload,
    )


def first_title(payload: dict[str, Any]) -> str:
    titles = payload.get("title") if isinstance(payload.get("title"), list) else []
    return str(titles[0]) if titles else ""


def sciverse_candidate(payload: dict[str, Any], query_kind: str, score: float) -> SourceCandidate:
    raw = dict(payload)
    doc_id = payload.get("doc_id")
    unique_id = payload.get("unique_id")
    raw["content_access"] = assess_sciverse_ai_ready(
        doc_id=str(doc_id) if doc_id else None,
        content_accessible=bool(payload.get("is_content_accessible")),
        provenance={"endpoint": "/meta-search", "source": "sciverse"},
    )
    raw["content_access"].update({
        "source": "sciverse",
        "doc_id": doc_id,
        "is_content_accessible": bool(payload.get("is_content_accessible")),
    })
    return SourceCandidate(
        source_name="sciverse",
        query_kind=query_kind,
        status="ready",
        source_record_id=str(unique_id or doc_id) if unique_id or doc_id else None,
        normalized_doi=normalize_doi(str(payload.get("doi"))) if payload.get("doi") else None,
        title=str(payload.get("title")) if payload.get("title") else None,
        year=safe_int(payload.get("publication_published_year")),
        venue=str(payload.get("publication_venue_name_unified"))
        if payload.get("publication_venue_name_unified")
        else None,
        candidate_score=score,
        license=str(payload.get("access_license")) if payload.get("access_license") else None,
        open_access_status=str(payload.get("access_oa_status"))
        if payload.get("access_oa_status")
        else None,
        source_url=first_string(payload.get("access_oa_url")),
        raw=raw,
    )


def sciverse_semantic_candidate(payload: dict[str, Any], score: float) -> SourceCandidate:
    """agentic-search hit -> SourceCandidate. The hit carries doc-level
    metadata (title/abstract/year/venue/citation_count) plus the matched
    full-text chunk; no DOI in this endpoint's payload — resolve on demand
    via meta-search when the caller needs a dedup key."""
    return SourceCandidate(
        source_name="sciverse-semantic",
        query_kind="semantic",
        status="ready",
        source_record_id=str(payload.get("doc_id") or "") or None,
        title=str(payload.get("title")) if payload.get("title") else None,
        year=safe_int(payload.get("publication_published_year")),
        venue=str(payload.get("publication_venue_name_unified"))
        if payload.get("publication_venue_name_unified")
        else None,
        candidate_score=score,
        raw={
            "doc_id": payload.get("doc_id"),
            "chunk_id": payload.get("chunk_id"),
            "chunk": payload.get("chunk"),
            "abstract": payload.get("abstract"),
            "citation_count": payload.get("citation_count"),
            "primary_topic": payload.get("primary_topic"),
            "endpoint": "/agentic-search",
        },
    )


def sciverse_results(response: object) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        raise SourceAdapterError("Sciverse meta-search response is not a JSON object")
    results = response.get("results") or []
    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, dict)]


def sciverse_score(query: str, payload: dict[str, Any]) -> float:
    relevance = payload.get("relevance_score")
    if isinstance(relevance, int | float):
        return float(relevance)
    return title_similarity(query, str(payload.get("title") or ""))


def assess_sciverse_ai_ready(
    *,
    doc_id: str | None = None,
    chunk_id: str | None = None,
    content_accessible: bool = False,
    content_status: str | None = None,
    content_text: str | None = None,
    content_error: str | None = None,
    locator: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    locator_data = locator or {}
    provenance_data = provenance or {}
    text_length = len((content_text or "").strip())
    quality_issues: list[str] = []

    if not doc_id:
        quality_issues.append("missing_doc_id")
    if content_status == "failed":
        quality_issues.append("content_failed")
    elif content_status == "empty":
        quality_issues.append("content_empty")
    elif content_status != "ready":
        quality_issues.append("content_not_verified")
    elif text_length < 120:
        quality_issues.append("content_too_short")

    if chunk_id is None and not locator_data.get("chunk_id"):
        quality_issues.append("missing_chunk_id")
    has_locator = any(locator_data.get(key) is not None for key in ("page_no", "page_idx", "offset"))
    if not has_locator:
        quality_issues.append("missing_locator")
    has_model = bool(provenance_data.get("model_name") or provenance_data.get("model_version"))
    if not has_model:
        quality_issues.append("missing_model_provenance")
    has_trace = bool(provenance_data.get("endpoint") or provenance_data.get("query") or provenance_data.get("raw_response_path"))
    if not has_trace:
        quality_issues.append("missing_trace_provenance")

    recommended_blockers = {
        "missing_doc_id",
        "content_failed",
        "content_empty",
        "content_not_verified",
        "content_too_short",
        "missing_chunk_id",
        "missing_locator",
        "missing_model_provenance",
        "missing_trace_provenance",
    }
    if not (recommended_blockers & set(quality_issues)):
        tier = "recommended"
        readiness = "ready"
        label = "推荐使用 Sciverse AI-ready"
        next_action = "use_sciverse_ai_ready"
    elif content_status in {"failed", "empty"} or "missing_doc_id" in quality_issues:
        tier = "local_mineru"
        readiness = "failed" if content_status == "failed" else "missing"
        label = "建议本地 MinerU"
        next_action = "run_local_mineru"
    elif content_accessible or doc_id:
        tier = "try"
        readiness = "partial"
        label = "可尝试 Sciverse AI-ready"
        next_action = "try_sciverse_ai_ready"
    else:
        tier = "local_mineru"
        readiness = "missing"
        label = "建议本地 MinerU"
        next_action = "run_local_mineru"

    return {
        "asset_kind": "sciverse_ai_ready",
        "quality_tier": tier,
        "readiness_status": readiness,
        "label": label,
        "next_action": next_action,
        "quality_issues": quality_issues,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "content_status": content_status,
        "content_length": text_length,
    }


def sciverse_hit(payload: dict[str, Any], query: str) -> dict[str, Any]:
    chunk_id = str(payload.get("chunk_id") or "")
    doc_id = str(payload.get("doc_id") or "")
    text = str(payload.get("chunk") or "")
    return {
        "source_type": "sciverse_chunk",
        "source_id": chunk_id or stable_fallback_id("sciverse_chunk", doc_id, text),
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": text,
        "locator": {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "offset": payload.get("offset"),
            "page_no": payload.get("page_no"),
        },
        "provenance": {
            "source_system": "sciverse",
            "endpoint": "/agentic-search",
            "query": query,
            "title": payload.get("title"),
            "source": payload.get("source"),
            "recall_source": payload.get("recall_source"),
            "score": payload.get("score"),
            "model_name": payload.get("model_name"),
            "model_version": payload.get("model_version"),
        },
        "raw": payload,
    }


def safe_int(value: object) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def first_string(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item:
                return item
    return None


def s2_candidate(payload: dict[str, Any], query_kind: str, score: float) -> SourceCandidate:
    external = payload.get("externalIds") if isinstance(payload.get("externalIds"), dict) else {}
    paper_id = payload.get("paperId")
    return SourceCandidate(
        source_name="s2",
        query_kind=query_kind,
        status="ready",
        source_record_id=str(paper_id) if paper_id else None,
        normalized_doi=normalize_doi(str(external["DOI"])) if external.get("DOI") else None,
        title=str(payload.get("title")) if payload.get("title") else None,
        year=int(payload["year"]) if payload.get("year") else None,
        venue=str(payload.get("venue")) or None,
        candidate_score=score,
        license=None,
        open_access_status=None,
        source_url=f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None,
        raw=dict(payload, s2_external_ids=external),
    )


def failed_candidate(
    source_name: str,
    query_kind: str,
    error_summary: str,
    normalized_doi: str | None = None,
) -> SourceCandidate:
    return SourceCandidate(
        source_name=source_name,
        query_kind=query_kind,
        status="failed",
        normalized_doi=normalized_doi,
        error_summary=error_summary,
    )


def blocked_candidate(
    source_name: str,
    query_kind: str,
    error_summary: str,
    normalized_doi: str | None = None,
) -> SourceCandidate:
    return SourceCandidate(
        source_name=source_name,
        query_kind=query_kind,
        status="blocked",
        normalized_doi=normalized_doi,
        error_summary=error_summary,
    )


def request_json(url: str, timeout_seconds: int) -> object:
    """HTTP GET returning parsed JSON. Polite-pool aware (2026-08-27):
    - OpenAlex: appends ?mailto= (~10x the anonymous rate budget);
    - Crossref: polite pool keys off the User-Agent containing a mailto
      (their documented convention);
    - retries 429/5xx with exponential backoff (OPENALEX_MAX_RETRIES=4 → up
      to ~15s total wait)."""
    polite = url
    if "api.openalex.org" in url and "mailto=" not in url:
        polite = url + ("&" if "?" in url else "?") + "mailto=" + quote(OPENALEX_MAILTO)
    headers = {"Accept": "application/json",
               "User-Agent": f"{USER_AGENT} (mailto:{OPENALEX_MAILTO})"}
    last_exc: Exception | None = None
    for attempt in range(max(0, OPENALEX_MAX_RETRIES) + 1):
        request = Request(polite, headers=headers)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_exc = exc
            if exc.code in (429, 500, 502, 503) and attempt < OPENALEX_MAX_RETRIES:
                time.sleep(min(2 ** attempt, 8))
                continue
            raise SourceAdapterError(f"HTTP {exc.code} {exc.reason} for {url}") from exc
        except URLError as exc:
            last_exc = exc
            if attempt < OPENALEX_MAX_RETRIES:
                time.sleep(min(2 ** attempt, 8))
                continue
            raise SourceAdapterError(f"Network error for {url}: {exc.reason}") from exc
        except Exception as exc:  # noqa: BLE001
            raise SourceAdapterError(str(exc)) from exc
    raise SourceAdapterError(f"exhausted retries for {url}: {last_exc}")


class _TokenBucket:
    """Client-side rate limiter (2026-09-26): Sciverse caps every endpoint
    at 30 req/min. Without pacing, our parallel fan-outs (gap_search 2-3
    queries, lineage_walk up to 6, subquery pools of 4) burst over the line,
    and a 429 would flow into the circuit breaker as a source FAILURE —
    3 in 120s trips it OPEN for 600s, blacking out the semantic channel for
    ten minutes (the A6 run2 poisoning shape). This bucket WAITS instead of
    failing: threads throttle naturally, the server-side limit is never
    reached, and the circuit never sees rate-limit noise."""

    def __init__(self, rate_per_min: float, capacity: int):
        self.rate = rate_per_min / 60.0
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, max_wait_s: float = 10.0) -> bool:
        deadline = time.monotonic() + max_wait_s
        while True:
            with self._lock:
                now = time.monotonic()
                self.tokens = min(self.capacity,
                                  self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
                need_s = (1.0 - self.tokens) / self.rate
            remaining = deadline - time.monotonic()
            if need_s >= remaining:
                return False
            time.sleep(min(need_s, remaining))


def _sciverse_bucket() -> "_TokenBucket":
    global _SCIVERSE_BUCKET
    if _SCIVERSE_BUCKET is None:
        rate = float(os.environ.get("SCIVERSE_RATE_PER_MIN", "30"))
        _SCIVERSE_BUCKET = _TokenBucket(rate, max(1, int(rate)))
    return _SCIVERSE_BUCKET


_SCIVERSE_BUCKET: _TokenBucket | None = None


def sciverse_request_json(
    method: str,
    path: str,
    *,
    payload: object | None = None,
    query: dict[str, object] | None = None,
    timeout_seconds: int,
) -> object:
    token = os.environ.get("SCIVERSE_API_TOKEN")
    if not token:
        raise SourceAdapterError("缺少 SCIVERSE_API_TOKEN，无法调用 Sciverse。")
    # pace to the documented per-endpoint limit BEFORE firing (default 30/min;
    # one shared bucket across endpoints — conservative but our usage is
    # dominated by agentic-search anyway). Waits up to 10s for a token; if
    # still dry the call proceeds anyway (server 429 is then the backstop)
    # rather than fabricating a source failure.
    _sciverse_bucket().acquire(max_wait_s=10.0)
    url = "https://api.sciverse.space" + path
    if query:
        url += "?" + urlencode(query)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    for attempt in range(2):  # one 429 backoff retry (belt: bucket should prevent it)
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 429 and attempt == 0:
                time.sleep(2.5)  # let the bucket refill a little, then retry once
                continue
            raise SourceAdapterError(f"HTTP {exc.code} {exc.reason} for {method} {url}") from exc
        except URLError as exc:
            raise SourceAdapterError(f"Network error for {method} {url}: {exc.reason}") from exc
        except Exception as exc:  # noqa: BLE001
            raise SourceAdapterError(str(exc)) from exc
    raise SourceAdapterError(f"exhausted 429 retries for {method} {url}")


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def venue_from_openalex(payload: dict[str, Any]) -> str | None:
    primary_location = payload.get("primary_location")
    if not isinstance(primary_location, dict):
        return None
    source = primary_location.get("source")
    if not isinstance(source, dict):
        return None
    name = source.get("display_name")
    return str(name) if name else None


def year_from_crossref(payload: dict[str, Any]) -> int | None:
    issued = payload.get("issued")
    if not isinstance(issued, dict):
        return None
    date_parts = issued.get("date-parts")
    if not isinstance(date_parts, list) or not date_parts:
        return None
    first = date_parts[0]
    if not isinstance(first, list) or not first:
        return None
    try:
        return int(first[0])
    except (TypeError, ValueError):
        return None


def title_similarity(query: str, title: str) -> float:
    query_terms = token_set(query)
    title_terms = token_set(title)
    if not query_terms or not title_terms:
        return 0.0
    return len(query_terms & title_terms) / len(query_terms | title_terms)


def token_set(text: str) -> set[str]:
    return {token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if token}


def stable_fallback_id(prefix: str, *parts: str) -> str:
    return prefix + ":" + "|".join(part for part in parts if part)[:80]


class SourceAdapterError(RuntimeError):
    pass
