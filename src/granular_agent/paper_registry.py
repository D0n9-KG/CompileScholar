"""HTTP client for the sci-evo-extract paper registry API.

Gives LogicKG uniform access: DOI/title -> paper_id -> fulltext + references
+ (future) hypergraph. This is Side B1 of the integration plan
(.research_tmp/PLAN_sci_evo_integration.md): LogicKG consumes the sci-evo-extract
HTTP API (Side A, commit 0da9d14 added the references/citations endpoints).

Design constraints (project discipline):
- stdlib `urllib` only — do NOT add requests/httpx to the LogicKG env.
- base_url via env SCIEVO_API_BASE (default http://127.0.0.1:8765/api).
- get_hypergraph returns None gracefully: the sci-evo side does not yet store
  LogicKG hypergraph artifacts (that closed loop is built in a later step).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_BASE_URL = os.environ.get(
    "SCIEVO_API_BASE", "http://127.0.0.1:8765/api"
)

# artifact kinds (sci-evo-extract mineru pipeline + reserved future kinds)
KIND_MINERU_MARKDOWN = "mineru_markdown"
KIND_LOGICKG_HYPERGRAPH = "logickg_hypergraph"  # produced by step-6 storage loop


class PaperRegistryError(RuntimeError):
    """Non-2xx response (other than 404) from the sci-evo-extract API."""


class PaperNotFound(PaperRegistryError):
    """404 from a paper-scoped endpoint."""


class PaperRegistryClient:
    """Thin HTTP client over the sci-evo-extract /api endpoints."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- low-level --------------------------------------------------------

    def _origin(self) -> str:
        parsed = urllib.parse.urlsplit(self.base_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404:
                raise PaperNotFound(f"{method} {url} -> 404: {detail}") from None
            raise PaperRegistryError(
                f"{method} {url} -> {exc.code}: {detail}"
            ) from None
        except urllib.error.URLError as exc:
            raise PaperRegistryError(
                f"cannot reach sci-evo-extract at {url}: {exc.reason}"
            ) from None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        return self._request("POST", path, body=body)

    # -- public API -------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """GET /health — liveness + storage config."""
        return self._get("/health")

    def resolve(
        self,
        doi: str | None = None,
        title: str | None = None,
        register: bool = False,
        process: bool = False,
        sources: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Resolve a DOI or title to a paper record (and optionally register it).

        First hits /library/search-resolve (dry-run) for candidates + any
        already-registered paper_id. When register=True and no paper_id is found,
        POSTs /library/acquisitions to register (process=True triggers MinerU,
        which is heavy — off by default).

        Returns the first result row (dict with paper_id, doi, title, year,
        registered_already, ...) or None if no candidates matched.
        """
        if not doi and not title:
            raise ValueError("resolve requires doi or title")
        query_type = "doi" if doi else "title"
        query = doi or title
        payload = self._post(
            "/library/search-resolve",
            {
                "query": query,
                "query_type": query_type,
                "metadata_sources": sources or ["openalex", "crossref"],
                "dry_run": True,
                "write_policy": "dry_run",
            },
        )
        results = payload.get("results") or []
        if not results:
            return None
        row = results[0]
        if register and not row.get("paper_id") and row.get("doi"):
            acq = self._post(
                "/library/acquisitions",
                {"doi": row["doi"], "title": row.get("title"), "process": process},
            )
            # acquisition returns paper_id at top level
            row = {**row, "paper_id": acq.get("paper_id"), "acquisition": acq}
        return row

    def get_paper(self, paper_id: str) -> dict[str, Any]:
        """GET /library/papers/{id} — paper detail + readiness + artifact manifest."""
        return self._get(f"/library/papers/{urllib.parse.quote(paper_id)}")

    def get_references(
        self,
        paper_id: str,
        source: str = "openalex",
        fetch: bool = False,
        doi: str | None = None,
    ) -> list[dict[str, Any]]:
        """GET /library/papers/{id}/references — outgoing references (papers it cites).

        fetch=True triggers a fresh OpenAlex lookup (needs the paper's DOI, or
        pass doi=) and upserts before returning. This is the citation-relation
        structural signal consumed by lift_corpus (step 3 of the goal).
        """
        params: dict[str, Any] = {"source": source}
        if fetch:
            params["fetch"] = "true"
        if doi:
            params["doi"] = doi
        payload = self._get(
            f"/library/papers/{urllib.parse.quote(paper_id)}/references", params=params
        )
        return payload.get("items", [])

    def get_citations(
        self, paper_id: str, source: str = "openalex"
    ) -> list[dict[str, Any]]:
        """GET /library/papers/{id}/citations — incoming citations (papers citing it).

        Requires prior registration via the sci-evo side (reverse-citation fetch
        not yet wired on the API).
        """
        payload = self._get(
            f"/library/papers/{urllib.parse.quote(paper_id)}/citations",
            params={"source": source},
        )
        return payload.get("items", [])

    def list_artifacts(self, paper_id: str) -> list[dict[str, Any]]:
        """GET /library/papers/{id}/artifacts — artifact manifest with content_url."""
        payload = self._get(f"/library/papers/{urllib.parse.quote(paper_id)}/artifacts")
        return payload.get("items", [])

    def get_fulltext(self, paper_id: str, kind: str = KIND_MINERU_MARKDOWN) -> str | None:
        """Fetch the fulltext artifact (default mineru_markdown) as text.

        Returns None if no ready/downloadable artifact of that kind exists.
        """
        artifact = self._find_artifact(paper_id, kind)
        if artifact is None:
            return None
        content_url = artifact.get("content_url")
        if not content_url:
            return None
        # content_url is an /api/... path relative to the server origin.
        url = f"{self._origin()}{content_url}"
        req = urllib.request.Request(url, headers={"Accept": "text/plain, */*"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raise PaperRegistryError(f"fulltext fetch {url} -> {exc.code}") from None
        except urllib.error.URLError as exc:
            raise PaperRegistryError(f"fulltext fetch {url} failed: {exc.reason}") from None

    def get_hypergraph(self, paper_id: str) -> dict[str, Any] | None:
        """Fetch the stored LogicKG hypergraph artifact for a paper, if present.

        Returns None when no hypergraph artifact exists yet (the storage loop is
        built in a later step). When present, returns the parsed JSON hypergraph.
        """
        artifact = self._find_artifact(paper_id, KIND_LOGICKG_HYPERGRAPH)
        if artifact is None:
            return None
        content_url = artifact.get("content_url")
        if not content_url:
            return None
        url = f"{self._origin()}{content_url}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise PaperRegistryError(f"hypergraph fetch {url} -> {exc.code}") from None
        except urllib.error.URLError as exc:
            raise PaperRegistryError(f"hypergraph fetch {url} failed: {exc.reason}") from None

    # -- helpers ----------------------------------------------------------

    def _find_artifact(
        self, paper_id: str, kind: str
    ) -> dict[str, Any] | None:
        for item in self.list_artifacts(paper_id):
            if item.get("kind") == kind and item.get("downloadable"):
                return item
        return None
