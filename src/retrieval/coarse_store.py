"""Two-tier extraction state store (batch-3, 2026-09-25 user directive:
sci-evo's mature SQLite registry becomes the system of record for the
investment ladder).

Tier 0  metadata only (title+abstract from a retrieval result)
Tier 1  coarse extraction — abstract -> lightweight records; registered
        here as a COARSE run; re-extraction is refused while the run is
        ready (the loop pays 11-16s once per paper, never twice)
Tier 2  deep extraction — full CompileScholar pipeline over acquired full
        text; registered as a DEEP run; supersedes the coarse tier

Backed by the existing library tables — no schema changes:
  papers            (upsert_paper: DOI/title dedup for free)
  extraction_runs   (run_id prefix COARSE_/DEEP_ distinguishes the tier)
  paper_tags        ("tier-coarse" / "tier-deep" / "promotion-candidate")

The deterministic promotion-candidate rule (no LLM): a coarse paper is a
candidate for Tier 2 when its records carry limitation signal (gap-
filling evidence value) or >=2 distinct in-corpus entities matched via
mentions backflow (multi-lineage anchor). The ANSWERING loop sees the
flag; the promotion itself is a library-growth action (acquisition ->
parse -> per-paper pipeline), fired by promote_to_deep.py — never inside
the latency-bound answer loop.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from retrieval.registry import (
    LibraryRegistry,
    LibraryRegistryError,
)


def _run_id(prefix: str, paper_id: str, seed: str) -> str:
    h = hashlib.md5(f"{paper_id}|{seed}".encode("utf-8")).hexdigest()[:10]
    return f"RUN_{prefix}_{h}"


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


class TierStore:
    def __init__(self, db_path: Path | str, library_root: Path | str):
        self.registry = LibraryRegistry(Path(db_path), Path(library_root))
        self.registry.initialize()

    # ---- tier 1 ----------------------------------------------------------

    def register_coarse(
        self,
        *,
        title: str,
        doi: str | None = None,
        year: int | None = None,
        abstract: str = "",
        records: list[dict] | None = None,
        matched_entities: list[str] | None = None,
    ) -> dict[str, Any]:
        """Register (or refuse to redo) a coarse extraction. Returns the
        paper's tier state; `reused=True` means a ready coarse run already
        existed and nothing was re-paid."""
        paper = self.registry.upsert_paper(
            doi=doi or None, title=title, year=year,
            identity_confidence=0.6 if not doi else 0.9)
        paper_id = paper["paper_id"]
        runs = self.registry.list_extraction_runs(paper_id)
        if any(r["run_id"].startswith("RUN_COARSE_") and r["status"] == "ready"
               for r in runs):
            return {**self.state(paper_id), "reused": True}
        if any(r["run_id"].startswith("RUN_DEEP_") and r["status"] == "ready"
               for r in runs):
            # already deep — coarse would be a downgrade, refuse
            return {**self.state(paper_id), "reused": True}

        records = records or []
        matched = matched_entities or []
        candidate = self.promotion_rule(records, matched)
        out_dir = self.registry.library_root / "papers" / paper_id / "coarse"
        out_dir.mkdir(parents=True, exist_ok=True)
        rec_path = out_dir / "records.json"
        rec_path.write_text(json.dumps(
            {"title": title, "doi": doi, "year": year,
             "abstract": abstract, "records": records,
             "matched_entities": matched,
             "promotion_candidate": candidate},
            ensure_ascii=False, indent=1), encoding="utf-8")
        self.registry.register_extraction_run(
            run_id=_run_id("COARSE", paper_id, _norm_title(title)),
            paper_id=paper_id,
            status="ready",
            input_artifact_ids=[],
            output_artifacts={"coarse_records": rec_path},
            error=None,
        )
        tags = ["tier-coarse"] + (["promotion-candidate"] if candidate else [])
        self.registry.set_paper_tags(paper_id, sorted(set(
            tags + [t for t in self.registry.list_paper_tags(paper_id)
                    if t.startswith("tier-") is False and
                    t != "promotion-candidate"])))
        return {**self.state(paper_id), "reused": False,
                "promotion_candidate": candidate}

    @staticmethod
    def promotion_rule(records: list[dict], matched_entities: list[str]) -> bool:
        """Deterministic Tier-2 trigger signal: limitation records (gap
        evidence) or a multi-lineage anchor (>=2 matched entities)."""
        has_limitation = any(r.get("kind") == "limitation" for r in records)
        return has_limitation or len(set(matched_entities)) >= 2

    # ---- tier 2 ----------------------------------------------------------

    def register_deep(
        self,
        *,
        paper_id: str,
        output_artifacts: dict[str, Path],
        error: str | None = None,
    ) -> dict[str, Any]:
        """Register a completed (or failed) deep extraction. A ready DEEP
        run is the terminal state — coarse/deep both refuse re-runs."""
        run_id = _run_id("DEEP", paper_id, "deep")
        self.registry.register_extraction_run(
            run_id=run_id, paper_id=paper_id,
            status="blocked" if error else "ready",
            input_artifact_ids=[],
            output_artifacts={k: Path(v) for k, v in output_artifacts.items()},
            error=error)
        if not error:
            tags = [t for t in self.registry.list_paper_tags(paper_id)
                    if t not in ("tier-coarse", "promotion-candidate")]
            self.registry.set_paper_tags(paper_id, sorted(set(tags + ["tier-deep"])))
        return self.state(paper_id)

    # ---- queries -----------------------------------------------------------

    def state(self, paper_id: str) -> dict[str, Any]:
        paper = self.registry.get_paper(paper_id)
        if not paper:
            return {"paper_id": None, "tier": "none"}
        runs = self.registry.list_extraction_runs(paper_id)
        deep = next((r for r in runs
                     if r["run_id"].startswith("RUN_DEEP_")
                     and r["status"] == "ready"), None)
        coarse = next((r for r in runs
                       if r["run_id"].startswith("RUN_COARSE_")
                       and r["status"] == "ready"), None)
        tier = "deep" if deep else ("coarse" if coarse else "none")
        return {
            "paper_id": paper_id,
            "title": paper.get("title"),
            "doi": paper.get("normalized_doi"),
            "year": paper.get("published_year"),
            "tier": tier,
            "promotion_candidate": "promotion-candidate" in
            self.registry.list_paper_tags(paper_id),
            "coarse_run": (deep or coarse or {}).get("run_id"),
            "output_artifacts": json.loads(
                (deep or coarse or {}).get("output_artifacts_json") or "{}"),
        }

    def find(self, *, doi: str | None = None, title: str | None = None):
        """Locate a paper by DOI, else by normalized title (title-keyed
        papers get stable paper_ids from upsert_paper)."""
        if doi:
            paper = self.registry.find_paper_by_doi(doi)
            if paper:
                return paper["paper_id"]
        if title:
            nt = _norm_title(title)
            for row in self.registry.list_papers():
                if _norm_title(row.get("title") or "") == nt:
                    return row["paper_id"]
        return None

    def pending_promotions(self) -> list[dict[str, Any]]:
        """Coarse papers without a ready deep run, flagged as candidates —
        the Tier-2 work queue for promote_to_deep."""
        out = []
        for row in self.registry.list_papers():
            pid = row["paper_id"]
            st = self.state(pid)
            if st["tier"] == "coarse" and st["promotion_candidate"]:
                out.append(st)
        return out
