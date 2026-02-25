from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.settings import settings


RouteDecision = Literal["auto_accept", "pending_review", "rejected"]

_SPACE_RE = re.compile(r"\s+")
_NORM_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")


@dataclass(frozen=True, slots=True)
class SchemaCandidate:
    candidate_type: str
    raw_label: str
    confidence: float
    evidence_chunk_id: str
    evidence_quote: str


@dataclass(frozen=True, slots=True)
class SchemaPatchProposal:
    patch_id: str
    candidate_type: str
    normalized_label: str
    aliases: tuple[str, ...]
    score: float
    evidence_count: int
    evidence_chunk_ids: tuple[str, ...]
    evidence_quotes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AcceptedSchemaPatch:
    version_id: str
    patch_id: str
    replay_chunk_ids: tuple[str, ...]


def _normalize_label(value: str) -> str:
    raw = str(value or "").strip().lower()
    raw = _NORM_RE.sub(" ", raw)
    raw = _SPACE_RE.sub(" ", raw).strip()
    return raw


def _safe_confidence(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _default_storage_dir() -> Path:
    backend_root = Path(__file__).resolve().parents[2]
    return backend_root / settings.storage_dir / "schema" / "evolution"


class SchemaEvolutionEngine:
    def __init__(
        self,
        *,
        storage_dir: Path | str | None = None,
        t_high: float | None = None,
        t_mid: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir is not None else _default_storage_dir()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_file = self.storage_dir / "candidates.jsonl"
        self.versions_dir = self.storage_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        self.t_high = _safe_confidence(
            float(t_high if t_high is not None else getattr(settings, "schema_evolution_t_high", 0.85))
        )
        self.t_mid = _safe_confidence(
            float(t_mid if t_mid is not None else getattr(settings, "schema_evolution_t_mid", 0.60))
        )
        if self.t_mid > self.t_high:
            self.t_mid = self.t_high

        self.enabled = bool(
            getattr(settings, "schema_evolution_enabled", True) if enabled is None else enabled
        )

    def register_candidates(self, candidates: list[SchemaCandidate]) -> int:
        rows = []
        for item in candidates:
            normalized = _normalize_label(item.raw_label)
            if not normalized:
                continue
            rows.append(
                {
                    "candidate_type": str(item.candidate_type or "").strip().lower() or "entity",
                    "raw_label": str(item.raw_label or "").strip(),
                    "normalized_label": normalized,
                    "confidence": _safe_confidence(item.confidence),
                    "evidence_chunk_id": str(item.evidence_chunk_id or "").strip(),
                    "evidence_quote": str(item.evidence_quote or "").strip(),
                    "created_at": datetime.now(tz=timezone.utc).isoformat(),
                }
            )
        if not rows:
            return 0

        with self.candidates_file.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)

    def build_patch_proposals(self, candidates: list[SchemaCandidate]) -> list[SchemaPatchProposal]:
        grouped: dict[tuple[str, str], list[SchemaCandidate]] = {}
        for item in candidates:
            candidate_type = str(item.candidate_type or "").strip().lower() or "entity"
            normalized = _normalize_label(item.raw_label)
            if not normalized:
                continue
            key = (candidate_type, normalized)
            grouped.setdefault(key, []).append(item)

        proposals: list[SchemaPatchProposal] = []
        for (candidate_type, normalized), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
            aliases = sorted(
                {
                    _SPACE_RE.sub(" ", str(row.raw_label or "").strip())
                    for row in rows
                    if str(row.raw_label or "").strip()
                }
            )
            chunk_ids = sorted(
                {
                    str(row.evidence_chunk_id or "").strip()
                    for row in rows
                    if str(row.evidence_chunk_id or "").strip()
                }
            )
            quotes = [
                str(row.evidence_quote or "").strip()
                for row in rows
                if str(row.evidence_quote or "").strip()
            ]
            confidence_values = [_safe_confidence(row.confidence) for row in rows]
            score = sum(confidence_values) / max(1, len(confidence_values))
            patch_seed = f"{candidate_type}\0{normalized}"
            patch_id = "patch:" + hashlib.sha256(patch_seed.encode("utf-8", errors="ignore")).hexdigest()[:16]

            proposals.append(
                SchemaPatchProposal(
                    patch_id=patch_id,
                    candidate_type=candidate_type,
                    normalized_label=normalized,
                    aliases=tuple(aliases),
                    score=round(score, 6),
                    evidence_count=len(rows),
                    evidence_chunk_ids=tuple(chunk_ids),
                    evidence_quotes=tuple(quotes[:8]),
                )
            )
        return proposals

    def route_proposal(self, proposal: SchemaPatchProposal) -> RouteDecision:
        if not self.enabled:
            return "rejected"
        if proposal.score >= self.t_high:
            return "auto_accept"
        if proposal.score >= self.t_mid:
            return "pending_review"
        return "rejected"

    def compute_incremental_replay_chunks(self, proposal: SchemaPatchProposal) -> list[str]:
        return sorted({str(x).strip() for x in proposal.evidence_chunk_ids if str(x).strip()})

    def accept_patch(self, proposal: SchemaPatchProposal) -> AcceptedSchemaPatch:
        version_id = f"schema-v{proposal.patch_id.split(':')[-1]}"
        replay_chunks = tuple(self.compute_incremental_replay_chunks(proposal))
        target = self.versions_dir / f"{version_id}.json"

        if not target.exists():
            payload = {
                "version_id": version_id,
                "patch_id": proposal.patch_id,
                "candidate_type": proposal.candidate_type,
                "normalized_label": proposal.normalized_label,
                "aliases": list(proposal.aliases),
                "score": proposal.score,
                "evidence_count": proposal.evidence_count,
                "evidence_chunk_ids": list(replay_chunks),
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return AcceptedSchemaPatch(
            version_id=version_id,
            patch_id=proposal.patch_id,
            replay_chunk_ids=replay_chunks,
        )
