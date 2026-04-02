from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Callable, Literal, Sequence

from pydantic import Field

from .models import ContractModel


SourceKind = Literal['md', 'txt']
EligibilityStatus = Literal['eligible', 'corpus_health_failure']
SamplingMode = Literal['fixed_regression', 'random_exploration']
Neo4jLookupStatus = Literal['ready', 'unavailable', 'skipped']

_PREFIX_RE = re.compile(r'^(\d+)_')


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _normalize_relative_ref(path: Path, corpus_root: Path) -> str:
    return str(path.resolve().relative_to(corpus_root.resolve())).replace('\\', '/')


def _extract_numeric_prefix(value: str) -> str | None:
    match = _PREFIX_RE.match(str(value or '').strip())
    if match:
        return match.group(1)
    return None


def _derive_corpus_paper_id(relative_ref: str, *, path: Path) -> str:
    for candidate in (path.stem, path.parent.name):
        prefix = _extract_numeric_prefix(candidate)
        if prefix:
            return prefix
    digest = hashlib.sha256(relative_ref.encode('utf-8', errors='ignore')).hexdigest()[:16]
    return f'hash:{digest}'


def _display_title(value: str) -> str:
    candidate = _PREFIX_RE.sub('', str(value or '').strip())
    candidate = candidate.replace('_', ' ').replace('-', ' ').strip()
    if candidate:
        return re.sub(r'\s+', ' ', candidate)
    fallback = str(value or '').strip().replace('_', ' ')
    return re.sub(r'\s+', ' ', fallback).strip() or 'Untitled corpus item'


def _probe_readable(path: Path) -> None:
    with path.open('rb') as handle:
        handle.read(1)


class FixedRegressionManifestEntry(ContractModel):
    corpus_paper_id: str
    display_title: str
    corpus_relative_ref: str
    selection_reason: str | None = None


class CorpusHealthIssue(ContractModel):
    issue_type: str
    source_path: str | None = None
    corpus_relative_ref: str | None = None
    detail: str
    exception_type: str | None = None


class CorpusInventoryEntry(ContractModel):
    corpus_paper_id: str
    display_title: str
    corpus_relative_ref: str
    md_path: str | None = None
    txt_path: str | None = None
    preferred_source_path: str | None = None
    preferred_source_kind: SourceKind | None = None
    eligibility_status: EligibilityStatus = 'eligible'
    neo4j_paper_id: str | None = None
    neo4j_paper_source: str | None = None
    neo4j_ingested: bool | None = None


class CorpusBatchExclusion(ContractModel):
    corpus_paper_id: str
    reason: str


class CorpusSamplingBatch(ContractModel):
    batch_id: str
    sampling_mode: SamplingMode
    built_at: str
    requested_count: int
    selected: list[CorpusInventoryEntry] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list)
    seed: int | None = None
    fixed_manifest_ref: str | None = None
    exclusions: list[CorpusBatchExclusion] = Field(default_factory=list)


class CorpusSamplingBundle(ContractModel):
    schema_version: str = 'v1'
    built_at: str = Field(default_factory=_utc_now_iso)
    corpus_root: str
    inventory_entries: list[CorpusInventoryEntry] = Field(default_factory=list)
    corpus_health_failures: list[CorpusHealthIssue] = Field(default_factory=list)
    fixed_regression_batch: CorpusSamplingBatch | None = None
    random_exploration_batch: CorpusSamplingBatch | None = None
    fixed_manifest_ref: str | None = None
    seed: int | None = None
    neo4j_lookup_status: Neo4jLookupStatus | None = None
    neo4j_lookup_error: str | None = None


class _EntryAccumulator:
    def __init__(self, corpus_paper_id: str) -> None:
        self.corpus_paper_id = corpus_paper_id
        self.title_candidates: list[str] = []
        self.relative_candidates: list[str] = []
        self.md_path: str | None = None
        self.txt_path: str | None = None
        self.md_readable = False
        self.txt_readable = False

    def add_candidate(
        self,
        *,
        source_kind: SourceKind,
        path: Path,
        relative_ref: str,
        readable: bool,
    ) -> None:
        self.title_candidates.append(_display_title(path.stem))
        self.relative_candidates.append(relative_ref)
        source_path = str(path.resolve())
        if source_kind == 'md':
            if self.md_path is None or readable and not self.md_readable:
                self.md_path = source_path
            self.md_readable = self.md_readable or readable
            return
        if self.txt_path is None or readable and not self.txt_readable:
            self.txt_path = source_path
        self.txt_readable = self.txt_readable or readable

    def to_entry(self) -> CorpusInventoryEntry:
        preferred_source_path: str | None = None
        preferred_source_kind: SourceKind | None = None
        if self.md_path and self.md_readable:
            preferred_source_path = self.md_path
            preferred_source_kind = 'md'
        elif self.txt_path and self.txt_readable:
            preferred_source_path = self.txt_path
            preferred_source_kind = 'txt'

        preferred_ref = None
        if preferred_source_path:
            preferred_ref = next(
                (
                    ref
                    for ref in self.relative_candidates
                    if str(Path(ref).name).lower() == Path(preferred_source_path).name.lower()
                ),
                None,
            )

        return CorpusInventoryEntry(
            corpus_paper_id=self.corpus_paper_id,
            display_title=self.title_candidates[0] if self.title_candidates else 'Untitled corpus item',
            corpus_relative_ref=preferred_ref or self.relative_candidates[0],
            md_path=self.md_path,
            txt_path=self.txt_path,
            preferred_source_path=preferred_source_path,
            preferred_source_kind=preferred_source_kind,
            eligibility_status='eligible' if preferred_source_path else 'corpus_health_failure',
        )


def scan_corpus_inventory(
    corpus_root: str | Path,
    *,
    probe_file: Callable[[Path], None] | None = None,
) -> CorpusSamplingBundle:
    root = _as_path(corpus_root)
    if not root.exists():
        raise FileNotFoundError(f'corpus root not found: {root}')
    if not root.is_dir():
        raise ValueError(f'corpus root is not a directory: {root}')

    resolved_root = root.resolve()
    file_probe = probe_file or _probe_readable
    health_failures: list[CorpusHealthIssue] = []
    grouped_entries: dict[str, _EntryAccumulator] = {}

    def _on_walk_error(exc: OSError) -> None:
        error_path = Path(exc.filename) if getattr(exc, 'filename', None) else resolved_root
        relative_ref = None
        try:
            relative_ref = _normalize_relative_ref(error_path, resolved_root)
        except Exception:
            relative_ref = None
        health_failures.append(
            CorpusHealthIssue(
                issue_type='walk_error',
                source_path=str(error_path),
                corpus_relative_ref=relative_ref,
                detail=str(exc),
                exception_type=type(exc).__name__,
            )
        )

    for dirpath, _, filenames in os.walk(resolved_root, onerror=_on_walk_error):
        directory = Path(dirpath)
        for filename in sorted(filenames):
            suffix = Path(filename).suffix.lower()
            if suffix not in {'.md', '.txt'}:
                continue

            path = directory / filename
            relative_ref = _normalize_relative_ref(path, resolved_root)
            corpus_paper_id = _derive_corpus_paper_id(relative_ref, path=path)
            accumulator = grouped_entries.setdefault(corpus_paper_id, _EntryAccumulator(corpus_paper_id))

            readable = True
            try:
                file_probe(path)
            except OSError as exc:
                readable = False
                health_failures.append(
                    CorpusHealthIssue(
                        issue_type='unreadable_file',
                        source_path=str(path.resolve()),
                        corpus_relative_ref=relative_ref,
                        detail=str(exc),
                        exception_type=type(exc).__name__,
                    )
                )

            accumulator.add_candidate(
                source_kind='md' if suffix == '.md' else 'txt',
                path=path,
                relative_ref=relative_ref,
                readable=readable,
            )

    inventory_entries = [
        grouped_entries[key].to_entry()
        for key in sorted(grouped_entries, key=lambda item: (item.startswith('hash:'), item))
    ]

    return CorpusSamplingBundle(
        corpus_root=str(resolved_root),
        inventory_entries=inventory_entries,
        corpus_health_failures=health_failures,
    )


def load_fixed_regression_manifest(path_like: str | Path) -> list[FixedRegressionManifestEntry]:
    path = _as_path(path_like)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise ValueError(f'fixed regression manifest must be a JSON array: {path}')
    return [FixedRegressionManifestEntry.model_validate(item) for item in payload]


def _eligible_entries(entries: Sequence[CorpusInventoryEntry]) -> list[CorpusInventoryEntry]:
    return [entry for entry in entries if entry.eligibility_status == 'eligible']


def build_fixed_regression_batch(
    entries: Sequence[CorpusInventoryEntry],
    manifest_entries: Sequence[FixedRegressionManifestEntry],
    *,
    fixed_manifest_ref: str | None = None,
    built_at: str | None = None,
) -> CorpusSamplingBatch:
    eligible_by_id = {entry.corpus_paper_id: entry for entry in _eligible_entries(entries)}
    missing_ids: list[str] = []
    selected: list[CorpusInventoryEntry] = []
    seen_ids: set[str] = set()

    for manifest_entry in manifest_entries:
        corpus_paper_id = manifest_entry.corpus_paper_id
        if corpus_paper_id in seen_ids:
            continue
        seen_ids.add(corpus_paper_id)
        entry = eligible_by_id.get(corpus_paper_id)
        if entry is None:
            missing_ids.append(corpus_paper_id)
            continue
        selected.append(entry)

    if missing_ids:
        missing_text = ', '.join(missing_ids)
        raise ValueError(f'fixed regression ids not found in eligible corpus inventory: {missing_text}')

    return CorpusSamplingBatch(
        batch_id='phase7-fixed-regression',
        sampling_mode='fixed_regression',
        built_at=built_at or _utc_now_iso(),
        requested_count=len(manifest_entries),
        selected=selected,
        selected_ids=[entry.corpus_paper_id for entry in selected],
        fixed_manifest_ref=fixed_manifest_ref,
    )


def build_random_exploration_batch(
    entries: Sequence[CorpusInventoryEntry],
    *,
    random_count: int,
    seed: int,
    fixed_ids: Sequence[str] | None = None,
    built_at: str | None = None,
) -> CorpusSamplingBatch:
    fixed_id_set = {str(corpus_paper_id) for corpus_paper_id in fixed_ids or [] if str(corpus_paper_id).strip()}
    eligible_pool = [
        entry
        for entry in sorted(
            _eligible_entries(entries),
            key=lambda item: (item.corpus_paper_id, item.corpus_relative_ref),
        )
        if entry.corpus_paper_id not in fixed_id_set
    ]
    if random_count > len(eligible_pool):
        raise ValueError(
            f'random exploration batch requested {random_count} papers but only {len(eligible_pool)} eligible papers remain'
        )

    rng = random.Random(int(seed))
    selected = rng.sample(eligible_pool, random_count)
    return CorpusSamplingBatch(
        batch_id='phase7-random-exploration',
        sampling_mode='random_exploration',
        built_at=built_at or _utc_now_iso(),
        requested_count=int(random_count),
        selected=selected,
        selected_ids=[entry.corpus_paper_id for entry in selected],
        seed=int(seed),
        exclusions=[
            CorpusBatchExclusion(corpus_paper_id=corpus_paper_id, reason='fixed_regression_exclusion')
            for corpus_paper_id in sorted(fixed_id_set)
        ],
    )


__all__ = [
    'CorpusBatchExclusion',
    'CorpusHealthIssue',
    'CorpusInventoryEntry',
    'CorpusSamplingBatch',
    'CorpusSamplingBundle',
    'EligibilityStatus',
    'FixedRegressionManifestEntry',
    'Neo4jLookupStatus',
    'SamplingMode',
    'SourceKind',
    'build_fixed_regression_batch',
    'build_random_exploration_batch',
    'load_fixed_regression_manifest',
    'scan_corpus_inventory',
]
