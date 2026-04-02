from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.research_logic import (
    FixedRegressionManifestEntry,
    build_fixed_regression_batch,
    build_random_exploration_batch,
    scan_corpus_inventory,
)


def _write_text(path: Path, text: str = 'content') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def _fixture_corpus(root: Path) -> Path:
    _write_text(root / '1001_Alpha_Paper' / '1001_Alpha_Paper.md', '# alpha')
    _write_text(root / 'txt' / '1001_Alpha_Paper.txt', 'alpha text')
    _write_text(root / '1002_Beta_Result' / '1002_Beta_Result.md', '# beta')
    _write_text(root / 'txt' / '1003_Gamma_Scan.txt', 'gamma text')
    _write_text(root / '1004_Delta_Study' / '1004_Delta_Study.md', '# delta')
    _write_text(root / 'txt' / '1005_Epsilon_Test.txt', 'epsilon text')
    _write_text(root / '1006_Zeta_Report' / '1006_Zeta_Report.md', '# zeta')
    return root


def test_scan_corpus_inventory_collapses_md_and_txt_variants_into_one_entry(tmp_path: Path) -> None:
    corpus_root = _fixture_corpus(tmp_path / 'corpus')

    bundle = scan_corpus_inventory(corpus_root)

    alpha = next(entry for entry in bundle.inventory_entries if entry.corpus_paper_id == '1001')

    assert len(bundle.inventory_entries) == 6
    assert alpha.display_title == 'Alpha Paper'
    assert alpha.md_path is not None
    assert alpha.txt_path is not None
    assert alpha.preferred_source_kind == 'md'
    assert alpha.corpus_relative_ref == '1001_Alpha_Paper/1001_Alpha_Paper.md'
    assert alpha.eligibility_status == 'eligible'


def test_fixed_and_random_batches_are_reproducible_and_do_not_overlap(tmp_path: Path) -> None:
    corpus_root = _fixture_corpus(tmp_path / 'corpus')
    bundle = scan_corpus_inventory(corpus_root)

    manifest_entries = [
        FixedRegressionManifestEntry(
            corpus_paper_id='1001',
            display_title='Alpha Paper',
            corpus_relative_ref='1001_Alpha_Paper/1001_Alpha_Paper.md',
            selection_reason='stable markdown-plus-text variant',
        ),
        FixedRegressionManifestEntry(
            corpus_paper_id='1002',
            display_title='Beta Result',
            corpus_relative_ref='1002_Beta_Result/1002_Beta_Result.md',
            selection_reason='stable markdown-only paper',
        ),
    ]

    fixed_batch = build_fixed_regression_batch(
        bundle.inventory_entries,
        manifest_entries,
        fixed_manifest_ref='docs/replay/corpus_sampling/phase7-fixed-regression-set.json',
    )
    random_batch_a = build_random_exploration_batch(
        bundle.inventory_entries,
        random_count=2,
        seed=17,
        fixed_ids=fixed_batch.selected_ids,
    )
    random_batch_b = build_random_exploration_batch(
        bundle.inventory_entries,
        random_count=2,
        seed=17,
        fixed_ids=fixed_batch.selected_ids,
    )

    assert fixed_batch.selected_ids == ['1001', '1002']
    assert random_batch_a.selected_ids == random_batch_b.selected_ids
    assert set(random_batch_a.selected_ids).isdisjoint(fixed_batch.selected_ids)
    assert [exclusion.model_dump(mode='json') for exclusion in random_batch_a.exclusions] == [
        {'corpus_paper_id': '1001', 'reason': 'fixed_regression_exclusion'},
        {'corpus_paper_id': '1002', 'reason': 'fixed_regression_exclusion'},
    ]


def test_scan_corpus_inventory_reports_traversal_and_unreadable_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus_root = _fixture_corpus(tmp_path / 'corpus')
    unreadable_path = _write_text(corpus_root / 'txt' / '1007_Unreadable_Paper.txt', 'blocked')
    original_walk = os.walk
    original_open = Path.open

    def fake_walk(top, topdown=True, onerror=None, followlinks=False):  # noqa: ANN001
        if onerror is not None:
            err = FileNotFoundError('broken nested directory')
            err.filename = str(Path(top) / 'broken-subdir')
            onerror(err)
        yield from original_walk(top, topdown=topdown, onerror=onerror, followlinks=followlinks)

    def fake_open(self, *args, **kwargs):  # noqa: ANN001
        if self.resolve() == unreadable_path.resolve():
            raise PermissionError('permission denied')
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(os, 'walk', fake_walk)
    monkeypatch.setattr(Path, 'open', fake_open)

    bundle = scan_corpus_inventory(corpus_root)

    issue_types = {issue.issue_type for issue in bundle.corpus_health_failures}
    unreadable_entry = next(entry for entry in bundle.inventory_entries if entry.corpus_paper_id == '1007')

    assert issue_types == {'unreadable_file', 'walk_error'}
    assert unreadable_entry.eligibility_status == 'corpus_health_failure'
    assert unreadable_entry.preferred_source_path is None
