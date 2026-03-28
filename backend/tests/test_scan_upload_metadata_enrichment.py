from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

from app.crossref.client import CrossrefResolveResult, CrossrefWork
from app.ingest.scan_upload import scan_upload
from app.ingest.upload_store import assembled_root, manifest_path
from app.settings import settings


class ScanUploadMetadataEnrichmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_storage_dir = settings.storage_dir
        self._tmpdir = tempfile.mkdtemp(prefix='logickg-metadata-enrichment-')
        settings.storage_dir = self._tmpdir

    def tearDown(self) -> None:
        settings.storage_dir = self._old_storage_dir
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_manifest(self, upload_id: str, doi_strategy: str) -> None:
        p = manifest_path(upload_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    'upload_id': upload_id,
                    'mode': 'folder',
                    'chunk_bytes': 1024 * 1024,
                    'files': [],
                    'doi_strategy': doi_strategy,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )

    def _write_unit_md(self, upload_id: str, rel_dir: str, md_name: str, content: str) -> None:
        root = assembled_root(upload_id)
        unit_dir = root / rel_dir
        (unit_dir / 'images').mkdir(parents=True, exist_ok=True)
        (unit_dir / 'images' / 'fig1.png').write_bytes(b'\x89PNG\r\n\x1a\n')
        (unit_dir / md_name).write_text(content, encoding='utf-8')

    @patch('app.ingest.scan_upload.Neo4jClient')
    @patch('app.ingest.scan_upload.CrossrefClient')
    def test_scan_upload_enriches_suspicious_metadata_from_doi_lookup(self, mock_crossref_cls, mock_neo4j_cls) -> None:
        upload_id = 'u_metadata_enrichment'
        self._write_manifest(upload_id, 'title_crossref')
        self._write_unit_md(
            upload_id,
            'paperA',
            'paper.md',
            '# 2.1 LJ Parameters\n\n龙新平 $^{1}$, 何碧 $^{2}$\n\nDOI: 10.1000/example-doi\n\nBody text.\n',
        )

        selected = CrossrefWork(
            doi='10.1000/example-doi',
            title='论 VLW 状态方程',
            year=1992,
            venue='爆炸学报',
            authors=['龙新平', '何碧', '蒋小华', '吴雄'],
            score=None,
        )
        mock_crossref = mock_crossref_cls.return_value
        mock_crossref.get_work_by_doi.return_value = selected

        mock_neo4j = mock_neo4j_cls.return_value.__enter__.return_value
        mock_neo4j.get_paper_basic.side_effect = KeyError('not found')

        out = scan_upload(upload_id)

        units = out.get('units') or []
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].get('title'), '论 VLW 状态方程')
        self.assertEqual(units[0].get('year'), 1992)
        self.assertEqual(units[0].get('doi'), '10.1000/example-doi')
        mock_crossref.get_work_by_doi.assert_called_once_with('10.1000/example-doi')
        mock_crossref.resolve_reference.assert_not_called()
