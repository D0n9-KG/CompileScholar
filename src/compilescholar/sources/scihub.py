# -*- coding: utf-8 -*-
"""The local Sci-Hub archive: DOI -> PDF bytes (internal use only; never named in a publication; CS main experiments
do not use it — INTEGRATED-SYSTEM-1005 §2.5, §4).

Index: an SQLite file (configs/local.yaml paths.scihub_index), table items(doi_norm, doi_raw, archive_id, inner_path,
size, crc) with idx_items_doi_norm, archives(id, rel_path, ...). doi_norm is already lowercase and URL-decoded, so the
lookup is an index point query on core.ids.normalize_doi(doi) (the old WHERE lower(doi_norm)=? scanned 87M rows,
22.8 s per query; measured 1.3 ms with the index). Archive: zip files under paths.scihub_archive; a member is read
and its CRC checked. One read-only connection per thread.
Coverage (measured 10-05): journals 2016-2020 45-59 % (big publishers 86-100 %), 2021 21 %, 2022 onward 0."""
from __future__ import annotations

import threading
import zipfile
import zlib
from dataclasses import dataclass

from ..core import ids, paths

_tl = threading.local()


@dataclass(frozen=True)
class Hit:
    doi: str
    archive: str        # relative to paths.scihub_archive
    inner_path: str
    size: int
    crc: int


def _con():
    c = getattr(_tl, "con", None)
    if c is None:
        import sqlite3
        c = sqlite3.connect(f"file:{paths.resource('scihub_index').as_posix()}?mode=ro", uri=True,
                            check_same_thread=False)
        _tl.con = c
    return c


def lookup(doi: str) -> Hit | None:
    d = ids.normalize_doi(doi)
    if not d:
        return None
    r = _con().execute("SELECT i.doi_norm, a.rel_path, i.inner_path, i.size, i.crc FROM items i "
                       "JOIN archives a ON a.id = i.archive_id WHERE i.doi_norm = ? LIMIT 1", (d,)).fetchone()
    return Hit(*r) if r else None


def read(hit: Hit) -> bytes:
    """The PDF bytes; raises ValueError when the member is missing or its CRC / size does not match the index."""
    with zipfile.ZipFile(paths.resource("scihub_archive") / hit.archive) as z:
        b = z.read(hit.inner_path)
    if len(b) != hit.size or (zlib.crc32(b) & 0xFFFFFFFF) != (hit.crc & 0xFFFFFFFF):
        raise ValueError(f"scihub member {hit.inner_path}: size/crc mismatch")
    return b


def pointer(hit: Hit) -> dict:
    """The asset pointer stored in the library (PDFs are not copied, §4 存储)."""
    return {"channel": "scihub_local", "doi": hit.doi, "archive": hit.archive, "inner_path": hit.inner_path,
            "size": hit.size, "crc": hit.crc}
