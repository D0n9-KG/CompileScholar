# -*- coding: utf-8 -*-
"""Phase B fast-tier check outside CS: GROBID (full) on the three non-CS PDFs the extraction review used (PRL `[n]`
without a references heading, Nat Commun `<sup>n</sup>`, JACS superscripts with letter sub-citations), from the
local Sci-Hub archive (internal use only). No HTML gold exists for these: reported are entries, linked citations,
citation sentences, and a printed sample to read. Run from PowerShell (UNC).
Output: runs/phaseB_fasttier/noncs/<doi-slug>.{pdf,tei.xml}
"""
from __future__ import annotations

import re
import sqlite3
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fasttier_compare as C  # noqa: E402

from compilescholar.core import paths  # noqa: E402

DOIS = ["10.1103/physrevlett.110.010403", "10.1038/ncomms10002", "10.1021/ja400020e"]
OUT = C.OUT / "noncs"


def fetch(doi: str) -> Path:
    dest = OUT / (re.sub(r"[^\w.]+", "_", doi) + ".pdf")
    if dest.exists():
        return dest
    con = sqlite3.connect(f"file:{paths.resource('scihub_index').as_posix()}?mode=ro", uri=True)
    row = con.execute("SELECT a.rel_path, i.inner_path FROM items i JOIN archives a ON a.id = i.archive_id "
                      "WHERE i.doi_norm = ?", (doi,)).fetchone()
    if row is None:
        raise LookupError(doi)
    with zipfile.ZipFile(paths.resource("scihub_archive") / row[0]) as z:
        dest.write_bytes(z.read(row[1]))
    return dest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for doi in DOIS:
        pdf = fetch(doi)
        tei = C._grobid_tei_coords(pdf, OUT / (pdf.stem + ".tei.xml"), C.GROBID.replace(":8070", ":8071"))
        links = C.pdf_cite_links(pdf)
        r = C.parse_tei_links(tei.read_bytes(), links, fill_only=None)
        sents = {}
        for k, s in r["pairs"]:
            sents.setdefault(s, []).append(k)
        print(f"{doi}: entries {len(r['entries'])}, pairs {len(r['pairs'])}, sentences {len(sents)}, "
              f"pdf cite links {len(links)} (used: {r['used_links']})")
        for s, ks in list(sents.items())[:4]:
            print(f"    [{', '.join(ks)}] {s[:200]}")
        for k in list(r["entries"])[:2]:
            print(f"    entry {k}: {r['entries'][k][:150]}")


if __name__ == "__main__":
    main()
