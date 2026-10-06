# -*- coding: utf-8 -*-
"""Phase B gate (INTEGRATED-SYSTEM-1005 §12): 20 cross-field PDFs (Nature, physics, chemistry, medicine, biology,
materials, geoscience, with formulas) through registry -> acquire -> fast + careful parse -> units.

DOIs: two per publisher prefix, drawn at random (fixed seed) from the local Sci-Hub index (internal use only), so the
sample is not chosen by hand. Steps: Crossref metadata into the registry, acquire (OpenAlex OA, then Sci-Hub),
fast tier (GROBID), careful tier (MinerU, shared service, 6 slots), then a per-paper report. Run from PowerShell (UNC).
Output: runs/gate_b/{dois.txt, report.json}"""
from __future__ import annotations

import collections
import json
import random
import sqlite3

from compilescholar.acquire import run as ACQ
from compilescholar.core import paths
from compilescholar.documents import assemble as AS
from compilescholar.documents import parse as P
from compilescholar.library import identity, import_crossref, store

PREFIXES = {"nature": "10.1038/nature", "nat_commun": "10.1038/ncomms", "science": "10.1126/science",
            "prl": "10.1103/physrevlett", "prb": "10.1103/physrevb", "jacs": "10.1021/ja", "angew": "10.1002/anie",
            "nejm": "10.1056/nejm", "lancet": "10.1016/s0140-6736", "cell": "10.1016/j.cell",
            "jgr": "10.1029/20", "acta_mater": "10.1016/j.actamat"}
OUT = paths.runs() / "gate_b"


def sample(seed: int = 20261006, per: int = 2, span: int = 20000) -> list[tuple[str, str]]:
    con = sqlite3.connect(f"file:{paths.resource('scihub_index').as_posix()}?mode=ro", uri=True)
    rng = random.Random(seed)
    out = []
    for field, pre in PREFIXES.items():
        got = set()
        for _ in range(per * 6):
            if len(got) >= per:
                break
            r = con.execute("SELECT doi_norm FROM items WHERE doi_norm >= ? AND doi_norm < ? LIMIT 1 OFFSET ?",
                            (pre, pre + "￿", rng.randrange(span))).fetchone()
            if r and r[0] not in got:
                got.add(r[0])
        out += [(field, d) for d in sorted(got)]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    picks = sample()
    (OUT / "dois.txt").write_text("\n".join(f"{f}\t{d}" for f, d in picks), encoding="utf-8")
    print(f"{len(picks)} DOIs", flush=True)
    import_crossref.run([d for _, d in picks])
    con = store.connect()
    pids = {d: identity.resolve(con, "doi", d) for _, d in picks}
    con.close()
    ACQ.acquire([p for p in pids.values() if p], workers=8)
    P.run("fast", paper_ids=[p for p in pids.values() if p])
    P.run("careful", paper_ids=[p for p in pids.values() if p], workers=6)
    con = store.connect()
    rep = []
    for field, d in picks:
        pid = pids[d]
        row = {"field": field, "doi": d, "paper_id": pid}
        if pid:
            fp = identity.first_public(con, pid)
            row["first_public"] = str(fp) if fp else None
            a = con.execute("SELECT channel, identity FROM assets WHERE paper_id=?", (pid,)).fetchone()
            row["asset"] = list(a) if a else None
            row["attempts"] = [list(x) for x in con.execute("SELECT channel, status, detail FROM attempts "
                                                            "WHERE paper_id=?", (pid,))]
            if a:
                doc = AS.get(con, pid, 0)
                f, c = doc["fast"], doc["careful"]
                row["text_date"] = doc["text_date"]
                if f:
                    row["fast"] = {"title": f["title"][:100], "sentences": len(f["sentences"]),
                                   "entries": len(f["entries"]), "cites": len(f["cites"])}
                if c:
                    row["careful"] = {"units": dict(collections.Counter(u["kind"] for u in c["units"])),
                                      "references": len(c["references"])}
        rep.append(row)
        print(json.dumps(row, ensure_ascii=False)[:400], flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = [r for r in rep if r.get("fast") and r.get("careful")]
    print(f"end to end: {len(ok)}/{len(rep)}", flush=True)


if __name__ == "__main__":
    main()
