# -*- coding: utf-8 -*-
"""Fill resolved titles for DOI / OpenAlex resolutions in the E1 sample (one OpenAlex batch call per 50 ids, 1 credit
each), then rewrite the sheet and its sha256. Labels must still be empty when this runs."""
import hashlib
import json
import sys

from compilescholar.core import paths
from compilescholar.sources import refgraph as RG

OUT = paths.REPO / "results" / "e1"


def main():
    rows = [json.loads(l) for l in open(OUT / "e1_sample.jsonl", encoding="utf-8")]
    dois = sorted({x["paper"][4:] for x in rows if x["paper"].startswith("doi:")})
    oas = sorted({x["paper"][9:] for x in rows if x["paper"].startswith("openalex:")})
    title = {}
    for i in range(0, len(dois), 50):
        d = RG._oa_get({"filter": "doi:" + "|".join("https://doi.org/" + x for x in dois[i:i + 50]), "per-page": 50,
                        "select": "doi,display_name,publication_year"}, 1)
        for w in (d or {}).get("results") or []:
            title["doi:" + (w.get("doi") or "").replace("https://doi.org/", "").lower()] = f"{w['display_name']} ({w.get('publication_year')})"
    for i in range(0, len(oas), 50):
        d = RG._oa_get({"filter": "openalex:" + "|".join(oas[i:i + 50]), "per-page": 50,
                        "select": "id,display_name,publication_year"}, 1)
        for w in (d or {}).get("results") or []:
            title["openalex:" + w["id"].split("/")[-1]] = f"{w['display_name']} ({w.get('publication_year')})"
    for x in rows:
        if x["paper"] in title:
            x["resolved_title"] = title[x["paper"]]
    with open(OUT / "e1_sample.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    md = ["# E1 annotation sheet (marker -> paper)\n",
          "For each row: does the bibliography entry the marker points to describe the same paper as **Resolved**? "
          "Label `Y` (same paper), `N` (different paper), or `?` (cannot tell). Leave `label` empty until annotating.\n"]
    for x in rows:
        md.append(f"\n## {x['id']} ({x['style']}, via {x['method']})\n\n**Marker** `{x['key']}` in survey `{x['survey']}`\n\n"
                  f"**Quote**: {x['quote']}\n\n**Entry**: {x['entry']}\n\n**Resolved**: `{x['paper']}` — {x['resolved_title']}\n\n"
                  f"label: \n")
    (OUT / "e1_sheet.md").write_text("".join(md), encoding="utf-8", newline="\n")
    h = {n: hashlib.sha256((OUT / n).read_bytes()).hexdigest() for n in ("e1_sample.jsonl", "e1_sheet.md")}
    (OUT / "e1_sample.sha256").write_text("".join(f"{v}  {k}\n" for k, v in h.items()), encoding="utf-8", newline="\n")
    missing = sum(1 for x in rows if not x.get("resolved_title") or str(x["resolved_title"]).startswith("("))
    print("titles filled; still missing:", missing, h)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
