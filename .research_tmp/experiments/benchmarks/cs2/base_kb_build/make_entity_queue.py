# -*- coding: utf-8 -*-
"""Generate entity_queue payloads from survey/coarse records so the
standard registry_growth chain can fold base-KB entity surfaces.

registry_growth reads {pid: {entity_queue: [{surface, paper_id, kind,
field}]}} — deep_extract normally fills this; survey_extract/coarse
don't. Here we synthesize it from:
  - survey_lineage: from_method/to_method (field=from_method/to_method)
  - survey_claim: claims_about
  - domain_snapshot: subject
  - survey_gap: subject
  - coarse records: subject + mentions (lineage hooks)
"""
import json
import re
import sys
from pathlib import Path

BASE = Path(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp"
            r"\experiments\benchmarks\cs2\base_kb")

_STOP = re.compile(r"^(the|a|an|this|these|we|our|it|its)$", re.I)


def _clean(s):
    s = (s or "").strip()
    if not s or len(s) < 2 or len(s) > 120:
        return None
    if _STOP.match(s):
        return None
    # 拒绝作者引用式（方法名纪律：这些不能成为实体）
    if re.match(r"^[A-Z][a-z]+ et al", s) or re.match(r"^\[?\d{4}\]?$", s):
        return None
    return s


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    queue_out = {}
    n_surf = 0
    survey = json.load(open(BASE / "records_survey.json",
                            encoding="utf-8"))
    for pid, payload in survey.items():
        q = []
        for r in payload.get("records", []):
            k = r.get("kind")
            if k == "survey_lineage":
                for f in ("from_method", "to_method"):
                    s = _clean(r.get(f))
                    if s:
                        q.append({"surface": s, "paper_id": pid,
                                  "kind": k, "field": f})
            elif k == "survey_claim":
                s = _clean(r.get("claims_about"))
                if s:
                    q.append({"surface": s, "paper_id": pid,
                              "kind": k, "field": "claims_about"})
            elif k in ("domain_snapshot", "survey_gap"):
                s = _clean(r.get("subject"))
                if s:
                    q.append({"surface": s, "paper_id": pid,
                              "kind": k, "field": "subject"})
        if q:
            queue_out[pid] = {"entity_queue": q}
            n_surf += len(q)
    coarse = json.load(open(BASE / "coarse_records.json",
                            encoding="utf-8"))
    for pid, payload in coarse.items():
        q = []
        for r in payload.get("records", []):
            s = _clean(r.get("subject"))
            if s:
                q.append({"surface": s, "paper_id": pid,
                          "kind": r.get("kind") or "coarse",
                          "field": "subject"})
            for m in r.get("mentions") or []:
                s = _clean(m)
                if s:
                    q.append({"surface": s, "paper_id": pid,
                              "kind": "mention", "field": "mentions"})
        if q:
            queue_out[pid] = {"entity_queue": q}
            n_surf += len(q)
    out = BASE / "entity_queue_basekb.json"
    json.dump(queue_out, open(out, "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"[queue] {len(queue_out)} papers, {n_surf} surfaces -> "
          f"{out.name}")


if __name__ == "__main__":
    main()
