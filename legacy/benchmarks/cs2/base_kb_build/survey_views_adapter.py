# -*- coding: utf-8 -*-
"""Adapt survey records (S1-S4) into the views pipeline (design §6).

survey_lineage -> standard lineage shape (from/to refs + provenance=
"survey") so the genealogy view carries survey edges in the SAME graph
(survey-claimed edges marked provenance="survey" for track separation).
survey_gap -> absence-view entries with absence_type="survey_claimed"
(fourth tier, strictly separated from corpus-derived tiers).
domain_snapshot / survey_claim -> consensus-layer view (new view key
"consensus", compare/card consume as band-background).

Output: base_kb/survey_records_adapted.json (records_slot-compatible
payloads) — merged into the base KB records before views compilation.
"""
import json
import re
import sys
from pathlib import Path

BASE = Path(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp"
            r"\experiments\benchmarks\cs2\base_kb")


def adapt_survey_records(records_survey: dict) -> dict:
    """{pid: {records: [S1-S4]}} -> {pid: {records: [adapted]}}."""
    out = {}
    n_adapted = {"lineage": 0, "absence": 0, "consensus": 0}
    for pid, payload in records_survey.items():
        recs = []
        for r in payload.get("records", []):
            k = r.get("kind")
            if k == "survey_lineage":
                recs.append({
                    "kind": "lineage",
                    "from_method": r.get("from_method"),
                    "to_method": r.get("to_method"),
                    "relation": r.get("relation"),
                    "from_method_ref": {"surface": r.get("from_method"),
                                        "canonical": None,
                                        "entity_id": None},
                    "to_method_ref": {"surface": r.get("to_method"),
                                      "canonical": None,
                                      "entity_id": None},
                    "scope": {},
                    "evidence_basis": "survey_claim",
                    "claim": r.get("claim"),
                    "epistemic": "survey-claimed",
                    "paper_id": r.get("paper_id"),
                    "id": r.get("id"),
                    "quote": r.get("quote"),
                    "chunk_id": r.get("chunk_id"),
                    "section": r.get("section"),
                    "provenance": "survey",
                })
                n_adapted["lineage"] += 1
            elif k == "survey_gap":
                recs.append({
                    "kind": "absence",
                    "subject": r.get("subject"),
                    "missing": r.get("gap_statement"),
                    "absence_type": "survey_claimed",   # 第四档
                    "gap_type": r.get("gap_type"),
                    "evidence": "survey states this gap verbatim",
                    "epistemic": "survey-claimed-absence",
                    "paper_id": r.get("paper_id"),
                    "id": r.get("id"),
                    "quote": r.get("quote"),
                    "chunk_id": r.get("chunk_id"),
                    "section": r.get("section"),
                    "provenance": "survey",
                })
                n_adapted["absence"] += 1
            elif k in ("domain_snapshot", "survey_claim"):
                # consensus layer: keep original kind, views adapter
                # builds the consensus view from these
                recs.append(r)
                n_adapted["consensus"] += 1
        out[pid] = {"records": recs}
    return out, n_adapted


def build_consensus_view(records_survey: dict) -> dict:
    """domain_snapshot + survey_claim -> consensus view (band 背景)."""
    by_entity = {}
    for pid, payload in records_survey.items():
        for r in payload.get("records", []):
            if r.get("kind") == "survey_claim":
                ent = (r.get("claims_about") or "").strip()
                if not ent:
                    continue
                by_entity.setdefault(ent, []).append({
                    "claim": r.get("claim"),
                    "conditions": r.get("conditions"),
                    "survey_paper": r.get("paper_id"),
                    "record_id": r.get("id"),
                    "quote": (r.get("quote") or "")[:200],
                })
            elif r.get("kind") == "domain_snapshot":
                subj = (r.get("subject") or "").strip()
                for c in r.get("claims") or []:
                    ent = (c.get("claims_about") or subj).strip()
                    by_entity.setdefault(ent, []).append({
                        "claim": c.get("claim"),
                        "conditions": c.get("conditions"),
                        "survey_paper": r.get("paper_id"),
                        "record_id": r.get("id"),
                        "quote": (r.get("quote") or "")[:200],
                        "snapshot_type": r.get("snapshot_type"),
                    })
    return {"kind": "consensus",
            "n_entities": len(by_entity),
            "by_entity": by_entity}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records_survey = json.load(open(BASE / "records_survey.json",
                                    encoding="utf-8"))
    adapted, n = adapt_survey_records(records_survey)
    json.dump(adapted, open(BASE / "survey_records_adapted.json", "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
    consensus = build_consensus_view(records_survey)
    json.dump(consensus, open(BASE / "views_consensus.json", "w",
                              encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[adapt] {n} | consensus entities: "
          f"{consensus['n_entities']}", flush=True)


if __name__ == "__main__":
    main()
