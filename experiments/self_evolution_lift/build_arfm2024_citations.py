# -*- coding: utf-8 -*-
"""Build citation relations for the ARFM2024 lift-eval corpus (step3 prep).

For each of the 19 granular-flow papers used in eval_lift_*papers.py:
  1. resolve title via sci-evo-extract search-resolve (OpenAlex) -> DOI
  2. register the paper (POST /library/acquisitions, process=False) if not yet
  3. fetch references (OpenAlex, upsert) -> list of cited DOIs/titles
  4. match cited DOIs back to the corpus (which of the 19 does this paper cite)

Output: .research_tmp/runs/ARFM2024/citations.json
  {paper_name: {doi, paper_id, n_refs, cites_corpus: [paper_name,...], raw_refs:[...]}}

Citation edges within the corpus (A cites B) become the step-3 structural signal
fed into judge_relation as a prior for extends/improves/background.

Requires the sci-evo-extract server on :8765.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from granular_agent.paper_registry import (  # noqa: E402
    PaperRegistryClient,
    PaperRegistryError,
)

# Author_Year_Title -> (author, year, clean title). The filename underscores
# encode spaces AND original punctuation; we restore by hand for 19 papers since
# automatic de-underscoring loses commas/colons that matter for OpenAlex match.
FILES = [
    "Bazant_2006_The_spot_model_for_random-packing_dynamics",
    "Bouzid_2013_Nonlocal_rheology_of_granular_flows_across_yield_c",
    "Bouzid_2015_Non-local_rheology_in_dense_granular_flows",
    "Gray_2005_A_theory_for_particle_size_segregation_in_shallow_",
    "Gray_2006_Particle-size_segregation_and_diffusive_remixing_i",
    "Hill_2014_Segregation_in_dense_sheared_flows__gravity,_tempe",
    "Jenkins_1983_A_theory_for_the_rapid_flow_of_identical,_smooth,_",
    "Jenkins_2010_Dense_inclined_flows_of_inelastic_spheres__tests_o",
    "Jop_2006_A_constitutive_law_for_dense_granular_flows",
    "Kamrin_2012_Nonlocal_constitutive_relation_for_steady_granular",
    "Kamrin_2015_Nonlocal_modeling_of_granular_flows_down_inclines",
    "Midi_2004_On_dense_granular_flows",
    "Pouliquen_1999_Scaling_laws_in_granular_flows_down_rough_inclined",
    "Sarkar_2008_Experimental_evidence_for_a_description_of_granula",
    "Savage_1988_Particle_size_segregation_in_inclined_chute_flow_o",
    "Savage_1998_Analyses_of_slow_high-concentration_flows_of_granu",
    "Tripathi_2013_Density_difference-driven_segregation_in_a_dense_g",
    "Tüzün_1979_Experimental_evidence_supporting_the_kinematic_mod",
    "Yoon_2006_The_influence_of_different_species'_granular_tempe",
]

# clean search titles (hand-restored from the truncated filenames).
# Underscore->space; collapse double underscores (broken word boundary) to space;
# trailing fragment from filename truncation is kept short — OpenAlex matches prefix.
TITLES = {
    "Bazant_2006_The_spot_model_for_random-packing_dynamics":
        "The spot model for random-packing dynamics",
    "Bouzid_2013_Nonlocal_rheology_of_granular_flows_across_yield_c":
        "Nonlocal rheology of granular flows across yield",
    "Bouzid_2015_Non-local_rheology_in_dense_granular_flows":
        "Non-local rheology in dense granular flows",
    "Gray_2005_A_theory_for_particle_size_segregation_in_shallow_":
        "A theory for particle size segregation in shallow granular free-surface flows",
    "Gray_2006_Particle-size_segregation_and_diffusive_remixing_i":
        "Particle-size segregation and diffusive remixing in granular avalanches",
    "Hill_2014_Segregation_in_dense_sheared_flows__gravity,_tempe":
        "Segregation in dense sheared flows: gravity, temperature and particle",
    "Jenkins_1983_A_theory_for_the_rapid_flow_of_identical,_smooth,_":
        "A theory for the rapid flow of identical, smooth, nearly elastic, circular disks",
    "Jenkins_2010_Dense_inclined_flows_of_inelastic_spheres__tests_o":
        "Dense inclined flows of inelastic spheres: tests of theoretical predictions",
    "Jop_2006_A_constitutive_law_for_dense_granular_flows":
        "A constitutive law for dense granular flows",
    "Kamrin_2012_Nonlocal_constitutive_relation_for_steady_granular":
        "Nonlocal constitutive relation for steady granular flow",
    "Kamrin_2015_Nonlocal_modeling_of_granular_flows_down_inclines":
        "Nonlocal modeling of granular flows down inclines",
    "Midi_2004_On_dense_granular_flows":
        "On dense granular flows",
    "Pouliquen_1999_Scaling_laws_in_granular_flows_down_rough_inclined":
        "Scaling laws in granular flows down rough inclined planes",
    "Sarkar_2008_Experimental_evidence_for_a_description_of_granula":
        "Experimental evidence for a description of granular flow by",
    "Savage_1988_Particle_size_segregation_in_inclined_chute_flow_o":
        "Particle size segregation in inclined chute flow of dry cohesionless granular material",
    "Savage_1998_Analyses_of_slow_high-concentration_flows_of_granu":
        "Analyses of slow high-concentration flows of granular materials",
    "Tripathi_2013_Density_difference-driven_segregation_in_a_dense_g":
        "Density difference-driven segregation in a dense granular flow",
    "Tüzün_1979_Experimental_evidence_supporting_the_kinematic_mod":
        "Experimental evidence supporting the kinematic model for the flow of granular",
    "Yoon_2006_The_influence_of_different_species'_granular_tempe":
        "The influence of different species' granular temperature upon segregation",
}

OUT = os.path.join(os.path.dirname(__file__), "..", "..", ".research_tmp", "runs"), "ARFM2024", "citations.json")


def main() -> int:
    client = PaperRegistryClient(timeout=60)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cached = {}
    if os.path.isfile(OUT):
        cached = json.load(open(OUT, encoding="utf-8"))

    # pass 1: resolve + register every paper, collect doi -> paper_name map
    doi2name: dict[str, str] = {}
    paper_records: dict[str, dict] = {}
    for name in FILES:
        rec = dict(cached.get(name, {}))
        title = TITLES.get(name)
        if not title:
            print(f"[SKIP] {name}: no title mapping")
            continue
        if rec.get("doi"):
            doi2name[rec["doi"]] = name
            paper_records[name] = rec
            print(f"[cached] {name}: {rec['doi']}")
            continue
        try:
            row = client.resolve(title=title)
        except PaperRegistryError as exc:
            print(f"[ERR resolve] {name}: {exc}")
            continue
        if not row or not row.get("doi"):
            print(f"[NODOI] {name}: title='{title}' -> no DOI")
            continue
        doi = row["doi"]
        # register (process=False: no MinerU; metadata only). resolve is dry-run.
        if not row.get("paper_id"):
            try:
                acq = client._post(
                    "/library/acquisitions",
                    {"doi": doi, "title": row.get("title"), "process": False},
                )
                pid = acq.get("paper_id")
            except PaperRegistryError as exc:
                print(f"[ERR acquire] {name}: {exc}")
                pid = None
        else:
            pid = row["paper_id"]
        rec = {"doi": doi, "paper_id": pid, "year": row.get("year"),
               "title": row.get("title")}
        doi2name[doi] = name
        paper_records[name] = rec
        print(f"[resolve] {name}: {doi} year={row.get('year')} pid={pid}")

    print(f"\nresolved {len(doi2name)}/{len(FILES)} papers; corpus DOIs known")

    # pass 2: fetch references for each, match cited DOIs back to corpus
    for name, rec in paper_records.items():
        if not rec.get("paper_id"):
            continue
        if rec.get("raw_refs"):
            print(f"[cached refs] {name}: {rec['n_refs']} refs, "
                  f"{len(rec.get('cites_corpus', []))} in-corpus")
            continue
        try:
            refs = client.get_references(rec["paper_id"], fetch=True, doi=rec["doi"])
        except PaperRegistryError as exc:
            print(f"[ERR refs] {name}: {exc}")
            continue
        cites_corpus = []
        for r in refs:
            rdoi = r.get("ref_doi")
            if rdoi and rdoi in doi2name and doi2name[rdoi] != name:
                cites_corpus.append(doi2name[rdoi])
        rec["n_refs"] = len(refs)
        rec["cites_corpus"] = sorted(set(cites_corpus))
        rec["raw_refs"] = refs
        print(f"[refs] {name}: {len(refs)} refs, {len(cites_corpus)} in-corpus "
              f"-> {rec['cites_corpus']}")
        json.dump(paper_records, open(OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    # summary: in-corpus citation edges
    edges = []
    for name, rec in paper_records.items():
        for tgt in rec.get("cites_corpus", []):
            edges.append((name, tgt))
    print(f"\n=== {len(edges)} in-corpus citation edges (A cites B) ===")
    for a, b in edges:
        print(f"  {a}  ->  {b}")

    json.dump(paper_records, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nsaved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
