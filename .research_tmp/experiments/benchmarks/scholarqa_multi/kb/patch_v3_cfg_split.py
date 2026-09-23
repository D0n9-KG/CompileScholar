"""One-off surgical registry correction (G3 investigation, 2026-09-24).

The registry merge folded "classifier-free guidance" (Ho & Salimans 2022,
own paper: Classifier-Free_Diffusion_Guidance) in as an ALIAS of the ADM
entity (Diffusion Models Beat GANs, Dhariwal & Nichol 2021). These are
different methods: ADM is an architecture, classifier guidance is the
ADM paper's steering technique, classifier-free guidance is the CFG
paper's technique. A standalone "Classifier-Free Guidance" entity
(f9e809929d69, owned by Stay_on_topic_with_Classifier-Free_Guidance)
already exists — this patch moves the CFG surface there and transfers
card ownership to the method's origin paper.

Run AFTER registry_dedup (operates on registry_v3.json in place).
"""

import json

KB = "."
ADM_ID = "b09c600fddc5"
CFG_ID = "f9e809929d69"
CFG_ORIGIN_PAPER = "Classifier-Free_Diffusion_Guidance"
SURFACE = "classifier-free guidance"


def main() -> None:
    path = f"{KB}/registry_v3.json"
    reg = json.load(open(path, encoding="utf-8"))
    by_id = {e["entity_id"]: e for e in reg["entities"]}
    adm, cfg = by_id[ADM_ID], by_id[CFG_ID]

    assert SURFACE in adm["aliases"], "precondition: surface sits in ADM aliases"
    adm["aliases"] = [a for a in adm["aliases"] if a != SURFACE]
    assert adm["in_corpus_paper_id"] != CFG_ORIGIN_PAPER

    if SURFACE not in cfg["aliases"]:
        cfg["aliases"] = cfg["aliases"] + [SURFACE]
    cfg["in_corpus_paper_id"] = CFG_ORIGIN_PAPER
    if CFG_ORIGIN_PAPER not in cfg["mention_papers"]:
        cfg["mention_papers"] = sorted(cfg["mention_papers"] + [CFG_ORIGIN_PAPER])
    cfg["mention_count"] = len(cfg["mention_papers"])

    si = reg["surface_index"]
    assert si.get(SURFACE) == ADM_ID
    si[SURFACE] = CFG_ID

    json.dump(reg, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("CFG split applied: surface moved ADM -> CFG entity, "
          f"ownership -> {CFG_ORIGIN_PAPER}")


if __name__ == "__main__":
    main()
