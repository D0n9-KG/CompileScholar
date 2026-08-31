# -*- coding: utf-8 -*-
"""Build the gold-ref <-> corpus alignment table (A2 granular arm prep).

Rule: a corpus paper IS the source paper when its TITLE matches the known
title of the cited work (we recognize the classics by title patterns), NOT
when it merely cites it. Output: .research_tmp/A2_granular_alignment.json
with entries {ref, corpus_id, title, status: confirmed|rejected|unclear}.
This is the DOI-alignment-table the gold audit demanded (arfm-gold-audit-
rebuild-principles), built at ref granularity.
"""
import json, os, re

BASE = "C:/Users/D0n9/Desktop/science_evo/data/upstream/remote_mineru/mineru_2355/papers"
GOLD = 'experiments/self_evolution_lift/gold/ARFM2024_gold.json'

# known titles for the classic refs (from the field's canonical bibliography;
# patterns, matched case-insensitively against block-0/1 title text)
TITLE_PATTERNS = {
    "MiDi 2004": r"On dense granular flows",
    "Jop et al. 2006": r"granular fluid|rheology.{0,30}granular|constitutive.{0,40}granular flow",
    "da Cruz et al. 2005": r"Rheophysics.{0,20}granular|granular.{0,30}rheolog",
    "Forterre & Pouliquen 2008": r"Flows of granular material",
    "Jenkins & Savage 1983": r"Theory for rapid flow|rapid flows? of granular",
    "Lun et al. 1984": r"Kinetic theories|simple shear flow of granular",
    "Garzo & Dufty 1999": r"Dense fluid|kinetic theory.{0,30}granular",
    "Jenkins & Berzi 2010": r"dense.{0,20}kinetic|Berzi",
    "Berzi 2014": r"kinetic theory.{0,40}(liquid|inertial)",
    "Savage & Lun 1988": r"Particle segregation|sieving",
    "Gray & Thornton 2005": r"mixture theory.{0,30}segregation|size segregation",
    "Gray & Chugunov 2006": r"particle size segregation|mapping approach",
    "Khakhar et al. 1997": r"Segregation.{0,30}granular|size segregation.{0,30}drum",
    "Sarkar & Khakhar 2008": r"segregation.{0,40}terms of|granular segregation",
    "Yoon & Jenkins 2006": r"segregation|kinetic",
    "Cundall & Strack 1979": r"discrete numerical model|ballistic|distinct element",
    "Henann & Kamrin 2013": r"nonlocal|flow arrest",
    "Kamrin & Koval 2012": r"nonlocality|granular fluid",
    "Bouzid et al. 2013": r"non-local|granular",
    "Aranson & Tsimring 2002": r"granular{0,10}flow|continuum",
    "Miller et al. 2013": r"rheology|granular",
    "Hill & Tan 2014": r"segregation",
}

g = json.load(open(GOLD, encoding='utf-8'))
refs = sorted({r for m in g['methods'] for r in m.get('refs', [])})

# corpus titles
titles = {}
for ppr in os.listdir(BASE):
    try:
        cl = json.load(open(os.path.join(BASE, ppr, "content_list.json"), encoding="utf-8"))
    except Exception:
        continue
    head = " ".join(str(it.get("text", "")) for it in cl[:2])[:300]
    titles[ppr] = head

out = []
for ref in refs:
    pat = TITLE_PATTERNS.get(ref)
    entry = {"ref": ref, "corpus_id": None, "status": "no-pattern", "title": ""}
    if pat:
        best = None
        for ppr, t in titles.items():
            if re.search(pat, t, re.IGNORECASE):
                # prefer also having the first author surname nearby
                first = re.split(r"[,&]| et al", ref)[0].strip()
                bonus = first.split()[-1] in t
                if best is None or bonus:
                    best = (ppr, t, bonus)
        if best:
            entry.update(corpus_id=best[0], title=best[1][:120],
                         status="confirmed" if best[2] else "title-only")
        else:
            entry["status"] = "pattern-miss"
    out.append(entry)

with open('.research_tmp/A2_granular_alignment.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
ok = [e for e in out if e['status'] == 'confirmed']
to = [e for e in out if e['status'] == 'title-only']
print(f"confirmed (title+author): {len(ok)}, title-only: {len(to)}, "
      f"no-pattern: {sum(1 for e in out if e['status']=='no-pattern')}, "
      f"pattern-miss: {sum(1 for e in out if e['status']=='pattern-miss')}")
for e in ok:
    print(f"  CONF {e['ref']:28s} -> {e['corpus_id']}")
