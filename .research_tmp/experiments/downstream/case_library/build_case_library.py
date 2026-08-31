# -*- coding: utf-8 -*-
"""Build the extraction FAILURE CASE LIBRARY (regression exam, 2026-08-26).

Raw material: 75 inline-judge (Claude subagent, 2026-08-26) failure records
from the 5 SOFTB bundles + the PASS edges (over-fix guard). One case = one
sentence-level watch span + expected behavior, scored by re-running
plan->execute->gate->verify->fix on a stored source chunk (10-min level,
no map_structure, frozen schema).

Outputs (this directory):
  cases_main.jsonl     scored dev cases + misc monitoring pool
  cases_holdout.jsonl  30% holdout (scored only), stratified by class x domain
  schema_frozen.json   union of the 5 SOFTB meta snapshots (frozen exam schema)
  README.md            usage + composition

Classification: hand-assigned by reading all 75 judge reasons (index = position
in _tmp_softb_inline_failures.json, glob-sorted by paper).
"""
import json, glob, os, random, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # GBK console guard

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from granular_agent.structure_mapper import load_paper_blocks, full_text_from_blocks

TMP = os.path.join(ROOT, ".research_tmp")
LIB = os.path.join(TMP, "case_library")
FAILS = os.path.join(TMP, "_tmp_softb_inline_failures.json")

PAPER_META = {  # domain / genre per the 5-paper calibration set
    "PPR_24493BE6E8C2": ("ml", "method-math"),        # DQN (Nature 2015)
    "PPR_149984584E26": ("granular", "experiment"),   # quasi-2D emulsion rearrangements
    "PPR_29EFA9BE34EC": ("granular", "short-classic"),# diffusional mixing
    "PPR_6048B48856F7": ("granular", "modeling"),     # rotating drum segregation model
    "PPR_A249CB1DCBA7": ("granular", "theory"),       # mu(I)-rheology compressibility
}

# ---------------------------------------------------------------------------
# Hand classification of the 75 failures.
# index -> (failure_class, expect_overrides)
# expect keys: bad={pattern[,role_contains]}, good_binding={pattern[,role_contains]},
#              good_alt=[patterns] (acceptable alternatives), watch=<override watch>
# Cases NOT in this map -> misc (auto bad spec from the judge's node binding,
# watch_only=False with auto spec but reported separately, not gating).
# ---------------------------------------------------------------------------
CLS = {
    # --- negation blindness (explicitly negated dependence -> positive edge) ---
    12:  ("negation", {"bad": {"pattern": "influences"}}),
    29:  ("negation", {"bad": {"pattern": "influences"}, "good_alt": ["absence_of_phenomenon"]}),
    34:  ("negation", {"bad": {"pattern": "influences"}, "good_alt": ["absence_of_phenomenon"]}),
    35:  ("negation", {"bad": {"pattern": "influences"}, "good_alt": ["absence_of_phenomenon"]}),
    36:  ("negation", {"bad": {"pattern": "influences"}, "good_alt": ["absence_of_phenomenon"]}),
    50:  ("negation", {"bad": {"pattern": "influences", "role_contains": {"source": ["type of material"]}},
                       "good_binding": {"pattern": "influences",
                                        "role_contains": {"source": ["diameter ratio"]}},
                       "good_alt": ["absence_of_phenomenon"]}),   # watch sentence has BOTH a negated and a positive clause
    53:  ("negation", {"bad": {"pattern": "constitutive_law", "role_contains": {"parameter": ["external"]}}}),
    62:  ("negation", {"bad": {"pattern": "compares", "role_contains": {"to": ["phi'"]}},
                       "good_binding": {"pattern": "compares", "role_contains": {"to": ["barker"]}}}),
    66:  ("negation", {"bad": {"pattern": "influences"}}),
    67:  ("negation", {"bad": {"pattern": "influences"}}),
    68:  ("negation", {"bad": {"pattern": "influences"}}),
    # --- pattern routing (right pattern existed in schema, wrong one used) ---
    18:  ("routing", {"bad": {"pattern": "compares"}, "good_binding": {"pattern": "agrees_with"}}),
    30:  ("routing", {"bad": {"pattern": "background"}, "good_binding": {"pattern": "validates_against"}}),
    40:  ("routing", {"bad": {"pattern": "compares"}, "good_binding": {"pattern": "equivalent_formulation"}}),
    52:  ("routing", {"bad": {"pattern": "extends"}, "good_binding": {"pattern": "adapts"}}),
    # --- role/polarity reversal on a real relation ---
    0:   ("polarity", {"bad": {"pattern": "background", "role_contains": {"from": ["previous"]}},
                       "good_binding": {"pattern": "background", "role_contains": {"from": ["spatial"]}}}),
    5:   ("polarity", {"bad": {"pattern": "influences", "role_contains": {"cause": ["pancake"]}}}),
    20:  ("polarity", {"bad": {"pattern": "influences", "role_contains": {"source": ["score"]}},
                       "good_binding": {"pattern": "influences",
                                        "role_contains": {"source": ["sequence"], "target": ["score"]}}}),
    38:  ("polarity", {"bad": {"pattern": "validates_against", "role_contains": {"object": ["system"]}},
                       "good_binding": {"pattern": "validates_against", "role_contains": {"object": ["theor"]}}}),
    41:  ("polarity", {"bad": {"pattern": "background"}}),
    55:  ("polarity", {"bad": {"pattern": "influences", "role_contains": {"source": ["volume"]}},
                       "good_binding": {"pattern": "influences",
                                        "role_contains": {"source": ["temperature"], "target": ["volume"]}}}),
    56:  ("polarity", {"bad": {"pattern": "influences",
                              "role_contains": {"source": ["temperature"], "target": ["shear"]}},
                       "good_binding": {"pattern": "influences",
                                        "role_contains": {"source": ["shear"], "target": ["temperature"]}}}),
    57:  ("polarity", {"bad": {"pattern": "background", "role_contains": {"from": ["kinetic theory"]}}}),
    # --- law-as-input (a law/equation bound to a quantity slot) ---
    26:  ("lawinput", {"bad": {"pattern": "constitutive_law", "role_contains": {"input": ["bellman"]}}}),
    39:  ("lawinput", {"bad": {"pattern": "constitutive_law", "role_contains": {"parameter": ["layer"]}},
                       "good_binding": {"pattern": "constitutive_law", "role_contains": {"output": ["composition"]}}}),
    72:  ("lawinput", {"bad": {"pattern": "constitutive_law", "role_contains": {"input": ["law"]}}}),
    # --- setup misextraction (setup/implementation/plot text read as claims) ---
    4:   ("setup", {"bad": {"pattern": "influences"}}),
    10:  ("setup", {"bad": {"pattern": "measures"}}),
    11:  ("setup", {"bad": {"pattern": "compares"}}),
    23:  ("setup", {"bad": {"pattern": "defines"}}),
    24:  ("setup", {"bad": {"pattern": "measures"}}),
    25:  ("setup", {"bad": {"pattern": "composed_of"}}),
}

_norm = lambda s: re.sub(r"\s+", " ", (s or "")).strip().lower()

def _find_chunk(full_text: str, watch: str, window: int = 1400) -> tuple[str, int] | None:
    """Locate watch in full_text (whitespace-normalized search); return a
    sentence-ish window around it."""
    ft = _norm(full_text)
    w = _norm(watch)
    key = w[:60] if len(w) > 60 else w
    pos = ft.find(key)
    if pos < 0:
        pos = ft.find(w[:40])
    if pos < 0:
        return None
    start = max(0, pos - window)
    end = min(len(ft), pos + len(w) + window)
    return ft[start:end], pos

def _auto_bad(nodes_field: str, pattern: str) -> dict:
    """misc auto bad spec: the first role=value surface in the judge's binding."""
    first = nodes_field.split(";")[0]
    val = first.split("=", 1)[1].strip() if "=" in first else first.strip()
    key = val[:40]
    return {"pattern": pattern, "surface_any": [key]} if key else {"pattern": pattern}

def main():
    os.makedirs(LIB, exist_ok=True)
    fails = json.load(open(FAILS, encoding="utf-8"))
    assert len(fails) == 75, f"expected 75 failures, got {len(fails)}"

    # ---- frozen exam schema = union of the 5 SOFTB snapshots ----
    merged = {"version": 1, "meta_nodes": {}, "patterns": {}, "meta_edges": [], "family_roots": {}}
    seen_edges = set()
    for p in sorted(glob.glob(os.path.join(TMP, "runs", "kernel_v2", "SOFTB_PPR_*", "meta_snapshot.json"))):
        d = json.load(open(p, encoding="utf-8"))
        merged["patterns"].update(d.get("patterns", {}))
        merged["meta_nodes"].update(d.get("meta_nodes", {}))
        for e in d.get("meta_edges", []):
            k = json.dumps(e, sort_keys=True)
            if k not in seen_edges:
                seen_edges.add(k); merged["meta_edges"].append(e)
        merged["family_roots"].update(d.get("family_roots", {}))
    json.dump(merged, open(os.path.join(LIB, "schema_frozen.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[lib] frozen schema: {len(merged['patterns'])} patterns (union of 5 snapshots)")

    # ---- full texts ----
    texts = {}
    for pid in PAPER_META:
        blocks = load_paper_blocks(pid)
        texts[pid] = full_text_from_blocks(blocks) if blocks else ""
        print(f"[lib] {pid}: {len(texts[pid])} chars")

    # ---- load concept graphs; failures join by index i into hyperedges ----
    cgs = {}
    for pid in PAPER_META:
        cg = json.load(open(os.path.join(TMP, "runs", "kernel_v2", f"SOFTB_{pid}",
                                         "concept_graph.json"), encoding="utf-8"))
        conc = {}
        for cid, c in cg.get("concepts", {}).items():
            sv = c.get("surface_variants") or [{}]
            conc[cid] = (sv[0].get("surface", "") if isinstance(sv[0], dict) else "")
        cgs[pid] = (cg.get("hyperedges", []), conc)

    def _edge_rec(pid, idx):
        hes, conc = cgs[pid]
        e = hes[idx]
        prov = (e.get("provenance") or [{}])[0]
        role_surfaces = [{ "role": r, "surface": conc.get(n, "")}
                         for n, r in zip(e.get("node_ids", []), e.get("node_roles", []))]
        return {"pattern": e.get("pattern_type", ""), "evidence": prov.get("evidence", ""),
                "section": prov.get("section", ""), "role_surfaces": role_surfaces}

    # ---- failure cases ----
    cases, n_fail_locate = [], 0
    for i, f in enumerate(fails):
        pid = re.sub(r"[\\/].*$", "", f["_ppr"])   # strip '\judge_inline.json' tail
        domain, genre = PAPER_META[pid]
        er = _edge_rec(pid, f["i"])
        watch = er["evidence"]
        loc = _find_chunk(texts[pid], watch)
        if loc is None:
            n_fail_locate += 1
            print(f"[lib] WARN cannot locate case {i} in {pid}: {watch[:60]!r}")
            continue
        chunk, _pos = loc
        cls, overrides = CLS.get(i, ("misc", {}))
        expect = dict(overrides)
        expect.setdefault("watch", watch)
        if cls == "misc":
            expect["bad"] = _auto_bad(f.get("nodes", ""), f["pattern_type"])
        cases.append({
            "case_id": f"C{i:03d}", "kind": "fail", "failure_class": cls,
            "paper_id": pid, "domain": domain, "genre": genre,
            "source": "SOFTB inline-judge 2026-08-26",
            "section": er["section"], "chunk": chunk,
            "bad_edge": {"pattern_type": f["pattern_type"],
                         "role_surfaces": er["role_surfaces"],
                         "judge_reason": f.get("reason", "")},
            "expect": expect, "scored": cls != "misc",
        })
    print(f"[lib] fail cases built: {len(cases)} (locate failures: {n_fail_locate})")

    # ---- pass-regression cases (over-fix guard): kept edges the judge PASSED ----
    rng = random.Random(20260827)
    pass_cases = []
    fail_idx = {}
    for f in fails:
        pid = re.sub(r"[\\/].*$", "", f["_ppr"])
        fail_idx.setdefault(pid, set()).add(f["i"])
    for pid, (domain, genre) in PAPER_META.items():
        hes, _conc = cgs[pid]
        ok_edges = []
        for idx, e in enumerate(hes):
            if idx in fail_idx.get(pid, set()):
                continue   # judged a failure
            prov = (e.get("provenance") or [{}])[0]
            ev = prov.get("evidence", "")
            if not ev or prov.get("paper_id") != pid:
                continue
            ok_edges.append({"pattern": e.get("pattern_type", ""), "ev": ev,
                             "idx": idx})
        # stratify by pattern, sample up to 5 per paper
        by_pat = {}
        for oe in ok_edges:
            by_pat.setdefault(oe["pattern"], []).append(oe)
        picked = []
        pats = sorted(by_pat)
        while len(picked) < min(5, len(ok_edges)) and pats:
            for pt in list(pats):
                if by_pat[pt]:
                    picked.append(by_pat[pt].pop(rng.randrange(len(by_pat[pt]))))
                if len(picked) >= 5:
                    break
        for oe in picked:
            loc = _find_chunk(texts[pid], oe["ev"])
            if loc is None:
                continue
            chunk, _ = loc
            pass_cases.append({
                "case_id": f"P{pid[-4:]}-{len(pass_cases):03d}", "kind": "pass",
                "failure_class": "pass_regression", "paper_id": pid,
                "domain": domain, "genre": genre,
                "source": "SOFTB inline-judge 2026-08-26 (passed edge)",
                "section": "", "chunk": chunk,
                "bad_edge": None,
                "expect": {"watch": oe["ev"],
                           "good_binding": {"pattern": oe["pattern"]}},
                "scored": True, "soft": True,   # single-run miss = variance, not hard fail
            })
    print(f"[lib] pass-regression cases: {len(pass_cases)}")

    # ---- stratified 30% holdout over scored cases (class x domain) ----
    scored = [c for c in cases + pass_cases if c["scored"]]
    cells = {}
    for c in scored:
        cells.setdefault((c["failure_class"], c["domain"]), []).append(c)
    holdout_ids = set()
    for key, lst in sorted(cells.items()):
        lst2 = sorted(lst, key=lambda c: c["case_id"])
        rng2 = random.Random(hash(key) & 0xFFFFFFFF)
        rng2.shuffle(lst2)
        n_h = max(1, round(len(lst2) * 0.3))
        holdout_ids.update(c["case_id"] for c in lst2[:n_h])
        print(f"[lib] cell {key}: {len(lst2)} scored, {n_h} -> holdout")
    main = [c for c in cases + pass_cases if c["case_id"] not in holdout_ids]
    hold = [c for c in cases + pass_cases if c["case_id"] in holdout_ids]

    for name, lst in (("cases_main.jsonl", main), ("cases_holdout.jsonl", hold)):
        with open(os.path.join(LIB, name), "w", encoding="utf-8") as fh:
            for c in lst:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"[lib] {name}: {len(lst)} cases")

if __name__ == "__main__":
    main()
