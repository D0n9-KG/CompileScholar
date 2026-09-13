# -*- coding: utf-8 -*-
"""Block 2: four-view compiler (+narrative layer, PaperScope lesson #3).

build_views(checked_records, registry, vocab, manifest) -> views dict:
  matrix    : (subject x metric) tables, cells banded by comparability
              (subject family / setup canonicals / budget magnitude)
  genealogy : entity nodes (with year) + lineage edges + transitive chains
              + as-of(t) filter support
  coverage  : entity x subject-family grid; extracted absences (tri-state)
              kept separate from derived empty cells; metadata flags
              (reports_variance / reports_ablation)
  cards     : per in-corpus entity method card incl. delta parsing
              (explicit record deltas + computed same-band pairwise deltas)
  narrative : shift-record chains + family evolution lines (thematic, not
              chronological流水账 — the PaperScope trend-类 failure fix)

Field-usage counters (FIELD_USES) feed the governance usage audit
(subtraction channel: fields never consumed by views are retirement
candidates).
"""
from __future__ import annotations

import re
from collections import defaultdict

FIELD_USES = defaultdict(int)


def _use(rec: dict, field: str):
    v = rec.get(field)
    if v not in (None, "", [], {}):
        FIELD_USES[field] += 1
    return v


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _ref_name(ref) -> str:
    """canonical name from a method_ref dict (or plain string)."""
    if isinstance(ref, dict):
        return ref.get("canonical") or ref.get("surface") or ""
    return str(ref or "")


def _ref_id(ref) -> str | None:
    return ref.get("entity_id") if isinstance(ref, dict) else None


def round_robin_by_paper(items):
    """IL-B4: interleave records so every paper gets a voice before any paper
    gets a second slot — file-order truncation silently dropped cross-paper
    evidence (A11: the Fedus 2020 counter-finding on PER sat late in file
    order and was cut by the findings k-cap; deterministic, no relevance
    model). Shared by views.build_cards and tools.findings."""
    by_p = defaultdict(list)
    for r in items:
        by_p[r.get("paper_id")].append(r)
    out, i = [], 0
    while any(i < len(v) for v in by_p.values()):
        for p in sorted(by_p):
            if i < len(by_p[p]):
                out.append(by_p[p][i])
        i += 1
    return out


def _num(s):
    """leading numeric value of a string ('87.5%'->87.5, '200M frames'->None-safe 200e6).
    LaTeX pre-cleanup: mineru renders numbers as '$2 9 8 0 \\pm 3 5$' — join
    digit-space-digit runs before parsing (measured bug 2026-09-05: parsed 2
    instead of 2980)."""
    s = str(s or "").replace("$", "").replace("\\,", "")
    s = re.sub(r"(?<=\d)[\s~]+(?=\d)", "", s)
    m = re.search(r"(\d[\d,]*\.?\d*)\s*([kKmMbB])?", s)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    mult = {"k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6, "b": 1e9, "B": 1e9}.get(m.group(2) or "", 1)
    return v * mult


def _budget_bucket(budget) -> str:
    n = _num(budget)
    if n is None:
        return "unspecified"
    low = str(budget).lower()
    unit_m = re.search(
        r"(frames|steps|episodes|hours|days|minutes|seconds|gpu|flops|samples|"
        r"interactions|epochs|tokens|particles|atoms|molecules|cells|cycles|"
        r"iterations|trials|runs|pa|kpa|mpa|kelvin|mol|kg)", low)
    if unit_m:
        unit = unit_m.group(1)
    else:
        # open fallback (granular-pilot adaptation): first alpha token AFTER the
        # number is taken as the unit — whitelist can never enumerate all domains
        tail = re.search(r"\d\s*([a-zμ/·°%]{2,14})", low)
        unit = tail.group(1) if tail else "units"
    import math
    mag = int(math.floor(math.log10(max(n, 1))))
    return f"1e{mag} {unit}"


def _year_of(rec_paper: str, manifest: dict) -> int | None:
    m = manifest.get(rec_paper) or {}
    return m.get("arxiv_year") or m.get("venue_year")


def flatten_records(records_by_paper: dict, exclude=("canary",)) -> list[dict]:
    out = []
    for pid, payload in records_by_paper.items():
        if pid in exclude:
            continue
        recs = payload.get("records", payload) if isinstance(payload, dict) else payload
        for r in recs:
            out.append(r)
    return out


# ---------- view A: comparison matrix ----------

def _eff_subject(r):
    """Effective subject for GROUPING views (matrix/coverage): governed dims
    first, then visible dims_new residue. PSV-IL-1 (2026-09-12): the IL-2
    postcheck fix moved non-vocab scalar subjects (dataset/task labels like
    SQuAD, MNLI, COCOHumanParts) out of dims into dims_new — typed queries get
    clean dims, but grouping views legitimately key on the residue labels
    (matrix collapsed 151->7 tables without this). Rows keep record_id for
    verification; residue labels remain visible pending vocab governance."""
    dims = r.get("dims") or {}
    meas = r.get("measure") or {}
    subj = dims.get("subject") or meas.get("subject")
    if not subj:
        dn = r.get("dims_new") or {}
        cand = dn.get("dims.subject") or dn.get("subject")
        if isinstance(cand, list) and cand:
            subj = cand[0]
    return subj


def build_matrix(recs, vocab, manifest):
    subj_families = {}
    for fam in vocab.get("subject", []):
        for m in fam.get("members", []):
            subj_families[_norm(m)] = fam.get("family") or m
    tables = defaultdict(lambda: defaultdict(list))  # (subject,metric) -> entity -> cells
    for r in recs:
        if r.get("kind") != "result":
            continue
        meas = _use(r, "measure") or {}
        ent = _ref_name(_use(r, "method_ref"))
        dims = r.get("dims") or {}
        subj = _norm(_eff_subject(r) or "")  # PSV-IL-1: dims + dims_new residue
        metric = _norm(_use(meas, "metric"))
        if not ent or not subj:
            continue
        band = {
            "subject_family": subj_families.get(subj, subj),
            "setup": sorted({_norm(x) for x in (dims.get("setup") or [])}),
            "budget_bucket": _budget_bucket(dims.get("budget")),
        }
        tables[(subj, metric)][ent].append({
            "value": meas.get("value"), "unit": meas.get("unit"),
            "direction": meas.get("direction"), "aggregation": _use(meas, "aggregation"),
            "timepoint": meas.get("timepoint"), "role": _use(r, "role"),
            "epistemic": _use(r, "epistemic"), "delta": r.get("delta"),
            "band": band, "paper_id": r.get("paper_id"), "record_id": r.get("id"),
            "quote": (r.get("quote") or "")[:200],
            "provenance": "extracted",
        })
    return {"tables": {f"{k[0]}||{k[1]}": dict(v) for k, v in tables.items()}}


# ---------- view B: method genealogy ----------

def build_genealogy(recs, registry, manifest):
    byid = {e["entity_id"]: e for e in registry.get("entities", [])}
    nodes = {}
    for e in registry.get("entities", []):
        year = None
        if e.get("in_corpus_paper_id"):
            year = _year_of(e["in_corpus_paper_id"], manifest)
        elif e.get("origin_year_cited"):
            year = (_use(e["origin_year_cited"], "value"))
        nodes[e["entity_id"]] = {"canonical": e["canonical"], "entity_type": e.get("entity_type"),
                                 "year": year, "in_corpus": bool(e.get("in_corpus_paper_id"))}
    edges = []
    for r in recs:
        if r.get("kind") != "lineage":
            continue
        rel = _use(r, "relation")
        f, t = r.get("from_method_ref"), r.get("to_method_ref")
        if not rel or not f or not t:
            continue
        fid, tid = _ref_id(f), _ref_id(t)
        edges.append({
            "from": fid or _norm(_ref_name(f)), "to": tid or _norm(_ref_name(t)),
            "from_name": _ref_name(f), "to_name": _ref_name(t),
            "relation": rel, "scope": r.get("scope") or {},
            "evidence_basis": _use(r, "evidence_basis"),
            "year": _year_of(r.get("paper_id"), manifest),
            "paper_id": r.get("paper_id"), "record_id": r.get("id"),
            "quote": (r.get("quote") or "")[:160], "provenance": "extracted",
        })
    # transitive chains over extends/improves/generalizes (ancestor closure)
    fwd = defaultdict(set)
    for e in edges:
        if e["relation"] in ("extends", "improves", "generalizes", "uses", "component_of"):
            fwd[e["from"]].add(e["to"])
    chains = {}
    for n in list(fwd):
        seen, stack = set(), [n]
        while stack:
            cur = stack.pop()
            for nxt in fwd.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        chains[n] = sorted(seen)
    return {"nodes": nodes, "edges": edges, "ancestor_closure": chains}


def as_of(genealogy, year: int):
    """as-of time travel: entities/edges visible at `year` (paper-year <= t;
    out-of-corpus cited-year <= t). Block-3 typed tool consumes this."""
    nodes = {k: v for k, v in genealogy["nodes"].items()
             if v["year"] is None or v["year"] <= year}
    edges = [e for e in genealogy["edges"] if (e["year"] or 0) <= year]
    return {"nodes": nodes, "edges": edges}


# ---------- view C: coverage map ----------

def build_coverage(recs, registry, vocab, manifest):
    fam_of = {}
    families = []
    for fam in vocab.get("subject", []):
        fname = fam.get("family") or (fam["members"][0] if fam.get("members") else None)
        if fname:
            families.append(fname)
            for m in fam.get("members", []):
                fam_of[_norm(m)] = fname
    grid = defaultdict(dict)          # entity -> family -> {n_results, papers}
    flags = defaultdict(lambda: {"reports_variance": False, "reports_ablation": False,
                                 "reports_repeats": False})
    # PSV4 (2026-09-12): coverage/absence layer = GOVERNED entities only.
    # The F24 table-channel added raw row-head surfaces ('Original', '32-bit
    # AdamW', dataset names) that inflated coverage entities 189->363 and
    # absences_derived 375->15512 via the entity x family cross product —
    # junk absence claims ("Original has no results on freelaw"). Absence
    # assertions require identity certainty: a surface enters coverage once
    # registry governance (round2/arbitration) resolves it. The MATRIX view
    # keeps raw surfaces (data layer — that is the table channel's point).
    _cov_si = {_norm(k) for k in (registry.get("surface_index", {}) or {})}
    for r in recs:
        if r.get("kind") != "result":
            continue
        ent = _ref_name(r.get("method_ref"))
        if _norm(ent) not in _cov_si:
            continue
        dims = r.get("dims") or {}
        # PSV4 (2026-09-12): grid keeps the PSV-IL-1 residue read (rich
        # descriptive coverage — dims-only collapsed the grid to 12 entities
        # because IL-2 moved dataset/task labels to dims_new). Governance is
        # applied at the CLAIM layer instead: derived absences are restricted
        # to vocab families below (absence claims need a governed taxonomy on
        # both axes; junk 'popular' families — 'unspecified' x52, 'mean' x15 —
        # inflated derived 375->8062 and fed find_gap with noise).
        _subj_n = _norm(_eff_subject(r))
        if not _subj_n:
            continue   # no subject anywhere: not a coverage datapoint
        fam = fam_of.get(_subj_n, _subj_n)
        cell = grid[ent].setdefault(fam, {"n_results": 0, "papers": set()})
        cell["n_results"] += 1
        cell["papers"].add(r.get("paper_id"))
        agg = str((r.get("measure") or {}).get("aggregation") or "")
        val_s = str((r.get("measure") or {}).get("value") or "")
        if ("std" in agg.lower() or "±" in val_s or "\\pm" in val_s
                or re.search(r"\+\s*[-−]|\bstderr\b", val_s + agg, re.I)):
            flags[ent]["reports_variance"] = True
        if r.get("role") == "ablation":
            flags[ent]["reports_ablation"] = True
        if dims.get("repeats"):
            flags[ent]["reports_repeats"] = True
    grid = {e: {f: {**c, "papers": sorted(c["papers"])} for f, c in fams.items()}
            for e, fams in grid.items()}
    extracted = []
    for r in recs:
        if r.get("kind") != "absence":
            continue
        extracted.append({
            "subject": _use(r, "subject"), "missing": _use(r, "missing"),
            "absence_type": _use(r, "absence_type"), "evidence": r.get("evidence"),
            "paper_id": r.get("paper_id"), "record_id": r.get("id"),
            "quote": (r.get("quote") or "")[:160], "provenance": "extracted",
        })
    # derived absences: entity evaluated in family F but sibling families with
    # >=3 evaluating entities stay empty for it -> "not evaluated" (derived)
    fam_counts = defaultdict(int)
    for e, fams in grid.items():
        for f in fams:
            fam_counts[f] += 1
    # PSV4: popular = governed vocab families only (see grid comment above)
    _gov_fams = {_norm(f) for f in families}
    popular = {f for f, c in fam_counts.items() if c >= 3 and _norm(f) in _gov_fams}
    derived = []
    for e, fams in grid.items():
        for f in popular - set(fams):
            derived.append({"entity": e, "subject_family": f,
                            "claim": f"no result records for {e} on {f} in corpus",
                            "provenance": "derived"})
    return {"grid": grid, "flags": dict(flags), "families": families,
            "absences_extracted": extracted, "absences_derived": derived}


# ---------- view D: method cards (incl. delta parsing) ----------

def build_cards(recs, registry, manifest, matrix):
    by_ent = defaultdict(list)
    # IL-B4 dossier pass: a card is the entity's CROSS-PAPER evidence dossier.
    # Findings whose scope/target resolved elsewhere but whose claim names a
    # registered entity attach to that entity too (A11: the Fedus 2020
    # counter-finding about PER is scoped to the replay study, not PER —
    # dossier must aggregate it). One compiled regex over all registry
    # surfaces, word-boundary anchored, deterministic.
    surfaces = {}
    for e in registry.get("entities", []):
        surfaces[_norm(e["canonical"])] = _norm(e["canonical"])
        for a in e.get("aliases", []):
            surfaces[_norm(a)] = _norm(e["canonical"])
    mention_pat = re.compile(r"\b(?:" + "|".join(
        sorted((re.escape(s) for s in surfaces), key=len, reverse=True)) + r")\b") \
        if surfaces else None
    for r in recs:
        # scope_ref is resolved+renamed to scope_ref_ref by slot.py (tool debt
        # fix 09-07: reading the raw name silently dropped ~2.4k findings from
        # cards — the rename mismatch the usage audit first surfaced)
        attached = False
        for f in ("method_ref", "from_method_ref", "scope_ref_ref"):
            ref = r.get(f)
            name = _ref_name(ref)
            if name:
                by_ent[_norm(name)].append(r)
                attached = True
                break
        if r.get("kind") == "finding":
            if not attached:
                tname = _ref_name(r.get("target_ref_ref"))
                if tname:
                    by_ent[_norm(tname)].append(r)
            if mention_pat:
                for mm in {_norm(x) for x in
                           mention_pat.findall(str(r.get("claim") or ""))}:
                    canon = surfaces.get(mm)
                    if canon:
                        by_ent[canon].append(r)
    cards = {}
    for e in registry.get("entities", []):
        if not e.get("in_corpus_paper_id"):
            continue
        pid = e["in_corpus_paper_id"]
        m = manifest.get(pid) or {}
        rs = by_ent.get(_norm(e["canonical"]), []) + \
            [r for a in e.get("aliases", []) for r in by_ent.get(_norm(a), [])]
        seen_ids, uniq = set(), []
        for r in rs:
            if r.get("id") not in seen_ids:
                seen_ids.add(r.get("id"))
                uniq.append(r)
        configs = [r for r in uniq if r["kind"] == "config"]
        results = [r for r in uniq if r["kind"] == "result"]
        # IL-B4 dossier ordering: findings that NAME the entity first (dossier
        # centrality — "PER does not significantly affect..." outranks passing
        # mentions), each group paper-round-robin (cross-paper breadth), then
        # the 40 cap. File order alone cut the decisive cross-paper evidence.
        findings_all = [r for r in uniq if r["kind"] == "finding"]
        enames = {_norm(e["canonical"]), *(_norm(a) for a in e.get("aliases", []))}
        direct, rest = [], []
        for r in findings_all:
            cn = _norm(r.get("claim"))
            (direct if any(n and n in cn for n in enames) else rest).append(r)
        findings = round_robin_by_paper(direct) + round_robin_by_paper(rest)
        lin_out = [r for r in uniq if r["kind"] == "lineage"
                   and _norm(_ref_name(r.get("from_method_ref"))) in
                   {_norm(e["canonical"]), *(_norm(a) for a in e.get("aliases", []))}]
        # delta parsing: explicit (record-carried) + computed (same band pairs)
        deltas_explicit = [{"record_id": r.get("id"), "delta": r.get("delta"),
                            "metric": (r.get("measure") or {}).get("metric"),
                            "quote": (r.get("quote") or "")[:120]}
                           for r in results if r.get("delta")]
        cards[e["entity_id"]] = {
            "canonical": e["canonical"], "aliases": e.get("aliases", []),
            "paper_id": pid, "title": m.get("title"),
            "year": m.get("arxiv_year"), "venue": m.get("venue"),
            "venue_year": m.get("venue_year"), "authors": m.get("authors"),
            "affiliations": m.get("affiliations"),
            "configs": [{"item": r.get("item"), "value": r.get("value"),
                         "role": r.get("role"), "record_id": r.get("id"),
                         "quote": (r.get("quote") or "")[:140]} for r in configs[:40]],
            "main_results": [{"subject": (r.get("dims") or {}).get("subject"),
                              "metric": (r.get("measure") or {}).get("metric"),
                              "value": (r.get("measure") or {}).get("value"),
                              "role": r.get("role"), "epistemic": r.get("epistemic"),
                              "record_id": r.get("id")}
                             for r in results if r.get("role") != "ablation"][:40],
            "ablations": [{"variant": (r.get("dims") or {}).get("variant"),
                           "delta": r.get("delta"),
                           "value": (r.get("measure") or {}).get("value"),
                           "record_id": r.get("id")}
                          for r in results if r.get("role") == "ablation"][:30],
            "findings": [{"claim": r.get("claim"), "claim_type": r.get("claim_type"),
                          "strength": r.get("strength"),
                          "epistemic": r.get("epistemic"),  # schema v1.4 provenance axis
                          "paper_id": r.get("paper_id"),
                          "record_id": r.get("id")}
                         for r in findings[:40]],
            "lineage_out": [{"relation": r.get("relation"),
                             "to": _ref_name(r.get("to_method_ref")),
                             "record_id": r.get("id")} for r in lin_out[:30]],
            "deltas_explicit": deltas_explicit[:30],
            "provenance": "compiled",
        }
    return {"cards": cards}


def compute_pair_deltas(matrix, direction_hint="higher_better"):
    """Cross-entity computed deltas within same (subject,metric,band) —
    provenance=derived, kept separate from record-carried deltas."""
    out = []
    for key, ents in matrix["tables"].items():
        subj, metric = key.split("||")
        by_band = defaultdict(dict)
        for ent, cells in ents.items():
            for c in cells:
                bk = (tuple(c["band"]["setup"]), c["band"]["budget_bucket"])
                v = _num(c["value"])
                if v is None:
                    continue
                prev = by_band[bk].get(ent)
                if prev is None or v > prev[0]:
                    by_band[bk][ent] = (v, c)
        for band, entsv in by_band.items():
            items = sorted(entsv.items(), key=lambda kv: -kv[1][0])
            for i in range(len(items) - 1):
                (a, (va, ca)), (b, (vb, cb)) = items[i], items[i + 1]
                out.append({"subject": subj, "metric": metric, "band": list(band),
                            "leader": a, "follower": b, "gap": round(va - vb, 4),
                            "leader_record": ca["record_id"], "follower_record": cb["record_id"],
                            "provenance": "derived"})
    return out


# ---------- narrative layer (PaperScope lesson #3) ----------

def build_narrative(recs, registry, manifest, genealogy):
    shifts = []
    for r in recs:
        if r.get("kind") == "shift":
            shifts.append({
                "from_state": _use(r, "from_state"), "to_state": _use(r, "to_state"),
                "driver": _use(r, "driver"), "scope": r.get("scope"),
                "time_range": r.get("time_range"), "source_type": r.get("source_type"),
                "year": _year_of(r.get("paper_id"), manifest),
                "paper_id": r.get("paper_id"), "record_id": r.get("id"),
                "quote": (r.get("quote") or "")[:200], "provenance": "extracted",
            })
    shifts.sort(key=lambda s: (s["scope"] or "", s["year"] or 9999))
    # thematic family lines: connected components over directed edges,
    # rendered per-theme in time order (NOT global chronological流水账)
    adj = defaultdict(list)
    for e in genealogy["edges"]:
        if e["relation"] in ("extends", "improves", "generalizes", "replaces"):
            adj[e["from_name"]].append(e)
    lines = []
    for src, edges in sorted(adj.items(), key=lambda kv: -len(kv[1])):
        edges = sorted(edges, key=lambda e: e["year"] or 9999)
        lines.append({
            "theme": src,
            "steps": [{"year": e["year"], "relation": e["relation"],
                       "from": e["from_name"], "to": e["to_name"],
                       "scope": e.get("scope"), "paper_id": e["paper_id"],
                       "record_id": e["record_id"], "quote": e["quote"]}
                      for e in edges],
            "provenance": "compiled",
        })
    return {"shifts": shifts, "family_lines": lines[:60]}


# ---------- driver ----------

def build_notation_index(recs):
    """v1.3: symbol/quantity definitions per paper (grounds formula-valued
    config/result records; formula-heavy domains produce these structurally)."""
    out = []
    for r in recs:
        if r.get("kind") != "notation":
            continue
        out.append({"paper_id": r.get("paper_id"), "symbol": _use(r, "symbol"),
                    "quantity": r.get("quantity"), "definition": _use(r, "definition"),
                    "unit": r.get("unit"),
                    "scope_ref": _ref_name(r.get("scope_ref_ref")),
                    "record_id": r.get("id"), "quote": (r.get("quote") or "")[:120],
                    "provenance": "extracted"})
    return out


def build_views(records_by_paper, registry, vocab, manifest, exclude=("canary",)):
    recs = flatten_records(records_by_paper, exclude=exclude)
    matrix = build_matrix(recs, vocab, manifest)
    genealogy = build_genealogy(recs, registry, manifest)
    coverage = build_coverage(recs, registry, vocab, manifest)
    cards = build_cards(recs, registry, manifest, matrix)
    narrative = build_narrative(recs, registry, manifest, genealogy)
    pair_deltas = compute_pair_deltas(matrix)
    notation = build_notation_index(recs)
    stats = {
        "n_records": len(recs),
        "matrix_tables": len(matrix["tables"]),
        "genealogy_edges": len(genealogy["edges"]),
        "genealogy_nodes_with_year": sum(1 for n in genealogy["nodes"].values() if n["year"]),
        "coverage_entities": len(coverage["grid"]),
        "absences_extracted": len(coverage["absences_extracted"]),
        "absences_derived": len(coverage["absences_derived"]),
        "cards": len(cards["cards"]),
        "pair_deltas_derived": len(pair_deltas),
        "shifts": len(narrative["shifts"]),
        "family_lines": len(narrative["family_lines"]),
        "notation": len(notation),
        "field_uses": dict(FIELD_USES),
    }
    return {"matrix": matrix, "genealogy": genealogy, "coverage": coverage,
            "cards": cards, "narrative": narrative, "pair_deltas": pair_deltas,
            "notation_index": notation, "stats": stats}


if __name__ == "__main__":
    import argparse, json, os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from kb_compiler.records.common import load_json, save_json
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    views = build_views(load_json(args.records, {}), load_json(args.registry, {}),
                        load_json(args.vocab, {}),
                        {r["paper_id"]: r for r in load_json(args.manifest, [])})
    save_json(views, args.out)
    print(json.dumps(views["stats"], ensure_ascii=False, indent=1))
