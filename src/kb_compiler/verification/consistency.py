# -*- coding: utf-8 -*-
"""L3 cross-record consistency: deterministic detection -> arbitration queue.

Four checks (block-2 obligations from the gold backtest, now delivered):
 1. same-paper drift   : same (entity, metric, subject, role) WITHIN one
                         comparability band (setup canonicals + budget bucket,
                         aligned with views.compiler.build_matrix) twice in one
                         paper with different values (batch1 note③: NGU 1344.0
                         vs 1354.4 — both kept, conflict DETECTED here,
                         resolution arbitrated). Cross-band value differences
                         are legitimate (different experiments), not drift
                         (tool debt fix 09-07: unbanded key overcounted 1437).
 2. cited-vs-source    : epistemic=cited value in paper X vs the source paper's
                         own demonstrated/stated value — numeric tolerance
                         MultiHiertt-style (rel 1%); mismatch -> queue
 3. condition equivalence candidates: bands whose setup strings differ but
                         normalize to the same vocab canonicals (batch1 oblig①
                         comparability banding input)
 4. relation symmetry  : concurrent_with/compares_with recorded one-directional
                         only, or motivated_by in both directions -> queue

Detection is deterministic; SEMANTIC resolution is NEVER automatic — output is
a queue for the arbitration channel (spec governance).
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

REL_TOL = 0.01  # MultiHiertt-style numeric tolerance (evaluation protocol §)


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _ref_name(ref):
    if isinstance(ref, dict):
        return ref.get("canonical") or ref.get("surface") or ""
    return str(ref or "")


def _num(s):
    s = str(s or "").replace("$", "").replace(",", "")
    s = re.sub(r"(?<=\d)[\s~]+(?=\d)", "", s)
    m = re.search(r"(-?\d+\.?\d*)\s*([kKmMbB])?", s)
    if not m:
        return None
    v = float(m.group(1))
    return v * {"k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6, "b": 1e9, "B": 1e9}.get(m.group(2) or "", 1)


def _close(a, b) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    return denom > 0 and abs(a - b) / denom <= REL_TOL


def flatten(records_by_paper, exclude=("canary",)):
    for pid, payload in records_by_paper.items():
        if pid in exclude:
            continue
        recs = payload.get("records", payload) if isinstance(payload, dict) else payload
        for r in recs:
            yield pid, r


def check_drift_same_paper(records_by_paper):
    from kb_compiler.views.compiler import _budget_bucket  # shared banding defn
    groups = defaultdict(list)
    for pid, r in flatten(records_by_paper):
        if r.get("kind") != "result":
            continue
        meas = r.get("measure") or {}
        dims = r.get("dims") or {}
        setup = tuple(sorted({_norm(x) for x in (dims.get("setup") or [])}))
        key = (pid, _norm(_ref_name(r.get("method_ref"))), _norm(meas.get("metric")),
               _norm(dims.get("subject")), _norm(r.get("role")),
               setup, _budget_bucket(dims.get("budget")))
        groups[key].append(r)
    out = []
    for key, rs in groups.items():
        vals = [( _num((x.get("measure") or {}).get("value")), x) for x in rs]
        vals = [(v, x) for v, x in vals if v is not None]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if not _close(vals[i][0], vals[j][0]):
                    out.append({
                        "check": "drift_same_paper", "paper_id": key[0], "entity": key[1],
                        "metric": key[2], "subject": key[3],
                        "band": {"setup": list(key[5]), "budget_bucket": key[6]},
                        "values": [str((vals[i][1].get("measure") or {}).get("value")),
                                   str((vals[j][1].get("measure") or {}).get("value"))],
                        "record_ids": [vals[i][1].get("id"), vals[j][1].get("id")],
                        "action": "keep_both_flag_conflict",  # batch1: both values may be
                        # legitimately different (abstract vs table) — resolution arbitrated
                    })
    return out


def check_cited_vs_source(records_by_paper, registry):
    """cited numbers in paper X vs source paper's own records."""
    own = {}   # (entity_norm, metric_norm) -> list of (value_num, record)
    cited = []
    for pid, r in flatten(records_by_paper):
        if r.get("kind") != "result":
            continue
        ent = _norm(_ref_name(r.get("method_ref")))
        metric = _norm((r.get("measure") or {}).get("metric"))
        v = _num((r.get("measure") or {}).get("value"))
        if not ent or v is None:
            continue
        if r.get("epistemic") == "cited":
            cited.append((pid, ent, metric, v, r))
        else:
            own.setdefault((ent, metric), []).append((v, r))
    # entity -> in_corpus paper (source of truth)
    ent_paper = {}
    for e in registry.get("entities", []):
        if e.get("in_corpus_paper_id"):
            ent_paper[_norm(e["canonical"])] = e["in_corpus_paper_id"]
            for a in e.get("aliases", []):
                ent_paper[_norm(a)] = e["in_corpus_paper_id"]
    out = []
    for pid, ent, metric, v, r in cited:
        src_paper = ent_paper.get(ent)
        if not src_paper:
            continue
        for (oent, ometric), vals in own.items():
            if oent != ent:
                continue
            for ov, orec in vals:
                if orec.get("paper_id") == src_paper and (ometric == metric or not ometric or not metric):
                    if not _close(v, ov):
                        out.append({
                            "check": "cited_vs_source", "citing_paper": pid,
                            "source_paper": src_paper, "entity": ent, "metric": metric,
                            "cited_value": str((r.get("measure") or {}).get("value")),
                            "source_value": str((orec.get("measure") or {}).get("value")),
                            "record_ids": [r.get("id"), orec.get("id")],
                            "action": "arbitration_queue",
                        })
    return out


def check_band_equivalence(vocab):
    """setup canonical alias classes — different surface, same canonical
    (comparability banding input; deterministic from vocab v1)."""
    classes = []
    for s in vocab.get("setup", []):
        aliases = [a for a in s.get("aliases", []) if _norm(a) != _norm(s["canonical"])]
        if aliases:
            classes.append({"canonical": s["canonical"], "equivalent_surfaces": aliases})
    return classes


def check_relation_symmetry(records_by_paper):
    pairs = defaultdict(list)
    for pid, r in flatten(records_by_paper):
        if r.get("kind") != "lineage":
            continue
        f, t = _norm(_ref_name(r.get("from_method_ref"))), _norm(_ref_name(r.get("to_method_ref")))
        pairs[tuple(sorted((f, t)))].append((f, t, r.get("relation"), pid, r.get("id")))
    out = []
    for (a, b), recs in pairs.items():
        rels = {(f, t, rel) for f, t, rel, _, _ in recs}
        for (f1, t1, r1) in rels:
            if r1 == "motivated_by" and (t1, f1, "motivated_by") in rels:
                out.append({"check": "symmetry_violation", "entities": [a, b],
                            "issue": "motivated_by in both directions",
                            "records": [x[4] for x in recs], "action": "arbitration_queue"})
        sym_rels = {"concurrent_with", "compares_with"}
        one_way = [(f1, t1, r1) for (f1, t1, r1) in rels if r1 in sym_rels]
        for (f1, t1, r1) in one_way:
            if (t1, f1, r1) not in rels:
                out.append({"check": "symmetric_single_sided", "entities": [f1, t1],
                            "relation": r1, "issue": "recorded one direction only (OK if single source; flagged for view dedup)",
                            "records": [x[4] for x in recs if x[2] == r1], "action": "note"})
    return out


def run_all(records_by_paper, registry, vocab):
    report = {
        "drift_same_paper": check_drift_same_paper(records_by_paper),
        "cited_vs_source": check_cited_vs_source(records_by_paper, registry),
        "band_equivalence_classes": check_band_equivalence(vocab),
        "relation_symmetry": check_relation_symmetry(records_by_paper),
    }
    report["summary"] = {k: len(v) for k, v in report.items()}
    report["arbitration_queue"] = [x for x in
                                   report["drift_same_paper"] + report["cited_vs_source"]
                                   + report["relation_symmetry"]
                                   if x.get("action") == "arbitration_queue"]
    return report


if __name__ == "__main__":
    sys.path.insert(0, "C:/Users/D0n9/Desktop/CompileScholar/src")
    import argparse
    from kb_compiler.records.common import load_json, save_json
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = run_all(load_json(args.records, {}), load_json(args.registry, {}),
                  load_json(args.vocab, {}))
    save_json(rep, args.out)
    print(json.dumps(rep["summary"], ensure_ascii=False), "| arbitration queue:",
          len(rep["arbitration_queue"]), flush=True)
