# -*- coding: utf-8 -*-
"""Unit tests for the paper-anchored derived-absence rule (AirQA stage-1
explosion fix): absence(e, f) is derived only when some paper evaluating e
ALSO evaluates family f on >=3 distinct entities. Cross-domain trivial pairs
must NOT be derived; the informative PS16 form must survive."""
import sys

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
from kb_compiler.views.compiler import build_coverage  # noqa: E402


def _res(ent, subj, pid):
    return {"kind": "result", "paper_id": pid,
            "method_ref": {"surface": ent, "canonical": ent},
            "dims": {"subject": subj}, "measure": {"value": "1.0"}}


VOCAB = {"subject": [{"family": "FamA", "members": ["DatasetA"]},
                     {"family": "FamB", "members": ["DatasetB"]},
                     {"family": "FamC", "members": ["DatasetC"]}]}
REGISTRY = {"surface_index": {f"m{i}": f"m{i}" for i in range(1, 8)},
            "entities": [{"canonical": f"m{i}", "aliases": [],
                          "entity_id": f"m{i}"} for i in range(1, 8)]}
MANIFEST = {f"p{k}": {"paper_id": f"p{k}", "title": f"P{k}"} for k in (1, 2)}


def _coverage(recs):
    return build_coverage(recs, REGISTRY, VOCAB, MANIFEST)


def test_paper_anchored_absence_derived():
    # P1: FamA has m1,m2,m3; FamB has m1,m2,m4 (3 entities in P1).
    # m3 is evaluated in P1 (FamA) but has no FamB -> derived (P1 itself
    # evaluates FamB on >=3 entities).
    recs = [_res("m1", "DatasetA", "p1"), _res("m2", "DatasetA", "p1"),
            _res("m3", "DatasetA", "p1"),
            _res("m1", "DatasetB", "p1"), _res("m2", "DatasetB", "p1"),
            _res("m4", "DatasetB", "p1")]
    cov = _coverage(recs)
    pairs = {(d["entity"], d["subject_family"]) for d in cov["absences_derived"]}
    assert ("m3", "FamB") in pairs, pairs


def test_cross_domain_trivial_not_derived():
    # P2 evaluates FamC on 3 entities (corpus-popular), but P1 (where m1
    # lives) has zero FamC evaluations -> absence(m1, FamC) is cross-domain
    # trivial and must NOT be derived.
    recs = [_res("m1", "DatasetA", "p1"), _res("m2", "DatasetA", "p1"),
            _res("m3", "DatasetA", "p1"),
            _res("m5", "DatasetC", "p2"), _res("m6", "DatasetC", "p2"),
            _res("m7", "DatasetC", "p2")]
    cov = _coverage(recs)
    pairs = {(d["entity"], d["subject_family"]) for d in cov["absences_derived"]}
    assert not any(f == "FamC" for _, f in pairs), pairs
    assert not any(e in ("m1", "m2", "m3") and f == "FamC" for e, f in pairs)


def test_family_below_anchor_threshold_not_derived():
    # FamB evaluated in P1 on only 2 entities -> below the >=3 anchor.
    recs = [_res("m1", "DatasetA", "p1"), _res("m2", "DatasetA", "p1"),
            _res("m3", "DatasetA", "p1"),
            _res("m1", "DatasetB", "p1"), _res("m2", "DatasetB", "p1")]
    cov = _coverage(recs)
    pairs = {(d["entity"], d["subject_family"]) for d in cov["absences_derived"]}
    assert not any(f == "FamB" for _, f in pairs), pairs


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
