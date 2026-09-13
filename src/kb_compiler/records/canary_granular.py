# -*- coding: utf-8 -*-
"""Granular-domain canary: same 5-fact + 4-trap structure as canary.py,
synthetic granular-rheology content. Tests the TRAP DETECTOR's domain
invariance (pre-reg GRANULAR-PILOT-PREREG.md 判读4)."""
from __future__ import annotations

import json
import re
import sys

CANARY_PID = "canary"

CANARY_TEXT = """# PSR: Pressure-Scaled Rheology for Dense Granular Flows

Alice Martin, Bob Chen. Institute of Soft Matter Mechanics.

## Abstract
We propose PSR, a pressure-scaled rheology for dense granular flows. At an
inertial number I = 0.05, PSR predicts a shear stress of 1.85 kPa, while the
classical mu(I) model gives 1.42 kPa under the same conditions. Unlike
particle-based approaches, PSR does not require contact resolution. The
agreement with simulation data holds only for quasi-static regimes (I < 0.1).

## 1. Introduction
Dense granular rheology is usually described by the inertial number framework.
PSR extends the classical mu(I) rheology of Jop et al. (2006) with a pressure
scaling term. Henann and Kamrin reported a fluidity of 0.42 in annular shear
geometries.

## 2. Model
PSR introduces a pressure scaling exponent gamma = 0.25. The interparticle
friction coefficient is set to 0.3 in all simulations. The model reduces to
mu(I) rheology in the limit of vanishing pressure gradient.

## 3. Simulations
We use discrete element simulations in a two-dimensional annular shear cell.

Table 1: Shear stress at I = 0.05 (2D annular cell).
| Model | Stress (kPa) | Strain Rate (1/s) |
| PSR | 1.85 | 0.05 |
| mu(I) | 1.42 | 0.05 |
| PSR-cohesive | 2.10 | 0.05 |

Table 1 shows that PSR reaches 1.85 kPa while the cohesive variant PSR-cohesive
reaches 2.10 kPa; the classical mu(I) model underpredicts with 1.42 kPa.

## 4. Related Work
Nonlocal rheologies were introduced by Kamrin and Koval (2012). Prior works
typically resolve particle contacts explicitly.

## 5. Conclusion
We presented PSR and showed improved stress prediction in 2D annular shear.
We do not evaluate on three-dimensional shear geometries; future work may
extend PSR to that setting.
"""


def _method_of(rec):
    ref = rec.get("method_ref") or rec.get("from_method_ref") or {}
    return (ref.get("surface") or "").lower() if isinstance(ref, dict) else ""


def _num_in(s, num):
    return num in re.sub(r"[,\s]", "", str(s or ""))


def score(records):
    out = {"facts": {}, "traps": {}}
    R = [r for r in records if r.get("kind") != "overflow"]

    # F1: PSR 1.85 kPa main result
    out["facts"]["F1_main_result"] = any(
        r["kind"] == "result" and "psr" in _method_of(r)
        and "cohesive" not in _method_of(r)
        and _num_in((r.get("measure") or {}).get("value"), "1.85")
        for r in R)
    # F2: mu(I) 1.42 bound to mu(I) as baseline
    out["facts"]["F2_baseline_binding"] = any(
        r["kind"] == "result"
        and ("mu(i)" in _method_of(r) or "μ(i)" in _method_of(r)
             or "classical" in _method_of(r))
        and _num_in((r.get("measure") or {}).get("value"), "1.42")
        for r in R)
    # F3: friction coefficient 0.3 config
    out["facts"]["F3_config"] = any(
        r["kind"] == "config" and _num_in(r.get("value"), "0.3")
        and "friction" in (str(r.get("item")) + str(r.get("quote"))).lower()
        for r in R)
    # F4: PSR extends mu(I)
    out["facts"]["F4_lineage"] = any(
        r["kind"] == "lineage" and "psr" in (
            (r.get("from_method_ref") or {}).get("surface") or "").lower()
        and ("mu(i)" in ((r.get("to_method_ref") or {}).get("surface") or "").lower()
             or "jop" in json.dumps(r, ensure_ascii=False).lower())
        and r.get("relation") in ("extends", "improves", "generalizes", "uses")
        for r in R)
    # F5: explicit absence — no 3D evaluation
    out["facts"]["F5_absence"] = any(
        r["kind"] == "absence"
        and re.search(r"three[- ]dimensional|3d", (str(r.get("missing"))
                      + str(r.get("subject")) + str(r.get("quote"))).lower())
        and r.get("absence_type") in ("explicitly_stated", "not_reported")
        for r in R)

    # T1: negation flip — PSR affirmatively given contact resolution
    t1 = False
    for r in R:
        blob = json.dumps(r, ensure_ascii=False).lower()
        if "contact" not in blob:
            continue
        if r["kind"] == "lineage" and "psr" in _method_of(r) \
                and r.get("relation") in ("uses", "extends", "component_of"):
            t1 = True
        if r["kind"] == "config" and "psr" in _method_of(r) \
                and "contact" in str(r.get("item", "")).lower():
            t1 = True
        if r["kind"] == "finding" and re.search(
                r"psr[^.\"]{0,60}(uses|requires|resolves)[^.\"]{0,40}contact", blob):
            t1 = True
    out["traps"]["T1_negation_flip"] = t1

    # T2: Henann-Kamrin fluidity 0.42 as stated/demonstrated or bound to PSR
    t2 = False
    for r in R:
        if r["kind"] == "result" and _num_in((r.get("measure") or {}).get("value"), "0.42"):
            if r.get("epistemic") in ("stated", "demonstrated", None, ""):
                t2 = True
            if "psr" in _method_of(r):
                t2 = True
    out["traps"]["T2_epistemic_flip"] = t2

    # T3: quasi-static-only claim without condition (condition/dims/dims_new)
    t3 = False
    for r in R:
        if r["kind"] != "finding":
            continue
        blob = (str(r.get("claim")) + str(r.get("quote"))).lower()
        if ("quasi-static" in blob or "i < 0.1" in blob or "i<0.1" in blob) \
                and ("only" in blob or "holds" in blob or "agreement" in blob):
            cond = json.dumps({k: r.get(k) for k in ("condition", "dims", "dims_new")},
                              ensure_ascii=False).lower()
            if not re.search(r"quasi|0\.1|static", cond):
                t3 = True
    out["traps"]["T3_condition_drop"] = t3

    # T4: wrong number-metric-method bindings
    t4 = False
    for r in R:
        if r["kind"] != "result":
            continue
        m = _method_of(r)
        meas = r.get("measure") or {}
        val = str(meas.get("value") or "")
        met = str(meas.get("metric") or "").lower()
        role = r.get("role")
        stress_metric = ("stress" in met) or ("kpa" in met)
        rate_metric = ("strain" in met) or ("rate" in met) or ("1/s" in met)
        if "psr" in m and "cohesive" not in m and role == "main_result" \
                and _num_in(val, "1.42"):
            t4 = True
        if ("mu(i)" in m or "μ(i)" in m) and _num_in(val, "1.85") \
                and role != "baseline_comparison":
            t4 = True
        if (_num_in(val, "1.85") or _num_in(val, "1.42") or _num_in(val, "2.10")) \
                and rate_metric:
            t4 = True
        if _num_in(val, "0.05") and stress_metric:
            t4 = True
    out["traps"]["T4_number_binding"] = t4

    out["fact_recall"] = f"{sum(out['facts'].values())}/5"
    out["trap_fired"] = sum(out["traps"].values())
    return out


if __name__ == "__main__":
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    recs = (data.get(CANARY_PID) or {}).get("records", [])
    print(json.dumps(score(recs), ensure_ascii=False, indent=1))
