# -*- coding: utf-8 -*-
"""Canary: synthetic sentinel paper with known ground truth + trap detector.

Design pattern carried from the legacy harness (canary batch monitoring),
rebuilt for schema v1.1. The canary paper goes through the SAME slot+postcheck
pipeline as real papers; scoring is rule-based (no LLM judge — deterministic).

Facts (recall side, 7): F1 main result / F2 baseline binding / F3 config /
F4 lineage / F5 explicit absence / F6 wide-table column binding (v2) /
F7 cited-source finding survives (v3, schema v1.4).
Traps (precision side, 6): T1 negation-as-affirmation / T2 cited-result
epistemic flip / T3 condition drop / T4 near-duplicate number binding /
T5 wide-table column-label misbinding (v2, the A7 form measured in the PS16
rebuild manual read) / T6 own-paper finding epistemic flip (v3).

Gates: v1 (SMOKE-PREREG.md): fact_recall >= 4/5, trap_fired == 0.
v2 (PSFIX-PREREG.md FX-D, 2026-09-11): fact_recall >= 5/6, trap_fired == 0.
v3 (schema v1.4, 2026-09-13): fact_recall >= 6/7, trap_fired == 0.
CANARY_TEXT changed in v2 (Table 2) and v3 (TRACE sentence in Related Work)
— scores across text versions are not comparable.
"""
from __future__ import annotations

import json
import re
import sys

CANARY_PID = "canary"

CANARY_TEXT = """# NOVA: Nested Overlap Value Aggregation for Sample-Efficient Control

Jane Doe, John Smith. Institute of Example Research.

## Abstract
We propose NOVA, a method that aggregates nested value estimates for
sample-efficient control. NOVA achieves 87.5 points on BenchX (mean over
3 seeds), outperforming the uniform replay baseline UNIFORM at 64.2 points.
Unlike prior work, NOVA does not use prioritized sampling. The improvement
over UNIFORM is significant only in sparse-reward settings.

## 1. Introduction
Sample efficiency remains a central challenge. NOVA extends LUNA (Doe et al.,
2019) with an overlap-aware aggregator that reuses nested value estimates.
LUNA reported 91.3 points on BenchY in its original publication.

## 2. Method
NOVA maintains a replay buffer of size 2M transitions. The overlap weight
beta is set to 0.1 in all experiments. The aggregator computes nested means
over value estimates, which stabilizes training under sparse rewards.

## 3. Experiments
We evaluate on BenchX under the standard protocol (mean over 3 seeds).

Table 1: BenchX results.
| Method | Score | Success Rate |
| NOVA | 87.5 | 0.91 |
| NOVA-no-overlap | 79.1 | 0.85 |
| UNIFORM | 64.2 | 0.72 |

Table 1 shows that NOVA reaches 87.5 while the ablation without the overlap
aggregator (NOVA-no-overlap) reaches 79.1, a drop of 8.4 points. UNIFORM
reaches 64.2.

We further evaluate transfer across tasks. Table 2 reports per-task scores.

Table 2: Multi-task transfer (accuracy).
<table><tr><td rowspan="2">Method</td><td colspan="2">NLP</td><td colspan="2">Vision</td></tr><tr><td>QA RoBERTa-L</td><td>Summ T5-B</td><td>CLS Swin-T</td><td>Seg ViT-S</td></tr><tr><td>NOVA</td><td>71.8</td><td>68.3</td><td>74.9</td><td>69.4</td></tr><tr><td>UNIFORM</td><td>65.2</td><td>61.7</td><td>66.8</td><td>63.5</td></tr></table>

Table 2 shows that NOVA reaches 74.9 on image classification (CLS Swin-T),
its strongest transfer task, while UNIFORM reaches 66.8 on the same task.

## 4. Related Work
Value aggregation for control was introduced by LUNA (Doe et al., 2019).
Prior methods typically rely on prioritized sampling to accelerate learning.
TRACE (Kim et al., 2020) argues that prioritized sampling introduces bias
into value estimates, and corrects it with per-sample importance weights.

## 5. Conclusion
We presented NOVA and showed strong results on BenchX. We do not evaluate on
continuous control tasks; future work may extend NOVA to that setting.
"""

# note: the canary text intentionally does NOT contain the sentence the
# negation trap would affirm ("NOVA uses prioritized sampling").


def _method_of(rec):
    ref = rec.get("method_ref") or rec.get("from_method_ref") or {}
    return (ref.get("surface") or "").lower() if isinstance(ref, dict) else ""


def _num_in(s, num):
    return num in re.sub(r"[,\s]", "", str(s or ""))


def _label_blob(rec):
    """Structural label fields of a result record (metric + dims + dims_new),
    lowercased. Excludes claim/quote: prose may mention other tasks legitimately
    (comparison sentences) and must not trigger binding traps."""
    meas = rec.get("measure") or {}
    parts = [str(meas.get("metric") or ""),
             json.dumps(rec.get("dims") or {}, ensure_ascii=False),
             json.dumps(rec.get("dims_new") or {}, ensure_ascii=False)]
    return " ".join(parts).lower()


def score(records):
    """Rule-based scoring. Returns dict with per-fact/per-trap verdicts."""
    out = {"facts": {}, "traps": {}}
    R = [r for r in records if r.get("kind") != "overflow"]

    # F1: NOVA 87.5 main result on BenchX
    out["facts"]["F1_main_result"] = any(
        r["kind"] == "result" and "nova" in _method_of(r)
        and "no-overlap" not in _method_of(r)
        and _num_in((r.get("measure") or {}).get("value"), "87.5")
        for r in R)
    # F2: UNIFORM 64.2 bound to UNIFORM (baseline role)
    out["facts"]["F2_baseline_binding"] = any(
        r["kind"] == "result" and "uniform" in _method_of(r)
        and _num_in((r.get("measure") or {}).get("value"), "64.2")
        for r in R)
    # F3: buffer size 2M config
    out["facts"]["F3_config"] = any(
        r["kind"] == "config" and _num_in(r.get("value"), "2")
        and "buffer" in (str(r.get("item")) + str(r.get("value"))).lower()
        for r in R)
    # F4: NOVA extends LUNA lineage
    out["facts"]["F4_lineage"] = any(
        r["kind"] == "lineage"
        and "nova" in ((r.get("from_method_ref") or {}).get("surface") or "").lower()
        and "luna" in ((r.get("to_method_ref") or {}).get("surface") or "").lower()
        and r.get("relation") in ("extends", "improves", "uses")
        for r in R)
    # F5: explicit absence — not evaluated on continuous control
    out["facts"]["F5_absence"] = any(
        r["kind"] == "absence"
        and "continuous control" in (str(r.get("missing")) + str(r.get("subject"))
                                      + str(r.get("quote"))).lower()
        and r.get("absence_type") in ("explicitly_stated", "not_reported")
        for r in R)
    # F6 (canary v2, PSFIX FX-D): wide-table correct column binding —
    # NOVA 74.9 bound to the CLS Swin-T column AND no other task-column label
    # in structural fields (contradictory bindings fail the fact and trip T5)
    def _f6(r):
        if r["kind"] != "result" or "nova" not in _method_of(r):
            return False
        if not _num_in((r.get("measure") or {}).get("value"), "74.9"):
            return False
        lab = _label_blob(r)
        if not (("swin" in lab) or ("cls" in lab)):
            return False
        return not any(w in lab for w in ("qa", "roberta", "summ", "t5", "seg", "vit"))
    out["facts"]["F6_widetable_binding"] = any(_f6(r) for r in R)

    # F7 (canary v3, schema v1.4 user-approved Option A): cited-source finding —
    # the TRACE (Kim et al., 2020) mechanism claim in Related Work must survive
    # as a finding with epistemic='cited'. Ground truth for the 53-drop KB-v3-era
    # class: the model wrote strength='cited', the frozen schema had no slot, the
    # enum gate dropped correct-intuition records. v1.4: template slot + FG7
    # migration; this fact verifies the channel end to end.
    def _f7(r):
        if r.get("kind") != "finding" or r.get("epistemic") != "cited":
            return False
        blob = (str(r.get("claim", "")) + " " + str(r.get("quote", ""))).lower()
        return ("trace" in blob or "kim" in blob
                or "importance weight" in blob or "bias" in blob)
    out["facts"]["F7_cited_finding"] = any(_f7(r) for r in R)

    # T1: negation flipped to affirmation (NOVA uses prioritized sampling)
    t1 = False
    for r in R:
        blob = json.dumps(r, ensure_ascii=False).lower()
        if "prioritized" not in blob:
            continue
        if r["kind"] == "lineage" and "nova" in _method_of(r) \
                and r.get("relation") in ("uses", "extends", "component_of"):
            t1 = True
        if r["kind"] == "config" and "nova" in _method_of(r) \
                and "prioritized" in str(r.get("item", "")).lower() \
                and "not" not in blob.split("prioritized")[0][-40:]:
            t1 = True
        if r["kind"] == "finding" and "nova" in blob \
                and re.search(r"nova[^.\"]{0,60}uses[^.\"]{0,40}prioritized", blob):
            t1 = True
    out["traps"]["T1_negation_flip"] = t1

    # T2: LUNA's 91.3 on BenchY extracted as stated/demonstrated (must be cited),
    # or bound to NOVA
    t2 = False
    for r in R:
        if r["kind"] == "result" and _num_in((r.get("measure") or {}).get("value"), "91.3"):
            if r.get("epistemic") in ("stated", "demonstrated", None, ""):
                t2 = True
            if "nova" in _method_of(r):
                t2 = True
    out["traps"]["T2_epistemic_flip"] = t2

    # T3: sparse-reward-only claim extracted WITHOUT its condition.
    # dims_new counts as preservation (v1 scorer calibration 2026-09-05:
    # canary terms are not in the RL vocab, so dims_new is the SPEC-CORRECT
    # home for "sparse-reward"; DSF put it there and was falsely trapped).
    t3 = False
    for r in R:
        if r["kind"] != "finding":
            continue
        blob = (str(r.get("claim")) + str(r.get("quote"))).lower()
        if "sparse" in blob and ("significant" in blob or "improvement" in blob
                                 or "only" in blob):
            cond = json.dumps({k: r.get(k) for k in ("condition", "dims", "dims_new")},
                              ensure_ascii=False).lower()
            if "sparse" not in cond:
                t3 = True
    out["traps"]["T3_condition_drop"] = t3

    # T4: WRONG number-metric-method bindings only (v1 scorer calibration:
    # 0.91/0.85/0.72 with metric=Success Rate are CORRECT records, not traps;
    # both v1 arms bound the whole table correctly and were falsely trapped).
    t4 = False
    for r in R:
        if r["kind"] != "result":
            continue
        m = _method_of(r)
        meas = r.get("measure") or {}
        val = str(meas.get("value") or "")
        met = str(meas.get("metric") or "").lower()
        role = r.get("role")
        success_metric = ("success" in met) or ("rate" in met)
        # main NOVA score bound to the ablation value
        if "nova" in m and "no-overlap" not in m and role == "main_result" \
                and _num_in(val, "79.1"):
            t4 = True
        # ablation bound to the main score
        if "no-overlap" in m and _num_in(val, "87.5") \
                and role != "baseline_comparison":
            t4 = True
        # score values under a success-rate metric or vice versa
        if (_num_in(val, "87.5") or _num_in(val, "79.1")
                or _num_in(val, "64.2")) and success_metric:
            t4 = True
        if _num_in(val, "0.91") and not success_metric and met:
            t4 = True
    out["traps"]["T4_number_binding"] = t4

    # T5 (canary v2, PSFIX FX-D): wide-table column-label misbinding (the A7
    # form: value+column correct but dims.subject bound to a neighbouring
    # column/task label). Deterministic value->column ground truth for Table 2;
    # fires only when structural label fields name a DIFFERENT task column.
    _T5_GROUPS = {"qa": ("qa", "roberta"), "summ": ("summ", "t5"),
                  "cls": ("cls", "swin"), "seg": ("seg", "vit")}
    _T5_VALUE_GROUP = {"71.8": "qa", "65.2": "qa", "68.3": "summ", "61.7": "summ",
                       "74.9": "cls", "66.8": "cls", "69.4": "seg", "63.5": "seg"}
    t5 = False
    for r in R:
        if r["kind"] != "result":
            continue
        val = str((r.get("measure") or {}).get("value") or "")
        group = next((g for v, g in _T5_VALUE_GROUP.items() if _num_in(val, v)), None)
        if group is None:
            continue
        lab = _label_blob(r)
        if not lab:
            continue  # no structural label -> recall miss (F6), not a trap
        for g, words in _T5_GROUPS.items():
            if g != group and any(w in lab for w in words):
                t5 = True
    out["traps"]["T5_widetable_misbinding"] = t5

    # T6 (canary v3, schema v1.4): own-paper finding epistemic flip — NOVA's
    # OWN mechanism/conclusion marked epistemic='cited' (mirror of T2 on the
    # finding axis; cited is for restatements of OTHER papers' claims only).
    t6 = False
    for r in R:
        if r.get("kind") != "finding" or r.get("epistemic") != "cited":
            continue
        blob = (str(r.get("claim", "")) + " " + str(r.get("quote", ""))).lower()
        if "stabiliz" in blob or ("nova" in blob and "trace" not in blob
                                  and "kim" not in blob
                                  and "importance weight" not in blob):
            t6 = True
    out["traps"]["T6_finding_epistemic_flip"] = t6

    out["fact_recall"] = f"{sum(out['facts'].values())}/7"
    out["trap_fired"] = sum(out["traps"].values())
    return out


if __name__ == "__main__":
    # score a checked-records file: python -m kb_compiler.records.canary REC.json
    data = load = json.load(open(sys.argv[1], encoding="utf-8"))
    recs = (data.get(CANARY_PID) or {}).get("records", [])
    print(json.dumps(score(recs), ensure_ascii=False, indent=1))
