# -*- coding: utf-8 -*-
"""Sentinel cases (INTEGRATED-SYSTEM-1005 §7.3) — MUST be re-run after every prompt change.

The canary paper carries the pre-integration canary v3's 7 facts and 6 traps, re-scored for schema v2
statements; the chemistry variant is MinerU-shaped and covers the four hazards the design names (superscripts,
an UNTITLED reference list, a base64 figure, a scientific-notation table); the two new traps ("the quote is not
a substring of the source", "a number was rewritten") test the unified final check itself — they are
deterministic, never call the LLM, and must never fire.

run() drives the SAME per-paper code path the build uses: documents.sciverse.doc adapter -> reading.chunks ->
passes.t1/t2/results/other_batch -> final_check. Scoring is rule-based (no LLM judge).
Gates: canary fact_recall >= 6/7, chem facts >= 2/3, LLM traps fired == 0, pipeline gates PASS."""
from __future__ import annotations

import json
import re

from ..documents import sciverse as SV
from ..llm.client import call_local
from . import final_check as FC
from . import passes as PS
from . import reading as RD
from .schema import Statement

CANARY_PID = "arxiv:9999.00001"      # synthetic but schema-legal (arXiv YYMM 9999 cannot exist)
CHEM_PID = "arxiv:9999.00002"
DATE = "2020-01-01"

CANARY_TEXT = """# NOVA: Nested Overlap Value Aggregation for Sample-Efficient Control

## Abstract
We propose NOVA, a method that aggregates nested value estimates for sample-efficient control. NOVA achieves
87.5 points on BenchX (mean over 3 seeds), outperforming the uniform replay baseline UNIFORM at 64.2 points.
Unlike prior work, NOVA does not use prioritized sampling. The improvement over UNIFORM is significant only in
sparse-reward settings.

## 1. Introduction
Sample efficiency remains a central challenge. NOVA extends LUNA (Doe et al., 2019) with an overlap-aware
aggregator that reuses nested value estimates. LUNA reported 91.3 points on BenchY in its original publication.

## 2. Method
NOVA maintains a replay buffer of size 2M transitions. The overlap weight beta is set to 0.1 in all experiments.
The aggregator computes nested means over value estimates, which stabilizes training under sparse rewards.

## 3. Experiments
We evaluate on BenchX under the standard protocol (mean over 3 seeds).

Table 1: BenchX results.
<table><tr><td>Method</td><td>Score</td><td>Success Rate</td></tr><tr><td>NOVA</td><td>87.5</td><td>0.91</td></tr><tr><td>NOVA-no-overlap</td><td>79.1</td><td>0.85</td></tr><tr><td>UNIFORM</td><td>64.2</td><td>0.72</td></tr></table>

Table 1 shows that NOVA reaches 87.5 while the ablation without the overlap aggregator (NOVA-no-overlap) reaches
79.1, a drop of 8.4 points. UNIFORM reaches 64.2.

We further evaluate transfer across tasks. Table 2 reports per-task scores.

Table 2: Multi-task transfer (accuracy).
<table><tr><td rowspan="2">Method</td><td colspan="2">NLP</td><td colspan="2">Vision</td></tr><tr><td>QA RoBERTa-L</td><td>Summ T5-B</td><td>CLS Swin-T</td><td>Seg ViT-S</td></tr><tr><td>NOVA</td><td>71.8</td><td>68.3</td><td>74.9</td><td>69.4</td></tr><tr><td>UNIFORM</td><td>65.2</td><td>61.7</td><td>66.8</td><td>63.5</td></tr></table>

Table 2 shows that NOVA reaches 74.9 on image classification (CLS Swin-T), its strongest transfer task, while
UNIFORM reaches 66.8 on the same task.

## 4. Related Work
Value aggregation for control was introduced by LUNA (Doe et al., 2019). Prior methods typically rely on
prioritized sampling to accelerate learning. TRACE (Kim et al., 2020) argues that prioritized sampling
introduces bias into value estimates, and corrects it with per-sample importance weights.

## 5. Conclusion
We presented NOVA and showed strong results on BenchX. We do not evaluate on continuous control tasks; future
work may extend NOVA to that setting.
"""
# the text intentionally does NOT contain the sentence the negation trap would affirm ("NOVA uses prioritized
# sampling")

CHEM_TEXT = """# Adsorption of Ca²⁺ on goethite at low temperature

We report a low-temperature adsorption study of Ca²⁺ on goethite. The surface density reached 4.2×10⁻⁵ mol per
gram at pH 3.0 [1]. The Ca²⁺ uptake follows a Langmuir isotherm with K = 1.2e-3 L per mg.

Table 1: Ca²⁺ uptake at pH 3.0.
<table><tr><td>Sample</td><td>Uptake (mol/g)</td></tr><tr><td>Goethite-A</td><td>4.2×10⁻⁵</td></tr><tr><td>Goethite-B</td><td>1.1×10⁻⁵</td></tr></table>

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAAQUlEQVR4nO3BMQEAAADCoPVPbQwfoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIC3AcUIAAF2lG1rAAAAAElFTkSuQmCC)

[1] A. Author, B. Second. J. Colloid Sci. 1998, 120, 45-67.
[2] C. Third. Surface chemistry of oxides. 2001.
[3] D. Fourth. Ionic adsorption review. 2010.
"""

_GROUPS = {"qa": ("qa", "roberta"), "summ": ("summ", "t5"), "cls": ("cls", "swin"), "seg": ("seg", "vit")}
_VALUE_GROUP = {"71.8": "qa", "65.2": "qa", "68.3": "summ", "61.7": "summ",
                "74.9": "cls", "66.8": "cls", "69.4": "seg", "63.5": "seg"}


def _norm_num(s) -> str:
    return re.sub(r"[,\s]", "", str(s or ""))


def _obj(s) -> str:
    return (s.meta.get("object") or "").lower()


def _val(s) -> str:
    return _norm_num(s.meta.get("value"))


def _labels(s) -> str:
    return ((s.condition or "") + " " + str(s.meta.get("metric") or "")).lower()


# ---------------------------------------------------------------- scoring (rule-based, schema v2)
def score_canary(stmts: list[Statement]) -> dict:
    R = [s for s in stmts if s.pass_name == "results"]
    T2 = [s for s in stmts if s.pass_name == "t2"]
    O = [s for s in stmts if s.pass_name == "other"]
    facts, traps = {}, {}
    facts["F1_main_result"] = any("nova" in _obj(s) and "no-overlap" not in _obj(s) and "87.5" in _val(s)
                                  for s in R)
    facts["F2_baseline_binding"] = any("uniform" in _obj(s) and "64.2" in _val(s) for s in R)
    facts["F3_config"] = any(
        s.facet == "config" and "buffer" in str((s.meta.get("config") or {}).get("item", "")).lower()
        and "2" in str((s.meta.get("config") or {}).get("value", "")) for s in stmts)
    facts["F4_lineage"] = any(
        s.role in ("extends", "improves", "uses")
        and (any("luna" in str(m.get("name") or "").lower() for m in s.meta.get("mentions") or [])
             or "luna" in s.text.lower()) for s in T2)
    facts["F5_absence"] = any(s.facet == "absence" and "continuous control" in (s.text + " " + s.quote).lower()
                              for s in stmts)

    def f6(s):
        if s.pass_name != "results" or "nova" not in _obj(s) or "74.9" not in _val(s):
            return False
        blob = _labels(s)
        return (("swin" in blob) or ("cls" in blob)) and not any(
            w in blob for g, ws in _GROUPS.items() if g != "cls" for w in ws)
    facts["F6_widetable_binding"] = any(f6(s) for s in R)
    facts["F7_cited_finding"] = any(
        s.kind == "other" and re.search(r"trace|kim|importance weight|bias", s.text.lower()) for s in O)

    def blob(s):
        return json.dumps({"t": s.text, "m": s.meta}, ensure_ascii=False).lower()

    _NEG = re.compile(r"\b(not|never|no|without|unlike|avoid\w*|eschew\w*|refrain\w*|doesn't|isn't|don't)\b")

    def t1_fired(s):
        """The trap is the NEGATION FLIP — a correctly extracted negation ("NOVA does NOT use prioritized
        sampling", facet=absence or with the negation inside the claim) is the pipeline working, not a trap."""
        if s.kind != "self" or s.facet in ("absence", "limitation"):
            return False
        b = blob(s)
        m = re.search(r"nova[^.\"!?]{0,60}?uses?[^.\"!?]{0,40}?prioritized", b)
        if m and not _NEG.search(b[m.start():m.end()]):
            return True
        return s.role == "uses" and any("prioritized" in str(x.get("name") or "").lower()
                                        for x in s.meta.get("mentions") or [])

    traps["T1_negation_flip"] = any(t1_fired(s) for s in stmts)
    traps["T2_epistemic_flip"] = (
        any(s.kind == "self" and "91.3" in _norm_num(s.text) and s.epistemic in ("stated", "demonstrated")
            for s in stmts)
        or any("91.3" in _val(s) for s in R))
    traps["T3_condition_drop"] = any(
        s.kind == "self" and s.facet == "result" and "sparse" in s.text.lower()
        and re.search(r"significant|improvement|only", s.text.lower())
        and "sparse" not in (s.condition or "").lower() for s in stmts)

    def t4(s):
        o, v, met = _obj(s), _val(s), str(s.meta.get("metric") or "").lower()
        succ = "success" in met or "rate" in met
        return (("nova" in o and "no-overlap" not in o and "79.1" in v)
                or ("no-overlap" in o and "87.5" in v)
                or (any(x in v for x in ("87.5", "79.1", "64.2")) and succ)
                or ("0.91" in v and bool(met) and not succ))
    traps["T4_number_binding"] = any(t4(s) for s in R)

    def t5(s):
        g = next((g for vv, g in _VALUE_GROUP.items() if vv in _val(s)), None)
        if g is None:
            return False
        blob = _labels(s)
        return (any(w in blob for gg, ws in _GROUPS.items() if gg != g for w in ws)
                and not any(w in blob for w in _GROUPS[g]))
    traps["T5_widetable_misbinding"] = any(s.pass_name == "results" and t5(s) for s in R)
    traps["T6_own_epistemic_flip"] = any(
        s.kind == "self" and s.epistemic == "cited" and re.search(r"87\.5|benchx", s.text.lower())
        and "91.3" not in s.text and "luna" not in s.text.lower() for s in stmts)
    return {"facts": facts, "traps": traps}


def score_chem(stmts: list[Statement]) -> dict:
    facts = {}
    facts["CF1_scinotation_binding"] = any(
        s.pass_name == "results" and "goethite-a" in _obj(s) and "4.2" in _val(s) for s in stmts)
    facts["CF2_superscript_quote_survives"] = any(
        re.search(r"langmuir|1\.2e-3", (s.text + " " + s.quote).lower()) for s in stmts)
    facts["CF3_untitled_refs_not_extracted"] = not any(
        re.search(r"j\.?\s*colloid|surface chemistry of oxides|ionic adsorption review",
                  (s.quote + " " + s.text).lower()) for s in stmts)
    traps = {"CT1_base64_in_output": any("iVBOR" in (s.text + s.quote) for s in stmts)}
    return {"facts": facts, "traps": traps}


# ---------------------------------------------------------------- runner (real LLM, same code path as the build)
def _full(pid, text):
    d = SV.doc(pid, text)
    return d, {"paper_id": pid, "source": "sciverse", "units": d["units"],
               "sentences": [dict(s, date=DATE, in_delta=0) for s in d["sentences"]]}


def _paper(pid, text, title, abstract, other_sentence=None, other_cited=None, chat=call_local) -> dict:
    d, full = _full(pid, text)
    inp = {"paper_id": pid, "title": title, "date": DATE, "source": "sentinel",
           "sentences": [{"n": i + 1, "text": s} for i, s in enumerate(SV.split_sentences(abstract))]}
    stmts, stats = [], {}
    s1, names, st = PS.t1(inp, chat=chat)
    stats["t1"] = st
    if s1 is not None:
        k, _, fst = FC.run(s1, " ".join(x["text"] for x in inp["sentences"]),
                           {f"{pid}@abs{x['n']}": x["text"] for x in inp["sentences"]}, chat=chat, item=pid)
        stmts += k
        stats["t1_fc"] = fst
    s2, own, st = PS.t2(pid, title, full, seed_own=names, chat=chat)
    stats["t2"] = st
    if s2 is not None:
        k, _, fst = FC.run(s2, " ".join(x["text"] for x in full["sentences"]),
                           {x["sid"]: x["text"] for x in full["sentences"]}, chat=chat, item=pid)
        stmts += k
        stats["t2_fc"] = fst
    tbs = RD.tables_of(full)
    s3, st = PS.results(pid, tbs, own_methods=list(set(names) | set(own)), default_date=DATE, chat=chat)
    k, _, fst = FC.run(s3, " ".join(f"{tb['html']} {tb.get('caption') or ''} "
                                    f"{' '.join(tb.get('context') or [])}" for tb in tbs),
                       {s.loc.get("sent_id"): s.quote for s in s3}, chat=chat, item=pid)
    stmts += k
    stats["results"] = {**st, **fst}
    if other_sentence:
        it = {"s": 1, "sentence": other_sentence, "date": DATE, "citing_key": f"{pid}@v1",
              "sid": f"{pid}#sent1", "in_delta": 0,
              "cites": [{"key": "b0", "cited": other_cited, "title": "", "raw": "", "self_cite": 0}]}
        s4, st = PS.other_batch(pid, [it], chat=chat)
        stats["other"] = st
        if s4 is not None:
            k, _, fst = FC.run(s4, other_sentence, {it["sid"]: other_sentence}, chat=chat, item=pid)
            stmts += k
            stats["other_fc"] = fst
    return {"statements": stmts, "stats": stats, "units": len(d["units"]), "tables": len(tbs),
            "has_refs": d["has_refs"]}


def _gate_traps() -> dict:
    """The two pipeline traps — deterministic, no LLM, no repair (chat refuses everything): the final check must
    discard a fabricated quote and a rewritten number."""
    src = "We propose NOVA, a method for sample-efficient control."
    base = dict(speaker=CANARY_PID, date=DATE, kind="self", about=CANARY_PID, role="proposes",
                facet="contribution", epistemic="stated", loc={"unit_id": "u", "sent_id": "s1"})
    bad_quote = Statement(**base, text="proposes NOVA", quote=src + " It also does robotics.")
    bad_num = Statement(**base, text="NOVA reaches 99.9 points on BenchX", quote=src)
    kept, disc, _ = FC.run([bad_quote, bad_num], src, {"s1": src}, chat=lambda *a, **k: None, item="gate")
    codes = {v[0].split(":")[0] for _, v in disc}
    return {"T7_quote_not_substring_caught": not kept and FC.QUOTE_NOT_IN_SOURCE in codes,
            "T8_number_rewritten_caught": FC.NUMBER_REWRITTEN in codes}


def run(chat=None, log=print, only: tuple = ()) -> dict:
    chat = chat or call_local
    out: dict = {"gates": _gate_traps()}
    if not only or "canary" in only:
        can = _paper(CANARY_PID, CANARY_TEXT,
                     "NOVA: Nested Overlap Value Aggregation for Sample-Efficient Control",
                     ("We propose NOVA, a method that aggregates nested value estimates for sample-efficient "
                      "control. NOVA achieves 87.5 points on BenchX (mean over 3 seeds), outperforming the "
                      "uniform replay baseline UNIFORM at 64.2 points. Unlike prior work, NOVA does not use "
                      "prioritized sampling. The improvement over UNIFORM is significant only in sparse-reward "
                      "settings."),
                     other_sentence=("TRACE (Kim et al., 2020) argues that prioritized sampling introduces bias "
                                     "into value estimates, and corrects it with per-sample importance weights."),
                     other_cited="stub:trace2020", chat=chat)
        sc = score_canary(can["statements"])
        out["canary"] = {**sc, "stats": can["stats"], "n_statements": len(can["statements"]),
                         "fact_recall": f"{sum(sc['facts'].values())}/{len(sc['facts'])}",
                         "traps_fired": sorted(k for k, v in sc["traps"].items() if v)}
    if not only or "chem" in only:
        ch = _paper(CHEM_PID, CHEM_TEXT, "Adsorption of Ca²⁺ on goethite at low temperature",
                    ("We report a low-temperature adsorption study of Ca²⁺ on goethite. The surface density "
                     "reached 4.2×10⁻⁵ mol per gram at pH 3.0 [1]. The Ca²⁺ uptake follows a Langmuir isotherm "
                     "with K = 1.2e-3 L per mg."),
                    other_sentence="The surface density reached 4.2×10⁻⁵ mol per gram at pH 3.0 [1].",
                    other_cited="stub:author1998", chat=chat)
        sc = score_chem(ch["statements"])
        out["chem"] = {**sc, "stats": ch["stats"], "n_statements": len(ch["statements"]),
                       "has_refs": ch["has_refs"],
                       "fact_recall": f"{sum(sc['facts'].values())}/{len(sc['facts'])}",
                       "traps_fired": sorted(k for k, v in sc["traps"].items() if v)}
    ok = all(out["gates"].values())
    if "canary" in out:
        ok = ok and sum(out["canary"]["facts"].values()) >= 6 and not out["canary"]["traps_fired"]
    if "chem" in out:
        ok = ok and sum(out["chem"]["facts"].values()) >= 2 and not out["chem"]["traps_fired"]
    out["pass"] = ok
    log(f"[sentinel] {'PASS' if ok else 'FAIL'}: "
        + json.dumps({k: (v if k == "gates" else {"facts": v["facts"], "traps_fired": v["traps_fired"],
                                                   "n": v["n_statements"]})
                      for k, v in out.items() if k != "pass"}, ensure_ascii=False, default=str))
    return out
