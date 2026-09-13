# -*- coding: utf-8 -*-
"""Stage 3: deterministic post-checks + repair-or-drop. Spec v1.1 §3 Stage 3.

Five gates (no LLM):
  1. required fields complete (schema.REQUIRED_FIELDS + quote rules)
  2. enum legality (relation/role/absence_type/epistemic/strength/...)
  3. quote verbatim -> loc resolution (exact / normalized / token-anchor fuzzy;
     fuzzy locator = design pattern carried from the legacy stack)
  4. numeric verbatim: numbers in measure.value/delta/config.value must appear
     (normalized) inside the quote; formula-shaped values take the full-string
     quote-containment channel (batch2 note 4)
  5. explicit dims legality: subject/setup/variant values must be in the vocab
     (canonical or alias) -> violation is NON-fatal: value moves to dims_new
     (arbitration queue), record survives (growth discipline, spec §2)

Fatal violations (1-4) -> ONE repair call with the violation + source chunk
(repair-not-kill) -> re-check -> still bad -> drop + log.

Usage:
  python -m kb_compiler.records.postcheck --records REC.json --texts DIR \
      --vocab V.json --out-dir DIR --model DeepSeek-V4-Flash [--dry]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter

from .common import call_json, load_corpus, load_json, save_json
from .schema import (RELATIONS, RESULT_ROLES, CONFIG_ROLES, ABSENCE_TYPES,
                     EPISTEMIC, FINDING_STRENGTH, FINDING_CLAIM_TYPES,
                     LINEAGE_EVIDENCE, SHIFT_SOURCES, DIRECTIONS, REQUIRED_FIELDS)

ENUM_FIELDS = {
    "relation": RELATIONS, "role_result": RESULT_ROLES, "role_config": CONFIG_ROLES,
    "absence_type": ABSENCE_TYPES, "epistemic": EPISTEMIC,
    "strength": FINDING_STRENGTH, "evidence_basis": LINEAGE_EVIDENCE,
    "source_type": SHIFT_SOURCES, "direction": DIRECTIONS,
}
_NUM_IN_VAL = re.compile(r"\d[\d,]*\.?\d*|\d+")
_FORMULA_CHARS = set("^_{}=+*/()αβγελητ")
FG7_MIGRATED = [0]  # schema v1.4 finding strength='cited' -> epistemic migration count


# ---------- text normalization with offset map ----------

def _norm_text(text: str, drop_ws: bool = False):
    """NFKC + lowercase normalization; returns (norm_str, idx_map) where
    idx_map[i] = original index of norm char i.
    drop_ws=False: whitespace collapsed to single spaces (token boundaries
    kept — used by the fuzzy locator).
    drop_ws=True: ALL whitespace removed — used by quote containment and
    numeric/formula channels (measured failure 2026-09-05: mineru renders
    LaTeX as 's _ { t }' while LLM quotes 's_{t}'; collapse-normalization
    missed 61 legitimate quotes on the icm/seed_PER trial)."""
    out, idx = [], []
    prev_space = False
    for i, ch in enumerate(text):
        if ch == "$":
            # mineru wraps LaTeX in $...$; LLM quotes usually drop the
            # delimiters (measured 2026-09-05: 'r _ { t } ^ { i }' vs
            # '$r_{t}^{i}$'). Symmetric strip keeps containment sound.
            continue
        c = unicodedata.normalize("NFKC", ch).lower()
        if c.isspace():
            if drop_ws:
                continue
            if not prev_space and out:
                out.append(" ")
                idx.append(i)
                prev_space = True
            continue
        prev_space = False
        out.append(c)
        idx.append(i)
    return "".join(out), idx


def _math_strip(s: str) -> str:
    """Formula channel: additionally remove LaTeX noise chars so 'r_t^i'
    matches 'r _ { t } ^ { i }' (batch2 note 4: formula values go through
    quote-containment, not numeric normalization)."""
    for ch in "{}\\$ \t":
        s = s.replace(ch, "")
    return s


# Deterministic LaTeX fold (matching layer ONLY — stored quotes stay verbatim).
# Measured 2026-09-05 on the icm trial: LLM quotes faithfully render macros as
# unicode ("0~\leq~\beta" -> "0 ≤ β") — equivalence, not content drift; without
# folding these burn repair calls and risk dropping good records.
_LATEX_MAP = {
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈", r"\times": "×",
    r"\cdot": "·", r"\infty": "∞", r"\propto": "∝", r"\sim": "~",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\kappa": "κ", r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν",
    r"\xi": "ξ", r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ",
    r"\upsilon": "υ", r"\phi": "φ", r"\varphi": "φ", r"\chi": "χ",
    r"\psi": "ψ", r"\omega": "ω", r"\Delta": "Δ", r"\Sigma": "Σ",
    r"\Omega": "Ω", r"\Phi": "Φ", r"\Psi": "Ψ", r"\Gamma": "Γ",
    r"\left": "(", r"\right": ")", r"\,": "", r"\;": "", r"\!": "",
    r"\ ": " ", r"\quad": " ", r"\qquad": " ",
}
_MACROS_SORTED = sorted(_LATEX_MAP, key=len, reverse=True)


def _fold_latex(text: str):
    """Returns (folded_text, fold_map) with fold_map[j] = original index of
    folded char j (macro replacement chars all map to the macro start; loc
    boundaries are therefore approximate at macro edges — acceptable for
    provenance display, documented)."""
    out, fmap = [], []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            hit = None
            for m in _MACROS_SORTED:
                if text.startswith(m, i):
                    hit = m
                    break
            if hit:
                rep = _LATEX_MAP[hit]
                out.append(rep)
                fmap.extend([i] * len(rep))
                i += len(hit)
                continue
        if ch == "~":
            out.append(" ")
            fmap.append(i)
            i += 1
            continue
        out.append(ch)
        fmap.append(i)
        i += 1
    return "".join(out), fmap


def _num_forms(s: str) -> set:
    """Normalized numeric forms of a string: commas stripped, trailing .0 kept
    and dropped variants, % -> both forms.

    FG6 (2026-09-11, PSFIX): dual channel = ws-preserved ∪ ws-stripped. The old
    ws-stripped-only channel concatenated space-separated numbers in table cells
    ("39.7 33.6" -> "39.733.6"), making legitimate values unmatchable and
    false-dropping table records (measured: 40/46 val1b + 55/65 main-rebuild
    number_not_in_quote violations were false positives; psfix_2026-09-11/
    num46_fixcheck.json). ws-stripped channel is kept for latex digit-spacing
    artifacts ("$6 6 . 5$" -> 66.5). Union is strictly more permissive:
    records passing the old gate cannot fail the new one."""
    forms = set()
    for base in (s.replace(",", ""), s.replace(",", "").replace(" ", "")):
        for m in _NUM_IN_VAL.finditer(base):
            t = m.group(0).rstrip(".")
            forms.add(t)
            if t.endswith(".0"):
                forms.add(t[:-2])
            elif "." not in t:
                forms.add(t + ".0")
    return forms


def _is_formula(s: str) -> bool:
    return any(c in _FORMULA_CHARS for c in s) and not _NUM_IN_VAL.fullmatch(s.strip())


# ---------- individual checks ----------

def check_record(rec: dict, normc: str, idx_c: list, normf: str, idx_f: list,
                 vocab_sets: dict):
    """Returns (violations: list[str], warnings: list[str], loc: dict|None).
    normc/idx_c = whitespace-collapsed (fuzzy locator); normf/idx_f =
    whitespace-free (containment + numeric/formula channels)."""
    v, w = [], []
    kind = rec.get("kind")
    # FG7 (schema v1.4, 2026-09-13, user-arbitrated Option A): deterministic
    # provenance migration BEFORE the gates. Historical form (53 drops in the
    # KB v3 era, largest enum-violation class): the model stuffed 'cited' into
    # finding.strength — semantic intuition correct (restatement of ANOTHER
    # paper's claim), frozen schema had no slot. v1.4 gives finding the
    # epistemic template slot; strength='cited' MIGRATES instead of dropping:
    # provenance -> epistemic, strength -> 'stated' (the citing paper at least
    # states it — faithful on both axes). Counter observable for run reports.
    if kind == "finding" and rec.get("strength") == "cited":
        if not rec.get("epistemic"):
            rec["epistemic"] = "cited"
        rec["strength"] = "stated"
        FG7_MIGRATED[0] += 1
    # 1. required fields
    for f in REQUIRED_FIELDS.get(kind, ()):
        val = rec.get(f)
        if val in (None, "", [], {}):
            v.append(f"missing_field:{f}")
    if kind == "result":
        meas = rec.get("measure") or {}
        for f in ("metric", "value"):
            if meas.get(f) in (None, ""):
                v.append(f"missing_field:measure.{f}")
    quote = (rec.get("quote") or "").strip()
    from .schema import QUOTE_REQUIRED_KINDS
    if kind in QUOTE_REQUIRED_KINDS and not quote:
        v.append("missing_field:quote")
    if kind == "absence" and rec.get("absence_type") in ("explicitly_stated", "cannot_tell") \
            and not quote:
        v.append("missing_field:quote(explicit absence needs quote)")
    # 2. enums
    def _enum(val, allowed, name):
        if val not in (None, "") and val not in allowed:
            v.append(f"enum:{name}={val!r}")
    _enum(rec.get("relation"), RELATIONS, "relation")
    _enum(rec.get("epistemic"), EPISTEMIC, "epistemic")
    _enum(rec.get("strength"), FINDING_STRENGTH, "strength")
    _enum(rec.get("claim_type"), FINDING_CLAIM_TYPES, "claim_type")
    _enum(rec.get("absence_type"), ABSENCE_TYPES, "absence_type")
    _enum(rec.get("evidence_basis"), LINEAGE_EVIDENCE, "evidence_basis")
    _enum(rec.get("source_type"), SHIFT_SOURCES, "source_type")
    if kind == "result":
        _enum(rec.get("role"), RESULT_ROLES, "role")
        _enum((rec.get("measure") or {}).get("direction"), DIRECTIONS, "direction")
    if kind == "config":
        _enum(rec.get("role"), CONFIG_ROLES, "role")
    # 3. quote -> loc (exact on whitespace-free norm, fuzzy on collapsed norm)
    loc = None
    qfold = _fold_latex(quote)[0] if quote else ""
    if quote:
        nqf, _ = _norm_text(qfold, drop_ws=True)
        pos = normf.find(nqf)
        if pos >= 0 and nqf:
            loc = {"char_start": idx_f[pos],
                   "char_end": idx_f[min(pos + len(nqf) - 1, len(idx_f) - 1)] + 1,
                   "match": "normalized"}
        else:
            import difflib
            nqc, _ = _norm_text(qfold)
            toks = [t for t in re.findall(r"[a-z0-9.%\-]+", nqc) if len(t) > 2]
            if len(toks) >= 3:
                anchor = max(toks, key=lambda t: (len(t), t))
                best = None
                for m in re.finditer(re.escape(anchor), normc):
                    s0 = max(0, m.start() - len(nqc))
                    window = normc[s0:m.start() + len(nqc)]
                    r = difflib.SequenceMatcher(None, nqc, window).ratio()
                    if r >= 0.85 and (best is None or r > best[0]):
                        best = (r, s0, window)
                if best:
                    r, s0, window = best
                    p2 = window.find(nqc[:20])
                    start = s0 + (p2 if p2 >= 0 else 0)
                    loc = {"char_start": idx_c[min(start, len(idx_c) - 1)],
                           "char_end": idx_c[min(start + len(nqc), len(idx_c) - 1)] + 1,
                           "match": f"fuzzy:{r:.2f}"}
            if loc is None:
                v.append("quote_not_in_text")
    # 3b. table_header verbatim containment (v2 table protocol: quote = data
    # row verbatim, table_header = header row verbatim; both must be in text)
    th = (rec.get("table_header") or "").strip()
    if th:
        nth, _ = _norm_text(_fold_latex(th)[0], drop_ws=True)
        if nth and nth not in normf:
            v.append("table_header_not_in_text")
    # 4. numeric verbatim (fatal) — values must appear in the quote
    if quote:
        nq_nums = _num_forms(qfold) | {qfold.replace(",", "").lower()}
        q_math = _math_strip(_norm_text(qfold, drop_ws=True)[0])
        vals = []
        if kind == "result":
            meas = rec.get("measure") or {}
            vals.append(str(meas.get("value") or ""))
            if rec.get("delta"):
                vals.append(str(rec["delta"]))
        elif kind == "config":
            vals.append(str(rec.get("value") or ""))
        for val in vals:
            val = val.strip()
            if not val:
                continue
            if _is_formula(val):
                nv = _math_strip(_norm_text(_fold_latex(val)[0], drop_ws=True)[0])
                if nv and nv not in q_math:
                    v.append(f"formula_not_in_quote:{val[:30]}")
                continue
            forms = _num_forms(val)
            if forms and not (forms & nq_nums):
                v.append(f"number_not_in_quote:{val[:30]}")
    # 5. dims legality (non-fatal -> dims_new)
    def _dims_check(dims, path="dims"):
        if not isinstance(dims, dict):
            return
        for dim, allowed in vocab_sets.items():
            val = dims.get(dim)
            vals = val if isinstance(val, list) else ([val] if val else [])
            keep = []
            for x in vals:
                xs = str(x).strip()
                if not xs:
                    continue
                if _norm_text(xs)[0] in allowed:
                    keep.append(x)
                else:
                    rec.setdefault("dims_new", {}).setdefault(path + "." + dim, []).append(xs)
                    w.append(f"dims_new:{dim}={xs[:40]}")
            if val is not None:
                if isinstance(val, list):
                    dims[dim] = keep
                elif keep:
                    dims[dim] = keep[0]
                else:
                    # PSFIX IL-2 (2026-09-11): scalar dim with no vocab-valid
                    # value must be REMOVED, not silently retained (old code
                    # kept the invalid label in dims while also copying it to
                    # dims_new — the A7 stray-label family, measured 15 records
                    # in nN8T Table 2). Value is preserved in dims_new above.
                    dims.pop(dim, None)
    for dpath in ("dims", "condition", "scope", "applicability"):
        _dims_check(rec.get(dpath), dpath)
    return v, w, loc


# ---------- repair ----------

REPAIR_PROMPT = """下面这条从论文抽取的记录未通过确定性后检。违规项：
{violations}

记录 JSON：
{record}

论文相关原文（用于重新核对）：
{source}

修复纪律（quote-first）：先找到能逐字支持该事实的原文句作为 quote（≤40词），再仅从 quote 填字段；数值必须在 quote 中逐字出现；枚举字段取值必须合法。表格记录：quote=逐字抄数据行整行（含分隔符），table_header=逐字抄表头行，严禁拼接自造。若原文根本不支持该记录，输出 {{"drop": true, "reason": "..."}}。
输出修复后的完整记录 JSON（保留 id/paper_id/chunk_id/section 不变）："""


def repair_record(rec, violations, text, model):
    start = max(0, (rec.get("chunk_char_start") or 0) - 500)
    source = text[start:start + 9000]
    prompt = (REPAIR_PROMPT.replace("{violations}", "\n".join(violations))
              .replace("{record}", json.dumps(rec, ensure_ascii=False)[:2500])
              .replace("{source}", source))
    obj = call_json(prompt, model, max_tokens=3000, retries=2)
    if not isinstance(obj, dict) or obj.get("drop"):
        return None
    for k in ("id", "paper_id", "chunk_id", "section", "chunk_char_start"):
        obj[k] = rec.get(k)
    return obj


# ---------- driver ----------

def build_vocab_sets(vocab):
    sets = {}
    subj = set()
    for fam in vocab.get("subject", []):
        for m in fam.get("members", []):
            subj.add(_norm_text(m)[0])
        if fam.get("family"):
            subj.add(_norm_text(fam["family"])[0])
    sets["subject"] = subj
    for dim in ("setup", "variant"):
        s = set()
        for e in vocab.get(dim, []):
            s.add(_norm_text(e["canonical"])[0])
            for a in e.get("aliases", []):
                s.add(_norm_text(a)[0])
        sets[dim] = s
    return sets


def _triage_skip(violations: list, triage: dict) -> bool:
    """FG3 (2026-09-10): evidence-based repair triage. Archive-wide rescue
    rates (gate1 arms + flagship ps16, n~2000 attempts): enum 0/122 (also 0 in
    combo violations), number_not_in_quote 6/122. Skip repair (drop directly,
    marked) when the record cannot be rescued: ANY enum violation, or ALL
    violations in the near-zero-rescue set. Config: triage_table.json.

    FG6 disclosure (2026-09-11): the 6/122 number rescue rate was measured
    under the pre-FG6 buggy numeric gate (ws-concatenation false positives,
    see _num_forms). Post-FG6 the residual number violations are true/
    conservative drops (name tokens, quote mis-select, fused cells) which
    repair also cannot legitimately fix — skip behavior intentionally kept."""
    classes = [v.split(":", 1)[0] for v in violations]
    if any(c in triage.get("skip_if_any", ()) for c in classes):
        return True
    skip_all = set(triage.get("skip_if_all_in", ()))
    return bool(classes) and all(c in skip_all for c in classes)


def run_postcheck(records_by_paper, texts, vocab, model, dry=False, triage=None):
    vocab_sets = build_vocab_sets(vocab)
    stats = Counter()
    checked, dropped, warnings = {}, [], []
    for pid, payload in records_by_paper.items():
        text = texts.get(pid, "")
        folded, fmap = _fold_latex(text)
        nc, ic = _norm_text(folded)
        idx_c = [fmap[j] for j in ic]
        nf, iff = _norm_text(folded, drop_ws=True)
        idx_f = [fmap[j] for j in iff]
        normc, normf = nc, nf
        out_recs = []
        for rec in payload.get("records", []):
            v, w, loc = check_record(rec, normc, idx_c, normf, idx_f, vocab_sets)
            warnings.extend({"paper_id": pid, "record_id": rec.get("id"), "w": x} for x in w)
            if loc:
                rec["loc"] = loc
            if not v:
                stats["first_pass"] += 1
                out_recs.append(rec)
                continue
            if dry:
                stats["would_repair"] += 1
                dropped.append({"paper_id": pid, "record": rec, "violations": v, "dry": True})
                continue
            if triage and _triage_skip(v, triage):
                stats["triage_skipped"] += 1
                stats["dropped"] += 1
                dropped.append({"paper_id": pid, "record": rec, "violations": v,
                                "triage": "skipped_repair"})
                continue
            v_orig = list(v)  # FG-archive fix (2026-09-10): dropped entries used to
            fixed = repair_record(rec, v, text, model)  # pair the ORIGINAL record with
            if fixed:  # the POST-REPAIR violations — misattribution found in FG1
                v2, w2, loc2 = check_record(fixed, normc, idx_c, normf, idx_f, vocab_sets)  # diagnosis (5/25 'passes_now' were this artifact).
                if not v2:
                    fixed["repair_history"] = {"violations": v}
                    if loc2:
                        fixed["loc"] = loc2
                    stats["repaired"] += 1
                    out_recs.append(fixed)
                    continue
                stats["dropped"] += 1
                dropped.append({"paper_id": pid, "record": rec,
                                "violations": v_orig,
                                "violations_after_repair": v2,
                                "record_after_repair": fixed})
                continue
            stats["dropped"] += 1
            dropped.append({"paper_id": pid, "record": rec, "violations": v_orig})
        for ov in payload.get("overflow", []):
            stats["overflow"] += 1
        checked[pid] = {"records": out_recs, "overflow": payload.get("overflow", []),
                        "entity_queue": payload.get("entity_queue", [])}
        stats["total"] += len(payload.get("records", []))
        print(f"[{pid}] in={len(payload.get('records', []))} pass={len(out_recs)} "
              f"(dry={dry})", flush=True)
    return checked, dropped, warnings, dict(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="DeepSeek-V4-Flash")
    ap.add_argument("--dry", action="store_true", help="no repair calls; count only")
    args = ap.parse_args()

    records = load_json(args.records, {})
    texts = dict(load_corpus(args.texts))
    vocab = load_json(args.vocab, {})
    checked, dropped, warnings, stats = run_postcheck(
        records, texts, vocab, args.model, dry=args.dry)
    tag = "_dry" if args.dry else ""
    save_json(checked, f"{args.out_dir}/records_checked{tag}.json")
    save_json(dropped, f"{args.out_dir}/dropped{tag}.json")
    save_json(warnings, f"{args.out_dir}/warnings{tag}.json")
    n = max(1, stats.get("total", 0))
    stats["rates"] = {
        "first_pass": round(stats.get("first_pass", 0) / n, 3),
        "repaired": round(stats.get("repaired", 0) / n, 3),
        "dropped": round(stats.get("dropped", 0) / n, 3),
        "residual_overflow": round(stats.get("overflow", 0) /
                                   max(1, n + stats.get("overflow", 0)), 3),
    }
    save_json(stats, f"{args.out_dir}/postcheck_stats{tag}.json")
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
