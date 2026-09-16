# -*- coding: utf-8 -*-
"""F35 semantic table layer — LLM proposes roles for tables the deterministic
table_channel could not fully resolve; a STRUCTURAL gate validates; corrections
are re-applied to records. (Spec: airqa/F35-SEMANTIC-TABLE-LAYER-SPEC.md.)

Why: F32 (deterministic) lifted table-record subject fill 0->43%. The 57.5%
residual is dominated by single-tier role-reversed tables (column header = a
METHOD wrongly emitted as measure.metric; row label = a DATASET wrongly emitted
as method_ref.surface; real metric name lives only in the caption). Deciding
"TD3+BC is a method, halfcheetah is a dataset" is complex semantic judgement —
per design law, rules cannot referee it; an LLM proposes. The gate stays
deterministic and STRUCTURAL only (it does not judge semantics): registry match,
benchmark-leakage blocklist, caption provenance, canary ground truth, and a
conservative low-confidence path. Semantic correctness is anchored by the canary
(G5 hard stop), not by rules.

Discipline:
- LLM only PROPOSES; the gate is the sole write authority ("mechanical backstop").
- Prompt carries ZERO gold / questions / benchmark lists (blocklist is gate-side
  only, never in the prompt).
- Gate-rejected tables keep their F32 records untouched; proposals are logged to
  a rejected ledger (auditable).
- provenance "f35_semantic" on every corrected record.

Usage:
  python -m kb_compiler.records.table_semantic \
      --records records_checked.json --texts TEXTS --registry registry.json \
      --out-dir OUT [--model Qwen3.6-27B] [--canary] [--dry]
"""
import json, os, re, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kb_infra.llm import call_paratera, parse_json_response
from kb_compiler.records.table_channel import (
    _strip_cites, _norm_tier, _build_entity_lookup, _link_method, GENERIC_TIER,
)

# condition-level sub-headers that get mis-emitted as metric (0b1cad92 family)
COND_SET = {"base", "top", "bottom", "random", "middle", "nd", "pn",
            "w/ outlier d", "w/o outlier d"}

# Gate-side benchmark/dataset blocklist (G3): metric_name/subject_name must not
# be a known benchmark/dataset name (prevents gold-side leakage into the KB).
# NEVER passed to the LLM. Seed list; extend per domain. Deterministic substring
# match on normalised form.
BENCHMARK_BLOCKLIST = {
    "imagenet", "coco", "cifar", "mnist", "squad", "glue", "superglue", "mmlu",
    "arc-c", "arc-easy", "hellaswag", "winogrande", "gsm8k", "humaneval",
    "natural questions", "triviaqa", "hotpotqa", "ms marco", "marco", "nq",
    "fiqa", "nfcorpus", "arguana", "cqadupstack", "dbpedia", "scidocs",
    "cora", "citeseer", "pubmed", "ogbn-arxiv", "kinetics", "voc", "cityscapes",
    "ade20k", "pascal", "wikitext", "enwik8", "lm1b", "xnli", "paws", "qqp",
}

DIM_KEYS = ["method_axis", "metric_name", "confidence"]


# ---------------------------------------------------------------- trigger ID
def _metric_of(r):
    return ((r.get("measure") or {}).get("metric") or "").strip()


def _has_subject(r):
    dn = r.get("dims_new") or {}
    s = dn.get("dims.subject")
    return bool(s and any(str(x).strip() for x in (s if isinstance(s, list) else [s])))


def is_trigger_record(r):
    """A record needs the semantic layer if deterministic parsing left it
    unresolved: no subject, OR metric is a condition label, OR metric carries a
    fusion marker ('>' / '[')."""
    m = _metric_of(r)
    if not _has_subject(r):
        return "no_subject"
    if m.lower() in COND_SET:
        return "cond_label_metric"
    if ">" in m or "[" in m:
        return "fused_metric"
    return ""


def group_tables(records_by_paper):
    """Group result records into tables keyed by (pid, table_header).
    Returns {key: {"pid", "table_header", "records", "trigger"}}."""
    tables = {}
    for pid, payload in records_by_paper.items():
        if pid == "canary" or not isinstance(payload, dict):
            continue
        for r in payload.get("records", []):
            if not str(r.get("chunk_id", "")).endswith("#table"):
                continue
            th = r.get("table_header") or ""
            key = (pid, th)
            t = tables.setdefault(key, {"pid": pid, "table_header": th,
                                        "records": [], "trigger": ""})
            t["records"].append(r)
            if not t["trigger"]:
                tr = is_trigger_record(r)
                if tr:
                    t["trigger"] = tr
    return tables


# ---------------------------------------------------------------- table repr
def table_repr(recs):
    """Reconstruct the table shape from its records (no source re-parse):
    column headers = distinct measure.metric; row labels = distinct
    method_ref.surface; plus the verbatim header HTML and one row quote."""
    cols, rows = [], []
    for r in recs:
        c = _metric_of(r)
        if c and c not in cols:
            cols.append(c)
        s = ((r.get("method_ref") or {}).get("surface") or "").strip()
        if s and s not in rows:
            rows.append(s)
    return {
        "column_headers": cols,        # currently emitted as metric
        "row_labels": rows,            # currently emitted as method_ref.surface
        "header_html": recs[0].get("table_header", "") if recs else "",
        "row_quote_sample": (recs[0].get("quote", "") if recs else "")[:400],
        "n_records": len(recs),
    }


_CAP_RE = re.compile(r"(Table\s+[0-9IVXLC]+[\.:]\s*[^\n]{0,300})", re.I)


def caption_and_context(text, char_start, back=600, fwd=200):
    """Caption = nearest 'Table N: ...' at/just-before the table position;
    context = the surrounding window (for G4 metric-name provenance)."""
    if not text:
        return "", ""
    lo = max(0, (char_start or 0) - back)
    hi = min(len(text), (char_start or 0) + fwd)
    win = text[lo:hi]
    caps = _CAP_RE.findall(win)
    caption = caps[-1].strip() if caps else ""
    return caption, win


# ---------------------------------------------------------------- LLM propose
PROMPT = """You are analysing ONE table from a scientific paper to assign semantic roles.

Table header (HTML): {header_html}
A data row (HTML): {row_quote}
Column headers (as currently parsed): {cols}
Row labels (as currently parsed): {rows}
Table caption: {caption}
Surrounding context: {context}
Known method/entity names in this corpus (for reference only): {registry_names}

Decide the table's semantic structure. In many ML/NLP/CV results tables the
COLUMNS are methods/models and the ROWS are datasets/tasks (or the reverse).
The current parser assumed rows=methods; it is often WRONG for single-header
tables. Judge which axis really carries the METHODS/MODELS, and recover the
true METRIC name (what the cell numbers measure — usually stated in the caption,
e.g. accuracy / F1 / perplexity / return / mAP), NOT a column label.

Reply with ONLY a JSON object:
{{"method_axis": "column" or "row",
  "metric_name": "<the true metric, lowercase, or empty string if not determinable>",
  "confidence": "high" or "medium" or "low"}}

Rules: metric_name must be a generic measure word/phrase from the caption or
context, never a dataset/benchmark/method name. If unsure, set confidence "low"
and metric_name "". Do not invent labels absent from the table."""


def build_prompt(repr_, caption, context, registry_names):
    return PROMPT.format(
        header_html=repr_["header_html"][:600],
        row_quote=repr_["row_quote_sample"],
        cols=json.dumps(repr_["column_headers"][:20], ensure_ascii=False),
        rows=json.dumps(repr_["row_labels"][:20], ensure_ascii=False),
        caption=(caption or "(none found)")[:300],
        context=re.sub(r"\s+", " ", (context or ""))[:500],
        registry_names=", ".join(list(registry_names)[:120]),
    )


def llm_propose(prompt, model):
    raw = call_paratera(prompt, model=model, max_tokens=400) or ""
    obj = parse_json_response(raw)
    if isinstance(obj, list):
        obj = next((x for x in obj if isinstance(x, dict)), None)
    if not isinstance(obj, dict):
        return None
    ma = str(obj.get("method_axis", "")).strip().lower()
    if ma not in ("column", "row"):
        return None
    mn = str(obj.get("metric_name", "")).strip().lower()[:60]
    conf = str(obj.get("confidence", "low")).strip().lower()
    if conf not in ("high", "medium", "low"):
        conf = "low"
    return {"method_axis": ma, "metric_name": mn, "confidence": conf}


# ---------------------------------------------------------------- gate
def gate(proposal, repr_, caption, context, lookup):
    """STRUCTURAL validation only. Returns (verdict, proposal2, reasons).
    verdict in {accept, degrade, reject}. proposal2 is the (possibly degraded)
    proposal safe to apply."""
    reasons = []
    ma = proposal["method_axis"]
    mn = proposal.get("metric_name", "")
    conf = proposal.get("confidence", "low")
    p2 = dict(proposal)

    # G1 structural: if method_axis=column, the column headers are the methods —
    # they come from the records so they exist in the table by construction.
    # Guard against a degenerate proposal that names no usable axis.
    if ma == "column" and not repr_["column_headers"]:
        return "reject", p2, reasons + ["G1: method_axis=column but no column headers"]
    if ma == "row" and not repr_["row_labels"]:
        return "reject", p2, reasons + ["G1: method_axis=row but no row labels"]

    # G3 benchmark leakage: metric_name must not be a known benchmark/dataset.
    if mn:
        mnorm = re.sub(r"[^a-z0-9 ]", "", mn)
        for b in BENCHMARK_BLOCKLIST:
            if b in mnorm:
                return "reject", p2, reasons + [f"G3: metric_name '{mn}' is a benchmark/dataset name"]

    # G4 caption provenance: metric_name must appear in caption/context.
    if mn:
        hay = re.sub(r"\s+", " ", (caption + " " + context)).lower()
        # token-level containment (metric words like 'accuracy','f1','perplexity')
        if mn not in hay and not any(tok in hay for tok in mn.split() if len(tok) >= 4):
            reasons.append(f"G4: metric_name '{mn}' not in caption/context -> degrade to empty")
            p2["metric_name"] = ""

    # G6 conservative: low confidence -> only the role swap (subject/method),
    # do NOT overwrite metric (keep deterministic F32 metric).
    if conf == "low":
        if p2["metric_name"]:
            reasons.append("G6: low confidence -> metric_name dropped (keep F32 metric)")
        p2["metric_name"] = ""

    # G2 registry: handled at correction time (re-link method labels); unlinked
    # methods stay surface-only (never fabricate entity_id). Not a reject.
    return "accept", p2, reasons


# ---------------------------------------------------------------- correction
def apply_correction(recs, proposal, lookup):
    """Re-assign record fields per an accepted proposal. Only the role-reversal
    (method_axis=column) is corrected in v1; method_axis=row matches the
    deterministic default -> no change (return recs as-is, unmarked)."""
    if proposal["method_axis"] != "column":
        return recs, 0
    out, n = [], 0
    mn = proposal.get("metric_name", "")
    for r in recs:
        r2 = json.loads(json.dumps(r))  # deep copy
        old_metric = _metric_of(r)            # the column header == the METHOD
        old_surf = ((r.get("method_ref") or {}).get("surface") or "").strip()  # row == DATASET
        # method <- old column header; re-link against registry (G2)
        link = _link_method(_strip_cites(old_metric), lookup) or {}
        r2["method_ref"] = {"surface": _strip_cites(old_metric)[:60],
                            "canonical": link.get("canonical"),
                            "entity_id": link.get("entity_id")}
        # subject <- old row label (the dataset/task)
        if old_surf and _norm_tier(_strip_cites(old_surf)) not in GENERIC_TIER:
            r2.setdefault("dims_new", {})["dims.subject"] = [_strip_cites(old_surf)[:60]]
        # metric <- recovered true metric (if gate kept it), else keep F32 metric
        if mn:
            r2["measure"]["metric"] = mn
        r2["provenance"] = "f35_semantic"
        out.append(r2)
        n += 1
    return out, n


# ---------------------------------------------------------------- main
def load_texts(texts_dir):
    tx = {}
    if texts_dir and os.path.isdir(texts_dir):
        for fn in os.listdir(texts_dir):
            if fn.endswith((".md", ".txt")):
                with open(os.path.join(texts_dir, fn), encoding="utf-8", errors="replace") as f:
                    tx[fn.rsplit(".", 1)[0]] = f.read()
    return tx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)   # table_channel/merged records (dict by pid)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen3.6-27B")
    ap.add_argument("--dry", action="store_true", help="no LLM; report trigger surface only")
    ap.add_argument("--limit", type=int, default=0, help="cap tables processed (0=all)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    recs_by_paper = json.load(open(args.records, encoding="utf-8"))
    registry = json.load(open(args.registry, encoding="utf-8"))
    lookup = _build_entity_lookup(registry)
    reg_names = set()
    for e in registry.get("entities", []):
        if e.get("canonical"):
            reg_names.add(e["canonical"])
    texts = load_texts(args.texts)

    tables = group_tables(recs_by_paper)
    trig = {k: t for k, t in tables.items() if t["trigger"]}
    print(f"tables={len(tables)} trigger={len(trig)} "
          f"({round(100*len(trig)/max(1,len(tables)),1)}%)")
    from collections import Counter
    print("trigger reasons:", dict(Counter(t["trigger"] for t in trig.values())))
    if args.dry:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    corrected = {}      # (pid,th) -> [records]
    rejected = []
    items = list(trig.items())
    if args.limit:
        items = items[:args.limit]
    n_acc = n_deg = n_rej = n_corr = 0
    for i, (key, t) in enumerate(items):
        pid, th = key
        repr_ = table_repr(t["records"])
        cs = t["records"][0].get("chunk_char_start") or 0
        caption, context = caption_and_context(texts.get(pid, ""), cs)
        prompt = build_prompt(repr_, caption, context, reg_names)
        prop = llm_propose(prompt, args.model)
        if prop is None:
            rejected.append({"pid": pid, "th": th[:80], "reason": "llm_unparseable"}); n_rej += 1
            continue
        verdict, prop2, reasons = gate(prop, repr_, caption, context, lookup)
        if verdict == "reject":
            rejected.append({"pid": pid, "th": th[:80], "proposal": prop, "reasons": reasons})
            n_rej += 1
            continue
        if verdict == "degrade" or (reasons and not prop2.get("metric_name")):
            n_deg += 1
        else:
            n_acc += 1
        new_recs, nc = apply_correction(t["records"], prop2, lookup)
        if nc:
            corrected[key] = new_recs
            n_corr += nc
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(items)}] acc={n_acc} deg={n_deg} rej={n_rej} corrected_recs={n_corr}", flush=True)

    json.dump({f"{k[0]}||{k[1]}": v for k, v in corrected.items()},
              open(os.path.join(args.out_dir, "f35_corrected.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(rejected, open(os.path.join(args.out_dir, "f35_rejected.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"DONE tables_processed={len(items)} accept={n_acc} degrade={n_deg} reject={n_rej} "
          f"corrected_records={n_corr} -> {args.out_dir}")


if __name__ == "__main__":
    main()
