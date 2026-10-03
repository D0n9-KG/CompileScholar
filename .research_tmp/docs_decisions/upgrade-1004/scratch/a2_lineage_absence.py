# -*- coding: utf-8 -*-
"""Lineage endpoints, absence mojibake, entity alignment, survey claim labels. Read-only."""
import json, os, re, sys, collections, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
V2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb_v2"
P = json.load(open(os.path.join(V2, "papers.json"), encoding="utf-8"))
R = json.load(open(os.path.join(V2, "records.json"), encoding="utf-8"))
C = collections.Counter
recs = [(pid, r) for pid, v in R.items() for r in v["records"]]

def norm(s): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").lower())).strip()
def end(x):
    if isinstance(x, dict): return x.get("canonical") or x.get("surface") or ""
    return str(x or "")
CIT = re.compile(r"\b[A-Z][A-Za-z\-]+(?: et al\.?| and [A-Z][A-Za-z\-]+)?,? \(?(19|20)\d\d[a-z]?\)?")
lin = [(pid, r) for pid, r in recs if r.get("kind") == "lineage"]
print("lineage", len(lin), C(r.get("relation") for _, r in lin).most_common())
print("lineage scope", C(str(r.get("scope")) for _, r in lin).most_common(6), "evidence_basis", C(str(r.get("evidence_basis")) for _, r in lin).most_common(6))
cit_end = sum(1 for _, r in lin if CIT.search(end(r.get("from_method_ref"))) or CIT.search(end(r.get("to_method_ref"))))
print("lineage with citation-string endpoint", cit_end)
# endpoint entity ids
eid_both = sum(1 for _, r in lin if all(isinstance(r.get(k), dict) and r[k].get("entity_id") for k in ("from_method_ref", "to_method_ref")))
print("lineage both endpoints have entity_id", eid_both)
# resolve endpoint to KB paper: does endpoint name appear as a method subject of a KB paper (method record, non-survey)?
method_owner = collections.defaultdict(set)
for pid, r in recs:
    if r.get("kind") == "method" and r.get("subject"):
        method_owner[norm(r["subject"])].add(pid)
title_norm = {norm(p["title"]): pid for pid, p in P.items()}
hit_method = hit_title = 0
for _, r in lin:
    a, b = norm(end(r.get("from_method_ref"))), norm(end(r.get("to_method_ref")))
    if (a in method_owner) or (b in method_owner): hit_method += 1
    if any(t.startswith(x + " ") or t == x for x in (a, b) if len(x) > 3 for t in ()): pass
print("lineage endpoint == some KB method-record subject (exact norm)", hit_method)
# acronym-prefix in title e.g. "BERT: ..."
pref = collections.defaultdict(set)
for pid, p in P.items():
    m = re.match(r"^\s*([A-Za-z0-9\-\+]{2,20})\s*:", p["title"])
    if m: pref[norm(m.group(1))].add(pid)
hp = sum(1 for _, r in lin if norm(end(r.get("from_method_ref"))) in pref or norm(end(r.get("to_method_ref"))) in pref)
print("KB titles with 'NAME:' prefix", len(pref), "lineage endpoint matches such prefix", hp)
# endpoints: distinct names, freq
ends = C(norm(end(r.get(k))) for _, r in lin for k in ("from_method_ref", "to_method_ref"))
print("distinct endpoint names", len(ends), "top", ends.most_common(15))
print("lineage sample:")
for pid, r in random.Random(1).sample(lin, 5):
    print("  ", (P[pid]["layer"]), end(r.get("from_method_ref")), "|", r.get("relation"), "|", end(r.get("to_method_ref")), "||", (r.get("quote") or "")[:160])
# ---- absence mojibake
ab = [(pid, r) for pid, r in recs if r.get("kind") == "absence"]
def moj(s):
    s = s or ""
    if not s: return None
    bad = len(re.findall(r"[\u00c0-\u00ff][\u0080-\u00bf]|[ÃÂâ€]|\ufffd|[\u4e00-\u9fff]{0}", s))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", s)); asc = len(re.findall(r"[A-Za-z]", s))
    return bad, cjk, asc
types = C(); ex = {}
for pid, r in ab:
    m = r.get("missing")
    if not isinstance(m, str) or not m.strip(): types["empty/nonstr"] += 1; continue
    bad, cjk, asc = moj(m)
    odd = len(re.findall(r"[\u0400-\u04ff\u0370-\u03ff\u0590-\u06ff\u3040-\u30ff\uac00-\ud7af\u0e00-\u0e7f\u2500-\u25ff\ue000-\uf8ff]", m))
    if "\ufffd" in m or bad > 2: t = "mojibake_latin1"
    elif odd > 3: t = "mojibake_other_script"
    elif cjk > 3: t = "chinese"
    elif asc > 10: t = "english"
    else: t = "other"
    types[t] += 1; ex.setdefault(t, m[:120])
print("absence missing types", types.most_common())
for t, s in ex.items(): print("  ex", t, repr(s))
mz = C(("zh" if isinstance(r.get("missing_zh"), str) and r["missing_zh"].strip() else "-", str(type(r.get("missing_translated")).__name__)) for _, r in ab)
print("missing_zh/translated", mz.most_common())
print("absence_type", C(r.get("absence_type") for _, r in ab).most_common(), "resolved_by set", sum(1 for _, r in ab if r.get("resolved_by")))
# ---- entity alignment
ent_p = collections.defaultdict(set); ent_name = {}
for pid, r in recs:
    for e in r.get("entity_refs") or []:
        if isinstance(e, dict) and e.get("entity_id"):
            ent_p[e["entity_id"]].add(pid); ent_name.setdefault(e["entity_id"], e.get("canonical") or e.get("surface"))
print("entity ids", len(ent_p), ">=2 papers", sum(1 for v in ent_p.values() if len(v) >= 2), ">=5", sum(1 for v in ent_p.values() if len(v) >= 5))
print("top shared", [(ent_name[k], len(v)) for k, v in sorted(ent_p.items(), key=lambda x: -len(x[1]))[:15]])
# ---- survey claim labels / snapshot types
print("survey_claim labels", C(r.get("semantic_label") for _, r in recs if r.get("kind") == "survey_claim").most_common())
print("snapshot types", C(r.get("snapshot_type") for _, r in recs if r.get("kind") == "domain_snapshot").most_common())
print("finding claim_type", C(r.get("claim_type") for _, r in recs if r.get("kind") in ("finding", "method", "limitation")).most_common(12))
# survey_claim with cited refs field?
sc = [r for _, r in recs if r.get("kind") == "survey_claim"]
print("survey_claim fields", C(k for r in sc for k in r).most_common(30))
