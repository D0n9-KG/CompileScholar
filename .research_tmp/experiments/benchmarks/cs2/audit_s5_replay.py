# -*- coding: utf-8 -*-
"""S5 audit replay: parse_notes line-by-line classification over batches 10-14
(production config: adapt_batches.py EvidenceStore WITHOUT texts_dir)."""
import json, re, sys, os, hashlib, collections

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\_shared\tools")
from report_adapter import (_NOTE_LINE, _NOTE_LINE_DASH, _NOTE_LINE_DASH_TAIL,
                            _BACKREF, EvidenceStore, parse_notes)

CS2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2"
BK = os.path.join(CS2, "base_kb")

def load_records():
    records = json.load(open(os.path.join(BK, "records_merged.json"), encoding="utf-8"))
    deep = json.load(open(os.path.join(BK, "deep_read_records.json"), encoding="utf-8"))
    for pid, payload in deep.items():
        if not (isinstance(payload, dict) and payload.get("records")):
            continue
        base_recs = (records.get(pid) or {}).get("records") or []
        seen = {r.get("id") for r in base_recs if r.get("id")}
        records[pid] = {"records": base_recs + [r for r in payload["records"] if r.get("id") not in seen]}
    return records

records = load_records()
manifest = json.load(open(os.path.join(BK, "manifest_all.json"), encoding="utf-8"))
store = EvidenceStore(records, manifest)  # production config
store_td = EvidenceStore(records, manifest, texts_dir=os.path.join(BK, "deep_read_texts"))

# index of ALL ids appearing in any base_kb records source (for miss classification)
all_kb_ids = set()
for fn in ["records_merged.json", "deep_read_records.json", "coarse_records.json",
           "coarse_records_full.json", "records_hub.json", "records_survey.json",
           "survey_records_adapted.json"]:
    p = os.path.join(BK, fn)
    if not os.path.exists(p):
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    for pid, payload in d.items():
        if isinstance(payload, dict):
            for r in payload.get("records", []) or []:
                if isinstance(r, dict) and r.get("id"):
                    all_kb_ids.add(r["id"])
all_paper_ids = {row["paper_id"] for row in manifest}
# also deep texts available set
texts_avail = set(os.listdir(os.path.join(BK, "deep_read_texts")))

_CORPUS_ROW = re.compile(r"^\s*-\s*\S+\s*\(\d{4}\)\s*$")

def drop_reason(s):
    if s.startswith("Q:"): return "header:Q"
    if s.lower().startswith("relevant corpus"): return "header:corpus-list-title"
    if _CORPUS_ROW.match(s): return "corpus-paper-row(no [id])"
    if s.startswith("Entity "): return "header:entity"
    if re.match(r"^[A-Z]\d+\s", s) or re.match(r"^[NX]\d+[^\.\s]", s): return "N/X-line-malformed(no dot)"
    if s.startswith("[plan]"): return "plan-line"
    if s.startswith("[unsourced]"): return "unsourced-form-unmatched"
    if _BACKREF.search(s): return "OTHER-but-has-[backref]"
    if s.startswith("-"): return "other-dash-line"
    return "other"

def classify_ref(ref):
    e = store.resolve(ref)
    if e is not None:
        return "rec" if e["tier"] == "full" and "#" not in ref else (
            "paper(title)" if e["tier"] == "title" else "anchor-in-records")
    if "#" in ref:
        pid = ref.partition("#")[0]
        f = hashlib.md5(pid.encode()).hexdigest() + ".txt"
        if f in texts_avail:
            return "chunk-anchor(texts_dir-given)"
        return "chunk-anchor(text-file-missing)"
    if ref in all_kb_ids: return "miss(id-in-KB-not-in-EvidenceStore)"
    if ref in all_paper_ids: return "miss(paper-in-manifest-not-in-store)"  # shouldn't happen
    return "miss(nowhere)"

def replay_line(line):
    s = line.strip()
    if not s: return ("blank", "", 0, [], "")
    orig = s
    form = None
    if s.startswith("[unsourced]"):
        rest = s[len("[unsourced]"):].lstrip()
        if _NOTE_LINE.match(rest):
            line = rest; form = "unsourced+N"
        elif not _BACKREF.search(s):
            return ("dropped", "unsourced-no-backref", 0, [], orig)
        else:
            line = s; form = "unsourced+ref(pass-through)"
    m = _NOTE_LINE.match(line)
    md = _NOTE_LINE_DASH.match(line) if not m else None
    mdt = _NOTE_LINE_DASH_TAIL.match(line) if not m and not md else None
    if not m and not md and not mdt:
        return ("dropped", drop_reason(s), 0, [], orig)
    if form is None:
        form = "N/X" if m else ("dash" if md else "dash-tail")
    body = (m.group(2) if m else (md.group(1) if md else mdt.group(0)))
    refs = _BACKREF.findall(body)
    text = _BACKREF.sub(" ", body, count=len(refs))
    text = re.sub(r"^\s*-\s*", "", text)
    # P1-5 anchor merge replay
    tail = text.split("|", 1)[1] if "|" in text else ""
    text0 = text.split("|")[0].strip()
    m_anchor = re.search(r'anchor\s*:\s*"([^"]+)"', tail)
    merged = dropped_anchor = 0
    if m_anchor:
        a = m_anchor.group(1).strip()
        na = set(re.findall(r"\d+\.?\d*", a)); nt = set(re.findall(r"\d+\.?\d*", text0))
        if (na - nt) or not text0: merged = 1
        else: dropped_anchor = 1
    inval = line.strip().startswith("X")
    # ref classification
    refcls = [classify_ref(r) for r in refs]
    return ("ok", form, len(refs), refcls, orig,
            {"merged": merged, "anchor_dropped": dropped_anchor,
             "has_anchor": 1 if m_anchor else 0, "invalidated": 1 if inval else 0})

BATCHES = [10, 11, 12, 13, 14]
report = {}
for b in BATCHES:
    rows = json.load(open(os.path.join(CS2, "arm_ours", f"answers_pilot_cs2batch{b}.json"), encoding="utf-8"))
    agg = collections.Counter()
    drop_samples = collections.defaultdict(list)
    ref_misses = collections.Counter()
    per_q = []
    for row in rows:
        nf = row.get("notes_final") or ""
        qstat = collections.Counter()
        for line in nf.splitlines():
            r = replay_line(line)
            if r[0] == "blank":
                agg["blank"] += 1; continue
            if r[0] == "dropped":
                agg["dropped"] += 1; agg["drop:" + r[1]] += 1
                qstat["dropped"] += 1
                if len(drop_samples[r[1]]) < 6:
                    drop_samples[r[1]].append(r[4][:130])
                if r[1] == "OTHER-but-has-[backref]":
                    qstat["dropped_with_ref"] += 1
            else:
                agg["ok"] += 1; agg["form:" + r[1]] += 1
                qstat["ok"] += 1
                for c in r[3]:
                    agg["ref:" + c] += 1
                    if c.startswith("miss") or "anchor" in c:
                        ref_misses[c] += 1
                for k in ("merged", "anchor_dropped", "has_anchor", "invalidated"):
                    agg[k] += r[5][k]
        per_q.append({"qid": row["id"], "q": row["question"][:60], **qstat})
    report[b] = {"agg": dict(agg), "drop_samples": {k: v for k, v in drop_samples.items()},
                 "per_q": per_q}

out = os.path.join(CS2, "audit_s5_replay_out.json")
json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for b in BATCHES:
    a = report[b]["agg"]
    ok = a.get("ok", 0); dr = a.get("dropped", 0)
    print(f"\n===== batch {b}: lines ok={ok} dropped={dr} (drop rate {dr/(ok+dr):.1%}) =====")
    for k in sorted(a):
        if k.startswith(("drop:", "form:")): print(f"  {k}: {a[k]}")
    for k in ("has_anchor", "merged", "anchor_dropped", "invalidated"):
        print(f"  {k}: {a.get(k, 0)}")
    for k in sorted(a):
        if k.startswith("ref:"): print(f"  {k}: {a[k]}")
print("\nsaved:", out)
