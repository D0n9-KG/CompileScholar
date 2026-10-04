# -*- coding: utf-8 -*-
"""S5 audit part 2: orphan cause classification + noref claims + miss refs."""
import json, re, sys, os, hashlib, collections

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\_shared\tools")
from report_adapter import parse_notes, EvidenceStore

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
store = EvidenceStore(records, manifest)              # production
store_td = EvidenceStore(records, manifest, texts_dir=os.path.join(BK, "deep_read_texts"))
texts_avail = set(os.listdir(os.path.join(BK, "deep_read_texts")))

def ref_class_prod(ref):
    e = store.resolve(ref)
    if e is not None:
        return "resolves"
    if "#" in ref:
        pid = ref.partition("#")[0]
        f = hashlib.md5(pid.encode()).hexdigest() + ".txt"
        return "anchor:texts_dir-would-resolve" if f in texts_avail else "anchor:text-file-missing"
    return "miss(nowhere)"

out = {}
for b in [10, 11, 12, 13, 14]:
    rows = json.load(open(os.path.join(CS2, "arm_ours", f"answers_pilot_cs2batch{b}.json"), encoding="utf-8"))
    binfo = {"orphan_candidate_claims": [], "noref_claims": [], "miss_refs": [],
             "counts": collections.Counter()}
    for row in rows:
        claims = parse_notes(row.get("notes_final") or "")
        for c in claims:
            if c["invalidated"] or not c["text"]:
                continue
            if not c["refs"]:
                binfo["noref_claims"].append({"qid": row["id"], "text": c["text"][:110]})
                binfo["counts"]["noref"] += 1
                continue
            cls = [ref_class_prod(r) for r in c["refs"]]
            for r, k in zip(c["refs"], cls):
                if k == "miss(nowhere)":
                    binfo["miss_refs"].append({"qid": row["id"], "ref": r})
                    binfo["counts"]["miss_ref"] += 1
            if all(k != "resolves" for k in cls):
                # usable claim whose every ref fails -> orphan sentence if cited
                cause = ("anchor-only(would-resolve-with-texts_dir)"
                         if any("would-resolve" in k for k in cls)
                         else ("anchor-only(text-missing)" if any("anchor" in k for k in cls)
                               else "all-refs-miss(nowhere)"))
                binfo["orphan_candidate_claims"].append({
                    "qid": row["id"], "refs": c["refs"], "cause": cause,
                    "text": c["text"][:110]})
                binfo["counts"]["orphan_candidate"] += 1
                binfo["counts"][cause] += 1
    out[b] = {"counts": dict(binfo["counts"]),
              "orphan_candidates": binfo["orphan_candidate_claims"],
              "noref": binfo["noref_claims"], "miss_refs": binfo["miss_refs"]}
    c = binfo["counts"]
    print(f"\n===== batch {b} =====")
    print(f"  noref claims (refs=[] parsed): {c.get('noref',0)}")
    print(f"  orphan-candidate claims (all refs unresolvable): {c.get('orphan_candidate',0)}")
    print(f"  breakdown: anchor-rescuable={c.get('anchor-only(would-resolve-with-texts_dir)',0)}"
          f" anchor-missing={c.get('anchor-only(text-missing)',0)}"
          f" all-miss={c.get('all-refs-miss(nowhere)',0)}")
    print(f"  total miss(nowhere) refs: {c.get('miss_ref',0)}")

json.dump(out, open(os.path.join(CS2, "audit_s5_orphan_out.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nsaved audit_s5_orphan_out.json")
