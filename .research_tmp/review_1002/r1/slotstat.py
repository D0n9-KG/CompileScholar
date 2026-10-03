import json, os, re, collections, sys
B = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2"
REC = {}
for f in ("records_merged.json", "deep_read_records.json"):
    for p in json.load(open(os.path.join(B,"base_kb",f), encoding="utf-8")).values():
        if isinstance(p, dict):
            for r in p.get("records") or []: REC[r.get("id")] = r
src = open(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\_shared\tools\evidence_gate2r_harness.py", encoding="utf-8").read()
ns = {"re": re}
exec(src[src.index("_SLOT_STOP = {"):src.index("def render_evidence_plan")], ns)
def status_fixed(s, notes):
    toks = ns["_slot_tokens"](s["need"]); hits = 0
    for rid in re.findall(r"\[((?:coarse:|svy_)?[0-9a-f]{12,16})\]", notes):
        r = REC.get(rid)
        if r and toks & ns["_slot_tokens"](str(r.get("subject") or "")+" "+str(r.get("claim") or "")): hits += 1
    if hits: return "C"
    nt = set(re.findall(r"[a-z]{4,}", notes.lower())); h = sum(1 for t in toks if t in nt)
    return "P" if h else "O"
for b in sys.argv[1:]:
    rows = json.load(open(os.path.join(B,"arm_ours",f"answers_pilot_cs2batch{b}.json"), encoding="utf-8"))
    sets = [x["slots"] for x in rows[0]["backfills"] if isinstance(x, dict) and x.get("slots")]
    cur = collections.Counter(); fix = collections.Counter(); allcov_cur = allcov_fix = 0
    for r in rows:
        qt = set(re.findall(r"[a-z]{4,}", r["question"].lower()))
        sl = max(sets, key=lambda S: len(qt & set(t for s in S for t in re.findall(r"[a-z]{4,}", s["need"].lower()))))
        notes = r["notes_final"] or ""
        a = [ns["_slot_status2"](s, notes, REC)[0] for s in sl]; f = [status_fixed(s, notes) for s in sl]
        cur.update(a); fix.update(f); allcov_cur += all(x=="C" for x in a); allcov_fix += all(x=="C" for x in f)
    print(f"{b}: current statuses {dict(cur)} all-COVERED q={allcov_cur}/15 || with coarse:/svy_ ids recognized {dict(fix)} all-COVERED q={allcov_fix}/15")
