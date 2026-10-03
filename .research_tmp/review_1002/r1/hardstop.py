import json, re, os, sys, collections
sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\_shared\tools")
A = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2"
# REC_INDEX equivalent: merged + deep records
REC = {}
for f in ("records_merged.json", "deep_read_records.json"):
    d = json.load(open(os.path.join(A, "base_kb", f), encoding="utf-8"))
    for pid, p in d.items():
        if isinstance(p, dict):
            for r in p.get("records") or []:
                if r.get("id"): REC[r["id"]] = r
print("REC size", len(REC))
import importlib.util
src = open(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\_shared\tools\evidence_gate2r_harness.py", encoding="utf-8").read()
# exec only the slot helpers (no heavy imports)
ns = {"re": re}
start = src.index("_SLOT_CLAIMS = {"); end = src.index("def decompose_question")
exec(src[start:end], ns)
start = src.index("_SLOT_STOP = {"); end = src.index("def render_evidence_plan")
exec(src[start:end], ns)
for b in sys.argv[1:]:
    rows = json.load(open(os.path.join(A, "arm_ours", f"answers_pilot_cs2batch{b}.json"), encoding="utf-8"))
    print(f"== {b}")
    for r in rows:
        g = r["gate"]; qid = r["id"][:8]
        bf = r.get("backfills") or []
        slots = [x for x in bf if isinstance(x, dict) and "slots" in x]
        mine = [x for x in slots if str(x.get("qid","")) [:8] == qid]
        sl = (mine[0]["slots"] if mine else None) or []
        notes = r.get("notes_final") or ""
        nl = [l for l in notes.split("\n") if re.match(r"\s*N\d+\.", l)]
        rids = set(re.findall(r"\[([0-9a-f]{12,16})\]", notes))
        st = [ns["_slot_status2"](s, notes, REC)[0] for s in sl]
        # generic-token COVERED: slot covered but the only overlapping tokens are generic
        gen = []
        for s in sl:
            toks = ns["_slot_tokens"](s["need"])
            ov = set()
            for rid in rids:
                rr = REC.get(rid)
                if rr: ov |= toks & ns["_slot_tokens"](str(rr.get("subject") or "")+" "+str(rr.get("claim") or ""))
            gen.append(sorted(ov)[:4])
        hs_step = next((t.get("at_step") for t in r["trajectory"] if t.get("f28_hard_stop")), None)
        print(f" {qid} hs={'Y@'+str(hs_step) if g.get('f28_hard_stop') else '-'} steps={r['steps']}/{r['cap']} Nlines={len(nl)} rids={len(rids)} slots={''.join(st) or 'none'} bf_slots={len(slots)} autos={g.get('f28_autos')} | ov={gen[:3]}")
