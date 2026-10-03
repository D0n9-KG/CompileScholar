import json, sys, collections, os
A = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\arm_ours"
for b in sys.argv[1:]:
    p = os.path.join(A, f"answers_pilot_cs2batch{b}.json")
    rows = json.load(open(p, encoding="utf-8"))
    tools = collections.Counter(); n = len(rows)
    fb = hs = hint = pre = autos = 0; steps = []; nslots = []; decfail = 0
    ext_q = 0; struct_q = 0
    for r in rows:
        g = r.get("gate") or {}
        fb += bool(g.get("fallback_compiled")); hs += bool(g.get("f28_hard_stop"))
        autos += g.get("f28_autos", 0); pre += g.get("plan_precheck", 0)
        steps.append(r.get("steps"))
        qt = collections.Counter()
        for t in r.get("trajectory") or []:
            if t.get("tool"): tools[t["tool"]] += 1; qt[t["tool"]] += 1
            if t.get("f28_external_hint"): hint += 1
        if any(k in qt for k in ("search_papers","gap_search","lineage_walk_ext","citation_graph")): ext_q += 1
        if any(k in qt for k in ("lineage","lineage_walk_ext","compare","find_gap","as_of","gap_search")): struct_q += 1
        bf = r.get("backfills") or []
    tot = sum(tools.values())
    print(f"== {b}: n={n} fallback={fb} hard_stop={hs} ext_hint={hint} f28_autos={autos} precheck={pre} steps_med={sorted(steps)[n//2]} q_with_ext={ext_q} q_with_struct={struct_q} calls={tot}")
    print("   ", ", ".join(f"{k}:{v}({100*v/tot:.0f}%)" for k, v in tools.most_common()))
