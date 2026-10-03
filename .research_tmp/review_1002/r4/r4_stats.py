# -*- coding: utf-8 -*-
"""R4 review: tool usage / termination / score correlates / variance / cross-arm.
Read-only over cs2 artifacts. Output -> r4_stats_out.json + stdout."""
import json, re, os, glob, collections, statistics as st, math, random

CS2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2"
ARM = os.path.join(CS2, "arm_ours")
BASE = os.path.join(CS2, "base_kb")
BATCHES = ['31a', '31b', '32a', '32b', '33a', '33b', '34a', '34b', '34c', '34d', '34e']


def qavg(s):
    ir = (s.get("ingredient_recall") or {}).get("ingredient_recall")
    ap = (s.get("answer_precision") or {}).get("answer_precision")
    c = s.get("citation") or {}
    cf, cr, cp = c.get("f1"), c.get("citation_recall"), c.get("citation_precision")
    vals = [v for v in (ir, ap, cf) if isinstance(v, (int, float))]
    return {"IR": ir, "AP": ap, "CT": cf, "CR": cr, "CP": cp,
            "G": sum(vals) / len(vals) if vals else None}


def scores(path):
    d = json.load(open(path, encoding="utf-8"))
    return {k: qavg(v) for k, v in d.items() if isinstance(v, dict) and v.get("ingredient_recall")}


# ---- record-id -> source tier ----
merged = json.load(open(os.path.join(BASE, "records_merged.json"), encoding="utf-8"))
deep = json.load(open(os.path.join(BASE, "deep_read_records.json"), encoding="utf-8"))
snapdeep = json.load(open(os.path.join(ARM, "kb_snapshot_writes", "deep_read_records.json"), encoding="utf-8"))
rid_src = {}
for pid, p in merged.items():
    for r in (p or {}).get("records") or []:
        rid = r.get("id")
        if not rid:
            continue
        src = r.get("source") or r.get("tier") or ""
        if rid.startswith("svy_") or r.get("kind") in ("survey_claim", "domain_snapshot"):
            t = "survey"
        elif rid.startswith("coarse:"):
            t = "coarse"
        else:
            t = "merged_other"
        rid_src[rid] = t
for pid, p in deep.items():
    for r in p.get("records") or []:
        rid_src[r.get("id")] = "deep_base"
for pid, p in snapdeep.items():
    for r in p.get("records") or []:
        rid_src.setdefault(r.get("id"), "deep_runtime")

TOK = re.compile(r'\[([^\[\]\n]{3,160})\]')


def classify(tok):
    if tok.startswith("ext:"):
        return "ext_candidate(abstract)"
    if "#" in tok:
        return "chunk_anchor(fulltext)"
    if tok in rid_src:
        return rid_src[tok]
    if tok.startswith("coarse:"):
        return "coarse"
    if tok.startswith("svy_"):
        return "survey"
    if re.fullmatch(r"[0-9a-f]{10,16}", tok):
        return "deep_runtime"   # runtime deep record id (not in base files)
    if tok.startswith(("sciverse_", "ext_", "arxiv")):
        return "paper_id(paper-level)"
    return "other"


def cited(answer):
    out = []
    for m in TOK.findall(answer or ""):
        for t in re.split(r"[;,]\s*", m):
            t = t.strip()
            if t and not re.fullmatch(r"[\d.]+", t):
                out.append(t)
    return out


STATE_TOOLS = {"lineage", "lineage_walk_ext", "compare", "find_gap", "gap_search", "as_of"}
EXT_TOOLS = {"search_papers", "admit_paper", "extract_paper", "deep_read", "gap_search", "citation_graph"}

res = {"per_batch": {}, "rows": []}
tool_tot = collections.Counter()
for b in BATCHES:
    A = json.load(open(os.path.join(ARM, f"answers_pilot_cs2batch{b}.json"), encoding="utf-8"))
    S = scores(os.path.join(CS2, f"direct_scores_ours_batch{b}_ds.json"))
    tc = collections.Counter()
    term = collections.Counter()
    src = collections.Counter()
    for a in A:
        traj = [t for t in a["trajectory"] if t.get("tool")]
        for t in traj:
            tc[t["tool"]] += 1
        g = a.get("gate") or {}
        if g.get("f28_hard_stop"):
            tm = "F28_hard_stop->fallback"
        elif g.get("fallback_compiled"):
            tm = "budget/forced->fallback"
        elif g.get("forced_notes_submit"):
            tm = "forced_submit(no fallback)"
        else:
            tm = "agent_submit"
        term[tm] += 1
        cs = cited(a["answer"])
        cc = collections.Counter(classify(t) for t in cs)
        src.update(cc)
        sc = S.get(a["id"], {})
        first = [t for t in traj if t["step"] == 1]
        res["rows"].append({
            "batch": b, "qid": a["id"], "steps": a.get("steps"), "term": tm,
            "n_calls": len(traj),
            "n_ext": sum(1 for t in traj if t["tool"] in ("search_papers", "admit_paper", "extract_paper")),
            "n_deep": sum(1 for t in traj if t["tool"] == "deep_read"),
            "n_state": sum(1 for t in traj if t["tool"] in STATE_TOOLS),
            "step1_obs_chars": sum(t["obs_chars"] for t in first),
            "step1_findings_nonempty": sum(1 for t in first if t["tool"] == "findings" and t["obs_chars"] > 300),
            "n_cites": len(cs), "cite_src": dict(cc),
            "note_rejects": (g.get("note_rejects") or 0),
            "cit_stripped": g.get("compile_citation_stripped") or 0,
            **{k: sc.get(k) for k in ("G", "IR", "AP", "CT", "CR", "CP")}})
    tool_tot.update(tc)
    n = sum(tc.values())
    gs = [S[a["id"]]["G"] for a in A if a["id"] in S and S[a["id"]]["G"] is not None]
    res["per_batch"][b] = {"calls": n, "tools": dict(tc.most_common()),
                           "term": dict(term), "cite_src": dict(src),
                           "G": round(st.mean(gs), 4) if gs else None, "n_scored": len(gs)}

rows = res["rows"]
print("=== tool totals (31a-34e) ===")
N = sum(tool_tot.values())
for t, c in tool_tot.most_common():
    print(f"{t:18s}{c:6d}  {100*c/N:5.1f}%")
print("N calls", N, "| state tools share",
      round(100 * sum(c for t, c in tool_tot.items() if t in STATE_TOOLS) / N, 2), "%")

print("\n=== termination ===")
tt = collections.Counter(r["term"] for r in rows)
print(dict(tt))
for tm in tt:
    gs = [r["G"] for r in rows if r["term"] == tm and r["G"] is not None]
    irs = [r["IR"] for r in rows if r["term"] == tm and r["IR"] is not None]
    cts = [r["CT"] for r in rows if r["term"] == tm and r["CT"] is not None]
    stp = [r["steps"] for r in rows if r["term"] == tm]
    print(f"  {tm:28s} n={len(gs):3d} G={st.mean(gs):.3f} IR={st.mean(irs):.3f} CT={st.mean(cts):.3f} steps={st.mean(stp):.1f}")
for b in BATCHES:
    print(" ", b, res["per_batch"][b]["term"], "G", res["per_batch"][b]["G"])

print("\n=== citation source mix (all 11 batches) ===")
cs_tot = collections.Counter()
for r in rows:
    cs_tot.update(r["cite_src"])
M = sum(cs_tot.values())
for k, v in cs_tot.most_common():
    print(f"  {k:26s}{v:6d} {100*v/M:5.1f}%")


# ---- correlations (Spearman) ----
def rank(x):
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0] * len(x)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spear(x, y):
    p = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(p) < 5:
        return None
    rx, ry = rank([a for a, _ in p]), rank([b for _, b in p])
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 3) if den else None


print("\n=== Spearman (all rows; and within-question demeaned) ===")
qmean = collections.defaultdict(list)
for r in rows:
    if r["G"] is not None:
        qmean[r["qid"]].append(r)
feat = ["step1_obs_chars", "step1_findings_nonempty", "n_ext", "n_deep", "n_state", "n_calls", "steps",
        "n_cites", "note_rejects", "cit_stripped"]
rows_s = [r for r in rows if r["G"] is not None]
for r in rows_s:
    r["fallback"] = 1 if "fallback" in r["term"] else 0
feat.append("fallback")
corr = {}
for f in feat:
    line = []
    for y in ("G", "IR", "CT"):
        line.append(spear([r[f] for r in rows_s], [r[y] for r in rows_s]))
    # within-question: demean feature and y by question
    dm_f, dm_y = [], []
    for q, rs in qmean.items():
        if len(rs) < 2:
            continue
        mf = st.mean(r[f] for r in rs)
        my = st.mean(r["G"] for r in rs)
        for r in rs:
            dm_f.append(r[f] - mf)
            dm_y.append(r["G"] - my)
    w = spear(dm_f, dm_y)
    corr[f] = line + [w]
    print(f"  {f:24s} G={line[0]} IR={line[1]} CT={line[2]} | within-q G={w}")

# variance decomposition: between-question vs within-question
allG = [r["G"] for r in rows_s]
tot_var = st.pvariance(allG)
within = []
for q, rs in qmean.items():
    if len(rs) >= 2:
        within.extend([r["G"] - st.mean(x["G"] for x in rs) for r in rs])
print("\nG total var", round(tot_var, 4), "within-question var", round(st.pvariance(within), 4),
      "-> share explained by question identity", round(1 - st.pvariance(within) / tot_var, 3))

# ---- replicate SD ----
print("\n=== per-question replicate SD ===")
pairs = {"A(31a-33b)": BATCHES[:6], "B(34a-34e)": BATCHES[6:]}
sd_out = {}
for name, bs in pairs.items():
    for fac in ("G", "IR", "AP", "CT"):
        sds = []
        for q, rs in qmean.items():
            v = [r[fac] for r in rs if r["batch"] in bs and r[fac] is not None]
            if len(v) >= 2:
                sds.append(st.stdev(v))
        pooled = math.sqrt(st.mean([s * s for s in sds])) if sds else None
        sd_out[(name, fac)] = pooled
        print(f"  {name:12s}{fac:3s} pooled within-q SD={pooled:.3f} (n_q={len(sds)})")
res["sd"] = {f"{a}|{b}": v for (a, b), v in sd_out.items()}
json.dump(res, open(os.path.join(os.path.dirname(__file__), "r4_stats_out.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=str)
