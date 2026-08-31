# -*- coding: utf-8 -*-
"""Multi-seed judge orchestrator: run judge_cross.py (API dual-judge) on all
MSEED bundles, 8-way parallel across bundles, then aggregate per prereg
(PREREG_multiseed_soft_routing.md).

Usage: python .research_tmp/multiseed_judge_all.py          # judge missing
       python .research_tmp/multiseed_judge_all.py --agg    # aggregate only
"""
import argparse, glob, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.join(".research_tmp", "runs", "kernel_v2")
PAPERS = ["PPR_24493BE6E8C2", "PPR_A249CB1DCBA7", "PPR_149984584E26",
          "PPR_6048B48856F7", "PPR_29EFA9BE34EC"]


def bundles():
    out = []
    for tag in [f"MSEED_S{s}R{r}" for s in (1, 2, 3) for r in (0, 1)]:
        for p in PAPERS:
            d = os.path.join(BASE, f"{tag}_{p}")
            if os.path.isdir(d):
                out.append((tag, p, d))
    return out


def run_judge(tag, p, d):
    jc = os.path.join(d, "judge_cross.json")
    if os.path.exists(jc) and os.path.getsize(jc) > 100:
        return (tag, p, "cached")
    try:
        r = subprocess.run([sys.executable, ".research_tmp/judge_cross.py",
                            f"{tag}_{p}"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=2400)
        ok = os.path.exists(jc)
        return (tag, p, "ok" if ok else f"FAIL: {r.stdout[-200:]} {r.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        return (tag, p, "TIMEOUT")


def aggregate():
    def _parse(fr):
        n, m = fr.split("/")
        return int(n), int(m)

    rows = []
    for tag, p, d in bundles():
        jc = os.path.join(d, "judge_cross.json")
        if not os.path.exists(jc):
            continue
        j = json.load(open(jc, encoding="utf-8"))
        s = j.get("summary", {})
        nb, ne = _parse(s.get("both_pass", "0/0"))
        ne_, _ = _parse(s.get("either_pass", "0/0"))
        rows.append({"tag": tag, "paper": p, "edges": ne, "both": nb,
                     "either": ne_})
    if not rows:
        print("no judge_cross results yet")
        return
    agg = {}
    for r in rows:
        agg.setdefault(r["tag"], []).append(r)
    print(f"{'tag':<14}{'papers':<8}{'edges':<8}{'both':<8}{'either':<8}")
    summary = {}
    for tag in sorted(agg):
        rs = agg[tag]
        n_edges = sum(x["edges"] for x in rs)
        n_both = sum(x["both"] for x in rs)
        n_eith = sum(x["either"] for x in rs)
        both = n_both / max(1, n_edges)
        eith = n_eith / max(1, n_edges)
        summary[tag] = {"edges": n_edges, "both": n_both, "either": n_eith,
                        "both_rate": round(both, 4), "either_rate": round(eith, 4)}
        print(f"{tag:<14}{len(rs):<8}{n_edges:<8}{n_both:<8}{n_eith:<8} "
              f"both={both:.1%} either={eith:.1%}")
    with open(os.path.join(".research_tmp", "multiseed_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    # prereg pairing: per seed R1 vs R0
    print("\n=== prereg pairing (both-pass rate) ===")
    for s in (1, 2, 3):
        r1 = summary.get(f"MSEED_S{s}R1", {}).get("both_rate")
        r0 = summary.get(f"MSEED_S{s}R0", {}).get("both_rate")
        if r1 is not None and r0 is not None:
            print(f"seed {s}: R1={r1:.1%} vs R0={r0:.1%} -> "
                  f"{'R1>=R0' if r1 >= r0 else 'R1<R0'} ({(r1-r0)*100:+.1f}pt)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", action="store_true")
    a = ap.parse_args()
    if a.agg:
        aggregate()
    else:
        bs = bundles()
        print(f"[judge] {len(bs)} bundles")
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(run_judge, t, p, d) for t, p, d in bs]
            for f in as_completed(futs):
                try:
                    t, p, msg = f.result()
                except Exception as e:
                    t, p, msg = "?", "?", f"CRASH: {e!r}"
                print(f"[judge] {t}_{p}: {msg}", flush=True)
        aggregate()
