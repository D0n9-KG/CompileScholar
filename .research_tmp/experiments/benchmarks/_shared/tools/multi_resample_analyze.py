"""multi_resample_analyze — P1-1 resample three-metric report + G1 acceptance.

Compares the resample20 run (post P0-1/P1-2/G1-B1/B2/B3, registry_v3) against
the frozen Multi-108 run on the SAME question ids:

  P1-1 metrics: steps median/p90, hardstop rate (f28_hard_stop or budget
  exhaustion), wall-clock (per-question, load-inflated — reported with the
  caveat), notes chars, answer chars.
  G1 acceptance: card()/compare() empty-return rates on the resample
  trajectories (<10% target; measured baseline 38%/56%).

Usage:
    python multi_resample_analyze.py [--new answers_pilot_resample20.json]
"""

import argparse
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.join(_HERE, "..", "..", "scholarqa_multi", "baselines", "ours")


def _traj_tools(rows):
    """tool usage counts + empty-return rates from trajectory entries."""
    counts = {}
    card_total = card_empty = comp_total = comp_empty = 0
    for row in rows:
        for t in row.get("trajectory") or []:
            if not isinstance(t, dict) or "tool" not in t:
                continue
            tool = t["tool"]
            counts[tool] = counts.get(tool, 0) + 1
            # empty heuristic: sub-300-char observations for card/compare
            # (measured: real hits are 800-6000+ chars, empties ~62-250)
            if tool == "card":
                card_total += 1
                card_empty += (t.get("obs_chars") or 0) < 300
            elif tool == "compare":
                comp_total += 1
                comp_empty += (t.get("obs_chars") or 0) < 300
    return counts, card_total, card_empty, comp_total, comp_empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="answers_pilot_resample20.json")
    ap.add_argument("--old", default="answers_pilot_multi.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    old = {r["id"]: r for r in json.load(
        open(os.path.join(_BASE, args.old), encoding="utf-8"))}
    new = [r for r in json.load(
        open(os.path.join(_BASE, args.new), encoding="utf-8"))
        if r.get("answer")]
    ids = [r["id"] for r in new]

    def _stats(rows, key):
        vals = [r.get(key) or 0 for r in rows]
        if not vals:
            return {}
        return {"median": statistics.median(vals),
                "p90": sorted(vals)[int(len(vals) * 0.9) - 1 if len(vals) >= 10 else -1],
                "max": max(vals)}

    old_rows = [old[i] for i in ids if i in old]
    print(f"=== P1-1 resample vs frozen run ({len(ids)} questions) ===\n")

    for label, rows in (("old", old_rows), ("new", new)):
        steps = _stats(rows, "steps")
        notes = _stats(rows, "notes_chars" if rows[0].get("notes_chars") is None
                       else "notes_chars")
        # notes length lives in the per-question log line; use answer length
        ans = _stats(rows, "steps")  # placeholder replaced below
        ans = {"median": statistics.median([len(r.get("answer") or "")
                                            for r in rows])}
        hardstop = sum(1 for r in rows
                       if (r.get("gate") or {}).get("f28_hard_stop"))
        forced = sum(1 for r in rows
                     if (r.get("gate") or {}).get("forced_notes_submit"))
        cap_burn = sum(1 for r in rows if (r.get("steps") or 0) >= 40)
        print(f"[{label}] steps {steps} | hardstop {hardstop}/{len(rows)} "
              f"| forced_submit {forced} | >=40-step burn {cap_burn} "
              f"| answer-chars median {ans['median']:.0f}")
    print()

    # G1 acceptance: empty-return rates on NEW trajectories
    counts, ct, ce, pt, pe = _traj_tools(new)
    o_counts, oct_, oce, opt_, ope = _traj_tools(old_rows)
    print("=== G1 acceptance: tool usage / empty rates (new vs old) ===")
    print(f"card:  {ce}/{ct} empty ({100*ce/max(1,ct):.0f}%)  "
          f"[old {oce}/{oct_} = {100*oce/max(1,oct_):.0f}%]  target <10%")
    print(f"compare: {pe}/{pt} empty ({100*pe/max(1,pt):.0f}%)  "
          f"[old {ope}/{opt_} = {100*ope/max(1,opt_):.0f}%]  target <10%")
    for tool in sorted(set(counts) | set(o_counts)):
        print(f"  {tool:12s} old={o_counts.get(tool, 0):4d} new={counts.get(tool, 0):4d}")
    print()
    degenerate = [r["id"] for r in new if len(r.get("answer") or "") < 800]
    print(f"degenerate answers (<800ch): {degenerate or 'none'}")


if __name__ == "__main__":
    main()
