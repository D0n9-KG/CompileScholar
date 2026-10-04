# -*- coding: utf-8 -*-
"""Batch-1 vs old-run comparison: quantify the fixes' effect.

Old run (answers_pilot_cs2dev20.json, first 15 questions): ran WITHOUT
  - text_index (search_text -> 192-char error loop, 73% duplicate calls)
  - grounding trim (148k tokens/step injection)
  - APC prefix ordering
Batch 1 (arm_ours_batch/answers_batch1.json, 5 questions): all fixes on.

Compares on the 4 overlapping questions + overall stats.
"""
import json
import sys
from pathlib import Path

CS2 = Path(__file__).resolve().parent


def load_rows(p):
    return {r["id"]: r for r in json.load(open(p, encoding="utf-8"))
            if (r.get("steps") or 0) > 0}


def traj_stats(row):
    traj = row.get("trajectory") or []
    from collections import Counter
    tools = Counter(t.get("tool") for t in traj)
    seen, dup, total = set(), 0, 0
    for t in traj:
        key = (t.get("tool"),
               json.dumps(t.get("args"), sort_keys=True)[:80])
        total += 1
        if key in seen:
            dup += 1
        seen.add(key)
    search_text_share = tools.get("search_text", 0) / max(1, total)
    return {"steps": row.get("steps"),
            "calls": total,
            "dup_rate": round(dup / max(1, total), 2),
            "search_text_share": round(search_text_share, 2),
            "answer_chars": len(row.get("answer") or ""),
            "notes_chars": len(row.get("notes_final") or "")}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    old = load_rows(CS2 / "arm_ours" / "answers_pilot_cs2dev20.json")
    new = load_rows(CS2 / "arm_ours_batch" / "answers_batch1.json")

    print(f"old rows: {len(old)} | batch1 rows: {len(new)}")
    print()
    overlap = [qid for qid in new if qid in old]
    print(f"=== 同题对比（{len(overlap)} 题重叠） ===")
    print(f"{'qid':<26} {'旧步数':>6} {'新步数':>6} {'旧重复率':>8} "
          f"{'新重复率':>8} {'旧ST占比':>8} {'新ST占比':>8}")
    for qid in overlap:
        o, n = traj_stats(old[qid]), traj_stats(new[qid])
        print(f"{qid[:24]:<26} {o['steps']:>6} {n['steps']:>6} "
              f"{o['dup_rate']:>8} {n['dup_rate']:>8} "
              f"{o['search_text_share']:>8} {n['search_text_share']:>8}")
    print()
    print("=== 答案质量抽查（重叠题新旧各看长度） ===")
    for qid in overlap[:3]:
        print(f"{qid[:20]}: old {len(old[qid].get('answer') or '')}ch "
              f"vs new {len(new[qid].get('answer') or '')}ch")


if __name__ == "__main__":
    main()
