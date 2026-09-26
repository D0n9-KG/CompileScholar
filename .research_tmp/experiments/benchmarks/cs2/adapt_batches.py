# -*- coding: utf-8 -*-
"""批 5-9 判分适配：answers_pilot_cs2batch*.json -> report_adapter ->
judge_input_ours_batch*.json（复用 judge_dev20.ours_sections 的管线，
参数化批号）。用法：python adapt_batches.py 5 6 7 8 9
"""
import json
import os
import sys
from pathlib import Path

CS2 = Path(__file__).resolve().parent
ARM_OURS = CS2 / "arm_ours"
BASE = CS2 / "base_kb"

sys.path.insert(0, str(CS2.parent / "_shared" / "tools"))
sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
os.environ.setdefault("LOCAL_SOCK_TIMEOUT", "900")


def _load_records_with_deep():
    """records_merged + deep_read_records 叠加（批 10 教训：深抽终化
    记录若不进 EvidenceStore，笔记里的深记录回指解析不了→空壳报告）。
    合并非替换：粗抽记录的回指仍然有效（deep payload 不含 coarse id）。"""
    records = json.load(open(BASE / "records_merged.json", encoding="utf-8"))
    dp = BASE / "deep_read_records.json"
    if dp.exists():
        deep = json.load(open(dp, encoding="utf-8"))
        for pid, payload in deep.items():
            if not (isinstance(payload, dict) and payload.get("records")):
                continue
            base_recs = (records.get(pid) or {}).get("records") or []
            seen = {r.get("id") for r in base_recs if r.get("id")}
            merged_recs = list(base_recs) + [
                r for r in payload["records"]
                if r.get("id") not in seen]
            records[pid] = {"records": merged_recs}
    return records


def adapt_batch(b: int) -> str:
    from report_adapter import (EvidenceStore, assemble, narrative_compile,
                                parse_notes)
    records = _load_records_with_deep()
    manifest = json.load(open(BASE / "manifest_all.json", encoding="utf-8"))
    store = EvidenceStore(records, manifest)
    rows = json.load(open(ARM_OURS / f"answers_pilot_cs2batch{b}.json",
                          encoding="utf-8"))
    out = []
    for row in rows:
        if not row.get("notes_final"):
            print(f"[b{b}] {row['id'][:14]}: no notes (弃答) — skipped",
                  flush=True)
            continue
        claims = parse_notes(row["notes_final"])
        try:
            draft = narrative_compile(row["question"], claims, store)
            report, diag = assemble(draft, claims, store)
            out.append({"qid": row["id"], "question": row["question"],
                        "sections": report["sections"], "diag": diag})
        except Exception as e:
            print(f"[b{b}] {row['id'][:14]}: adapter FAIL {str(e)[:80]}",
                  flush=True)
    outp = CS2 / f"judge_input_ours_batch{b}.json"
    json.dump(out, open(outp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[b{b}] {len(out)}/{len(rows)} adapted -> {outp.name}", flush=True)
    return str(outp)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for b in [int(x) for x in sys.argv[1:]] or [5, 6, 7, 8, 9]:
        adapt_batch(b)
