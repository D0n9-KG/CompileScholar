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
    # FULLCHAIN-AUDIT C1：texts_dir 接线（批11 chunk 锚修复是死代码——
    # 三个调用方都没传）+ views_cs2 内嵌记录加载（33 个真实 id 断链源）
    store = EvidenceStore(records, manifest,
                          texts_dir=str(BASE / "deep_read_texts"),
                          views_path=str(BASE / "views_cs2.json"))
    rows = json.load(open(ARM_OURS / f"answers_pilot_cs2batch{b}.json",
                          encoding="utf-8"))
    out = []
    # FULLCHAIN-AUDIT C3：整题归零防护——历史上 8 题因笔记无 N 形态行
    # 在适配器异常（no usable claims）后从 judge_input 消失→四指标全零
    # 且 degenerate=False 无人知。现在：(a) 异常时降级产出（title-only
    # 引用的最小报告，判分可咬合而非缺席）；(b) 结构化 ledger 落盘
    # （adapt_failures.jsonl，不再只是 console print）。
    failures = []
    for row in rows:
        if not row.get("notes_final"):
            print(f"[b{b}] {row['id'][:14]}: no notes (弃答) — degraded",
                  flush=True)
            failures.append({"qid": row["id"], "reason": "no_notes",
                             "notes_chars": 0})
            continue
        claims = parse_notes(row["notes_final"])
        try:
            draft = narrative_compile(row["question"], claims, store)
            report, diag = assemble(draft, claims, store)
            out.append({"qid": row["id"], "question": row["question"],
                        "sections": report["sections"], "diag": diag})
        except Exception as e:
            reason = f"adapter FAIL: {type(e).__name__}: {str(e)[:120]}"
            print(f"[b{b}] {row['id'][:14]}: {reason} — degraded", flush=True)
            failures.append({"qid": row["id"], "reason": reason,
                             "notes_chars": len(row["notes_final"])})
            # 降级产出：从笔记里救出全部能解析的 paper_id 回指，组成
            # 最小 sections（判分不缺席；citation 走 title 档）
            import re as _re
            pids = list(dict.fromkeys(_re.findall(
                r"\[([0-9A-Za-z_]{20,120})\]", row["notes_final"])))
            if pids:
                cites = []
                for i, pid in enumerate(pids[:12], 1):
                    meta = store.paper_meta(pid) if pid in store.by_paper else {}
                    cites.append({"id": f"[{i}]", "snippets": [],
                                  "title": meta.get("title") or pid,
                                  "metadata": meta})
                text = (" ".join(f"[{i}]" for i in range(1, len(cites) + 1)))
                out.append({"qid": row["id"], "question": row["question"],
                            "sections": [{"title": "Findings",
                                          "text": text,
                                          "citations": cites}],
                            "diag": {"degraded": True, "reason": reason}})
    if failures:
        fp = CS2 / f"adapt_failures_batch{b}.jsonl"
        with open(fp, "a", encoding="utf-8") as f:
            for x in failures:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"[b{b}] {len(failures)} degraded rows -> {fp.name}", flush=True)
    outp = CS2 / f"judge_input_ours_batch{b}.json"
    json.dump(out, open(outp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[b{b}] {len(out)}/{len(rows)} adapted -> {outp.name}", flush=True)
    return str(outp)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for b in [int(x) for x in sys.argv[1:]] or [5, 6, 7, 8, 9]:
        adapt_batch(b)
