# -*- coding: utf-8 -*-
"""CS2 dev20 three-arm judge driver.

Feeds each arm's answers through the official sqa scorer with GLM-5.3
(memorized_solver mode: offline answers -> .eval log). Per arm:
  ours      answers_pilot_cs2dev20.json -> report_adapter(notes->sections)
  harness   answers_harness_dev20.json  -> sections JSON already in result
  elicit    answers_elicit_dev20.json   -> sections (adapted)

Usage:
  python judge_dev20.py <arm>   # builds the task file, prints inspect cmd
"""
import json
import sys
from pathlib import Path

CS2 = Path(__file__).resolve().parent
ARM_OURS = CS2 / "arm_ours"
ARM_HARNESS = CS2 / "arm_harness"
ARM_MEM = CS2 / "arm_memorized"
RUBRICS = CS2.parent / "scholarqa_multi" / "sqa2_rubrics_v1_recomputed.json"


def ours_sections():
    """ours answers -> report_adapter -> CS2 sections (per question)."""
    sys.path.insert(0, str(CS2.parent / "_shared" / "tools"))
    sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
    from report_adapter import parse_notes, EvidenceStore, \
        narrative_compile, assemble
    import os
    os.environ.setdefault("LOCAL_SOCK_TIMEOUT", "900")
    BASE = CS2 / "base_kb"
    records = json.load(open(BASE / "records_merged.json",
                              encoding="utf-8"))
    # 批10 教训：深抽终化记录叠加（合并非替换——粗抽回指仍有效）
    _dp = BASE / "deep_read_records.json"
    if _dp.exists():
        _deep = json.load(open(_dp, encoding="utf-8"))
        for _pid, _payload in _deep.items():
            if not (isinstance(_payload, dict) and _payload.get("records")):
                continue
            _base = (records.get(_pid) or {}).get("records") or []
            _seen = {r.get("id") for r in _base if r.get("id")}
            records[_pid] = {"records": list(_base) + [
                r for r in _payload["records"] if r.get("id") not in _seen]}
    manifest = json.load(open(BASE / "manifest_all.json",
                              encoding="utf-8"))
    # FULLCHAIN-AUDIT C1：texts_dir + views_cs2 接线（与 adapt_batches 同）
    store = EvidenceStore(records, manifest,
                          texts_dir=str(BASE / "deep_read_texts"),
                          views_path=str(BASE / "views_cs2.json"))
    rows = json.load(open(ARM_OURS / "answers_pilot_cs2dev20.json",
                          encoding="utf-8"))
    out = []
    for row in rows:
        if not row.get("notes_final"):
            continue
        claims = parse_notes(row["notes_final"])
        try:
            draft = narrative_compile(row["question"], claims, store)
            report, diag = assemble(draft, claims, store)
            out.append({"qid": row["id"], "question": row["question"],
                        "sections": report["sections"],
                        "adapter_diag": diag})
        except Exception as e:
            print(f"[adapter] {row['id']}: {str(e)[:80]}", flush=True)
    return out


def harness_sections():
    rows = json.load(open(ARM_HARNESS / "answers_harness_dev20.json",
                          encoding="utf-8"))
    out = []
    for r in rows:
        if not r.get("ok"):
            continue
        res = r.get("result") or ""
        i, j = res.find("{"), res.rfind("}") + 1
        try:
            obj = json.loads(res[i:j])
            if obj.get("sections"):
                out.append({"qid": r["qid"], "question": r["question"],
                            "sections": obj["sections"]})
        except Exception:
            pass
    return out


def elicit_sections():
    rows = json.load(open(ARM_MEM / "answers_elicit_dev.json",
                          encoding="utf-8"))
    return [{"qid": r["qid"], "question": r["question"],
             "sections": r["sections"]} for r in rows]


ARMS = {"ours": ours_sections, "harness": harness_sections,
        "elicit": elicit_sections}


def main():
    arm = sys.argv[1]
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = ARMS[arm]()
    # dev20 只取前 20 题（rubrics 顺序）
    rubrics = json.load(open(RUBRICS, encoding="utf-8"))[:20]
    rmap = {q["case_id"][:24]: q for q in rubrics}
    rows20 = [r for r in rows if r["qid"] in rmap]
    outp = CS2 / f"judge_input_{arm}_dev20.json"
    json.dump(rows20, open(outp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[{arm}] {len(rows20)}/20 questions with sections -> {outp.name}")
    print(f"next: build task file per row & run inspect eval "
          f"(scorer_model=openai-api/glm/GLM-5.3)")


if __name__ == "__main__":
    main()
