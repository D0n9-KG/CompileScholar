# -*- coding: utf-8 -*-
"""P0-5 修复版:中文缺口翻译落到 records 层(而非只写 views)。

病根(2026-10-01 发现):translate_absences.py 只把翻译写进
views_cs2.json——但 views 是 records 重编译产物,后续任何重编译
(实体归一/引用桥都触发)把翻译全部冲掉。实测:views 里 275 条中文
missing 复活、missing_en=0(旧字段名也没人写了)。

修复:翻译写回 records_merged.json 的 absence 记录本体
(missing=英文,missing_zh=中文原文,missing_translated=True),
重编译链天然携带。幂等:已有 missing_zh 的记录跳过。

用法:PYTHONUTF8=1 python translate_absences_v2.py [n_par]
之后:重编译 views(compiler→patch_cards→backfill_years)。
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

CS2 = os.path.dirname(os.path.abspath(__file__))
BASE_KB = os.path.join(CS2, "base_kb")
SRC = r"C:\Users\D0n9\Desktop\CompileScholar\src"
sys.path.insert(0, SRC)

from kb_infra.llm import call_paratera  # noqa: E402

MODEL = "DeepSeek-V4.1-Flash"


def _cn(s):
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(s or ""))


def _translate(a):
    prompt = f"""Translate this research-gap statement from Chinese to concise academic English. This is a "what is missing / what has not been solved" statement about the subject "{a.get('subject')}" from a survey paper. Keep technical terms as-is.

Chinese: {a.get('missing')}

Output ONLY the English translation, one line, no quotes."""
    try:
        out = call_paratera(prompt, model=MODEL, max_tokens=250,
                            temperature=0.0, enable_thinking=False)
    except Exception as e:
        return None, str(e)[:60]
    if not out:
        return None, "no response"
    t = " ".join(out.strip().splitlines()).strip().strip('"')
    return (t if 10 < len(t) < 600 else None), (t[:60] if t else "empty")


def main():
    n_par = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    src = os.path.join(BASE_KB, "records_merged.json")
    merged = json.load(open(src, encoding="utf-8"))
    # 收集中文 absence 记录
    jobs = []
    for pid, payload in merged.items():
        if not isinstance(payload, dict):
            continue
        for r in payload.get("records") or []:
            if r.get("kind") == "absence" and _cn(r.get("missing")) \
                    and not r.get("missing_zh"):
                jobs.append(r)
    print(f"[translate] {len(jobs)} chinese absence records in records_merged")
    if not jobs:
        return

    done = fail = 0
    with ThreadPoolExecutor(max_workers=n_par) as ex:
        for r, (t, note) in zip(jobs, ex.map(_translate, jobs)):
            if t:
                r["missing_zh"] = r["missing"]
                r["missing"] = t
                r["missing_translated"] = True
                done += 1
            else:
                r["_translate_failed"] = note
                fail += 1
            if (done + fail) % 40 == 0:
                print(f"  ...{done + fail}/{len(jobs)}", flush=True)
    tmp = src + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    os.replace(tmp, src)
    print(f"[translate] done={done} failed={fail} -> records_merged.json")
    print("[translate] 下一步: 重编译 views(compile→patch_cards→backfill_years)")


if __name__ == "__main__":
    main()
