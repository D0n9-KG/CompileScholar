# -*- coding: utf-8 -*-
"""P0-5（FIX-PLAN v2）：中文缺口清洗（275 条）。

形态（实测）：subject 全英文（实体名不受影响），missing 为中文综述的
中文转述——而 KB 的检索/匹配通道（gap_search token/embedding、
report 消费）全英文。quote 里通常有英文原文（缺口出处），但转述是
抽取器自己写的，直接弃中文会丢信息——翻译成英文入库，中文原文保留
在 missing_zh（双语，可审计）。

用法：PYTHONUTF8=1 python translate_absences.py [n_par]
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

VIEWS = os.path.join(BASE_KB, "views_cs2.json")
MODEL = "DeepSeek-V4.1-Flash"


def _cn(s):
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(s or ""))


def _atomic_write(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


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
    views = json.load(open(VIEWS, encoding="utf-8"))
    absences = (views.get("coverage") or {}).get("absences_extracted") or []
    todo = [a for a in absences
            if _cn(a.get("missing")) and not a.get("missing_en")]
    print(f"[translate] {len(todo)} chinese absences to translate")
    if not todo:
        return
    jobs = {i: a for i, a in enumerate(todo)}

    def run(i):
        t, note = _translate(jobs[i])
        return i, t, note

    done = fail = 0
    with ThreadPoolExecutor(max_workers=n_par) as ex:
        for i, t, note in ex.map(run, sorted(jobs)):
            a = jobs[i]
            if t:
                a["missing_zh"] = a["missing"]   # 原文保留（可审计）
                a["missing"] = t                  # 英文进主通道
                a["missing_translated"] = True
                done += 1
            else:
                fail += 1
                a["_translate_failed"] = note
            if (done + fail) % 40 == 0:
                print(f"  ...{done + fail}/{len(todo)}", flush=True)
    _atomic_write(views, VIEWS)
    print(f"[translate] done={done} failed={fail}")
    # 残余中文（含 subject 为中文的极端情况）
    left = sum(1 for a in absences if _cn(a.get("missing")))
    print(f"[translate] chinese remaining in missing: {left}")


if __name__ == "__main__":
    main()
