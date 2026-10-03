# -*- coding: utf-8 -*-
import json, os, re, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
V2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb_v2"
R = json.load(open(os.path.join(V2, "records.json"), encoding="utf-8"))
ab = [r for v in R.values() for r in v["records"] if r.get("kind") == "absence"]
C = collections.Counter()
ex = {}
for r in ab:
    m = r.get("missing") or ""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", m)); lat1 = len(re.findall(r"[\u00c0-\u00ff]", m))
    asc = len(re.findall(r"[A-Za-z]", m)); rep = m.count("\ufffd")
    # classic utf8-as-gbk/latin1 mojibake markers
    mj = len(re.findall(r"(?:Ã.|Â.|â€|锟斤拷|鍙|鐨|浠|璇|绋)", m))
    t = ("replacement_char" if rep else "latin1_mojibake" if lat1 > 2 else "gbk_mojibake_markers" if mj > 2
         else "chinese" if cjk > 3 else "english" if asc > 10 else "other")
    C[t] += 1; ex.setdefault(t, m[:80])
print(C.most_common()); [print(" ", k, repr(v)) for k, v in ex.items()]
# can we round-trip: is any string decodable as gbk->utf8 mojibake? try encode gbk then decode utf8
fix = 0
for r in ab:
    m = r.get("missing") or ""
    try:
        f = m.encode("gbk").decode("utf-8")
        if f != m: fix += 1
    except Exception:
        pass
print("strings that are valid 'utf8 bytes misread as gbk' (round-trip changes):", fix)
q_en = sum(1 for r in ab if len(re.findall(r"[A-Za-z]", r.get("quote") or "")) > 20)
print("absence with English quote", q_en, "/", len(ab))
s_en = sum(1 for r in ab if len(re.findall(r"[A-Za-z]", r.get("subject") or "")) > 3)
print("absence with English subject", s_en)
print("sample:", [(r.get("subject"), r.get("missing")[:40], (r.get("quote") or "")[:80]) for r in ab[:3]])
