import sys, json, copy
sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\review_1002\r3")
from kbload import load
from kb_compiler.views.tools import KBTools
t, records, views, man = load()
for q in ["transformer", "BERT", "dropout"]:
    r = t.card(q); s = json.dumps(r, ensure_ascii=False)
    print("AS-IS", q, len(s), s[:160])
v2 = dict(views); v2["cards"] = {"cards": views["cards"]}
t2 = KBTools(v2, t.registry if hasattr(t,"registry") else json.load(open(r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2\base_kb\registry_v2.json",encoding="utf-8")), {}, man, records)
for q in ["transformer", "BERT", "dropout"]:
    r = t2.card(q); s = json.dumps(r, ensure_ascii=False)
    print("FIXED", q, len(s), s[:160])
