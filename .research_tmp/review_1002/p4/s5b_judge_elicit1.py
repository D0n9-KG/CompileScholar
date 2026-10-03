# -*- coding: utf-8 -*-
"""Judge the single-citation Elicit resample with the same s5 prompt."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from common import P4, calls
from s5_judge import judge
items = json.load(open(os.path.join(P4, "s3b_elicit1.json"), encoding="utf-8"))
with ThreadPoolExecutor(4) as ex:
    out = list(ex.map(judge, items))
json.dump(out, open(os.path.join(P4, "s5b_judged_elicit1.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
from collections import Counter
print(Counter(o["judge"].get("label") for o in out), calls())
