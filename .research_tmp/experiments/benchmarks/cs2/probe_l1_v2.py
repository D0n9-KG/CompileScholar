# -*- coding: utf-8 -*-
"""Probe v2: cache-hit path + upgraded matcher + L2 daemon exit behavior."""
import json
import os
import sys
import time

SRC = r"C:\Users\D0n9\Desktop\CompileScholar\src"
SHARED = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\_shared\tools"
CS2 = os.path.dirname(os.path.abspath(__file__))
BASE_KB = os.path.join(CS2, "base_kb")
sys.path.insert(0, SRC)
sys.path.insert(0, SHARED)

os.environ.setdefault("LOCAL_MAX_CONCURRENT", "10")
os.environ.setdefault("LOCAL_LARGE_MAX_CONCURRENT", "6")
os.environ.setdefault("LLM_CALL_LOG", os.path.join(CS2, "arm_ours", "ledger_probe_l1v2.jsonl"))

from external_tools import ExternalTools, attach_external_tools  # noqa: E402

views = json.load(open(os.path.join(BASE_KB, "views_cs2.json"), encoding="utf-8"))
manifest = {r["paper_id"]: r for r in json.load(
    open(os.path.join(BASE_KB, "manifest_all.json"), encoding="utf-8"))}

class _KB:
    def __init__(self):
        self.records = {}

kb = _KB()
attach_external_tools(kb, views, manifest,
                      deep_cache_path=os.path.join(BASE_KB, "deep_read_cache.jsonl"),
                      deep_text_dir=os.path.join(BASE_KB, "deep_read_texts"))
ext = kb._ext_tools

PID = "sciverse_fpga_based_autonomous_gps_disciplined_oscillatorsfor_wireless_sensor_network_nod"

# 1) re-deep_read the CACHED paper with the upgraded keyword matcher:
#    new sections -> fresh L1 extraction from cache-miss chunks only
t0 = time.time()
obs = kb.deep_read(PID, sections=["pps", "oscillator stability"])
print(f"\n=== cached paper, keyword sections ({time.time()-t0:.0f}s) ===")
print(json.dumps({k: v for k, v in obs.items() if k != "section_miss"},
                 ensure_ascii=False, indent=1)[:1200])
if obs.get("section_miss"):
    print("section_miss:", obs["section_miss"][:300])

# 2) L2 must be running in background; process must NOT hang at exit.
#    Give L2 ~60s head start, report state, then exit (daemon threads die).
time.sleep(60)
n_lines = sum(1 for _ in open(os.path.join(BASE_KB, "deep_read_cache.jsonl"),
                              encoding="utf-8"))
print(f"\ncache lines after 60s: {n_lines}")
print("L2 running:", sorted(p[:44] for p in ExternalTools._L2_RUNNING))
print("records in kb:", len(kb.records.get(PID, {}).get("records", [])))
print("exiting now — if this line prints and the process ends, daemon exit OK")
