# -*- coding: utf-8 -*-
"""Live probe: deep_read L1 directed mode end-to-end on a batch-9 paper.

Tests: (a) sections param path, (b) cache round-trip (second call hits
cache), (c) records land in records_target, (d) L2 background starts.
"""
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
os.environ.setdefault("LLM_CALL_LOG", os.path.join(CS2, "arm_ours", "ledger_probe_l1.jsonl"))

from external_tools import ExternalTools, attach_external_tools  # noqa: E402

views = json.load(open(os.path.join(BASE_KB, "views_cs2.json"), encoding="utf-8"))
manifest = {r["paper_id"]: r for r in json.load(
    open(os.path.join(BASE_KB, "manifest_all.json"), encoding="utf-8"))}

# a fake KBTools-like holder with a records dict (as attach does)
class _KB:
    def __init__(self):
        self.records = {}

kb = _KB()
attach_external_tools(kb, views, manifest,
                      deep_cache_path=os.path.join(BASE_KB, "deep_read_cache.jsonl"),
                      deep_text_dir=os.path.join(BASE_KB, "deep_read_texts"))
ext = kb._ext_tools
ext._current_question = ("how does the FPGA GPS-disciplined oscillator "
                         "generate PPS timing signals and what oscillator "
                         "stability is achieved")

PID = "sciverse_fpga_based_autonomous_gps_disciplined_oscillatorsfor_wireless_sensor_network_nod"
t0 = time.time()
obs = kb.deep_read(PID, sections=[" oscillator stability ", "pps"])
print(f"\n=== call 1 (sections) {time.time()-t0:.0f}s ===")
print(json.dumps(obs, ensure_ascii=False, indent=1)[:1800])
print("\nrecords landed:", PID in kb.records,
      "n=", len(kb.records.get(PID, {}).get("records", [])))

t0 = time.time()
obs2 = kb.deep_read("sciverse_research_on_absolute_calibration_of_gnss_receiver_delay_through_clock_steering_c")
print(f"\n=== call 2 (default question-top3) {time.time()-t0:.0f}s ===")
print(json.dumps(obs2, ensure_ascii=False, indent=1)[:1800])

# wait a bit for L2 background to make progress, then check cache file
time.sleep(20)
n_lines = sum(1 for _ in open(os.path.join(BASE_KB, "deep_read_cache.jsonl"), encoding="utf-8"))
print(f"\ncache file lines after 20s: {n_lines}")
print("L2 running:", sorted(p[:40] for p in ExternalTools._L2_RUNNING))
