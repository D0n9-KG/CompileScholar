# -*- coding: utf-8 -*-
"""Probe: reproduce batch-9 deep_read obs=248 on the GNSS papers."""
import json
import os
import sys

SRC = r"C:\Users\D0n9\Desktop\CompileScholar\src"
SHARED = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\_shared\tools"
CS2 = os.path.dirname(os.path.abspath(__file__))
BASE_KB = os.path.join(CS2, "base_kb")
sys.path.insert(0, SRC)
sys.path.insert(0, SHARED)

os.environ.setdefault("LLM_CALL_LOG", os.path.join(CS2, "arm_ours", "ledger_probe_l1.jsonl"))

from external_tools import ExternalTools  # noqa: E402

views = json.load(open(os.path.join(BASE_KB, "views_cs2.json"), encoding="utf-8"))
manifest = {r["paper_id"]: r for r in json.load(
    open(os.path.join(BASE_KB, "manifest_all.json"), encoding="utf-8"))}

ext = ExternalTools(views, manifest, "local:Qwen3.8-27B")
ext._manifest_path = os.path.join(BASE_KB, "manifest_all.json")

pids = [
    "sciverse_research_on_absolute_calibration_of_gnss_receiver_delay_through_clock_steering_c",
    "sciverse_fpga_based_autonomous_gps_disciplined_oscillatorsfor_wireless_sensor_network_nod",
]
for pid in pids:
    m = manifest.get(pid) or {}
    print(f"== {pid[:60]}")
    print("   manifest keys:", sorted(k for k in m.keys())[:10])
    print("   doc_id:", m.get("doc_id"), "| arxiv_id:", m.get("arxiv_id"))
    print("   title:", (m.get("title") or "")[:80])

# do NOT run the full deep_read (slow); first check the full-text channels
print("\n--- channel probe ---")
from sci_evo_extract.library.sources import SciverseClient  # noqa: E402
sc = SciverseClient()
for pid in pids:
    m = manifest.get(pid) or {}
    doc_id = m.get("doc_id")
    if not doc_id:
        print(f"{pid[:50]}: NO doc_id")
        continue
    try:
        resp = sc.read_content(doc_id=doc_id, offset=0, limit=10000)
        t = str(resp.get("text") or "")
        print(f"{pid[:50]}: first page {len(t)} chars, more={resp.get('more')}")
    except Exception as e:
        print(f"{pid[:50]}: SciverseClient ERROR {type(e).__name__}: {str(e)[:150]}")
