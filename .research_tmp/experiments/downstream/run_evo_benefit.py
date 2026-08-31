# -*- coding: utf-8 -*-
"""Evolution-benefit paired experiment (2026-08-27, user-approved):

Does an EVOLVED schema extract better than the SEED schema on the same papers?
The load-bearing evidence for the offline-evolve / online-frozen architecture.

Arms (same 5 acceptance papers, same judge):
  EVOB_SEED — fresh agent (seed schema, zero evolution)
  EVOB_ML   — agent loaded with ML schema v1 (A2M_EVO1 29 patterns)
Both arm=full BUT the schema load happens before paper 1 and evolves further
in-run for EVOB_ML — to isolate SCHEMA effect we keep both arms' evolution
ON (same treatment) so the only difference is the starting schema.
Domain stays 'granular flow physics' (the acceptance papers are granular) —
this ALSO tests cross-domain transfer of the ML-evolved schema.

Usage: python .research_tmp/run_evo_benefit.py <arm: seed|ml>
"""
import json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

from granular_agent.agent import GranularFlowAgent
from granular_agent.llm_client import CALL_LOG, call_log_summary

PAPERS = ["PPR_24493BE6E8C2", "PPR_A249CB1DCBA7", "PPR_149984584E26",
          "PPR_6048B48856F7", "PPR_29EFA9BE34EC"]


def main(arm: str):
    agent = GranularFlowAgent(domain="granular flow physics",
                               llms=["DeepSeek-V4-Flash"])
    if arm == "ml":
        ok = agent.load_meta(".research_tmp/ml_schema_v1.json")
        if not ok:
            print("FATAL: ml schema load failed")
            sys.exit(1)
        agent._initial_seed_pats = set(agent.meta_hg.patterns.keys())
    elif arm != "seed":
        print(f"unknown arm {arm}")
        sys.exit(1)

    for pid in PAPERS:
        t0 = time.time()
        res = agent.process_paper_via_kernel(pid, arm="full")
        dt = time.time() - t0
        if res.get("error"):
            print(f"[{arm}] {pid}: ERROR {res['error']}")
            continue
        print(f"[{arm}] {pid}: {res['n_hyperedges']} edges, "
              f"{res['total_patterns_after']} patterns, {dt:.0f}s", flush=True)
        # rename bundle to tagged name
        import shutil
        src = os.path.join(".research_tmp", "runs", "kernel_v2", pid)
        dst = os.path.join(".research_tmp", "runs", "kernel_v2", f"EVOB_{arm.upper()}_{pid}")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        if os.path.isdir(src):
            shutil.move(src, dst)
    # persist the final schema of this arm for inspection
    agent.save_meta(f".research_tmp/evob_{arm}_schema_final.json")
    print(f"[{arm}] final schema saved")


if __name__ == "__main__":
    main(sys.argv[1])
