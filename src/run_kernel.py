"""Official kernel-pipeline runner (D5) — replaces all _tmp run scripts.

Usage (repo root):
  python src/run_kernel.py --papers PPR_24493BE6E8C2 [--papers PPR_xxx ...]
      [--domain ml] [--arm full] [--seed 1] [--tag mytag]

- Real MinerU blocks (no mock) via load_paper_blocks / MINERU_BASE
- Sets LLM_RUN_ID + per-run JSONL call log (provenance, D3)
- Bundle auto-saved by _save_kernel_bundle to .research_tmp/runs/kernel_v2/
  then renamed to {tag}_{paper} when --tag is given
- Prints a provenance summary (model x calls x tokens) per run (efficiency
  metrics: wall time / calls / tokens are first-class design goals)
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="Kernel pipeline runner")
    ap.add_argument("--papers", action="append", required=True,
                    help="paper ids (PPR_xxx or any MINERU_BASE dir name); repeatable")
    ap.add_argument("--domain", default="granular", help="domain tag (default granular)")
    ap.add_argument("--arm", default="full",
                    choices=["full", "add_only", "no_intra_dag", "frozen"],
                    help="ablation arm (default full)")
    ap.add_argument("--seed", type=int, default=None, help="LLM seed param (D4)")
    ap.add_argument("--tag", default="", help="bundle dir prefix (e.g. PROBE_B)")
    ap.add_argument("--model", default="DeepSeek-V4-Flash", help="extractor model")
    args = ap.parse_args()

    from granular_agent.agent import GranularFlowAgent
    from granular_agent.llm_client import CALL_LOG, call_log_summary

    run_id = f"{args.tag}_{args.arm}_{int(time.time())}" if args.tag else \
        f"{args.arm}_{int(time.time())}"
    os.environ["LLM_RUN_ID"] = run_id
    if args.seed is not None:
        os.environ["LLM_SEED"] = str(args.seed)
    # per-paper call log inside each bundle dir is set below (needs the dir)

    agent = GranularFlowAgent(domain=args.domain, llms=[args.model])

    for paper_id in args.papers:
        bundle_root = os.path.join(".research_tmp", "runs", "kernel_v2")
        os.makedirs(bundle_root, exist_ok=True)
        log_path = os.path.join(bundle_root, f"{run_id}_{paper_id}_calls.jsonl")
        os.environ["LLM_CALL_LOG"] = log_path

        t0 = time.time()
        res = agent.process_paper_via_kernel(paper_id, arm=args.arm)
        dt = time.time() - t0

        # tag-rename the auto-saved bundle dir
        if args.tag:
            src = os.path.join(bundle_root, paper_id)
            dst = os.path.join(bundle_root, f"{args.tag}_{paper_id}")
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            if os.path.isdir(src):
                shutil.move(src, dst)

        if res.get("error"):
            print(f"[run_kernel] {paper_id}: ERROR {res['error']}")
            continue
        summary = call_log_summary()
        n_calls = summary["total_calls"]
        tok = sum(v["prompt_tokens"] + v["completion_tokens"]
                  for v in summary["by_model"].values())
        print(f"[run_kernel] {paper_id} arm={args.arm}: "
              f"{res['n_sections']} sections, {res['n_concepts']} concepts, "
              f"{res['n_hyperedges']} hyperedges, "
              f"v{res['version_before']}->{res['version_after']} "
              f"({len(agent._get_kernel().tbox.patterns)} patterns) | "
              f"{dt:.0f}s, {n_calls} LLM calls, {tok} tokens -> "
              f"{run_id}_{paper_id}_calls.jsonl")


if __name__ == "__main__":
    main()
