# -*- coding: utf-8 -*-
"""Stage runner with Intern-quota-aware channel switching (user directive
09-21: Intern ~100M tokens total, may die at ANY time; switching back to
local must be seamless).

Policy:
- Channel order: intern:qwen3.8-27b first, local:Qwen3.8-27B fallback.
- Pre-stage budget check: ledger-summed Intern tokens >= INTERN_SOFT_CAP
  (default 85M, reserving headroom under the ~100M quota) -> start on local.
- Mid-stage death: ChannelDeadError (loud abort, guard commit 8d392f32) ->
  rerun the SAME stage on the next channel. Stages must be resume-safe or
  cheap-deterministic (registry: embeddings cached, block calls redone;
  slot: per-paper resume built in).
- Ledger accounting printed before/after each attempt (provider-tagged).

Usage:
  python run_stage.py --name registry -- python -m kb_compiler.records.registry \
      --cards ... --out-dir ... --merge-mode blocked --manifest ...
(model placeholder {MODEL} in the command is substituted per attempt)
"""
import argparse
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(BASE, "kb")

CHANNELS = ["intern:qwen3.8-27b", "local:Qwen3.8-27B"]
INTERN_SOFT_CAP = int(os.environ.get("INTERN_SOFT_CAP", str(85_000_000)))


def ledger_tokens(log_path):
    per = {}
    if not os.path.exists(log_path):
        return per
    for line in open(log_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("ok"):
            k = r.get("provider") or "?"
            per[k] = per.get(k, 0) + (r.get("prompt_tokens") or 0) \
                + (r.get("completion_tokens") or 0)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ledger", default=os.path.join(KB, "ledger_build.jsonl"))
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd or "{MODEL}" not in " ".join(cmd):
        print("ERROR: command must contain the {MODEL} placeholder")
        return 2

    tok = ledger_tokens(args.ledger)
    intern_used = tok.get("intern", 0)
    print(f"[{args.name}] ledger so far: "
          + ", ".join(f"{k}={v/1e6:.2f}M" for k, v in sorted(tok.items()))
          + f" | intern soft cap {INTERN_SOFT_CAP/1e6:.0f}M", flush=True)

    start = 0
    if intern_used >= INTERN_SOFT_CAP:
        print(f"[{args.name}] intern budget exhausted ({intern_used/1e6:.1f}M "
              f">= cap) — starting on local", flush=True)
        start = 1

    for ci in range(start, len(CHANNELS)):
        model = CHANNELS[ci]
        real = [c.replace("{MODEL}", model) for c in cmd]
        env = os.environ.copy()
        env.update({"PYTHONPATH": r"C:/Users/D0n9/Desktop/CompileScholar/src",
                    "PYTHONIOENCODING": "utf-8",
                    "LLM_CALL_LOG": args.ledger,
                    "LLM_RUN_ID": f"multi-{args.name}",
                    "LLM_SOCK_TIMEOUT": "300", "LLM_WALL_TIMEOUT": "600"})
        if model.startswith("local"):
            env.setdefault("LOCAL_MAX_CONCURRENT", "16")
        else:
            env.setdefault("INTERN_MAX_CONCURRENT", "8")
        print(f"[{args.name}] attempt on {model}: {' '.join(real[:6])}...",
              flush=True)
        p = subprocess.run(real, env=env,
                           cwd=r"C:/Users/D0n9/Desktop/CompileScholar/src")
        tok = ledger_tokens(args.ledger)
        print(f"[{args.name}] {model} exited rc={p.returncode} | ledger: "
              + ", ".join(f"{k}={v/1e6:.2f}M" for k, v in sorted(tok.items())),
              flush=True)
        if p.returncode == 0:
            print(f"[{args.name}] STAGE DONE on {model}", flush=True)
            return 0
        # ChannelDeadError surfaces as a traceback + rc=1; any nonzero rc on
        # intern gets one local retry (conservative: local rerun is cheap and
        # deterministic). Nonzero on LOCAL = real failure -> stop.
        if model.startswith("local"):
            print(f"[{args.name}] failed on last channel — STOP (resume-safe "
                  f"rerun after fixing)", flush=True)
            return p.returncode
        print(f"[{args.name}] channel failed/exhausted — switching to next",
              flush=True)
    return 3


if __name__ == "__main__":
    main()
