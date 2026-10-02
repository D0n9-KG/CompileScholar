# -*- coding: utf-8 -*-
"""P2-10 watchdog（FIX-PLAN v2）：判分/答题挂死自动 resume。

前科：批 17b 一题 LLM 调用挂死 8h（ledger 零活动但进程活着）；
harness 臂 1800s 超时 60/93（今天资源竞争饿死）。inspect 路径挂过 2 次。

机制：轮询目标进程的产出文件 mtime + ledger 尾部时间戳——
超过 stall_minutes 无任何活动 → kill + 按环境变量原样重启（resume
机制由各 runner 自带：answers 文件存在即续跑）。

用法（另一个终端，与目标任务并行跑）：
  python watchdog.py --pattern "cs2_runner" --stall 20 \\
      --restart "PYTHONUTF8=1 OURS_TAG=xxx python cs2_runner.py"
"""
import argparse
import json
import os
import re
import subprocess
import time


def newest_activity(run_cwd):
    """目标目录里所有产出文件（answers/ledger/log）的最新 mtime。"""
    best = 0.0
    for name in os.listdir(run_cwd):
        if not re.match(r"(answers_|ledger_|judge_|direct_scores|arm_)", name):
            continue
        p = os.path.join(run_cwd, name)
        if os.path.isfile(p):
            try:
                best = max(best, os.path.getmtime(p))
            except OSError:
                pass
    return best


def find_pid(pattern):
    out = subprocess.run(
        ["powershell", "-Command",
         f"(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\")"
         f".CommandLine | Select-String '{pattern}'"],
        capture_output=True, text=True).stdout
    return out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True,
                    help="进程命令行匹配串（如 cs2_runner）")
    ap.add_argument("--watch-dir", default=".",
                    help="产出文件目录（默认当前目录）")
    ap.add_argument("--stall", type=int, default=20,
                    help="停滞分钟数（默认 20）")
    ap.add_argument("--restart", default="",
                    help="重启命令（含环境变量前缀；空=只报警不重启）")
    args = ap.parse_args()

    print(f"[watchdog] 监视 '{args.pattern}'，停滞>{args.stall}min 触发，"
          f"目录 {os.path.abspath(args.watch_dir)}", flush=True)
    while True:
        time.sleep(60)
        alive = find_pid(args.pattern)
        if not alive:
            continue   # 进程不在（可能正常结束）——不动作
        act = newest_activity(args.watch_dir)
        stall_min = (time.time() - act) / 60 if act else 999
        if stall_min > args.stall:
            print(f"[watchdog] 停滞 {stall_min:.0f}min — "
                  f"{'kill+重启' if args.restart else '报警'}", flush=True)
            subprocess.run(
                ["powershell", "-Command",
                 f"Get-CimInstance Win32_Process -Filter \"Name like "
                 f"'python%'\" | Where-Object {{$_.CommandLine -like "
                 f"'*{args.pattern}*'}} | ForEach-Object {{ Stop-Process "
                 f"-Id $_.ProcessId -Force }}"], capture_output=True)
            if args.restart:
                subprocess.Popen(args.restart, shell=True, cwd=os.getcwd())
            time.sleep(120)   # 重启缓冲


if __name__ == "__main__":
    main()
