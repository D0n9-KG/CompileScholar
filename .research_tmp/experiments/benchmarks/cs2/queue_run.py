# -*- coding: utf-8 -*-
"""P2-9 串行化队列（FIX-PLAN v2）：adapt/judge 竞态踩了 4 次——
adapt_batches 重跑会覆盖 judge_input_*，与在跑的 direct_judge 并发时
读旧写新互相踩。本脚本把 adapt→judge 排队执行（同一批的 adapt 完成后
才起 judge；不同批串行）。

用法：
  python queue_run.py 30c        # 单批：adapt 30c → judge 30c
  python queue_run.py 30c 30d    # 多批顺序
"""
import subprocess
import sys

def run(cmd, desc):
    print(f"\n=== {desc}: {' '.join(cmd)} ===", flush=True)
    r = subprocess.run(cmd, cwd=sys.path[0])
    if r.returncode != 0:
        print(f"✗ {desc} 失败 (exit {r.returncode})——队列中止")
        sys.exit(r.returncode)

def main():
    batches = sys.argv[1:]
    if not batches:
        print(__doc__)
        sys.exit(1)
    for b in batches:
        run([sys.executable, "adapt_batches.py", b], f"adapt {b}")
        run([sys.executable, "direct_judge.py",
             f"judge_input_ours_batch{b}.json",
             f"direct_scores_ours_batch{b}_ds.json", "8"],
            f"judge {b}")
        print(f"✓ 批 {b} 完成")

if __name__ == "__main__":
    main()
