#!/usr/bin/env bash
# S2 IP 级限流解除后自动续跑（10-03 晚：三臂并行把无 key 的 S2 打到 429，连续 20/20 失败）。
# 每 10 分钟发 1 个探测请求；连续 3 次 200 才判定恢复（避免刚解封就被再次打爆）。
# 恢复后严格串行用 S2：① DSB ours 剩余题 → ② CS2 test（run_test_final.sh：[ours r1 ‖ harness] → [ours r2 ‖ GPTR]）。
# 用法：bash resume_after_s2.sh   （日志 resume_after_s2.log）
set -u
cd "$(dirname "$0")"
export PYTHONUTF8=1
probe() {
  curl -s -o /dev/null -w "%{http_code}" -m 20 --noproxy '*' \
    "https://api.semanticscholar.org/graph/v1/paper/search/match?query=attention%20is%20all%20you%20need&fields=paperId"
}
ok=0
while [ $ok -lt 3 ]; do
  c=$(probe)
  echo "$(date '+%H:%M:%S') probe $c (consecutive ok=$ok)" >> resume_after_s2.log
  if [ "$c" = "200" ]; then ok=$((ok+1)); sleep 60; else ok=0; sleep 600; fi
done
echo "$(date '+%H:%M:%S') S2 RECOVERED" >> resume_after_s2.log
python -W ignore run_vnext_dsb.py --sys vnext_v2 --workers 2 >> run_vnext_dsb_v2.log 2>&1
echo "GEN_DONE" >> run_vnext_dsb_v2.log
echo "$(date '+%H:%M:%S') DSB ours generation done" >> resume_after_s2.log
bash run_test_final.sh
echo "$(date '+%H:%M:%S') CS2 test final done" >> resume_after_s2.log
