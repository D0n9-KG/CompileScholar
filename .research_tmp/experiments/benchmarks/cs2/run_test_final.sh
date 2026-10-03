#!/usr/bin/env bash
# CS2 test 终测（冻结配置 FREEZE_CS2_TEST_1003.json，git 19fc12af）。自跑臂全部同 27B、同截止 2025-05、同判分器。
#
# 调度（10-03 实测后修订）：瓶颈是外部 API，不是 27B。
#   - S2（无 key，IP 级 ~1 req/s）只有我们的管线用（引文扩展）。两遍 ours 并行时引文扩展 90s 预算排队耗尽，
#     test 前 15 题 cite 证据中位 1 条（dev 冻结时 10 条）、8/15 题为 0——测的不是冻结系统，那批结果已作废
#     （_discarded_contended_1003/）。→ 同一时刻只跑一个用 S2 的臂。
#   - harness（MCP）与 GPTR（external_tools）只用 Sciverse，可与 ours 并行（Sciverse 30/min 由跨进程令牌桶节流）。
# 顺序：[ours r1 ‖ harness] → [ours r2 ‖ GPTR]。存档臂另判。
# 用法：bash run_test_final.sh   （可重入：各 runner 都支持断点续跑）
set -u
cd "$(dirname "$0")"
export PYTHONUTF8=1 CS2_SPLIT=test KNOWLEDGE_CUTOFF=2025-05

ours() {
  tag=test100_r$1
  python -W ignore run_vnext.py --split test --offset 0 --limit 100 --tag $tag --workers 6 >> run_${tag}.log 2>&1
  python direct_judge.py judge_input_vnext_${tag}.json direct_scores_vnext_${tag}_ds.json 6 >> judge_${tag}.log 2>&1
  echo "ARM_DONE ours ${tag}" >> run_test_final.log
}

harness() {
  CS2_OFFSET=0 CS2_LIMIT=100 HARNESS_FANOUT=6 HARNESS_OUT=answers_harness_test100.json \
    python -W ignore harness_arm_run.py >> harness_test100.log 2>&1
  python harness_to_judge.py arm_harness/answers_harness_test100.json judge_input_harness_test100.json \
    --split test --offset 0 --limit 100 >> harness_test100.log 2>&1
  python direct_judge.py judge_input_harness_test100.json direct_scores_harness_test100_ds.json 6 >> judge_harness_test100.log 2>&1
  echo "ARM_DONE harness" >> run_test_final.log
}

gptr() {
  GPTR_OUT=answers_gptr_test100.json GPTR_FANOUT=8 python -W ignore cs2_gptr.py >> gptr_test100.log 2>&1
  python direct_judge.py arm_gptr/judge_input_gptr_test100.json direct_scores_gptr_test100_ds.json 6 >> judge_gptr_test100.log 2>&1
  echo "ARM_DONE gptr" >> run_test_final.log
}

ours 1 & harness &
wait
ours 2 & gptr &
wait
echo ALLDONE >> run_test_final.log
