#!/usr/bin/env bash
# CS2 test 终测（冻结配置 FREEZE_CS2_TEST_1003_v9b.json）：v9b = 多来源引文扩展（OpenAlex 批量为主）+ ~1000 词。
# 自跑臂全部同 27B、同截止 2025-05、同判分器（DeepSeek-V4.1-Flash）。
# 外部配额：Sciverse 30/min（跨进程文件令牌桶，所有臂共享）；OpenAlex 每日 10k credits（ours 每题 ~50）；
#           S2 无 key 仅作补充（429 → 全进程冷却 20 min）。
# 顺序：[ours r1 ‖ harness(fanout 2)] → ours r2（r2 复用 r1 的 OpenAlex 缓存，额度减半）。
# GPTR test 已在 10-03 21:25 那次（未经同意自启动的 v8 run）中答完 100 题，GPTR 臂与我们的配置无关，结果有效，此处不重跑。
# harness 同一次已有 12 题正常完成（保留），5 题因配额排队超时（重跑，超时放宽到 3600s）。
# 存档臂（OpenAI DR / SciSpace / Elicit）已另判，不在此脚本。
# 用法：bash run_test_final.sh   （可重入：各 runner 都支持断点续跑）
set -u
cd "$(dirname "$0")"
export PYTHONUTF8=1 CS2_SPLIT=test KNOWLEDGE_CUTOFF=2025-05
export SCIVERSE_SHARED_BUCKET="$HOME/.sciverse_bucket.json" SCIVERSE_MAX_WAIT_S=600

ours() {
  tag=test100_r$1
  python -W ignore run_vnext.py --split test --offset 0 --limit 100 --tag $tag --cite --workers 4 >> run_${tag}.log 2>&1
  python direct_judge.py judge_input_vnext_${tag}.json direct_scores_vnext_${tag}_ds.json 6 >> judge_${tag}.log 2>&1
  echo "ARM_DONE ours ${tag}" >> run_test_final.log
}

harness() {
  CS2_OFFSET=0 CS2_LIMIT=100 HARNESS_FANOUT=2 HARNESS_TIMEOUT_S=3600 HARNESS_OUT=answers_harness_test100.json \
    python -W ignore harness_arm_run.py >> harness_test100.log 2>&1
  python harness_to_judge.py arm_harness/answers_harness_test100.json judge_input_harness_test100.json \
    --split test --offset 0 --limit 100 >> harness_test100.log 2>&1
  python direct_judge.py judge_input_harness_test100.json direct_scores_harness_test100_ds.json 6 >> judge_harness_test100.log 2>&1
  echo "ARM_DONE harness" >> run_test_final.log
}

gptr() {
  GPTR_OUT=answers_gptr_test100.json GPTR_FANOUT=4 python -W ignore cs2_gptr.py >> gptr_test100.log 2>&1
  python direct_judge.py arm_gptr/judge_input_gptr_test100.json direct_scores_gptr_test100_ds.json 6 >> judge_gptr_test100.log 2>&1
  echo "ARM_DONE gptr" >> run_test_final.log
}

ours 1 & harness &
wait %1
ours 2
wait
echo ALLDONE >> run_test_final.log
