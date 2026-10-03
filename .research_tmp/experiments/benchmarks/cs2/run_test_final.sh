#!/usr/bin/env bash
# CS2 test 终测（冻结配置 FREEZE_CS2_TEST_1003.json，git 19fc12af）。自跑臂全部同 27B、同截止 2025-05、同判分器。
# 10-03 并发压测：27B 在 96 路内吞吐仍升、48 路内延迟基本不涨 → 三臂并行（ours×2 + harness），GPTR 随后。
# 外部检索配额靠跨进程共享节流（Sciverse 30/min 文件令牌桶、S2 1 req/s 文件锁），并行不会叠加超限。
# 存档臂（OpenAI DR / SciSpace / Elicit）另由 judge_input_*_test.json 判分，不在此脚本。
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

ours 1 & ours 2 & harness &
wait

GPTR_OUT=answers_gptr_test100.json GPTR_FANOUT=8 python -W ignore cs2_gptr.py >> gptr_test100.log 2>&1
python direct_judge.py arm_gptr/judge_input_gptr_test100.json direct_scores_gptr_test100_ds.json 6 >> judge_gptr_test100.log 2>&1
echo "ARM_DONE gptr" >> run_test_final.log
echo ALLDONE >> run_test_final.log
