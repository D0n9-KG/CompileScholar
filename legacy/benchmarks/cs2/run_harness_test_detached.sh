#!/usr/bin/env bash
# CS2 test harness 臂（独立进程，不受会话后台 2h 上限影响）。可重入：harness_arm_run 跳过已完成题。
set -u
cd "$(dirname "$0")"
export PYTHONUTF8=1 CS2_SPLIT=test KNOWLEDGE_CUTOFF=2025-05 SCIVERSE_SHARED_BUCKET="$HOME/.sciverse_bucket.json" SCIVERSE_MAX_WAIT_S=600
CS2_OFFSET=0 CS2_LIMIT=100 HARNESS_FANOUT=6 HARNESS_TIMEOUT_S=3600 HARNESS_OUT=answers_harness_test100.json \
  python -W ignore harness_arm_run.py >> harness_test100.log 2>&1
python harness_to_judge.py arm_harness/answers_harness_test100.json judge_input_harness_test100.json \
  --split test --offset 0 --limit 100 >> harness_test100.log 2>&1
python direct_judge.py judge_input_harness_test100.json direct_scores_harness_test100_ds.json 6 >> judge_harness_test100.log 2>&1
echo "ARM_DONE harness" >> run_test_final.log
