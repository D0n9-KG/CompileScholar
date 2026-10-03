#!/usr/bin/env bash
# CS2 test 终测（冻结配置 FREEZE_CS2_TEST_1003.json，git 19fc12af）。自跑臂全部同 27B、同截止 2025-05、同判分器。
# 顺序执行（共用 27B）：ours 两遍（n=2 估方差）→ harness（经 cc_compat_proxy）→ GPTR。每臂答完即判分。
# 存档臂（OpenAI DR / SciSpace / Elicit）另由 judge_input_*_test.json 判分，不在此脚本。
# 用法：bash run_test_final.sh   （可重入：各 runner 都支持断点续跑）
set -u
cd "$(dirname "$0")"
export PYTHONUTF8=1 CS2_SPLIT=test KNOWLEDGE_CUTOFF=2025-05

for rep in 1 2; do
  tag=test100_r${rep}
  python -W ignore run_vnext.py --split test --offset 0 --limit 100 --tag $tag --workers 3 >> run_${tag}.log 2>&1
  python direct_judge.py judge_input_vnext_${tag}.json direct_scores_vnext_${tag}_ds.json 6 >> judge_${tag}.log 2>&1
  echo "ARM_DONE ours ${tag}" >> run_test_final.log
done

CS2_OFFSET=0 CS2_LIMIT=100 HARNESS_FANOUT=2 HARNESS_OUT=answers_harness_test100.json \
  python -W ignore harness_arm_run.py >> harness_test100.log 2>&1
python harness_to_judge.py arm_harness/answers_harness_test100.json judge_input_harness_test100.json \
  --split test --offset 0 --limit 100 >> harness_test100.log 2>&1
python direct_judge.py judge_input_harness_test100.json direct_scores_harness_test100_ds.json 6 >> judge_harness_test100.log 2>&1
echo "ARM_DONE harness" >> run_test_final.log

GPTR_OUT=answers_gptr_test100.json GPTR_FANOUT=4 python -W ignore cs2_gptr.py >> gptr_test100.log 2>&1
python direct_judge.py arm_gptr/judge_input_gptr_test100.json direct_scores_gptr_test100_ds.json 6 >> judge_gptr_test100.log 2>&1
echo "ARM_DONE gptr" >> run_test_final.log
echo ALLDONE >> run_test_final.log
