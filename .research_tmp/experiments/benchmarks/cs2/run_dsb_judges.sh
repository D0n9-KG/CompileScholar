#!/bin/bash
# DSB 双臂判分接力（FIX-PLAN v2 并行小项）：STORM 剩余 30 题 → harness 66 题全量。
# 串行发起（adapt/judge 串行纪律）；judge=Paratera DeepSeek-V4.1-Flash
# （大小写敏感！deepseek-v4-1-flash 会 400 model not found）。
# 用法: bash run_dsb_judges.sh
set -e
ENVP="C:/Users/D0n9/Desktop/CompileScholar/.env"
DSB="C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/deepscholar/dsb"
PY="$DSB/../venv311/Scripts/python.exe"
PKEY=$(grep "^PARATERA_API_KEY=" "$ENVP" | cut -d= -f2- | tr -d ' \r')
PBASE=$(grep "^PARATERA_BASE_URL=" "$ENVP" | cut -d= -f2- | tr -d ' \r')
export OPENAI_API_KEY="$PKEY" OPENAI_API_BASE="$PBASE"
cd "$DSB"

# 1) STORM 剩余（36-65；0-35 已有 nc_batch1-3）
IDS=$(seq 36 65 | tr '\n' ' ')
PYTHONUTF8=1 "$PY" -m eval.main --modes storm --evals nugget_coverage \
  --input-folder "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/cs2/arm_storm_dsb/indexed" \
  --output-folder results_storm_rest --file-id $IDS --model-name DeepSeek-V4.1-Flash

# 2) harness 全量（66 目录）
PYTHONUTF8=1 "$PY" -m eval.main --modes storm --evals nugget_coverage \
  --input-folder "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/cs2/arm_harness_dsb/indexed" \
  --output-folder results_harness_dsb --model-name DeepSeek-V4.1-Flash

echo "ALL DSB JUDGES DONE"
