# -*- coding: utf-8 -*-
"""P2-11 依赖三件套检查（FIX-PLAN v2）：pip 装包后 openai/litellm/inspect
连锁断 3 次的前科——装包后跑本脚本，全绿再继续。

用法：python deps_check.py
"""
import sys

FAILED = []

def check(name, stmt):
    try:
        exec(stmt, {})
        print(f"  OK  {name}")
    except Exception as e:
        print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:80]}")
        FAILED.append(name)

print("依赖三件套检查（pip 装包后必跑）：")
check("openai", "from openai import OpenAI; OpenAI(api_key='x')")
check("litellm", "import litellm; litellm.get_model_info('gpt-3.5-turbo') if False else None")
check("inspect_ai", "import inspect_ai; from inspect_ai.scorer import score")

# 附带常用链
check("report_adapter 链", "import sys; sys.path.insert(0, r'..\\_shared\\tools'); import report_adapter")
check("harness 链", "import sys; sys.path.insert(0, r'..\\_shared\\tools'); import evidence_gate2r_harness")
check("nltk punkt", "import nltk; nltk.data.find('tokenizers/punkt_tab')")

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项失败——先修再继续（通常是 pip 装包把某个依赖升/降级了）")
    sys.exit(1)
print("✓ 全部通过")
