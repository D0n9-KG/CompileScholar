# -*- coding: utf-8 -*-
"""Merge-judgment probe: the gate-3 over-merge families, judged by
(1) intern qwen3.8-27b thinking-OFF, (2) intern thinking-ON,
(3) local Qwen3.8-27B thinking-OFF. Legacy DeepSeek-V4-Flash separated all
of these correctly; we need a free-channel configuration that matches.
"""
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")

ENV = {}
for line in open(r"C:/Users/D0n9/Desktop/CompileScholar/.env", encoding="utf-8"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        ENV[k.strip()] = v.strip()

SURFACES = [
    ("commitpackft", "dataset"), ("humanevalpack", "dataset"),
    ("octopack", "dataset"), ("octocoder", "dataset"), ("octogeex", "dataset"),
    ("doremi", "method"), ("iterated doremi", "method"),
    ("groove", "method"), ("algorithmic regret", "mechanism"), ("ar", "mechanism"),
    ("upomdp", "method"), ("meta-upomdp", "method"),
    ("qlora", "method"), ("quantized lora", "method"),          # easy positive
    ("gptq", "method"), ("post-training quantization gptq", "method"),  # positive
]

PROMPT = """下面是从 16 篇论文卡片收集的实体表面名（编号列表）。每行格式：
[编号] 表面名 | 类型

任务：把指向**同一实体**的表面名合并成组（缩写/全称/大小写/连字符变体/常见别名）。规则：
1. 只合并确信是同一实体的；有关联但不同的实体绝不合并（例如某方法的两个不同版本若文内明确区分，保持分开；不同方法即使同源也分开；同族的不同数据集/基准分开）。
2. 每个编号必须出现在恰好一个组里。
3. canonical 选组内最通用的写法。
4. 拿不准是否同一实体时，不合并（各自成组）。

输出 JSON：{"groups": [{"canonical": 编号, "members": [编号,...], "entity_type": "method|mechanism|practice|out_of_corpus"}]}
只输出 JSON。

实体列表：
{lines}"""


def call(base, key, model, thinking_off, timeout=300):
    payload = {"model": model,
               "messages": [{"role": "user", "content": PROMPT.replace(
                   "{lines}", "\n".join(f"[{i}] {s} | {t}"
                                        for i, (s, t) in enumerate(SURFACES)))}],
               "temperature": 0.0, "max_tokens": 4000}
    if thinking_off == "intern-off":
        payload["thinking"] = {"type": "disabled"}
    elif thinking_off == "local-off":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    m = resp["choices"][0]["message"]
    return m.get("content") or "", resp.get("usage") or {}


def judge(raw, label):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        print(f"[{label}] UNPARSEABLE: {raw[:120]}")
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        print(f"[{label}] BAD JSON: {m.group(0)[:120]}")
        return None
    groups = obj.get("groups") or []
    assigned = {}
    for g in groups:
        for i in g.get("members") or []:
            assigned[i] = g.get("canonical")
    pairs = []
    for i in range(len(SURFACES)):
        for j in range(i + 1, len(SURFACES)):
            if assigned.get(i) is not None and assigned.get(i) == assigned.get(j):
                pairs.append((SURFACES[i][0], SURFACES[j][0]))
    bad = [p for p in pairs if p not in {("qlora", "quantized lora"),
                                          ("gptq", "post-training quantization gptq")}]
    good = [p for p in pairs if p not in bad]
    print(f"[{label}] merged pairs ok={len(good)} WRONG={len(bad)}")
    for a, b in good:
        print(f"   ok: {a} + {b}")
    for a, b in bad:
        print(f"   WRONG: {a} + {b}")
    return bad


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ib, ik = ENV["INTERN_BASE_URL"], ENV["INTERN_API_KEY"]
    lb, lk = ENV["LOCAL_BASE_URL"], ENV.get("LOCAL_API_KEY", "local")
    raw, usage = call(ib, ik, "qwen3.8-27b", "intern-off")
    print("intern thinking-OFF usage:", usage.get("completion_tokens"))
    judge(raw, "intern-off")
    raw, usage = call(ib, ik, "qwen3.8-27b", "intern-on", timeout=600)
    print("\nintern thinking-ON usage:", usage)
    judge(raw, "intern-on")
    raw, usage = call(lb, lk, "Qwen3.8-27B", "local-off", timeout=600)
    print("\nlocal thinking-OFF usage:", usage.get("completion_tokens"))
    judge(raw, "local-off")
