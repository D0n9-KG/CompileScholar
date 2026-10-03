# -*- coding: utf-8 -*-
"""CS2 harness arm: Claude Code + local 27B + retrieval MCP (open mode).

Arm definition (design slot 3): matched-model deep-research arm —
Claude Code as the agent harness, Qwen3.8-27B the model, retrieval via
the open-mode MCP (Sciverse semantic + before_year=2025 cutoff).

Per question: fresh claude -p session, CS2 official prompt (json_to_
sample's format from task.py), output parsed for sections JSON.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CS2 = Path(__file__).resolve().parent
RUBRICS = CS2.parent / "scholarqa_multi" / "sqa2_rubrics_v1_recomputed.json"
ARM = CS2 / "arm_harness"
CLAUDE_EXE = (Path(os.environ.get("APPDATA", "")) / "npm" /
              "node_modules" / "@anthropic-ai" / "claude-code" / "bin" /
              "claude.exe")
MCP_CONFIG = CS2.parent / "_shared" / "mcp" / "mcp_config_open.json"

# 官方 prompt（task.py json_to_sample 逐字——excerpt_prompt=True 分支）
PROMPT_TMPL = """Generate a report answering the following research question. Be sure to include inline citations for each claim. Return your result as valid JSON with a single key `sections` which is a list of sections, each having keys `title`, `text`, and `citations`. Each entry in `citations` should have a JSON list of `snippets` extracted from the reference document and an `id`, each of which appears exactly in the text. Each `id` should be an inline citation as it appears in the text (with wrapping parentheses or square brackets if appropriate). Each citation should have a `title` if one is available. Any additional information about the citation should go under `metadata`. Do not create a References section.

Here is an example `section` to help you with formatting:

        {{
          "title": "Background",
          "text": "Convolutional neural networks (CNNs) have achieved state-of-the-art results in image classification [1][2].",
          "citations": [
            {{
              "id": "[1]",
              "snippets": ["CNNs have become the standard for many visual tasks."],
              "title": "ImageNet Classification with Deep Convolutional Neural Networks",
              "metadata": {{
                "authors": "Krizhevsky, A. et al.",
                "year": 2012,
                "arxiv": "1207.0580"
              }}
            }}
          ]
        }}

Question: {question}"""


def _ensure_proxy(port: int = 8765):
    import socket
    import subprocess as _sp
    s = socket.socket()
    try:
        s.settimeout(1)
        s.connect(("127.0.0.1", port))
        return
    except OSError:
        pass
    finally:
        s.close()
    _sp.Popen([sys.executable, str(CS2.parent / "_shared" / "mcp" / "cc_compat_proxy.py"), "--port", str(port)],
              stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    time.sleep(2)


def run_q(question: str, timeout_s: int = 1800) -> dict:
    _ensure_proxy()
    env = {**os.environ.copy(),
           # 经 cc_compat_proxy（127.0.0.1:8765）转发到 GPUStack：Claude Code 2.1.287 在 messages 中插入
           # role=system 消息，GPUStack 报 400 "System message must be at the beginning"；代理只把它并入
           # 顶层 system 字段（语义等价）。代理由 _ensure_proxy() 自动拉起。
           "ANTHROPIC_BASE_URL": os.environ.get("HARNESS_ANTHROPIC_BASE_URL", "http://127.0.0.1:8765"),
           "NO_PROXY": "127.0.0.1,localhost,192.168.199.73",
           "ANTHROPIC_AUTH_TOKEN": _local_key(),
           "ANTHROPIC_MODEL": "Qwen3.8-27B"}
    cmd = [str(CLAUDE_EXE), "-p",
           PROMPT_TMPL.format(question=question),
           "--model", "Qwen3.8-27B",
           "--output-format", "json",
           "--max-turns", "40",
           "--mcp-config", str(MCP_CONFIG),
           # 隔离用户级 ~/.claude/settings.json：其 env 块把 ANTHROPIC_BASE_URL 指到第三方网关，
           # 会覆盖本进程传入的本地 27B 地址（2026-10-03 实测：全部 503 "No available channel"）
           "--setting-sources", "project",
           "--strict-mcp-config",
           "--allowedTools",
           "mcp__retrieval__search_papers,mcp__retrieval__fetch_chunk"]
    # 中性工作目录（home 与仓库之外）：cwd 在 home 下时 Claude Code 沿祖先目录把 ~/.claude/CLAUDE.md
    # 当项目指令注入、并注入本仓库 git status（10-03 抓包实测），与任务无关的个人指令会污染 harness 臂
    cwd = Path(os.environ.get("HARNESS_CWD", r"C:\cs2_harness_cwd"))
    cwd.mkdir(exist_ok=True)
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout_s, cwd=str(cwd))
        out = p.stdout or ""
        try:
            envelope = json.loads(out.strip().splitlines()[-1])
            return {"ok": True, "result": envelope.get("result", ""),
                    "num_turns": envelope.get("num_turns"),
                    "is_error": envelope.get("is_error")}
        except Exception:
            return {"ok": False, "err": "envelope parse",
                    "raw": out[:500]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "err": "timeout"}
    except Exception as e:
        return {"ok": False, "err": str(e)[:150]}


def _local_key():
    for line in open(CS2.parents[3] / ".env", encoding="utf-8"):
        m = re.match(r"^LOCAL_API_KEY\s*=\s*(.+)$", line.strip())
        if m:
            return m.group(1).strip()
    return ""


def main():
    limit = int(os.environ.get("CS2_LIMIT", "20"))
    offset = int(os.environ.get("CS2_OFFSET", "0"))   # 批次轮转用
    fanout = int(os.environ.get("HARNESS_FANOUT", "2"))  # 服务器余量实测
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ARM.mkdir(exist_ok=True)
    # 官方划分：dev=rubrics_v1、test=rubrics_v2；输出文件可配置（截止重跑不覆盖历史产物）
    split = os.environ.get("CS2_SPLIT", "dev")
    rub = RUBRICS if split == "dev" else RUBRICS.with_name("sqa2_rubrics_v2_recomputed.json")
    rubrics = json.load(open(rub,
                             encoding="utf-8"))[offset:offset + limit]

    out_path = ARM / os.environ.get("HARNESS_OUT", "answers_harness_dev20.json")
    done = {}
    if out_path.exists():
        done = {r["qid"]: r for r in json.load(open(out_path,
                                                    encoding="utf-8"))}
    results = list(done.values())

    def _good(r):
        # Claude Code 把 API 层失败也包成 ok 信封（is_error=True / result="API Error: ..."）——不算完成
        return r.get("ok") and not r.get("is_error") and not str(r.get("result", "")).startswith("API Error")

    todo = [q for q in rubrics
            if q["case_id"][:24] not in done or not _good(done[q["case_id"][:24]])]
    results = [r for r in results if _good(r)]
    print(f"[harness] {len(done)} done, {len(todo)} todo, "
          f"fanout={fanout}", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def one(q):
        qid = q["case_id"][:24]
        t0 = time.time()
        r = run_q(q["question"])
        r["qid"] = qid
        r["question"] = q["question"]
        r["elapsed_s"] = round(time.time() - t0, 1)
        return r

    lock = __import__("threading").Lock()
    with ThreadPoolExecutor(max_workers=fanout) as ex:
        futs = [ex.submit(one, q) for q in todo]
        for fut in as_completed(futs):
            r = fut.result()
            with lock:
                results.append(r)
                json.dump(results, open(out_path, "w",
                                        encoding="utf-8"),
                          ensure_ascii=False, indent=1)
            print(f"[harness] {r['qid']} -> ok={r['ok']} "
                  f"turns={r.get('num_turns')} {r['elapsed_s']}s",
                  flush=True)
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"[harness] done: {n_ok}/{len(results)} ok -> {out_path}")


if __name__ == "__main__":
    main()
