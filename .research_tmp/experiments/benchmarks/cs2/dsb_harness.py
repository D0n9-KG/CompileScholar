# -*- coding: utf-8 -*-
"""DeepScholar-Bench harness 臂（Claude Code+27B+检索 MCP），与新管线 run_vnext_dsb.py 同题同截止同判分。

10-03 重写（旧版问题见 review-1002-full-audit #21：≥11/63 题自述"WebSearch 不可用"后闭卷作答；
无每题截止；用户级 settings 注入；GPUStack 响应不合规 1 轮即失败）：
- 题集 = p6/oracle_inputs.json 的 48 个唯一题（与 run_vnext_dsb / judge_nuggets 同口径），题面 = 官方 query 模板
  （含目标论文摘要与"只引用 <发表日> 之前的 arXiv 文献"）。
- 每题知识截止 = 目标论文发表年月：为每题写一份 MCP 配置（KNOWLEDGE_CUTOFF 注入 MCP 进程环境，服务端强制）。
- 与 CS2 harness 同隔离：--setting-sources project --strict-mcp-config、中性 cwd、经 cc_compat_proxy（127.0.0.1:8765）。
- 产物：p6/gen/<SYS>/<gt_dir>.md（正文，judge_nuggets.py 直接判）+ arm_harness_dsb/<SYS>.json（信封/轮数/耗时）。
  API 层失败（is_error / "API Error"）不算完成，续跑会重做；最终仍失败的题在判分时按空文本计 0。
用法：python dsb_harness.py --sys harness_v2 [--limit N] [--fanout 2]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CS2 = Path(__file__).resolve().parent
P6 = CS2.parents[2] / "review_1002" / "p6"
ARM = CS2 / "arm_harness_dsb"
MCP_BASE = CS2.parent / "_shared" / "mcp" / "mcp_config_open.json"
CLAUDE_EXE = (Path(os.environ.get("APPDATA", "")) / "npm" / "node_modules" / "@anthropic-ai" / "claude-code" /
              "bin" / "claude.exe")
sys.path.insert(0, str(CS2))
from harness_arm_run import _ensure_proxy, _local_key  # noqa: E402


def _mcp_config(cut: str) -> Path:
    cfg = json.load(open(MCP_BASE, encoding="utf-8"))
    for s in cfg["mcpServers"].values():
        s.setdefault("env", {})["KNOWLEDGE_CUTOFF"] = cut
    p = ARM / "mcp_cfg" / f"mcp_{cut}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(p, "w", encoding="utf-8"), indent=1)
    return p


def run_q(prompt: str, cut: str, timeout_s: int = 1800) -> dict:
    _ensure_proxy()
    env = {**os.environ.copy(),
           "ANTHROPIC_BASE_URL": os.environ.get("HARNESS_ANTHROPIC_BASE_URL", "http://127.0.0.1:8765"),
           "NO_PROXY": "127.0.0.1,localhost,192.168.199.73",
           "ANTHROPIC_AUTH_TOKEN": _local_key(),
           "ANTHROPIC_MODEL": "Qwen3.8-27B"}
    cmd = [str(CLAUDE_EXE), "-p", prompt, "--model", "Qwen3.8-27B", "--output-format", "json",
           "--max-turns", "40", "--mcp-config", str(_mcp_config(cut)),
           "--setting-sources", "project", "--strict-mcp-config",
           "--allowedTools", "mcp__retrieval__search_papers,mcp__retrieval__fetch_chunk"]
    cwd = Path(os.environ.get("HARNESS_CWD", r"C:\cs2_harness_cwd"))
    cwd.mkdir(exist_ok=True)
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout_s, cwd=str(cwd))
        try:
            envelope = json.loads((p.stdout or "").strip().splitlines()[-1])
            return {"ok": True, "result": envelope.get("result", ""), "num_turns": envelope.get("num_turns"),
                    "is_error": envelope.get("is_error")}
        except Exception:
            return {"ok": False, "err": "envelope parse", "raw": (p.stdout or "")[:500]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "err": "timeout"}
    except Exception as e:
        return {"ok": False, "err": str(e)[:150]}


def _good(r):
    return r.get("ok") and not r.get("is_error") and not str(r.get("result", "")).startswith("API Error")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", default="harness_v2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fanout", type=int, default=2)
    a = ap.parse_args()
    rows = json.load(open(P6 / "oracle_inputs.json", encoding="utf-8"))
    if a.limit:
        rows = rows[:a.limit]
    gen = P6 / "gen" / a.sys
    gen.mkdir(parents=True, exist_ok=True)
    ARM.mkdir(exist_ok=True)
    done_p = ARM / f"{a.sys}.json"
    done = {r["gt_dir"]: r for r in json.load(open(done_p, encoding="utf-8"))} if done_p.exists() else {}
    todo = [r for r in rows if not _good(done.get(r["gt_dir"], {}))]
    print(f"[harness-dsb] {len(rows) - len(todo)} done, {len(todo)} todo, fanout={a.fanout}", flush=True)

    def one(r):
        cut = (r.get("published_date") or "")[:7]
        t0 = time.time()
        res = run_q(r["query"], cut)
        res.update({"gt_dir": r["gt_dir"], "qid": r["qid"], "cutoff": cut, "elapsed_s": round(time.time() - t0, 1)})
        if _good(res):
            text = res.get("result") or ""
            m = re.search(r"^#{1,3} ", text, re.M)  # 剥掉正文前的说明行（"Here is ..."）
            if m and m.start() > 0:
                text = text[m.start():]
            open(gen / f"{r['gt_dir']}.md", "w", encoding="utf-8").write(text)
        return res

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(a.fanout) as ex:
        for f in as_completed([ex.submit(one, r) for r in todo]):
            r = f.result()
            done[r["gt_dir"]] = r
            json.dump(list(done.values()), open(done_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"[harness-dsb] gt={r['gt_dir']} good={_good(r)} turns={r.get('num_turns')} "
                  f"{len(r.get('result') or '')}ch {r['elapsed_s']}s {r.get('err', '')}", flush=True)
    print(f"[harness-dsb] good {sum(1 for r in done.values() if _good(r))}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
