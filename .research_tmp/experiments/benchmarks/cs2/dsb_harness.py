# -*- coding: utf-8 -*-
"""DeepScholar-Bench harness 臂（Claude Code+27B+检索 MCP）——DSB 三臂对照
的通用 agent 臂。

任务形态：给定论文 abstract → 生成 Related Works 综述段（含引用）。
与 CS2 harness 臂同配置（Claude Code CLI+27B+开放检索 MCP），仅题目源
和 prompt 换成 DSB 的 Related Works 任务。

产物落 DSB 官方 parser 期望的形态（每题目录 search_ai 风格?——harness
没有官方 parser，产物直接是 markdown 文章 → 落 storm 同款
storm_gen_article.md 文件名，复用 StormParser 的通用 markdown+引用解析）。

用法：python dsb_harness.py --limit 5 / python dsb_harness.py
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
CS2 = Path(_HERE)
DSB = CS2.parent / "deepscholar" / "dsb"
ARM = CS2 / "arm_harness_dsb"
CLAUDE_EXE = (Path(os.environ.get("APPDATA", "")) / "npm" /
              "node_modules" / "@anthropic-ai" / "claude-code" / "bin" /
              "claude.exe")
MCP_CONFIG = CS2.parent / "_shared" / "mcp" / "mcp_config_open.json"

PROMPT_TMPL = """Write a Related Works section for an academic paper, given the paper's abstract below. Cite the most relevant prior literature with inline citations. Use markdown format with sections where appropriate.

{abstract}

Requirements:
- Cover the main research threads the paper builds on (cite specific prior work)
- Inline citations in the form [Author et al., Year] or [1] with a references list at the end
- Survey-style prose, well organized into thematic subsections"""


def _local_key():
    for line in open(CS2.parents[3] / ".env", encoding="utf-8"):
        m = __import__("re").match(r"^LOCAL_API_KEY\s*=\s*(.+)$", line.strip())
        if m:
            return m.group(1).strip()
    return ""


def run_q(abstract: str, timeout_s: int = 1800) -> dict:
    env = {**os.environ.copy(),
           "ANTHROPIC_BASE_URL": "http://192.168.199.73",
           "ANTHROPIC_AUTH_TOKEN": _local_key(),
           "ANTHROPIC_MODEL": "Qwen3.8-27B"}
    cmd = [str(CLAUDE_EXE), "-p",
           PROMPT_TMPL.format(abstract=abstract),
           "--model", "Qwen3.8-27B",
           "--output-format", "json",
           "--max-turns", "40",
           "--mcp-config", str(MCP_CONFIG),
           "--allowedTools",
           "mcp__retrieval__search_papers,mcp__retrieval__fetch_chunk"]
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout_s, cwd=str(ARM))
        out = p.stdout or ""
        try:
            envelope = json.loads(out.strip().splitlines()[-1])
            return {"ok": True, "result": envelope.get("result", ""),
                    "num_turns": envelope.get("num_turns"),
                    "is_error": envelope.get("is_error")}
        except Exception:
            return {"ok": False, "err": "envelope parse", "raw": out[:500]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "err": "timeout"}
    except Exception as e:
        return {"ok": False, "err": str(e)[:150]}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    ARM.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(
        DSB / "dataset" / "papers_with_related_works.csv", encoding="utf-8")))
    rows = rows[:args.limit]

    # gt 映射（arxiv qid → 数字 index——与 StormParser 的 file_id 对齐）
    gt_map = {}
    for idx in os.listdir(DSB / "dataset" / "gt_nuggets_outputs"):
        p = DSB / "dataset" / "gt_nuggets_outputs" / idx / "res.json"
        if p.exists():
            gt_map[json.load(open(p, encoding="utf-8")).get("qid", "")] = idx

    done_path = ARM / "answers_harness_dsb.json"
    done = {}
    if done_path.exists():
        done = {r["qid"]: r for r in
                json.load(open(done_path, encoding="utf-8"))}
    todo = [r for r in rows if r["arxiv_id"] not in done]
    print(f"[harness-dsb] {len(done)} done, {len(todo)} todo", flush=True)

    fanout = int(os.environ.get("HARNESS_DSB_FANOUT", "1"))
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def one(r):
        qid = r["arxiv_id"]
        t0 = time.time()
        res = run_q(r["abstract"])
        res["qid"] = qid
        res["title"] = r["title"]
        res["elapsed_s"] = round(time.time() - t0, 1)
        # 产物落 StormParser 形态（index 目录 + storm_gen_article.md）
        idx = gt_map.get(qid)
        if idx and res.get("ok"):
            out_dir = ARM / "indexed" / idx
            out_dir.mkdir(parents=True, exist_ok=True)
            # 产物清洗：Claude Code 常在正文前加说明行（"下面是..."）——
            # 剥到第一个 markdown 标题；引用格式混合式（[Author, Year]）保留
            # （StormParser 的 citation 正则认 [text](url)，无 url 的标记
            # 按 nugget/organization 判分仍进正文——不丢内容）
            text = res.get("result", "")
            import re as _re
            m = _re.search(r"^#{1,3} ", text, _re.M)
            if m and m.start() > 0:
                text = text[m.start():]
            with open(out_dir / "storm_gen_article.md", "w",
                      encoding="utf-8") as f:
                f.write(text)
        return res

    lock = __import__("threading").Lock()
    with ThreadPoolExecutor(max_workers=fanout) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for fut in as_completed(futs):
            r = fut.result()
            with lock:
                done[r["qid"]] = r
                json.dump(list(done.values()),
                          open(done_path, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
            print(f"[harness-dsb] {r['qid'][:12]} ok={r.get('ok')} "
                  f"{len(r.get('result') or '')}ch {r.get('elapsed_s')}s "
                  f"{r.get('err', '')[:60]}", flush=True)
    print(f"[harness-dsb] answers saved: {done_path}", flush=True)


if __name__ == "__main__":
    main()
