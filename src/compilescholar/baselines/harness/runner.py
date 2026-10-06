# -*- coding: utf-8 -*-
"""Same-model agent-harness arm: Claude Code + local 27B + retrieval MCP (moved from cs2/harness_arm_run.py and
cs2/dsb_harness.py; the claude command line, isolation flags, prompt and result parsing are unchanged).

Isolation (each measured to matter on 10-03):
  - `--setting-sources project --strict-mcp-config`: the user-level ~/.claude/settings.json env block would otherwise
    redirect ANTHROPIC_BASE_URL to a third-party gateway;
  - neutral cwd outside home and outside the repo: under home, Claude Code walks up and injects ~/.claude/CLAUDE.md
    and this repo's git status as project instructions;
  - requests go through the compat proxy (proxy.py), which only relocates mid-conversation system messages and
    repairs two non-conformant response fields of the serving layer.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from ...core import paths, secrets

CS2_PROMPT = """Generate a report answering the following research question. Be sure to include inline citations for each claim. Return your result as valid JSON with a single key `sections` which is a list of sections, each having keys `title`, `text`, and `citations`. Each entry in `citations` should have a JSON list of `snippets` extracted from the reference document and an `id`, each of which appears exactly in the text. Each `id` should be an inline citation as it appears in the text (with wrapping parentheses or square brackets if appropriate). Each citation should have a `title` if one is available. Any additional information about the citation should go under `metadata`. Do not create a References section.

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

ALLOWED_TOOLS = "mcp__retrieval__search_papers,mcp__retrieval__fetch_chunk"
MODEL = "Qwen3.8-27B"


def claude_exe() -> Path:
    return Path(os.environ.get("CLAUDE_EXE") or Path(os.environ.get("APPDATA", "")) / "npm" / "node_modules" /
                "@anthropic-ai" / "claude-code" / "bin" / "claude.exe")


def ensure_proxy(port: int = 8765):
    s = socket.socket()
    try:
        s.settimeout(1)
        s.connect(("127.0.0.1", port))
        return
    except OSError:
        pass
    finally:
        s.close()
    subprocess.Popen([sys.executable, "-m", "compilescholar.baselines.harness.proxy", "--port", str(port)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)


def mcp_config(cutoff: str, out_dir: Path) -> Path:
    """MCP config for one knowledge cutoff (the server enforces it; models never pass before_year themselves)."""
    cfg = {"mcpServers": {"retrieval": {
        "command": sys.executable, "args": ["-m", "compilescholar.baselines.harness.mcp_server"],
        "env": {"RETRIEVAL_MCP_MODE": "open", "KNOWLEDGE_CUTOFF": cutoff,
                "SCIVERSE_SHARED_BUCKET": os.environ.get("SCIVERSE_SHARED_BUCKET")
                or os.path.join(os.path.expanduser("~"), ".sciverse_bucket.json"),
                "SCIVERSE_MAX_WAIT_S": "600"}}}}
    p = out_dir / f"mcp_{cutoff}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(p, "w", encoding="utf-8"), indent=1)
    return p


def run_q(prompt: str, mcp_cfg: Path, timeout_s: int = 3600, max_turns: int = 40) -> dict:
    ensure_proxy()
    local_host = (secrets.get("LOCAL_BASE_URL") or "").split("://")[-1].split("/")[0].split(":")[0]
    env = {**os.environ.copy(),
           "ANTHROPIC_BASE_URL": os.environ.get("HARNESS_ANTHROPIC_BASE_URL", "http://127.0.0.1:8765"),
           "NO_PROXY": ",".join(x for x in ("127.0.0.1", "localhost", local_host) if x),
           "ANTHROPIC_AUTH_TOKEN": secrets.get("LOCAL_API_KEY", ""),
           "ANTHROPIC_MODEL": MODEL}
    cmd = [str(claude_exe()), "-p", prompt, "--model", MODEL, "--output-format", "json", "--max-turns", str(max_turns),
           "--mcp-config", str(mcp_cfg), "--setting-sources", "project", "--strict-mcp-config",
           "--allowedTools", ALLOWED_TOOLS]
    cwd = paths.resource("harness_cwd")
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


def good(r: dict) -> bool:
    """Claude Code wraps API-layer failures in an ok envelope (is_error / "API Error: ...") — those are not answers."""
    return bool(r.get("ok")) and not r.get("is_error") and not str(r.get("result", "")).startswith("API Error")


def to_cs2_sections(result: str) -> list[dict]:
    """First JSON object in the harness result -> its `sections` (unparseable -> [] and the question scores 0)."""
    i, j = result.find("{"), result.rfind("}") + 1
    try:
        obj = json.loads(result[i:j]) if i >= 0 and j > i else {}
        return obj.get("sections") or []
    except Exception:
        return []


def default_arm_dir() -> Path:
    return paths.runs() / "harness"
