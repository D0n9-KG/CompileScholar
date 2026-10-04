# -*- coding: utf-8 -*-
"""One-off: copy the harness proxy + retrieval MCP server into compilescholar.baselines.harness (W6 S3)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
B = REPO / ".research_tmp" / "experiments" / "benchmarks" / "_shared" / "mcp"
DST = REPO / "src" / "compilescholar" / "baselines" / "harness"

p = (B / "cc_compat_proxy.py").read_text(encoding="utf-8")
p = p.replace("用法：python cc_compat_proxy.py [--port 8765] [--upstream http://192.168.199.73]",
              "用法：python -m compilescholar.baselines.harness.proxy [--port 8765] [--upstream <LOCAL_BASE_URL 去掉 /v1>]")
p = p.replace('import urllib.request\n\nUP = "http://192.168.199.73"\n',
              'import urllib.request\n\nfrom ...core import secrets\n\n\n'
              'def _default_upstream() -> str:\n'
              '    """Same host as the local chat endpoint (LOCAL_BASE_URL), without the OpenAI-style /v1 suffix."""\n'
              '    u = (secrets.get("LOCAL_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")\n'
              '    return u[:-3] if u.endswith("/v1") else u\n\n\n'
              'UP = _default_upstream()\n')
assert "_default_upstream" in p
(DST / "proxy.py").write_text(p, encoding="utf-8")

m = (B / "retrieval_mcp.py").read_text(encoding="utf-8")
a, b = m.index("CS_REPO = Path("), m.index("from mcp.server.fastmcp import FastMCP")
m = m[:a] + ("from ...core import cutoff as _cutoff\nfrom ...core import secrets\n"
             "from ...sources.sciverse import SciverseClient\n\n") + m[b:]
a, b = m.index("def _load_env_token()"), m.index("# ================================================================ corpus 模式")
m = m[:a] + 'def _load_env_token() -> str | None:\n    return secrets.get("SCIVERSE_API_TOKEN")\n\n\n' + m[b:]
a, b = m.index("def _sciverse():"), m.index("def _open_search(")
m = m[:a] + "def _sciverse():\n    return SciverseClient(token=_load_env_token())\n\n\n" + m[b:]
m = m.replace("    from cutoff import cutoff as _c0\n", "    _c0 = _cutoff.cutoff\n")
m = m.replace("    from cutoff import allowed as _allowed, cutoff as _cut\n", "    _allowed, _cut = _cutoff.allowed, _cutoff.cutoff\n")
m = m.replace('"command":"python","args":["<本文件>"]', '"command":"python","args":["-m","compilescholar.baselines.harness.mcp_server"]')
m = m.replace("SCIVERSE_API_TOKEN（open 模式；缺省从 CompileScholar/.env 读）", "SCIVERSE_API_TOKEN（open 模式；缺省从仓库 .env 读）")
assert "from cutoff" not in m and "sci_evo_extract" not in m and "CS_REPO" not in m
(DST / "mcp_server.py").write_text(m, encoding="utf-8")
print("ok")
