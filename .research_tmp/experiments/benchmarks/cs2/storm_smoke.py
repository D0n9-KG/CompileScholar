# -*- coding: utf-8 -*-
"""STORM local-backend smoke v3 (checklist #6) — GPUStack httpx fix first."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpustack_httpx_fix  # noqa: F401  MUST precede httpx-based imports
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ENV = {}
for line in open(r"C:\Users\D0n9\Desktop\CompileScholar\.env",
                 encoding="utf-8"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        ENV[k.strip()] = v.strip()

from knowledge_storm import VLLMClient

client = VLLMClient(
    model="Qwen3.8-27B", port="", model_type="chat",
    url="http://192.168.199.73/v1", api_key=ENV["LOCAL_API_KEY"],
    temperature=0.7, top_p=0.95,
)
client.base_url = "http://192.168.199.73/v1/"
for i in range(3):
    try:
        out = client.basic_request(
            "Say OK.", model="Qwen3.8-27B", max_tokens=30)
        print(f"[storm] call {i+1} OK:",
              str(out.choices[0].message.content)[:60])
    except Exception as e:
        print(f"[storm] call {i+1} FAIL:",
              type(e).__name__, str(e)[:120])
        break
