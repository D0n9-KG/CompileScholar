# -*- coding: utf-8 -*-
"""GLM-5.3 json_schema probe v2: bigger budget, inspect finish_reason."""
import json
import urllib.request

ENV = {}
for line in open(r"C:\Users\D0n9\Desktop\CompileScholar\.env", encoding="utf-8"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        ENV[k.strip()] = v.strip()

BASE = ENV["PARATERA_BASE_URL"]
KEY = ENV["PARATERA_API_KEY"]
MODEL = "GLM-5.3"


def call(payload, label):
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
            msg = d["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            print(label, "=> finish:", d["choices"][0].get("finish_reason"),
                  "| reasoning_len:", len(reasoning),
                  "| content:", repr(content[:150]))
            return d
    except Exception as e:
        print(label, "=> FAIL:", str(e)[:300])
        return None


schema = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["verdict"],
    "additionalProperties": False,
}

call({"model": MODEL,
      "messages": [{"role": "user",
                    "content": 'Is Paris in France? Answer per the schema.'}],
      "max_tokens": 4000,
      "response_format": {"type": "json_schema", "json_schema": {
          "name": "verdict", "strict": True, "schema": schema}}},
     "json_schema-strict/4k")

call({"model": MODEL,
      "messages": [{"role": "user",
                    "content": 'Is Paris in France? Answer per the schema.'}],
      "max_tokens": 4000,
      "response_format": {"type": "json_object"}}, "json_object/4k")

# thinking 参数显式关闭（若服务端支持 chat_template_kwargs 或 thinking 字段）
call({"model": MODEL,
      "messages": [{"role": "user",
                    "content": 'Is Paris in France? Answer per the schema.'}],
      "max_tokens": 4000,
      "thinking": {"type": "disabled"},
      "response_format": {"type": "json_schema", "json_schema": {
          "name": "verdict", "strict": True, "schema": schema}}},
     "json_schema+thinking-off")
