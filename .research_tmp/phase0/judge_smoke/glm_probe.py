# -*- coding: utf-8 -*-
"""GLM-5.3 via Paratera: connectivity + json_schema strict probe."""
import json
import urllib.request

ENV = {}
for line in open(r"C:\Users\D0n9\Desktop\CompileScholar\.env", encoding="utf-8"):
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        ENV[k.strip()] = v.strip()

BASE = ENV["PARATERA_BASE_URL"]
KEY = ENV["PARATERA_API_KEY"]


def call(payload, label):
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
            msg = d["choices"][0]["message"]
            content = msg.get("content") or ""
            print(label, "=> OK | model:", d.get("model"),
                  "| reasoning:", bool(msg.get("reasoning_content")),
                  "| content:", repr(content[:150]))
            return d
    except Exception as e:
        body = ""
        if hasattr(e, "read"):
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
        print(label, "=> FAIL:", str(e)[:200], body)
        return None


MODEL = "GLM-5.3"  # 大小写敏感：服务端模型清单为准（glm-5.3 报 400）
call({"model": MODEL,
      "messages": [{"role": "user", "content": "Reply with exactly: JUDGE OK"}],
      "max_tokens": 300}, "plain")

schema = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["verdict"],
    "additionalProperties": False,
}
call({"model": MODEL,
      "messages": [{"role": "user",
                    "content": 'Is Paris in France? Answer per the schema.'}],
      "max_tokens": 300,
      "response_format": {"type": "json_schema", "json_schema": {
          "name": "verdict", "strict": True, "schema": schema}}},
     "json_schema-strict")

call({"model": MODEL,
      "messages": [{"role": "user",
                    "content": 'Is Paris in France? Answer per the schema.'}],
      "max_tokens": 300,
      "response_format": {"type": "json_object"}}, "json_object")
