"""本地 GPUStack Qwen3.8-27B 最小客户端（关思考、直连不走代理）。"""
import os, re, time, httpx
from openai import OpenAI
ENV = r"C:\Users\D0n9\Desktop\CompileScholar\.env"
def _env(k):
    for line in open(ENV, encoding="utf-8"):
        m = re.match(r"^\s*" + k + r"\s*=\s*(.+?)\s*$", line)
        if m: return m.group(1)
    return None
BASE = (_env("LOCAL_BASE_URL") or "").rstrip("/")
if not BASE.endswith("/v1"): BASE = BASE + "/v1" if "/v1" not in BASE else BASE
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",192.168.199.73,localhost,127.0.0.1"
_client = OpenAI(api_key=_env("LOCAL_API_KEY") or "x", base_url=BASE,
                 http_client=httpx.Client(trust_env=False, timeout=900))
MODEL = "Qwen3.8-27B"
def chat(prompt, system=None, max_tokens=6000, temperature=0.3, retries=3):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    for i in range(retries):
        try:
            r = _client.chat.completions.create(model=MODEL, messages=msgs, max_tokens=max_tokens,
                    temperature=temperature, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return r.choices[0].message.content or "", r.usage
        except Exception as e:
            if i == retries - 1: raise
            time.sleep(5 * (i + 1))
