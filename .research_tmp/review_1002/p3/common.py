"""P3 共用：本地 27B / DeepSeek 客户端（urllib 直连，绕开注册表代理问题）。"""
import json, os, re, time, urllib.request
ROOT = "C:/Users/D0n9/Desktop/CompileScholar"
P3 = ROOT + "/.research_tmp/review_1002/p3"
SNAP = "//192.168.199.138/Share400T/pub/LLM_Data/data/JournalPapers/arXiv_Dataset/arxiv-metadata-oai-snapshot.json"
ENV = {}
for line in open(ROOT + "/.env", encoding="utf-8"):
    m = re.match(r"^\s*([A-Z_]+)\s*=\s*(.*)$", line.rstrip("\n"))
    if m: ENV[m.group(1)] = m.group(2).strip()
_noproxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def _post(url, key, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
          headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    return json.loads(_noproxy.open(req, timeout=timeout).read())

def local27b(prompt, max_tokens=4000, temperature=0.0, timeout=600, retries=3):
    body = {"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}}
    for i in range(retries):
        try:
            r = _post("http://192.168.199.73/v1/chat/completions", ENV["LOCAL_API_KEY"], body, timeout)
            return r["choices"][0]["message"]["content"]
        except Exception as e:
            if i == retries - 1: raise
            time.sleep(3 * (i + 1))

DS_CALLS = {"n": 0}
def deepseek(prompt, max_tokens=2000, temperature=0.0, timeout=300, retries=3):
    DS_CALLS["n"] += 1
    base = ENV["PARATERA_BASE_URL"].rstrip("/")
    body = {"model": "DeepSeek-V4.1-Flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    opener = urllib.request.build_opener()  # 公网走系统默认
    for i in range(retries):
        try:
            req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json", "Authorization": "Bearer " + ENV["PARATERA_API_KEY"]})
            r = json.loads(opener.open(req, timeout=timeout).read())
            return r["choices"][0]["message"]["content"]
        except Exception as e:
            if i == retries - 1: raise
            time.sleep(3 * (i + 1))

def parse_json(s):
    s = re.sub(r"^```(json)?|```$", "", s.strip(), flags=re.M).strip()
    m = re.search(r"[\[{].*[\]}]", s, re.S)
    return json.loads(m.group(0)) if m else None
