# -*- coding: utf-8 -*-
"""Claude Code → GPUStack 兼容代理（harness 臂用）。

Claude Code 2.1.287（10-02 自动更新）在 messages 数组里插入 role="system" 的消息（不在开头），GPUStack 的
Anthropic 兼容层按 OpenAI 聊天模板转换时报 400 "System message must be at the beginning"——harness 臂因此全部
失败（实测 10-03）。本代理只做一件事：把 messages 中所有 role="system" 消息的文本并入顶层 system 字段（追加到
末尾，保持原顺序），其余请求字节原样转发。不改动模型可见内容，只改位置（语义等价）。

响应侧只做 Anthropic Messages 格式规范化（10-03 抓包实测 GPUStack 两处不合规，harness 题 1 轮即失败）：
- 无参数工具调用（如 list_corpus）的 tool_use 块缺 `input` 字段 → Claude Code 报
  "Tool use input must be a string or object"；补 `input: {}`。
- 流式 thinking_delta 用 `text` 而非规范的 `thinking` 字段、thinking 块起始缺 `thinking` 字段 → 流式解析失败，
  Claude Code 退回非流式重发（每轮白算一遍）；改名/补空串。
不改动任何模型生成的内容。

用法：python cc_compat_proxy.py [--port 8765] [--upstream http://192.168.199.73]
"""
import argparse
import http.server
import json
import os
import threading
import urllib.error
import urllib.request

UP = "http://192.168.199.73"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
DUMP = os.environ.get("CC_PROXY_DUMP")  # 调试：把每个 /v1/messages 请求与响应原文写到该目录
_N = [0]
_LOCK = threading.Lock()


def _texts(content):
    if isinstance(content, str):
        return [content]
    return [c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text"]


def fix(body: bytes) -> bytes:
    try:
        d = json.loads(body)
    except Exception:
        return body
    msgs = d.get("messages")
    if not isinstance(msgs, list) or not any(isinstance(m, dict) and m.get("role") == "system" for m in msgs):
        return body
    extra = [t for m in msgs if m.get("role") == "system" for t in _texts(m.get("content"))]
    d["messages"] = [m for m in msgs if m.get("role") != "system"]
    sysf = d.get("system")
    if isinstance(sysf, list):
        d["system"] = sysf + [{"type": "text", "text": t} for t in extra if t]
    else:
        d["system"] = "\n\n".join([s for s in [sysf or ""] + extra if s])
    return json.dumps(d, ensure_ascii=False).encode("utf-8")


def _norm_block(b):
    if isinstance(b, dict):
        if b.get("type") == "tool_use" and not isinstance(b.get("input"), (dict, str)):
            b["input"] = {}
        if b.get("type") == "thinking" and "thinking" not in b:
            b["thinking"] = b.pop("text", "") or ""
    return b


def norm_json(data: bytes) -> bytes:
    try:
        d = json.loads(data)
    except Exception:
        return data
    if isinstance(d, dict) and isinstance(d.get("content"), list):
        d["content"] = [_norm_block(b) for b in d["content"]]
        return json.dumps(d, ensure_ascii=False).encode("utf-8")
    return data


def norm_sse_line(line: bytes) -> bytes:
    if not line.startswith(b"data:"):
        return line
    try:
        ev = json.loads(line[5:].strip())
    except Exception:
        return line
    t = ev.get("type")
    if t == "content_block_start":
        _norm_block(ev.get("content_block"))
    elif t == "content_block_delta":
        dl = ev.get("delta") or {}
        if dl.get("type") == "thinking_delta" and "thinking" not in dl:
            dl["thinking"] = dl.pop("text", "") or ""
    else:
        return line
    return b"data: " + json.dumps(ev, ensure_ascii=False).encode("utf-8")


class H(http.server.BaseHTTPRequestHandler):
    def _forward(self, method):
        n = int(self.headers.get("content-length", 0) or 0)
        body = self.rfile.read(n) if n else None
        if body is not None and self.path.startswith("/v1/messages"):
            body = fix(body)
        hdr = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "accept-encoding")}
        req = urllib.request.Request(UP + self.path, data=body, method=method, headers=hdr)
        dump = None
        if DUMP and body is not None and self.path.startswith("/v1/messages"):
            with _LOCK:
                _N[0] += 1
                k = _N[0]
            os.makedirs(DUMP, exist_ok=True)
            open(os.path.join(DUMP, f"{k:04d}_req.json"), "wb").write(body)
            dump = open(os.path.join(DUMP, f"{k:04d}_resp.txt"), "wb")
        try:
            r = _OPENER.open(req, timeout=900)
            code, rh = r.status, r.headers
        except urllib.error.HTTPError as e:
            r, code, rh = e, e.code, e.headers
        self.send_response(code)
        for k, v in rh.items():
            if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                self.send_header(k, v)
        stream = "text/event-stream" in (rh.get("content-type") or "")
        if stream:
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            buf = b""
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not chunk:
                    break
                if dump:
                    dump.write(chunk)
                buf += chunk
                # 按整行改写（SSE 事件以换行分隔；半行留到下个 chunk）
                *lines, buf = buf.split(b"\n")
                if not lines:
                    continue
                out = b"\n".join(norm_sse_line(ln) for ln in lines) + b"\n"
                self.wfile.write(f"{len(out):x}\r\n".encode() + out + b"\r\n")
                self.wfile.flush()
            if buf:
                out = norm_sse_line(buf)
                self.wfile.write(f"{len(out):x}\r\n".encode() + out + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        else:
            data = r.read()
            if dump:
                dump.write(data)
            if code == 200 and self.path.startswith("/v1/messages"):
                data = norm_json(data)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        if dump:
            dump.close()

    def do_POST(self):
        self._forward("POST")

    def do_GET(self):
        self._forward("GET")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--upstream", default=UP)
    a = ap.parse_args()
    UP = a.upstream.rstrip("/")
    http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
