# -*- coding: utf-8 -*-
"""Claude Code → GPUStack 兼容代理（harness 臂用）。

Claude Code 2.1.287（10-02 自动更新）在 messages 数组里插入 role="system" 的消息（不在开头），GPUStack 的
Anthropic 兼容层按 OpenAI 聊天模板转换时报 400 "System message must be at the beginning"——harness 臂因此全部
失败（实测 10-03）。本代理只做一件事：把 messages 中所有 role="system" 消息的文本并入顶层 system 字段（追加到
末尾，保持原顺序），其余请求字节原样转发。不改动模型可见内容，只改位置（语义等价）。

用法：python cc_compat_proxy.py [--port 8765] [--upstream http://192.168.199.73]
"""
import argparse
import http.server
import json
import urllib.error
import urllib.request

UP = "http://192.168.199.73"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


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


class H(http.server.BaseHTTPRequestHandler):
    def _forward(self, method):
        n = int(self.headers.get("content-length", 0) or 0)
        body = self.rfile.read(n) if n else None
        if body is not None and self.path.startswith("/v1/messages"):
            body = fix(body)
        hdr = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "accept-encoding")}
        req = urllib.request.Request(UP + self.path, data=body, method=method, headers=hdr)
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
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        else:
            data = r.read()
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

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
