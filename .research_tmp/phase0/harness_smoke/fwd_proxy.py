# -*- coding: utf-8 -*-
"""Tiny logging forward proxy: logs request bodies to GPUStack /v1/responses
so we can see exactly what codex sends. Run: python fwd_proxy.py 8791
"""
import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

UPSTREAM = "http://192.168.199.73"
LOG = open("fwd_proxy.log", "a", encoding="utf-8")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _forward(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        LOG.write("\n===== %s %s =====\n" % (self.command, self.path))
        for k, v in self.headers.items():
            if k.lower() not in ("authorization",):
                LOG.write("H %s: %s\n" % (k, v))
        try:
            pretty = json.dumps(json.loads(body), ensure_ascii=False, indent=1)
            LOG.write(pretty[:20000] + "\n")
        except Exception:
            LOG.write(repr(body[:5000]) + "\n")
        LOG.flush()
        req = urllib.request.Request(
            UPSTREAM + self.path, data=body, method=self.command
        )
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "accept-encoding"):
                req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            LOG.write("UPSTREAM ERROR %s: %s\n" % (e.code, data[:2000]))
            LOG.flush()
            self.send_response(e.code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
    HTTPServer(("127.0.0.1", port), H).serve_forever()
