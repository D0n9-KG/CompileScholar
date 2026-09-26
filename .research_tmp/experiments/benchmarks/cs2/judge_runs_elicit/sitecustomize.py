# Paratera 网关 httpx keep-alive bug 修复（同 GPUStack 同款）：
# 判分进程所有 httpx 请求强制 Connection: close
import httpx

_orig_send = httpx.Client.send

def _send_close(self, request, **kwargs):
    request.headers["Connection"] = "close"
    return _orig_send(self, request, **kwargs)

httpx.Client.send = _send_close
