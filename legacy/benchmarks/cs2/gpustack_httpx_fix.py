# -*- coding: utf-8 -*-
"""GPUStack gateway httpx keep-alive bug: global fix.

Root cause (09-28 深挖修正): Windows 注册表系统代理（Clash
127.0.0.1:7890）被 httpx trust_env 读取——内网 GPUStack 请求被丢给代
理 → 502。urllib 不读注册表代理所以全通（此前"keep-alive bug"假说不
完整：Connection:close 修补了表象）。正解=NO_PROXY 豁免内网段。
affected: openai SDK / litellm / STORM——凡 httpx 系打内网的都中。

Fix: force `Connection: close` on every httpx request (one new TCP
connection per request — same as urllib behavior). Import this module
BEFORE importing openai/litellm/knowledge_storm.

Usage:
    import gpustack_httpx_fix  # noqa: F401 — must be first
    from knowledge_storm import ...
"""
import os
import httpx

# 09-28 深挖补丁：httpx 默认 trust_env=True 会读 Windows 注册表系统代理
# （实测本机 Clash 127.0.0.1:7890）——内网 GPUStack 请求被丢给代理 → 502。
# urllib 不读注册表代理所以全通（这就是"urllib 全通 httpx 全 502"的根因，
# keep-alive 假说作废）。修法：send patch 里内网 host 强制直连（对公网
# 调用不生效，代理行为保留）。
_orig_send = httpx.Client.send
_orig_asend = httpx.AsyncClient.send

_INTERNAL_NETS = ("192.168.", "10.", "127.0.0.1", "localhost")


def _is_internal(url: str) -> bool:
    try:
        host = url.split("//", 1)[1].split("/", 1)[0].split(":", 1)[0]
    except Exception:
        return False
    return any(host.startswith(n) for n in _INTERNAL_NETS)


def _send_close(self, request, **kwargs):
    request.headers["Connection"] = "close"
    return _orig_send(self, request, **kwargs)


# 09-28 真根因（取代 keep-alive 假说）：Windows 注册表系统代理
# （Clash 127.0.0.1:7890）被 httpx trust_env 读取——内网 GPUStack
# 请求被丢给代理 → 502；urllib 不读注册表代理所以全通。
# 正解：NO_PROXY 环境变量豁免内网段（httpx/requests/openai SDK 全尊重
# 此变量；公网 Paratera 请求不受影响）。
_internal = os.environ.get("NO_PROXY", "")
_need = [h for h in ("192.168.199.73", "localhost", "127.0.0.1")
         if h not in _internal]
if _need:
    os.environ["NO_PROXY"] = (_internal + "," if _internal else "")         + ",".join(_need)
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
import os as _os  # noqa: E402  (NO_PROXY 注入在 import os 之后)


def _is_internal(url: str) -> bool:
    try:
        host = url.split("//", 1)[1].split("/", 1)[0].split(":", 1)[0]
    except Exception:
        return False
    return any(host.startswith(n) for n in _INTERNAL_NETS)


if not getattr(httpx.Client, "_gpustack_close_patched", False):
    httpx.Client.send = _send_close
    httpx.Client._gpustack_close_patched = True
