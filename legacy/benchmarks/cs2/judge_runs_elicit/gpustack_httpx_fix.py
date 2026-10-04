# -*- coding: utf-8 -*-
"""GPUStack gateway httpx keep-alive bug: global fix.

Root cause (measured 2026-09-28): GPUStack's gateway mishandles httpx
keep-alive connection REUSE — first request on a pooled connection
succeeds, subsequent ones 502/timeout. urllib (new connection per
request) never triggers it. All httpx-based tools are affected:
openai SDK, litellm, STORM VLLMClient, and very likely the earlier
codex CLI 502s (same signature).

Fix: force `Connection: close` on every httpx request (one new TCP
connection per request — same as urllib behavior). Import this module
BEFORE importing openai/litellm/knowledge_storm.

Usage:
    import gpustack_httpx_fix  # noqa: F401 — must be first
    from knowledge_storm import ...
"""
import httpx

_orig_send = httpx.Client.send


def _send_close(self, request, **kwargs):
    request.headers["Connection"] = "close"
    return _orig_send(self, request, **kwargs)


if not getattr(httpx.Client, "_gpustack_close_patched", False):
    httpx.Client.send = _send_close
    httpx.Client._gpustack_close_patched = True
