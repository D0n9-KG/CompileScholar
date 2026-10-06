# -*- coding: utf-8 -*-
"""Phase B: MinerU 9602 (task router) throughput with client-side throttling and retry.

The router reports its capacity at /health (max_concurrent_requests, summed over GPU workers). The client keeps at
most `slots` tasks in flight, submits through POST /tasks (async), polls GET /tasks/{id}, fetches GET
/tasks/{id}/result (zip: markdown + content_list). 409 / 429 / 503 on submit -> exponential backoff and resubmit;
a failed task is resubmitted up to 3 times. Output: runs/phaseB_fasttier/mineru/<id>.zip, mineru_runs.jsonl.
Usage: python mineru_throughput.py <slots> [limit]
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pymupdf

from compilescholar.core import paths

OUT = paths.runs() / "phaseB_fasttier"
FORM = {"backend": "vlm-vllm-async-engine", "parse_method": "auto", "lang_list": "en", "formula_enable": "true",
        "table_enable": "true", "return_md": "true", "return_content_list": "true", "return_middle_json": "false",
        "return_model_output": "false", "return_images": "false", "response_format_zip": "true"}
RETRY_STATUS = {409, 429, 502, 503}


def health(base: str) -> dict:
    return httpx.get(f"{base}/health", timeout=10, trust_env=False).json()


def parse_one(client: httpx.Client, base: str, pdf: Path, dest: Path) -> dict:
    rec = {"pdf": pdf.name, "submits": 0, "busy": 0}
    t0 = time.time()
    for attempt in range(3):
        backoff = 1.0
        while True:                                  # submit until accepted
            rec["submits"] += 1
            with open(pdf, "rb") as f:
                r = client.post(f"{base}/tasks", files={"files": (pdf.name, f, "application/pdf")}, data=FORM)
            if r.status_code in RETRY_STATUS:
                rec["busy"] += 1
                time.sleep(backoff)
                backoff = min(30.0, backoff * 2)
                continue
            r.raise_for_status()
            break
        tid = r.json()["task_id"]
        t_sub = time.time()
        while True:
            st = client.get(f"{base}/tasks/{tid}").json()
            if st["status"] in ("completed", "failed"):
                break
            time.sleep(2)
        if st["status"] == "completed":
            z = client.get(f"{base}/tasks/{tid}/result")
            z.raise_for_status()
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(z.content)
            tmp.replace(dest)
            started = st.get("started_at")
            rec.update(status="completed", seconds=time.time() - t0, task_seconds=time.time() - t_sub,
                       bytes=len(z.content), started_at=started, completed_at=st.get("completed_at"))
            return rec
        rec.setdefault("errors", []).append(st.get("error"))
    rec.update(status="failed", seconds=time.time() - t0)
    return rec


def main(slots: int, limit: int | None = None) -> None:
    base = paths._local_paths()["mineru_file_parse"].rstrip("/")     # a URL; paths.resource() returns a Path
    h = health(base)
    print(f"router {h['version']}: max_concurrent {h['max_concurrent_requests']}, queued {h['queued_tasks']}, "
          f"processing {h['processing_tasks']}, workers {len(h['servers'])}", flush=True)
    pdfs = sorted((OUT / "pdf").glob("*.pdf"))[:limit]
    dest = OUT / "mineru"
    dest.mkdir(exist_ok=True)
    todo = [p for p in pdfs if not (dest / (p.stem.removesuffix("v1") + ".zip")).exists()]
    pages = {p.name: pymupdf.open(p).page_count for p in todo}
    lock = threading.Lock()
    log = open(OUT / "mineru_runs.jsonl", "a", encoding="utf-8")
    client = httpx.Client(timeout=httpx.Timeout(60, read=600), trust_env=False)

    def one(p):
        rec = parse_one(client, base, p, dest / (p.stem.removesuffix("v1") + ".zip"))
        rec["pages"] = pages[p.name]
        rec["slots"] = slots
        with lock:
            log.write(json.dumps(rec) + "\n")
            log.flush()
        return rec
    t0 = time.time()
    with ThreadPoolExecutor(slots) as ex:
        res = list(ex.map(one, todo))
    wall = time.time() - t0
    ok = [r for r in res if r["status"] == "completed"]
    pg = sum(r["pages"] for r in ok)
    print(f"slots {slots}: {len(ok)}/{len(res)} completed, {pg} pages in {wall:.0f}s -> {pg / wall:.2f} pages/s, "
          f"{len(ok) / wall * 86400:.0f} papers/day; busy responses {sum(r['busy'] for r in res)}; "
          f"median task {sorted(r['task_seconds'] for r in ok)[len(ok) // 2]:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else None)
