# -*- coding: utf-8 -*-
"""Multi-108 corpus mineru parsing (stage 3 of task #11).

Takes the fetched PDFs from corpus/pdfs/ and sends them through the
self-deployed mineru server (same protocol as mineru500_self.py: VLM backend,
3 concurrent, return_md). Sciverse .md files are ALREADY parsed text — they
skip mineru entirely and are copied to the texts/ dir as-is.

Output: corpus/texts/<paper_id>.md (the only artifact the build pipeline
consumes). Resume-safe: skips existing .md files.
"""
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from pathlib import Path

BASE = Path(r"C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/scholarqa_multi")
REPORT = BASE / "corpus" / "fetch_report.json"
TEXTS = BASE / "corpus" / "texts"
SERVER = "http://192.168.199.73:9600"
CONCURRENCY = 3

FIELDS = {
    "output_dir": "./output",
    "lang_list": "en",
    "backend": "vlm-vllm-async-engine",
    "parse_method": "auto",
    "formula_enable": "true",
    "table_enable": "true",
    "return_md": "true",
    "return_middle_json": "false",
}


def post_pdf(pdf_path):
    boundary = f"----logickg-{os.urandom(8).hex()}"
    body = []
    for k, v in FIELDS.items():
        body += [f"--{boundary}\r\n".encode(),
                 f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode(),
                 str(v).encode(), b"\r\n"]
    body += [f"--{boundary}\r\n".encode(),
             f'Content-Disposition: form-data; name="files"; filename="{os.path.basename(pdf_path)}"\r\n'.encode(),
             "Content-Type: application/pdf\r\n\r\n".encode(),
             open(pdf_path, "rb").read(), b"\r\n",
             f"--{boundary}--\r\n".encode()]
    req = urllib.request.Request(SERVER + "/file_parse", data=b"".join(body),
                                 method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def get_json(path, timeout=120):
    with urllib.request.urlopen(SERVER + path, timeout=timeout) as r:
        return json.load(r)


def parse_one(paper_id, pdf_path, fmt):
    md_path = TEXTS / f"{paper_id}.md"
    if md_path.exists() and md_path.stat().st_size > 5000:
        return "exists"
    if fmt == "md":
        # Sciverse .md files are already parsed text — just copy
        shutil.copyfile(pdf_path, md_path)
        return "copied"
    # PDF: send to mineru
    try:
        r1 = post_pdf(str(pdf_path))
        task_id = r1.get("task_id")
        if not task_id:
            return f"fail:no_task_id"
        t0 = time.time()
        while time.time() - t0 < 900:
            t = get_json(f"/tasks/{task_id}")
            st = t.get("status")
            if st == "completed":
                break
            if st in ("failed", "error"):
                return f"fail:{st}:{str(t.get('error'))[:60]}"
            time.sleep(10)
        else:
            return "fail:timeout"
        res = get_json(f"/tasks/{task_id}/result")
        results = res.get("results") or {}
        md = None
        for v in results.values():
            if isinstance(v, dict) and v.get("md_content"):
                md = v["md_content"]
                break
        if not md or len(md) < 2000:
            return f"fail:no_md(len={len(md) if md else 0})"
        md_path.write_text(md, encoding="utf-8")
        return "ok"
    except Exception as e:
        return f"fail:{repr(e)[:80]}"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    TEXTS.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    # build worklist: paper_id -> (pdf_path, fmt)
    # NOTE: fetch_report paths may be stale (pre-rename LogicKG dir, or a
    # broken relative join). All source files live in corpus/pdfs/, so
    # resolve by basename there. Dedupe: two paper_ids can share one file.
    pdfs_dir = BASE / "corpus" / "pdfs"
    work, seen = [], set()
    unresolved = []
    for v in report.values():
        if v.get("ok") and v.get("path") and v.get("format"):
            src = pdfs_dir / Path(v["path"].replace("\\", "/")).name
            if not src.exists():
                unresolved.append(v["path"])
                continue
            pid = v.get("paper_id") or src.stem
            if pid in seen:
                continue
            seen.add(pid)
            work.append((pid, src, v["format"]))
    if unresolved:
        print(f"WARNING: {len(unresolved)} ok-entries with no file in pdfs/:", flush=True)
        for p in unresolved:
            print(f"  {p}", flush=True)
    print(f"mineru: {len(work)} papers to parse "
          f"({sum(1 for _, _, f in work if f == 'md')} sciverse .md, "
          f"{sum(1 for _, _, f in work if f == 'pdf')} pdf)", flush=True)
    lock = threading.Lock()
    stats = {}

    def run(item):
        pid, path, fmt = item
        st = parse_one(pid, path, fmt)
        with lock:
            stats[st.split(":")[0]] = stats.get(st.split(":")[0], 0) + 1
            if st.startswith("fail"):
                print(f"  FAIL {pid}: {st[:80]}", flush=True)
        return pid, st

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for i, (pid, st) in enumerate(ex.map(run, work)):
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(work)}] {stats}", flush=True)
    print(f"FINAL: {stats}", flush=True)


if __name__ == "__main__":
    main()
