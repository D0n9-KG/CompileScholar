# -*- coding: utf-8 -*-
"""Tier 1 -> 2 promotion pipeline (batch-3): take a coarse-extracted
external paper the answering loop flagged as core evidence and run the
FULL CompileScholar extraction over it.

Pipeline per paper:
  1. tier store lookup (kb/growth_library.db) — paper must be coarse;
     a ready DEEP run refuses re-promotion (the whole point of the store)
  2. full-text acquisition via the sci-evo chain (arXiv PDF -> OA ->
     Sciverse -> local DOI archive), identity-verified
  3. PDF -> markdown via the self-hosted mineru server (Sciverse .md is
     copied directly — same protocol as the Multi corpus battle)
  4. per-paper kb chain on a GROWTH dir (never touches the frozen Multi
     corpus): cards -> deep_extract -> postcheck
  5. DEEP run registered in the tier store; records land in
     kb/growth/records_growth.json for the views merge (growth overlay)

Usage:
  python promote_to_deep.py --title "..." [--doi ...]     one explicit paper
  python promote_to_deep.py --next                        oldest pending candidate
  python promote_to_deep.py --list                        pending queue
  python promote_to_deep.py --title "..." --acquire-only  stop after step 3
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
_CS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(BASE))))
SRC = os.path.join(_CS, "src")
_SCIEVO = r"C:\Users\D0n9\Desktop\sci-evo-extract\src"
for _p in (SRC, _SCIEVO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

KB = os.path.join(BASE, "kb")
GROWTH = os.path.join(KB, "growth")
GROWTH_TEXTS = os.path.join(GROWTH, "texts")
MODEL = "local:Qwen3.8-27B"
MINERU_SERVER = "http://192.168.199.73:9600"
MINERU_TOKEN = "***REMOVED-MINERU_TOKEN***"

FIELDS = {"enable_formula": "true", "language": "en", "enable_table": "true",
          "model_version": "pipeline", "return_md": "true",
          "token": MINERU_TOKEN}


# ---------------- 1-2. tier store + acquisition ----------------

def get_store():
    from sci_evo_extract.library.coarse_store import TierStore
    return TierStore(os.path.join(KB, "growth_library.db"),
                     os.path.join(KB, "growth_library"))


def acquire(paper: dict, out_dir: str) -> str:
    """Full-text acquisition chain -> local file path (or raise)."""
    from sci_evo_extract.library.acquisition_chain import acquire_fulltext
    r = acquire_fulltext(title=paper["title"], doi=paper.get("doi"),
                         out_dir=out_dir, timeout_seconds=180)
    for a in r.get("attempts", []):
        print(f"  [acquire] {a['channel']}: {'ok' if a['ok'] else a.get('reason', 'miss')}",
              flush=True)
    if r["status"] != "ready":
        raise RuntimeError(f"acquisition gap: {r['attempts'][-1].get('reason')}")
    return r["path"]


# ---------------- 3. mineru parse ----------------

def parse_pdf(pdf_path: str, out_md: str) -> None:
    boundary = f"----cs-{os.urandom(8).hex()}"
    body = []
    for k, v in FIELDS.items():
        body += [f"--{boundary}\r\n".encode(),
                 f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode(),
                 str(v).encode(), b"\r\n"]
    body += [f"--{boundary}\r\n".encode(),
             f'Content-Disposition: form-data; name="files"; '
             f'filename="{os.path.basename(pdf_path)}"\r\n\r\n'.encode(),
             "Content-Type: application/pdf\r\n\r\n".encode(),
             open(pdf_path, "rb").read(), b"\r\n",
             f"--{boundary}--\r\n".encode()]
    req = urllib.request.Request(
        MINERU_SERVER + "/file_parse", data=b"".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        task_id = json.load(r).get("task_id")
    if not task_id:
        raise RuntimeError("mineru: no task_id")
    t0 = time.time()
    while time.time() - t0 < 900:
        with urllib.request.urlopen(
                MINERU_SERVER + f"/tasks/{task_id}", timeout=120) as r:
            st = json.load(r).get("status")
        if st == "completed":
            break
        if st in ("failed", "error"):
            raise RuntimeError(f"mineru {st}")
        time.sleep(10)
    else:
        raise RuntimeError("mineru timeout")
    with urllib.request.urlopen(
            MINERU_SERVER + f"/tasks/{task_id}/result", timeout=120) as r:
        results = json.load(r).get("results") or {}
    md = next((v["md_content"] for v in results.values()
               if isinstance(v, dict) and v.get("md_content")), None)
    if not md or len(md) < 2000:
        raise RuntimeError(f"mineru: no usable md (len={len(md or '')})")
    open(out_md, "w", encoding="utf-8").write(md)


# ---------------- 4. per-paper kb chain (growth dir) ----------------

def run_kb_chain(paper_id: str, pid: str) -> dict:
    """cards -> deep_extract -> postcheck on the growth dir. The frozen
    Multi corpus/kb is never touched; growth records accumulate in
    records_growth.json for the views merge."""
    os.makedirs(GROWTH_TEXTS, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
        "LLM_CALL_LOG": os.path.join(GROWTH, "ledger_growth.jsonl"),
        "LLM_RUN_ID": f"growth-{paper_id}",
        "LOCAL_MAX_CONCURRENT": "8",
        "LOCAL_SOCK_TIMEOUT": "900", "LLM_WALL_TIMEOUT": "1200",
        "LLM_PROVIDER_ALLOWLIST": "local", "KB_EMBED_PROVIDER": "local",
    })
    stages = [
        ("cards", ["python", "-m", "kb_compiler.records.cards",
                   "--texts", GROWTH_TEXTS, "--manifest",
                   os.path.join(GROWTH, "manifest_growth.json"),
                   "--out", os.path.join(GROWTH, "cards_growth.json"),
                   "--model", MODEL]),
        ("deep_extract", ["python", "-m", "kb_compiler.records.deep_extract",
                          "--texts", GROWTH_TEXTS,
                          "--manifest", os.path.join(GROWTH, "manifest_growth.json"),
                          "--cards", os.path.join(GROWTH, "cards_growth.json"),
                          "--out", os.path.join(GROWTH, "records_growth.json"),
                          "--model", MODEL, "--pool", "4"]),
        ("postcheck", ["python", "-m", "kb_compiler.records.postcheck",
                       "--records", os.path.join(GROWTH, "records_growth.json"),
                       "--texts", GROWTH_TEXTS,
                       "--out-dir", os.path.join(GROWTH, "postcheck"),
                       "--model", MODEL]),
    ]
    for name, cmd in stages:
        log = open(os.path.join(GROWTH, f"stage_{name}.log"), "a",
                   encoding="utf-8", errors="replace")
        t0 = time.time()
        p = subprocess.run(cmd, cwd=SRC, env=env, stdout=log,
                           stderr=subprocess.STDOUT)
        log.close()
        print(f"  [{name}] {'OK' if p.returncode == 0 else 'FAIL'} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if p.returncode != 0:
            raise RuntimeError(f"stage {name} failed — see growth logs")
    checked = os.path.join(GROWTH, "postcheck", "records_checked.json")
    n = len(json.load(open(checked, encoding="utf-8")))
    return {"records_checked": n, "artifacts": {
        "records": checked,
        "cards": os.path.join(GROWTH, "cards_growth.json")}}


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title")
    ap.add_argument("--doi")
    ap.add_argument("--next", action="store_true",
                    help="promote the oldest pending candidate")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--acquire-only", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store = get_store()

    if args.list:
        for st in store.pending_promotions():
            print(f"  {st['paper_id']}  {st['title']}  "
                  f"(tier={st['tier']})")
        return

    paper = None
    if args.next:
        pend = store.pending_promotions()
        if not pend:
            print("[promote] no pending candidates")
            return
        pid = pend[0]["paper_id"]
    elif args.title:
        pid = store.find(doi=args.doi, title=args.title)
        if not pid:
            # not yet coarse — register it first (abstract may be unknown
            # here; the coarse pass must have run via the answer loop)
            print("[promote] paper not in tier store — run extract_paper "
                  "first (coarse pass), or pass the abstract")
            return
    else:
        ap.error("need --title/--doi or --next")
    st = store.state(pid)
    print(f"[promote] {st['title']} (tier={st['tier']}, "
          f"candidate={st['promotion_candidate']})", flush=True)
    if st["tier"] == "deep":
        print("[promote] already deep — refusing (no re-extraction)")
        return

    # growth manifest for this paper (cards/deep_extract need one)
    os.makedirs(GROWTH, exist_ok=True)
    gm_path = os.path.join(GROWTH, "manifest_growth.json")
    gm = json.load(open(gm_path, encoding="utf-8")) if os.path.exists(gm_path) else []
    row = {"paper_id": st["paper_id"], "title": st["title"],
           "doi": st.get("doi"), "subject": "growth", "year": st.get("year")}
    gm = [r for r in gm if r["paper_id"] != row["paper_id"]] + [row]
    json.dump(gm, open(gm_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # 2-3. acquire + parse
    pdf_dir = os.path.join(GROWTH, "pdfs")
    path = acquire(row, pdf_dir)
    print(f"[acquire] {path}", flush=True)
    md_path = os.path.join(GROWTH_TEXTS, row["paper_id"] + ".md")
    os.makedirs(GROWTH_TEXTS, exist_ok=True)
    if path.lower().endswith(".md"):
        import shutil
        shutil.copyfile(path, md_path)
        print("[parse] copied (already markdown)", flush=True)
    else:
        parse_pdf(path, md_path)
        print(f"[parse] mineru -> {os.path.basename(md_path)} "
              f"({os.path.getsize(md_path)} bytes)", flush=True)
    if args.acquire_only:
        print("[done] acquire-only mode stops here")
        return

    # 4. per-paper kb chain
    res = run_kb_chain(row["paper_id"], pid)
    print(f"[kb-chain] {res['records_checked']} checked records", flush=True)

    # 5. register deep
    out = store.register_deep(paper_id=pid,
                              output_artifacts=res["artifacts"])
    print(f"[deep] registered: tier={out['tier']}", flush=True)


if __name__ == "__main__":
    main()
