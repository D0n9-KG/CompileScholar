# -*- coding: utf-8 -*-
"""Batch-acquire the remaining ML succession papers via sci-evo-extract API.
Flow per paper (validated on Double DQN): POST acquisition -> find the ready
arxiv candidate -> confirm it (downloads PDF + runs MinerU VLM)."""
import json, time, urllib.request

API = "http://127.0.0.1:8000/api/library"
DOIS = {
    "10.48550/arxiv.1511.06581": "Dueling",
    "10.48550/arxiv.1511.05952": "Prioritized Experience Replay",
    "10.48550/arxiv.1602.01783": "A3C",
    "10.48550/arxiv.1710.02298": "Rainbow",
}


def req(method, path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


for doi, name in DOIS.items():
    print(f"=== {name} ({doi}) ===", flush=True)
    d = req("POST", "/acquisitions", {"doi": doi, "process": True})
    pid = d["paper_id"]
    # fetch candidates, find arxiv-ready one
    time.sleep(2)
    p = req("GET", f"/papers/{pid}")
    cand = next((c for c in p["source_candidates"]
                 if c["source_name"] == "arxiv" and c["status"] == "ready"), None)
    if not cand:
        print(f"  no ready arxiv candidate; statuses: "
              f"{[(c['source_name'], c['status']) for c in p['source_candidates']]}", flush=True)
        continue
    print(f"  confirming {cand['candidate_id']} (MinerU runs, may take minutes)...", flush=True)
    r = req("POST", f"/papers/{pid}/source-candidates/{cand['candidate_id']}/confirm",
            {"process": True}, timeout=600)
    print(f"  status={r.get('status')} pdf={r.get('pdf_asset_id')}", flush=True)
print("DONE")
