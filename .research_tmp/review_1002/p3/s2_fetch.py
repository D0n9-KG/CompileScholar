"""Step2: 抓 arXiv HTML（arxiv.org/html 优先，ar5iv 兜底），抽出表格(caption+行文本)。≤1 req/s。"""
import json, os, re, sys, time, subprocess
from bs4 import BeautifulSoup
sys.path.insert(0, "."); from common import P3
os.makedirs(P3 + "/html", exist_ok=True)
cands = json.load(open(P3 + "/s1_candidates.json", encoding="utf-8"))
def get(url):
    r = subprocess.run(["curl", "-s", "-L", "--max-time", "60", "-A", "Mozilla/5.0 (academic research probe)", "-w", "\n%{http_code}", url],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    body, _, code = r.stdout.rpartition("\n")
    return code.strip(), body
def tables_of(html):
    soup = BeautifulSoup(html, "lxml"); out = []
    for i, fig in enumerate(soup.select("figure.ltx_table, figure.ltx_float")):
        tabs = fig.find_all("table")
        if not tabs: continue
        cap = fig.find(["figcaption"]) 
        cap = " ".join(cap.get_text(" ").split()) if cap else ""
        rows = []
        for tab in tabs:
            for tr in tab.find_all("tr"):
                cells = [" ".join(td.get_text(" ").split()) for td in tr.find_all(["td", "th"])]
                if any(cells): rows.append(" | ".join(cells))
        if rows: out.append({"table_id": f"T{i}", "caption": cap[:600], "rows": rows[:80]})
    return out
log = []
for k, c in enumerate(cands):
    pid = c["id"]; fn = f"{P3}/html/{pid.replace('/','_')}.json"
    if os.path.exists(fn): log.append(json.load(open(fn, encoding="utf-8"))["src"]); continue
    res = {"id": pid, "src": None, "tables": []}
    for src, url in (("arxiv", f"https://arxiv.org/html/{pid}"), ("ar5iv", f"https://ar5iv.labs.arxiv.org/html/{pid}")):
        code, body = get(url); time.sleep(1.1)
        if code == "200" and "ltx_" in body:
            t = tables_of(body)
            res = {"id": pid, "src": src, "tables": t}
            break
    json.dump(res, open(fn, "w", encoding="utf-8"), ensure_ascii=False)
    log.append(res["src"])
    if k % 10 == 0: print(k, pid, res["src"], len(res["tables"]), flush=True)
from collections import Counter
print("fetch:", Counter(log))
