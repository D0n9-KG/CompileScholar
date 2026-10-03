"""Step3: 结果/消融表 → 受控对比元组（本地 27B）。紧凑格式：列模式(数据集/指标/设置)+每行(方法/角色/值列表)，代码里展平。"""
import json, os, re, sys, glob, concurrent.futures as cf
sys.path.insert(0, "."); from common import P3, local27b, parse_json
KEEP = re.compile(r"perform|result|compar|ablat|overall|recommend|effect|baseline|accura|HR@|NDCG|Recall|MRR", re.I)
PROMPT = """You are reading ONE table from a recommender-systems paper and must convert it into a compact machine-readable form.
Paper title: {title}
Table caption: {caption}
Table rows (cells separated by |):
{rows}

Return JSON only:
{{"is_result_table": true|false,
  "columns": [{{"col": <int index of the value column in your row arrays, from 0>, "dataset": "...", "metric": "e.g. NDCG@10", "higher_is_better": true|false, "setting": "condition that must match for comparability, e.g. full-ranking / sampled-100 negatives / backbone / dim; '' if none"}} , ...],
  "rows": [{{"method": "exact name as written", "role": "proposed|baseline|ablation", "values": [numbers or null, aligned with columns order]}}, ...]}}
Rules:
- is_result_table=false (columns=[], rows=[]) for dataset statistics, hyperparameter, notation or runtime-only tables.
- If datasets are stacked vertically (dataset/metric as row labels, methods as columns), TRANSPOSE: each method becomes a row, each (dataset,metric) a column.
- Exclude improvement %, gain, p-value, std-dev columns/rows. Strip % signs and asterisks/daggers from numbers.
- Keep at most 24 columns (prefer @10/@20 metrics) and at most 30 rows.
- Copy method and dataset names exactly as written."""
def flatten(pid, tid, j):
    out = []
    if not j or not j.get("is_result_table"): return out
    cols = j.get("columns") or []
    for r in j.get("rows") or []:
        vals = r.get("values") or []
        for k, c in enumerate(cols):
            if k >= len(vals) or vals[k] is None: continue
            try: v = float(str(vals[k]).replace("%", "").replace("*", ""))
            except Exception: continue
            out.append({"paper": pid, "table_id": tid, "method": r.get("method"), "role": r.get("role"),
                        "dataset": c.get("dataset"), "metric": c.get("metric"), "setting": c.get("setting") or "",
                        "higher_is_better": c.get("higher_is_better", True), "value": v})
    return out
def jobs_list():
    jobs = []
    for fn in glob.glob(P3 + "/html/*.json"):
        d = json.load(open(fn, encoding="utf-8"))
        for t in d["tables"]:
            txt = t["caption"] + " " + " ".join(t["rows"][:3])
            if not KEEP.search(txt): continue
            if sum(bool(re.search(r"\d\.\d", r)) for r in t["rows"]) < 2: continue
            jobs.append((d["id"], t))
    return jobs
def run(job, pap):
    pid, t = job; out = f"{P3}/tuples/{pid.replace('/','_')}__{t['table_id']}.json"
    if os.path.exists(out): return "cached"
    rows = "\n".join(t["rows"][:60])[:9000]
    raw = local27b(PROMPT.format(title=pap.get(pid, {}).get("title", ""), caption=t["caption"], rows=rows), max_tokens=6000, timeout=900)
    try: j = parse_json(raw)
    except Exception: j = None
    tup = flatten(pid, t["table_id"], j) if j else []
    json.dump({"paper": pid, "table_id": t["table_id"], "caption": t["caption"], "rows": t["rows"][:60], "raw_ok": j is not None,
               "result": j, "tuples": tup}, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    return "ok" if j else "parse_fail"
if __name__ == "__main__":
    pap = {c["id"]: c for c in json.load(open(P3 + "/s1_candidates.json", encoding="utf-8"))}
    os.makedirs(P3 + "/tuples", exist_ok=True)
    jobs = jobs_list(); print("tables to extract:", len(jobs), flush=True)
    from collections import Counter
    c = Counter()
    with cf.ThreadPoolExecutor(12) as ex:
        for i, r in enumerate(ex.map(lambda jb: run(jb, pap), jobs)):
            c[r] += 1
            if i % 25 == 0: print(i, dict(c), flush=True)
    print("extract:", dict(c))
