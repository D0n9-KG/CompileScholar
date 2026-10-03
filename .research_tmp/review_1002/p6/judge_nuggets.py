"""DSB nugget_coverage 判分（官方 Nuggetizer.assign 原样调用：同 prompt/窗口 10/模型/温度 0），
补两件官方 evaluator 不做的事：①保存每题每个 nugget 的 assignment 明细 ②题间并行+调用计数。
官方 nugget_coverage = strict_all_score（只计 support）。gt 用 res.json 的 supported_nuggets 与 query。

用法：
  python judge_nuggets.py --system B0 --texts gen/B0 [--ids 0 1 ...] [--workers 8] [--tag run1]
  texts 目录下每题一个 <gt_dir>.md（纯正文）。输出 judged/<system>[_tag].json
"""
import argparse, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

DSB = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\deepscholar\dsb"
sys.path.insert(0, os.path.join(DSB, "eval", "nuggetizer", "src"))
ENV = r"C:\Users\D0n9\Desktop\CompileScholar\.env"


def _env(k):
    for line in open(ENV, encoding="utf-8"):
        m = re.match(r"^\s*" + k + r"\s*=\s*(.+?)\s*$", line)
        if m:
            return m.group(1)


os.environ["OPENAI_API_KEY"] = _env("PARATERA_API_KEY")
os.environ["OPENAI_API_BASE"] = _env("PARATERA_BASE_URL")

from nuggetizer.core.types import ScoredNugget  # noqa: E402
from nuggetizer.models.nuggetizer import Nuggetizer  # noqa: E402
from nuggetizer.core.metrics import calculate_nugget_scores  # noqa: E402

MODEL = "DeepSeek-V4.1-Flash"
CALLS = {"n": 0, "err": 0}
_lock = threading.Lock()
_local = threading.local()


def _nuggetizer():
    if getattr(_local, "nz", None) is None:
        nz = Nuggetizer(model=MODEL, log_level=0)
        orig = nz.assigner_llm.client.chat.completions.create

        def counted(*a, **k):
            with _lock:
                CALLS["n"] += 1
            try:
                return orig(*a, **k)
            except Exception:
                with _lock:
                    CALLS["err"] += 1
                raise
        nz.assigner_llm.client.chat.completions.create = counted
        _local.nz = nz
    return _local.nz


def judge_one(gt_dir, text):
    gt = json.load(open(os.path.join(DSB, "dataset", "gt_nuggets_outputs", gt_dir, "res.json"), encoding="utf-8"))
    nuggets = [ScoredNugget(text=n["text"], importance=n.get("importance", "vital"))
               for n in gt.get("supported_nuggets", [])]
    t0 = time.time()
    assigned = _nuggetizer().assign(gt.get("query", ""), text or "", nuggets)
    nl = [{"text": n.text, "importance": n.importance, "assignment": n.assignment} for n in assigned]
    m = calculate_nugget_scores(gt_dir, nl)
    return {"gt_dir": gt_dir, "qid": gt["qid"], "nugget_coverage": m.strict_all_score,
            "strict_vital": m.strict_vital_score, "all_partial": m.all_score,
            "vital_partial": m.vital_score, "n_nuggets": len(nl),
            "n_failed": sum(1 for x in nl if x["assignment"] == "failed"),
            "words": len((text or "").split()), "judge_s": round(time.time() - t0, 1),
            "nuggets": nl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ids = a.ids or sorted([f[:-3] for f in os.listdir(a.texts)
                           if f.endswith(".md") and f[:-3].isdigit()], key=int)
    out_path = os.path.join("judged", f"{a.system}{('_' + a.tag) if a.tag else ''}.json")
    os.makedirs("judged", exist_ok=True)
    done = {}
    if os.path.exists(out_path):
        done = {r["gt_dir"]: r for r in json.load(open(out_path, encoding="utf-8"))}
    todo = [i for i in ids if i not in done]
    print(f"[judge {a.system}] {len(done)} done, {len(todo)} todo", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(judge_one, i, open(os.path.join(a.texts, i + ".md"), encoding="utf-8").read()): i for i in todo}
        for f in as_completed(futs):
            i = futs[f]
            try:
                r = f.result()
            except Exception as e:
                print(f"  {i} ERROR {str(e)[:120]}", flush=True)
                continue
            with _lock:
                done[i] = r
                json.dump(sorted(done.values(), key=lambda x: int(x["gt_dir"])),
                          open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"  {i} nc={r['nugget_coverage']:.3f} n={r['n_nuggets']} failed={r['n_failed']} "
                  f"{r['judge_s']}s calls={CALLS['n']}", flush=True)
    rs = [done[i] for i in ids if i in done]
    if rs:
        print(f"[judge {a.system}] n={len(rs)} mean nugget_coverage={sum(r['nugget_coverage'] for r in rs)/len(rs):.4f} "
              f"calls={CALLS['n']} call_errors={CALLS['err']}", flush=True)


if __name__ == "__main__":
    main()
