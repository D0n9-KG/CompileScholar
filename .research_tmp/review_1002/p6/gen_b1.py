"""B1「DeepScholar-base 式」：官方 DeepScholar-base 的 intro 生成阶段（官方评分对象 intro.md）原样跑在 oracle
文献集上——跳过 search/filter（oracle 已给金标文献），generate_intro_section → generate_section_summary_with_citations
→ lotus sem_agg（官方 section_writer_instructions + intro_section_instructions + citation_guidelines）。LM=本地 27B。
docs_df 列与官方 search 产出一致：title/url/snippet/date/authors/query/context（context 格式照抄 agentic_search.py:100）。
输出 gen/B1/<gt_dir>.md：官方 _postprocess_citation 产出的 markdown 链接再按 DeepScholarBaseParser 规则转回 [n]。"""
import asyncio, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

DSB = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\deepscholar\dsb"
sys.path.insert(0, DSB)
os.environ["NO_PROXY"] = "192.168.199.73,localhost,127.0.0.1"
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)
ENV = r"C:\Users\D0n9\Desktop\CompileScholar\.env"


def _env(k):
    for line in open(ENV, encoding="utf-8"):
        m = re.match(r"^\s*" + k + r"\s*=\s*(.+?)\s*$", line)
        if m:
            return m.group(1)


import pandas as pd  # noqa: E402
import lotus  # noqa: E402
from lotus.models import LM  # noqa: E402
from deepscholar_base.final_generation import generate_intro_section  # noqa: E402
from deepscholar_base.configs import Configs  # noqa: E402

_base = _env("LOCAL_BASE_URL").rstrip("/")
BASE = _base if _base.endswith("/v1") else _base + "/v1"
OUT = os.path.join("gen", "B1")


def make_lm():
    return LM(model="openai/Qwen3.8-27B", api_base=BASE, api_key=_env("LOCAL_API_KEY"),
              max_tokens=6000, temperature=0.3, max_ctx_len=32000,
              extra_body={"chat_template_kwargs": {"enable_thinking": False}}, timeout=900)


def docs_df_for(row):
    recs = []
    for i, r in enumerate(row["refs"], 1):
        url = r["url"] or f"https://ref.local/{i}"
        snip = r["abstract"] or ""
        recs.append({"title": r["title"], "url": url, "snippet": snip, "date": None,
                     "authors": r["authors"], "query": row["query"],
                     "context": f"{r['title']}[{url}]: {snip}"})
    return pd.DataFrame(recs)


# 官方 _postprocess_citation 产出 "\[[作者' 日期](url)\]"；作者串可含换行与 "]"（BibTeX 作者表），
# 所以按 "](http" 锚定 url，再向左吃到最近的 "\[[" 外壳。
from citeconv import to_numbered as _to_numbered  # noqa: E402


def to_numbered(text, df):
    return _to_numbered(text)


def one(row):
    t0 = time.time()
    lm = make_lm()
    # 坑：Configs.initialize_lms 派生各阶段 LM 时 pop 掉 max_completion_tokens→回落 512 截断；显式给齐
    cfg = Configs(lm=lm, generation_lm=lm, taxonomize_lm=lm, filter_lm=lm, search_lm=lm)
    df = docs_df_for(row)
    intro = asyncio.run(generate_intro_section(row["query"], df, "", cfg))
    text = to_numbered(intro, df)
    open(os.path.join(OUT, row["gt_dir"] + ".md"), "w", encoding="utf-8").write(text)
    open(os.path.join(OUT, row["gt_dir"] + ".raw.md"), "w", encoding="utf-8").write(intro)
    return row["gt_dir"], len(text.split()), round(time.time() - t0, 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = json.load(open("oracle_inputs.json", encoding="utf-8"))
    ids = set(sys.argv[1:])
    rows = [r for r in rows if (not ids or r["gt_dir"] in ids)
            and not os.path.exists(os.path.join(OUT, r["gt_dir"] + ".md"))]
    print(f"[B1] todo {len(rows)}", flush=True)
    log = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(one, r): r["gt_dir"] for r in rows}
        for f in as_completed(futs):
            try:
                d, w, s = f.result()
                log.append({"gt_dir": d, "words": w, "gen_s": s})
                print(f"  {d} words={w} {s}s", flush=True)
            except Exception as e:
                print(f"  {futs[f]} ERROR {str(e)[:200]}", flush=True)
    old = json.load(open("gen_b1_log.json")) if os.path.exists("gen_b1_log.json") else []
    json.dump(old + log, open("gen_b1_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
