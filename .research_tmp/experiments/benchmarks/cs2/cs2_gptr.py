# -*- coding: utf-8 -*-
"""CS2 GPT Researcher 臂——成熟 DR 开源产品代表（三臂格局更新：STORM 在
CS2 形态错配降级 DSB 专用，GPT Researcher 补 CS2 第三臂）。

公平性契约（与 ours/harness/STORM 同款）：
  - LLM：本地 GPUStack Qwen3.8-27B（OPENAI_BASE_URL 直连——官方支持）
  - 检索：Sciverse 同通道（CustomRetriever 插件式接入——返回
    {url, raw_content} 契约，raw_content=Sciverse 摘要全文）
  - 判分：同一套 direct_judge；出口适配 [1][2] 编号引用 → CS2 JSON

GPT Researcher pipeline（官方默认不动）：
  planner 生成子问题 → 并行执行 agents 检索+浏览 → 聚合成报告
  （report_type=research_report 默认形态）

用法：python cs2_gptr.py --smoke / python cs2_gptr.py
"""
import argparse
import asyncio
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(
    _HERE, "..", "_shared", "tools")))
sys.path.insert(0, _HERE)
os.environ.setdefault("LOCAL_MAX_CONCURRENT", "16")

# GPUStack 代理豁免（NO_PROXY——gptr 走 httpx/openai SDK）
import gpustack_httpx_fix  # noqa: F401

BASE_DIR = os.path.join(_HERE, "arm_gptr")
ANSWERS = os.path.join(BASE_DIR, "answers_gptr_cs2.json")

MODEL = "Qwen3.8-27B"


# ---------------------------------------------------------------------------
# Sciverse retriever（GPT Researcher CustomRetriever 契约）
# ---------------------------------------------------------------------------

class SciverseRetriever:
    """GPT Researcher retriever 插件：query → [{url, raw_content}]。
    requires_scraping=False（内容已随检索返回，无需再抓网页——与官方
    custom retriever 契约一致）。"""

    requires_scraping = False

    def __init__(self, query: str, query_domains=None):
        self.query = query
        from external_tools import ExternalTools
        self._ext = ExternalTools({}, {}, "local:Qwen3.8-27B")

    def search(self, max_results: int = 5):
        try:
            r = self._ext.search_papers(self.query, max_results)
            out = []
            for p in (r.get("papers") or [])[:max_results]:
                title = p.get("title") or "untitled"
                doi = p.get("doi")
                # 第 11 坑配套：url 必须是真 http(s)——报告里 md 链接引用
                # 会带着 url 进正文/判分，sciverse:// 假协议伤害引用
                # 可信度。无 doi 用 Google Scholar 检索 URL（稳定可点）。
                url = (f"https://doi.org/{doi}" if doi else
                       "https://scholar.google.com/scholar?q=" +
                       re.sub(r"\s+", "+", title[:80]))
                content = (f"{title}. {p.get('abstract') or ''}").strip()
                if len(content) < 80:
                    continue   # 无摘要的命中不成证据
                out.append({"url": url, "raw_content": content})
            return out
        except Exception as e:
            print(f"[sciverse-retriever] {str(e)[:100]}", flush=True)
            return []


# ---------------------------------------------------------------------------
# 配置 + 出口适配
# ---------------------------------------------------------------------------

def _env_from_dotenv():
    env_path = os.path.normpath(os.path.join(
        _HERE, "..", "..", "..", "..", ".env"))
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() in ("LOCAL_BASE_URL", "LOCAL_API_KEY",
                                 "SCIVERSE_API_KEY"):
                    os.environ.setdefault(k.strip(), v.strip())


def build_config():
    from gpt_researcher.config.config import Config
    _env_from_dotenv()
    # API 路由走环境变量（litellm/openai SDK 读 OPENAI_*）——27B 直连 GPUStack
    os.environ["OPENAI_API_KEY"] = os.environ.get("LOCAL_API_KEY", "")
    os.environ["OPENAI_BASE_URL"] = os.environ.get("LOCAL_BASE_URL", "")
    c = Config()
    # v0.16 Config 是属性对象（无 kwargs 构造）——setattr 注入
    c.fast_llm_provider = "openai"
    c.smart_llm_provider = "openai"
    c.fast_llm_model = MODEL
    c.smart_llm_model = MODEL
    c.strategic_llm_model = MODEL   # 默认 gpt-5.4（独立字段不随 smart 覆盖）
    c.strategic_llm_provider = "openai"
    c.retriever = "custom"            # SciverseRetriever（插件点注入）
    # v0.16: cfg.retrievers（复数列表）优先于 cfg.retriever——两个都设
    c.retrievers = ["custom"]
    c.max_search_results_per_query = 5
    c.total_words = 800
    return c


_NUM_CITE = re.compile(r"\[(\d+)\]")
# 第 10 坑（09-30）：v0.16 report prompt 用 APA markdown 超链接引用
# （([in-text citation](url)) 句尾形态），不是 [1] 编号——旧 parser 只认
# 编号 → cites=0。md 链接正则（普通链接+括号包裹链接两种形态）。
_MD_CITE = re.compile(r"\(?\[([^\]\[]{2,120})\]\((https?://[^)\s]+)\)?\)?")
_MD_CITE_PAREN = re.compile(r"\(\[([^\]\[]{2,120})\]\((https?://[^)\s]+)\)\)")


def report_to_cs2(report_md: str, sources=None) -> list[dict]:
    """GPT Researcher 报告 → CS2 sections/citations。两种引用形态：
    ① 正文 [1][2] 编号 + 文末 sources 列表（v0.15-）
    ② v0.16 APA markdown 超链接（([in-text citation](url)) 句尾）
    url 是稳定键：同 url 多次引用共用一条，title 取首次出现的链接文字。"""
    # 形态②：markdown 链接直接建 url→citation 映射
    url_map = {}   # url -> {"title":..., "n": seq}
    for m in _MD_CITE_PAREN.finditer(report_md):
        t, u = m.group(1).strip(), m.group(2)
        if u not in url_map and t and not t.lower().startswith("http"):
            url_map[u] = {"title": t[:120]}
    for m in re.finditer(r"\[([^\]\[]{2,120})\]\((https?://[^)\s]+)\)",
                         report_md):
        t, u = m.group(1).strip(), m.group(2)
        if u not in url_map and t and not t.lower().startswith("http"):
            url_map[u] = {"title": t[:120]}
    # 形态①：正文编号 + 文末 sources 列表（行首锚定——正文句中 [1]
    # 不是 sources 行）
    src_map = {}
    for m in re.finditer(r"^\s*\[(\d+)\][\s\-–]*(.+?)$",
                         report_md, re.M):
        n, rest = int(m.group(1)), m.group(2).strip()
        if n in src_map or not rest:
            continue
        title = re.sub(r"\[|\]\(.*?\)|https?://\S+", "", rest).strip()[:120]
        url_m = re.search(r"(https?://\S+)", rest)
        src_map[n] = {"title": title or f"source_{n}",
                      "url": url_m.group(1) if url_m else ""}
    # 分节
    sections = []
    cur_title, cur_lines = "Answer", []
    for line in report_md.splitlines():
        if line.startswith("## "):
            if any(l.strip() for l in cur_lines):
                sections.append((cur_title, "\n".join(cur_lines)))
            cur_title, cur_lines = line[3:].strip() or "Section", []
        else:
            cur_lines.append(line)
    if any(l.strip() for l in cur_lines):
        sections.append((cur_title, "\n".join(cur_lines)))

    out = []
    for title, text in sections:
        if title.lower() in ("references", "reference", "sources",
                             "table of contents"):
            continue
        if not text.strip():
            continue
        cites, seen = [], {}
        # 形态②：markdown 超链接引用（v0.16）——url 为键，节内去重，
        # 重编号 [1]..[n]；链接文字保留在正文（原样不动，judges 可见）
        for m in _MD_CITE.finditer(text):
            u = m.group(2)
            if u in seen:
                continue
            info = url_map.get(u) or {"title": u[:100]}
            if u not in url_map:
                url_map[u] = info
            seen[u] = len(cites) + 1
            cites.append({
                "id": f"[{seen[u]}]",
                "snippets": [],
                "title": info["title"],
                "metadata": {"url": u},
            })
        # 形态①：[n] 编号引用
        for m in _NUM_CITE.finditer(text):
            n = int(m.group(1))
            if ("md", n) in seen or n not in src_map:
                continue
            seen[("md", n)] = len(cites) + 1
            info = src_map[n]
            cites.append({
                "id": f"[{seen[('md', n)]}]",
                # GPT Researcher 无 snippet 抽取——title 档（0.5）；
                # raw_content 可作 snippet（retriever 契约里有全文）
                "snippets": [],
                "title": info["title"],
                "metadata": {"url": info["url"]},
            })
        text2 = _NUM_CITE.sub(
            lambda m: f"[{seen[('md', int(m.group(1)))]}]"
            if ("md", int(m.group(1))) in seen else m.group(0), text)
        out.append({"title": title, "text": text2, "citations": cites})
    return out


def load_cs2_questions():
    src = os.path.normpath(os.path.join(
        _HERE, "..", "scholarqa_multi", "sqa2_rubrics_v1_recomputed.json"))
    qs = json.load(open(src, encoding="utf-8"))
    # 全量 100 题（GPTR 臂与 ours/harness 同尺）；GPTR_LIMIT 可截断
    n = int(os.environ.get("GPTR_LIMIT", str(len(qs))))
    return [{"qid": q["case_id"][:24], "question": q["question"]}
            for q in qs[:n]]


async def run(smoke: bool = False):
    from gpt_researcher import GPTResearcher
    from gpt_researcher.retrievers.custom.custom import CustomRetriever
    # 注入 Sciverse 版 CustomRetriever（同位置替换——官方插件点）
    import gpt_researcher.retrievers.custom.custom as _cust
    _cust.CustomRetriever = SciverseRetriever

    os.makedirs(BASE_DIR, exist_ok=True)
    # env 路由（原 build_config 内——cfg 直改方案后 run 也要设置）
    _env_from_dotenv()
    os.environ["OPENAI_API_KEY"] = os.environ.get("LOCAL_API_KEY", "")
    os.environ["OPENAI_BASE_URL"] = os.environ.get("LOCAL_BASE_URL", "")
    # 第 12 坑（09-30 冒烟实锤×2）：gpt_researcher 的 openai provider 分支
    # ChatOpenAI(**kwargs) 不传 request_timeout（langchain 默认 None=无限
    # 等）——单请求可挂 90+ 分钟（进程活 CPU 微动但永不返回）。openrouter
    # 分支有 180s 默认而 openai 分支漏了。LLM_KWARGS 是官方逃生舱，直通
    # provider_kwargs → ChatOpenAI 构造器。
    # 第 13 坑（同日确诊）：27B 默认开思考模式——max_tokens 被思考内容
    # 吃光后 content=null（实测 2000 token 预算→9643 字符思考→0 输出），
    # 全部 "LLM returned empty response" 的根因。extra_body 关思考
    # （langchain ChatOpenAI 透传机制），实测关后 8410ch 正常输出。
    os.environ.setdefault("LLM_KWARGS",
                          '{"request_timeout": 180, "extra_body": '
                          '{"chat_template_kwargs": {"enable_thinking": false}}}')
    questions = load_cs2_questions()
    if smoke:
        questions = questions[:1]
    done = {}
    if os.path.exists(ANSWERS):
        done = {r["qid"]: r for r in
                json.load(open(ANSWERS, encoding="utf-8"))}
    todo = [q for q in questions if q["qid"] not in done]
    print(f"[gptr-cs2] {len(todo)} to answer ({len(done)} resumed)",
          flush=True)

    # 题间并行（09-30 实测加）：GPUStack 27B 压测显示 12 路并发吞吐线性
    # 扩展（447 tok/s）远未饱和；GPTR 原串行 100 题要 2-3 天。每题独立
    # 无共享状态，天然可并行。GPTR_FANOUT 控制（缺省 4——与 ours 臂
    # 同量级，两者合计 8-9 路仍在服务端舒适区）。answers 逐题落盘
    # （resume 兼容）；写盘串行化（同一 asyncio 锁）。
    fanout = int(os.environ.get("GPTR_FANOUT", "4"))
    write_lock = __import__("asyncio").Lock()

    async def one(q):
        qid, text = q["qid"], q["question"]
        try:
            researcher = GPTResearcher(
                query=text, report_type="research_report",
                report_format="markdown", report_source="web")
            # GPTResearcher.__init__ 不收 config 参数（内部 Config(config_path)
            # 重读默认）——构造后直接改 researcher.cfg（v0.16 唯一注入点）
            cfg = researcher.cfg
            cfg.fast_llm_provider = "openai"
            cfg.smart_llm_provider = "openai"
            cfg.fast_llm_model = MODEL
            cfg.smart_llm_model = MODEL
            cfg.strategic_llm_model = MODEL
            cfg.strategic_llm_provider = "openai"
            cfg.retriever = "custom"
            cfg.retrievers = ["custom"]
            # agent.__init__ 已把 retrievers 固化成类列表（构造后改 cfg 太晚）
            # ——直接替换 researcher.retrievers 列表为 SciverseRetriever
            researcher.retrievers = [SciverseRetriever]
            cfg.max_search_results_per_query = 5
            cfg.total_words = 800
            await researcher.conduct_research()
            # 第 9 坑（09-30）：v0.16 write_report 是 async coroutine——不
            # await 拿到的是 coroutine 对象，getattr(.content) 恒 None →
            # str(coroutine) 67ch 假报告。await 之。
            report_content = researcher.write_report(
                # 第 11 坑（09-30 批37 题实测）：27B 不遵守 v0.16 的 APA
                # markdown 超链接引用格式——37 题全零引用。custom_prompt
                # 强制引用指令（write_report 透传到 generate_report）。
                custom_prompt=(
                    "CITATION REQUIREMENT (mandatory): every factual claim "
                    "MUST end with a markdown hyperlink citation in the "
                    "form ([Source Title](url)) where Source Title is the "
                    "exact title of a retrieved source document and url is "
                    "its url from the context. A claim without a citation "
                    "is a defect. Include citations for statistics, "
                    "findings, and attributions alike."))
            if hasattr(report_content, "__await__"):
                report_content = await report_content
            if not isinstance(report_content, str):
                # v0.16 write_report 可能返回带 metadata 的对象——取 .content
                report_content = getattr(report_content, "content", None) or str(report_content)
            sections = report_to_cs2(report_content or "")
            async with write_lock:
                done[qid] = {"qid": qid, "question": text,
                             "report_chars": len(report_content or ""),
                             "sections": sections}
                json.dump(list(done.values()), open(ANSWERS, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
            ncit = sum(len(s["citations"]) for s in sections)
            print(f"[gptr-cs2] {qid[:10]} report={len(report_content or '')}ch "
                  f"sections={len(sections)} cites={ncit}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            async with write_lock:
                done[qid] = {"qid": qid, "question": text,
                             "err": f"{type(e).__name__}: {str(e)[:200]}"}
                json.dump(list(done.values()), open(ANSWERS, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)

    import asyncio
    sem = asyncio.Semaphore(fanout)

    async def bounded(q):
        async with sem:
            await one(q)

    await asyncio.gather(*[bounded(q) for q in todo])

    judge_input = [{"qid": r["qid"], "question": r["question"],
                    "sections": r["sections"]}
                   for r in done.values() if r.get("sections")]
    jip = os.path.join(BASE_DIR, "judge_input_gptr_cs2.json")
    json.dump(judge_input, open(jip, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[gptr-cs2] judge input: {jip} ({len(judge_input)} rows)",
          flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(smoke=args.smoke))


if __name__ == "__main__":
    main()
