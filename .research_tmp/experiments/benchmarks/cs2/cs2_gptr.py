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
                url = (f"https://doi.org/{doi}" if doi else
                       f"sciverse://{re.sub(r'[^a-z0-9]+', '-', title.lower())[:60]}")
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


def report_to_cs2(report_md: str, sources=None) -> list[dict]:
    """GPT Researcher 报告 → CS2 sections/citations。引用形态：正文 [1][2]
    编号 + 文末 sources 列表（title+url）——编号到 sources 顺序映射。"""
    # sources 解析（文末列表：- [1] title url 或markdown链接）
    src_map = {}
    for m in re.finditer(r"\[(\d+)\]\s*(.+?)(?:\n|$)", report_md):
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
        for m in _NUM_CITE.finditer(text):
            n = int(m.group(1))
            if n in seen or n not in src_map:
                continue
            seen[n] = len(cites) + 1
            info = src_map[n]
            cites.append({
                "id": f"[{seen[n]}]",
                # GPT Researcher 无 snippet 抽取——title 档（0.5）；
                # raw_content 可作 snippet（retriever 契约里有全文）
                "snippets": [],
                "title": info["title"],
                "metadata": {"url": info["url"]},
            })
        text2 = _NUM_CITE.sub(
            lambda m: f"[{seen[int(m.group(1))]}]"
            if int(m.group(1)) in seen else m.group(0), text)
        out.append({"title": title, "text": text2, "citations": cites})
    return out


def load_cs2_questions():
    src = os.path.normpath(os.path.join(
        _HERE, "..", "scholarqa_multi", "sqa2_rubrics_v1_recomputed.json"))
    qs = json.load(open(src, encoding="utf-8"))
    return [{"qid": q["case_id"][:24], "question": q["question"]}
            for q in qs[10:15]]


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

    for q in todo:
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
            report_content = researcher.write_report()
            if not isinstance(report_content, str):
                # v0.16 write_report 可能返回带 metadata 的对象——取 .content
                report_content = getattr(report_content, "content", None) or str(report_content)
            sections = report_to_cs2(report_content or "")
            done[qid] = {"qid": qid, "question": text,
                         "report_chars": len(report_content or ""),
                         "sections": sections}
            ncit = sum(len(s["citations"]) for s in sections)
            print(f"[gptr-cs2] {qid[:10]} report={len(report_content or '')}ch "
                  f"sections={len(sections)} cites={ncit}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            done[qid] = {"qid": qid, "question": text,
                         "err": f"{type(e).__name__}: {str(e)[:200]}"}
        json.dump(list(done.values()), open(ANSWERS, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

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
