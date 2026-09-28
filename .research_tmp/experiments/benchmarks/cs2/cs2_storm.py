# -*- coding: utf-8 -*-
"""CS2 STORM 臂——成熟 DR/综述生成开源产品代表（三范式第三臂）。

公平性契约：
  - STORM 官方默认配置不动（max_conv_turn=3/max_perspective=3/
    max_search_queries_per_turn=3/search_top_k=3/brief 模式/四阶段管线）
  - 仅两处替换（均为官方扩展点）：
    ① 检索器 YouRM → SciverseRM（学术检索替换通用 web 搜索；
       snippets 语义镜像 YouRM——搜索结果摘要片段级，不喂全文）
    ② LM 全组件 → 本地 GPUStack Qwen3.8-27B（LitellmModel+thinking off）
  - 判分：同一套 direct_judge；出口适配 storm_gen_article.md 的
    [title](url) inline citation → CS2 sections/citations JSON

用法：
  python cs2_storm.py --smoke      # 1 题冒烟
  python cs2_storm.py              # 5 题（批 15 同题）
"""
import argparse
import json
import os
import re
import sys

os.environ.setdefault("LOCAL_MAX_CONCURRENT", "16")

import dspy  # noqa: E402  (STORM retriever 基类)

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.normpath(os.path.join(_HERE, "..", "_shared", "tools"))
_SRC = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "..", "src"))
BASE_DIR = os.path.join(_HERE, "arm_storm")
ANSWERS = os.path.join(BASE_DIR, "answers_storm_cs2.json")

sys.path.insert(0, _SRC)
sys.path.insert(0, _SHARED)
os.environ.setdefault("LOCAL_MAX_CONCURRENT", "16")

import dspy  # noqa: E402  (STORM retriever 基类)

# GPUStack httpx keep-alive bug（详见 gpustack_httpx_fix.py 文档字符串）：
# litellm 走 httpx 对 GPUStack 稳定 502——必须在 knowledge_storm 导入前挂补丁
sys.path.insert(0, _HERE)
import gpustack_httpx_fix  # noqa: E402,F401

MODEL = "Qwen3.8-27B"


# ---------------------------------------------------------------------------
# SciverseRM —— STORM 官方 Retriever 扩展点的学术检索器
# ---------------------------------------------------------------------------

class SciverseRM(dspy.Retrieve):
    """dspy.Retrieve 子类，接口与 YouRM 完全同构：forward(query) 返回
    [{title, url, snippets, description}]。snippets=Sciverse 摘要级命中
    （镜像 YouRM 的搜索摘要片段语义——不喂全文，保持 STORM 原生证据
    形态）。"""

    def __init__(self, k=3):
        super().__init__(k=k)
        self.usage = 0

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"SciverseRM": usage}

    def forward(self, query_or_queries, exclude_urls=None):
        from external_tools import ExternalTools
        ext = _get_ext()
        queries = ([query_or_queries]
                   if isinstance(query_or_queries, str) else query_or_queries)
        self.usage += len(queries)
        collected = []
        for query in queries:
            try:
                r = ext.search_papers(query, self.k)
                for p in (r.get("papers") or [])[:self.k]:
                    title = p.get("title") or "untitled"
                    # url：有 doi 用 doi.org，否则用 Sciverse 文档标识
                    doi = p.get("doi")
                    url = (f"https://doi.org/{doi}" if doi
                           else f"sciverse://{re.sub(r'[^a-z0-9]+', '-', title.lower())[:60]}")
                    if url in (exclude_urls or []):
                        continue
                    snippets = []
                    abstract = (p.get("abstract") or "").strip()
                    if abstract:
                        # 镜像 YouRM snippets 形态：片段列表（搜索摘要）
                        snippets = [abstract[:1000]]
                    if not snippets:
                        continue   # 无摘要的命中不成证据（YouRM 同行为）
                    collected.append({
                        "title": title,
                        "url": url,
                        "snippets": snippets,
                        "description": (p.get("source") or "sciverse"),
                    })
            except Exception as e:
                import logging
                logging.error(f"SciverseRM query {query!r}: {e}")
        # YouRM 同款返回形态：裸 list[dict]（Retriever.retrieve 直接迭代
        # dict——包 dspy.Prediction 会让迭代 yield 字段名，YouRM 实测
        # 返回裸列表）
        return collected


_EXT = None


def _get_ext():
    global _EXT
    if _EXT is None:
        from external_tools import ExternalTools
        _EXT = ExternalTools({}, {}, "local:Qwen3.8-27B")
    return _EXT


# ---------------------------------------------------------------------------
# LM 配置：全组件本地 27B（LitellmModel 官方扩展点）
# ---------------------------------------------------------------------------

def build_lm_configs():
    from knowledge_storm import STORMWikiLMConfigs
    from knowledge_storm.lm import LitellmModel
    import inspect

    # .env 加载（"KEY = value" 形态，strip 容错）
    env_path = os.path.join(_SRC, "..", ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() in ("LOCAL_BASE_URL", "LOCAL_API_KEY"):
                    os.environ.setdefault(k.strip(), v.strip())

    base = (os.environ.get("LOCAL_BASE_URL")
            or "http://127.0.0.1:8000/v1").rstrip("/")
    key = os.environ.get("LOCAL_API_KEY", "local")

    # LitellmModel(model, api_key, **kwargs)——api_base/extra_body 走
    # **kwargs 直透 litellm.completion（LM.__call__ 组装 self.kwargs）
    lm = LitellmModel(
        model=f"openai/{MODEL}", api_key=key,
        api_base=base,
        # thinking off（Qwen 家族唯一验证形态）+ litellm 必要参数
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        timeout=300)
    # max_tokens 放宽（默认 1000——STORM 的 article_gen 长输出会被掐）
    lm.kwargs["max_tokens"] = 4000

    cfg = STORMWikiLMConfigs()
    # 五组件同模型（官方推荐：conv_simulator/question_asker 用便宜模型
    # 的建议是成本优化——我们全部同 27B 保持与 ours/harness 同尺）
    cfg.conv_simulator_lm = lm
    cfg.question_asker_lm = lm
    cfg.outline_gen_lm = lm
    cfg.article_gen_lm = lm
    cfg.article_polish_lm = lm
    return cfg


# ---------------------------------------------------------------------------
# 题目 + 出口适配（storm_gen_article.md → CS2 JSON）
# ---------------------------------------------------------------------------

def load_cs2_questions():
    src = os.path.normpath(os.path.join(
        _HERE, "..", "scholarqa_multi", "sqa2_rubrics_v1_recomputed.json"))
    qs = json.load(open(src, encoding="utf-8"))
    return [{"qid": q["case_id"][:24], "question": q["question"]}
            for q in qs[10:15]]


_NUM_CITE = re.compile(r"\[(\d+)\]")
_REF_RE = re.compile(r"^\s*[-*]?\s*\[(\d+)\]\s*(.*)$", re.M)



def article_to_cs2(article_md: str, url_to_info: dict | None = None) -> list[dict]:
    """STORM 产物 → CS2 sections/citations。polished 文章引用是 [n] 数字
    标记；url_to_info.json 给 n → {url, title, snippets}（snippets=检索
    摘要片段，verbatim 语义）。"""
    idx_map = (url_to_info or {}).get("url_to_unified_index") or {}
    info_map = (url_to_info or {}).get("url_to_info") or {}
    n2info = {}
    for url, n in idx_map.items():
        n = int(n)  # regex group 是 str，统一 int 键
        info = info_map.get(url) or {}
        n2info[n] = {"title": info.get("title") or url,
                     "snippets": info.get("snippets") or [],
                     "url": url}
    sections = []
    cur_title, cur_lines = "Answer", []
    for line in article_md.splitlines():
        if line.startswith("## ") or line.startswith("# "):
            if any(l.strip() for l in cur_lines):
                sections.append((cur_title, chr(10).join(cur_lines)))
            cur_title = line.lstrip("#").strip() or "Section"
            cur_lines = []
        else:
            cur_lines.append(line)
    if any(l.strip() for l in cur_lines):
        sections.append((cur_title, chr(10).join(cur_lines)))

    out = []
    for title, text in sections:
        if title.lower() in ("references", "reference", "summary"):
            continue
        if not text.strip():
            continue
        cites, seen = [], {}
        for m in _NUM_CITE.finditer(text):
            n = int(m.group(1))
            if n in seen or n not in n2info:
                continue
            seen[n] = len(cites) + 1
            info = n2info[n]
            cites.append({
                "id": "[" + str(seen[n]) + "]",
                "snippets": [s[:400] for s in info["snippets"][:3]],
                "title": info["title"],
                "metadata": {"url": info["url"]},
            })
        text2 = _NUM_CITE.sub(
            lambda mm: "[" + str(seen[int(mm.group(1))]) + "]"
            if int(mm.group(1)) in seen else mm.group(0), text)
        out.append({"title": title, "text": text2, "citations": cites})
    return out



# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(smoke: bool = False):
    import threading
    from knowledge_storm import (STORMWikiRunner,
                                 STORMWikiRunnerArguments)
    os.makedirs(BASE_DIR, exist_ok=True)
    questions = load_cs2_questions()
    if smoke:
        questions = questions[:1]

    done = {}
    if os.path.exists(ANSWERS):
        done = {r["qid"]: r for r in
                json.load(open(ANSWERS, encoding="utf-8"))}
    todo = [q for q in questions if q["qid"] not in done]
    print(f"[storm-cs2] {len(todo)} to answer ({len(done)} resumed)",
          flush=True)

    rm_cls = SciverseRM

    for q in todo:
        qid, text = q["qid"], q["question"]
        # STORM 的 topic=研究主题；CS2 问题→topic 直接用题面
        out_dir = os.path.join(BASE_DIR, qid)
        os.makedirs(out_dir, exist_ok=True)
        try:
            # Windows 路径 sanitize：STORM 把 topic（CS2 题面含 ?/: 等
            # 非法字符）拼进输出目录名——先清洗成安全形式
            def _safe_topic(t):
                return re.sub(r'[^A-Za-z0-9_\- ]+', '', t)[:80].strip().replace(' ', '_') or 'topic'
            args = STORMWikiRunnerArguments(
                output_dir=out_dir,
                # 官方默认全保留（显式写出防上游漂移）
                max_conv_turn=3, max_perspective=3,
                max_search_queries_per_turn=3,
                search_top_k=3, retrieve_top_k=3,
                max_thread_num=10)
            runner = STORMWikiRunner(args, build_lm_configs(), rm_cls(k=3))
            _topic = _safe_topic(text)
            runner.run(
                topic=_topic,
                ground_truth_url="",
                do_research=True, do_generate_outline=True,
                do_generate_article=True, do_polish_article=True)
            # 产物直接在 run() 内部落盘（storm_gen_article.txt +
            # polished 变体；老版 API 的 post_run/end 已不存在）
            art_path = None
            for root, _dirs, files in os.walk(out_dir):
                if "storm_gen_article_polished.txt" in files:
                    art_path = os.path.join(root, "storm_gen_article_polished.txt")
                    break
                if "storm_gen_article.txt" in files:
                    art_path = os.path.join(root, "storm_gen_article.txt")
                    break
            if art_path is None:
                raise FileNotFoundError("storm_gen_article not produced")
            article = open(art_path, encoding="utf-8").read()
            uti_path = os.path.join(os.path.dirname(art_path), "url_to_info.json")
            uti = json.load(open(uti_path, encoding="utf-8")) if os.path.exists(uti_path) else {}
            sections = article_to_cs2(article, uti)
            done[qid] = {"qid": qid, "question": text,
                         "article_path": art_path,
                         "sections": sections,
                         "article_chars": len(article)}
            print(f"[storm-cs2] {qid[:12]} article={len(article)}ch "
                  f"sections={len(sections)}", flush=True)
        except Exception as e:
            import traceback
            done[qid] = {"qid": qid, "question": text,
                         "err": f"{type(e).__name__}: {str(e)[:200]}"}
            traceback.print_exc()
        json.dump(list(done.values()), open(ANSWERS, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    # 判分输入文件（direct_judge 直接吃）
    judge_input = [{"qid": r["qid"], "question": r["question"],
                    "sections": r["sections"]}
                   for r in done.values() if r.get("sections")]
    jip = os.path.join(BASE_DIR, "judge_input_storm_cs2.json")
    json.dump(judge_input, open(jip, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[storm-cs2] judge input: {jip} ({len(judge_input)} rows)",
          flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
