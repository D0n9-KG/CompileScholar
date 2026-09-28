# -*- coding: utf-8 -*-
"""DeepScholar-Bench STORM 臂——同 CS2 STORM 部署复用（题目源换成
Related Works 生成任务）。

任务形态：给定论文 abstract → 生成 Related Works 综述段（含引用）。
判分走官方 eval/parsers/storm.py（直接吃 storm_gen_article.md 产物——
零适配，这是选 STORM 的红利）。

用法：python dsb_storm.py --smoke / python dsb_storm.py
"""
import argparse
import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "_shared", "tools")))

from cs2_storm import (SciverseRM, build_lm_configs, _safe_topic,
                       article_to_cs2)  # 复用 CS2 臂全部组件

DSB = os.path.normpath(os.path.join(_HERE, "..", "deepscholar", "dsb"))
BASE_DIR = os.path.join(_HERE, "arm_storm_dsb")
ANSWERS = os.path.join(BASE_DIR, "answers_storm_dsb.json")


def load_dsb_questions(limit=None):
    """DeepScholar 前 N 题：abstract 输入 → Related Works 任务。"""
    rows = list(csv.DictReader(open(
        os.path.join(DSB, "dataset", "papers_with_related_works.csv"),
        encoding="utf-8")))
    if limit:
        rows = rows[:limit]
    out = []
    for r in rows:
        # 官方任务 prompt（deepscholar_base README 的 query 形态）
        prompt = (
            "Your task is to write a Related Works section for an academic "
            "paper given the paper's abstract. You should cite the most "
            "relevant prior literature.\n\n"
            f"Abstract: {r['abstract']}")
        out.append({"qid": r["arxiv_id"], "title": r["title"],
                    "prompt": prompt,
                    "gold_related_works": r.get("clean_latex_related_works", "")})
    return out


def run(smoke=False, limit=None):
    from knowledge_storm import STORMWikiRunner, STORMWikiRunnerArguments
    os.makedirs(BASE_DIR, exist_ok=True)
    questions = load_dsb_questions(limit=1 if smoke else limit)
    if smoke:
        questions = questions[:1]

    done = {}
    if os.path.exists(ANSWERS):
        done = {r["qid"]: r for r in
                json.load(open(ANSWERS, encoding="utf-8"))}
    todo = [q for q in questions if q["qid"] not in done]
    print(f"[storm-dsb] {len(todo)} to answer ({len(done)} resumed)",
          flush=True)

    for q in todo:
        qid = q["qid"]
        out_dir = os.path.join(BASE_DIR, qid.replace("/", "_"))
        os.makedirs(out_dir, exist_ok=True)
        try:
            # topic=sanitize 的论文标题（STORM 以 topic 驱动研究——
            # Related Works 的主题就是这篇论文的主题）
            _topic = _safe_topic(q["title"])
            args = STORMWikiRunnerArguments(
                output_dir=out_dir,
                max_conv_turn=3, max_perspective=3,
                max_search_queries_per_turn=3,
                search_top_k=3, retrieve_top_k=3,
                max_thread_num=10)
            runner = STORMWikiRunner(args, build_lm_configs(), SciverseRM(k=3))
            runner.run(
                topic=_topic, ground_truth_url="",
                do_research=True, do_generate_outline=True,
                do_generate_article=True, do_polish_article=True)
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
            # 官方 parser 消费形态：每题目录下 storm_gen_article.md
            # （eval/parsers/storm.py 的 _load_file 找这个名字）
            md_path = os.path.join(out_dir, "storm_gen_article.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(article)
            done[qid] = {"qid": qid, "title": q["title"],
                         "article_path": art_path, "md_path": md_path,
                         "article_chars": len(article)}
            print(f"[storm-dsb] {qid} article={len(article)}ch", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            done[qid] = {"qid": qid, "title": q["title"],
                         "err": f"{type(e).__name__}: {str(e)[:200]}"}
        json.dump(list(done.values()), open(ANSWERS, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print(f"[storm-dsb] answers saved: {ANSWERS}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    run(smoke=args.smoke, limit=args.limit)


if __name__ == "__main__":
    main()
