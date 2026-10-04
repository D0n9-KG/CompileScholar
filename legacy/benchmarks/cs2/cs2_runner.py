# -*- coding: utf-8 -*-
"""CS2 OUR-arm runner: gate2r answer stack over the CS2 hot-start base KB.

Adapts multi_ours_run.py's KB-loading to the CS2 base_kb products:
  - KB = Tier-1 coarse (7,824) + survey records (S1-S4) + hub deep records
  - questions = CS2 dev rubrics (question text only, gold-blind)
  - answering model = local:Qwen3.8-27B (arm purity)
  - OPEN-SET = 1 (broker external retrieval + growth, CS2 is open-world)

Output rows feed report_adapter.py -> CS2 sections JSON -> official
scorer (GLM-5.3 judge).
"""
from __future__ import annotations

import json
import os
import re
import sys

SRC = r"C:\Users\D0n9\Desktop\CompileScholar\src"
STAGEB = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\stageB"
SHARED = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\_shared\tools"
CS2 = os.path.dirname(os.path.abspath(__file__))
BASE_KB = os.path.join(CS2, "base_kb")
ARM = os.path.join(CS2, "arm_ours")

sys.path.insert(0, SRC)
sys.path.insert(0, STAGEB)
sys.path.insert(0, SHARED)

MODEL = os.environ.get("OURS_MODEL", "local:Qwen3.8-27B")
TAG = os.environ.get("OURS_TAG", "cs2dev")
# --tag 命令行参数优先(2026-10-01:PowerShell run_in_background 的 env
# 块传递三次失效——31b/32a/32b 的 OURS_TAG/OURS_ANSWERS 都没到子进程,
# 32b 因此被 resume 判定已答而整批空转。命令行参数是可靠通道)
for _i, _a in enumerate(sys.argv):
    if _a == "--tag" and _i + 1 < len(sys.argv):
        TAG = sys.argv[_i + 1]
        os.environ["OURS_TAG"] = TAG
        del sys.argv[_i:_i + 2]
        break

os.environ.setdefault("LLM_CALL_LOG", os.path.join(ARM, f"ledger_ours_{TAG}.jsonl"))
os.environ.setdefault("LLM_RUN_ID", f"ours-{TAG}")
os.environ.setdefault("LLM_SOCK_TIMEOUT", "900")
os.environ.setdefault("LLM_WALL_TIMEOUT", "1200")
os.environ.setdefault("LOCAL_SOCK_TIMEOUT", "900")
# deep_read L1（2026-09-28）：大输出 chunk 抽取走独立车道（默认 2 车道
# 把 18 chunks 串成 48 分钟——SIFT 实测）；GPUStack 甜点 16 路（VERDICT-
# 0.5），答题本身 ~1 路/步，给深抽让出 6 车道
os.environ.setdefault("LOCAL_MAX_CONCURRENT", "10")
os.environ.setdefault("LOCAL_LARGE_MAX_CONCURRENT", "6")
os.environ.setdefault("KB_EMBED_PROVIDER", "local")
os.environ.setdefault("KB_OPEN_SET", "1")  # CS2 = open-world by design
# search_text 断口修复（批11拦截）：harness 在模块导入时就把
# PS53_TEXT_INDEX 读进 _TEXT_INDEX_DIR（默认=已不存在的 archive 路径），
# multi_ours_run 的 setdefault 在答题期才执行=太晚。必须在 import
# evidence_gate2r_harness 之前设好。全部 CS2 批次的 search_text 192
# 字符错误 obs 根因即此（模型笔记"search_text errored (infra)"）。
os.environ.setdefault("PS53_TEXT_INDEX", os.path.join(ARM, "text_index"))

import evidence_gate2r_harness as _HF  # noqa: E402
sys.modules["evidence_pilot_a1r2_harness"] = _HF
import evidence_pilot_a1r2 as D  # noqa: E402
D.H = _HF
D.R2 = ARM
D.H.MODEL = MODEL
# R-A 纪律（同 multi_ours_run）：无全局 token cap——per-question 预算
# 由步数 cap 管；全局 cap 在 CS2（每题独立报告任务）没有意义
D.TOKEN_CAP_R3 = int(os.environ.get("G2_CAP", "999999999999"))

ANSWERS = os.environ.get("OURS_ANSWERS",
                          os.path.join(ARM, f"answers_{TAG}.json"))


# ---------------- CS2 dev questions (gold-blind: question only) ----------------

def build_cs2_qfile(limit=None, split="dev", offset=0) -> str:
    """CS2 rubrics -> PS-shape qfile. Gold-blind: only question text used."""
    src = os.path.join(os.path.dirname(CS2),
                       "scholarqa_multi", "sqa2_rubrics_v1_recomputed.json")
    qs = json.load(open(src, encoding="utf-8"))
    qs = qs[offset:offset + limit] if limit else qs[offset:]
    out = []
    for q in qs:
        out.append({"qid": q["case_id"][:24],
                    "prompt_type": "aggregation",
                    "question": q["question"],
                    "gold_answer": ""})
    os.makedirs(ARM, exist_ok=True)
    path = os.path.join(ARM, f"questions_cs2_{split}.json")
    json.dump({"questions": out}, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return path


# ---------------- KB build: base_kb products ----------------

def build_tools_cs2():
    """KBTools over the CS2 base KB (coarse + survey + hub records)."""
    from kb_compiler.views.tools import KBTools

    # 批10 复盘（2026-09-28）：改载 records_merged.json（键桥接的完整
    # KB：sciverse 需求池粗抽+survey refs 粗抽+hub 深抽，paper_id 键）。
    # 旧路径载 coarse_records.json（title[:40] 键）——需求池 793 篇的
    # findings(paper_id="sciverse_...") 全空（Q1 三篇 in-topic 论文查
    # 0 记录→诚实弃答）；1896 键桥接工作落在 merged 里而 runner 从未
    # 加载它。7,816/7,819 coarse 记录 id 已被 merged 覆盖。
    records = {}
    merged = json.load(open(os.path.join(BASE_KB, "records_merged.json"),
                            encoding="utf-8"))
    for pid_key, payload in merged.items():
        if isinstance(payload, dict) and payload.get("records"):
            records[pid_key] = payload
    # FULLCHAIN-AUDIT B10：深读终化记录进运行时 KB——2,419 条深记录与
    # records_merged 零交集，typed tools 每批冷启动对它们全盲（批 N 深抽
    # 的知识批 N+1 只能靠 search_text 摸到）。「KB 随探索生长」在查询层
    # 兑现。合并非替换（粗抽记录保留）。
    deep_rp = os.path.join(BASE_KB, "deep_read_records.json")
    if os.path.exists(deep_rp):
        try:
            deep = json.load(open(deep_rp, encoding="utf-8"))
            for pid, payload in deep.items():
                if not (isinstance(payload, dict) and payload.get("records")):
                    continue
                base_recs = records.get(pid, {}).get("records") or []
                seen = {r.get("id") for r in base_recs if r.get("id")}
                records[pid] = {"records": list(base_recs) + [
                    r for r in payload["records"] if r.get("id") not in seen]}
        except Exception as e:
            print(f"[cs2] deep records merge skip: {str(e)[:80]}", flush=True)
    views = json.load(open(os.path.join(BASE_KB, "views_cs2.json"),
                           encoding="utf-8"))
    registry = json.load(open(os.path.join(BASE_KB, "registry_v2.json"),
                              encoding="utf-8"))
    vocab = json.load(open(os.path.join(BASE_KB, "dim_vocab_cs2.json"),
                           encoding="utf-8"))
    manifest = {r["paper_id"]: r for r in json.load(
        open(os.path.join(BASE_KB, "manifest.json"), encoding="utf-8"))}
    kb = KBTools(views, registry, vocab, manifest, records,
                 emb_cache_path=os.path.join(ARM, "emb_cache_records.bin"))
    if os.environ.get("KB_OPEN_SET", "1") == "1":
        from external_tools import attach_external_tools
        # KB_SNAPSHOT（FIX-PLAN v2 ②B 裁定：答题模式关生长）：=1 时全部
        # 可写产物重定向到 ARM 下的影子目录——答题期深读只进本题
        # EvidenceStore/影子层，库文件（deep_read_records/backflow_edges/
        # manifest_all/deep_cache/growth_library）只读不漂移。100 题全量
        # 看到同一个库（"单一版本"宣称），KB size 论文里是个固定数。
        # 缺省 0 = 开发期旧行为（深读累积进库，批间漂移）。
        if os.environ.get("KB_SNAPSHOT", "0") == "1":
            _snap = os.path.join(ARM, "kb_snapshot_writes")
            os.makedirs(_snap, exist_ok=True)
            _w = lambda name: os.path.join(_snap, name)  # noqa: E731
            print(f"[cs2] KB_SNAPSHOT=1: growth writes -> {_snap}", flush=True)
        else:
            _w = lambda name: os.path.join(BASE_KB, name)  # noqa: E731
        _ext = attach_external_tools(kb, views, manifest, model=MODEL,
                              registry=registry, blocklist=None,
                              backflow_path=_w("backflow_edges.jsonl"),
                              replay_from=os.path.join(
                                  BASE_KB, "backflow_edges.jsonl"),
                              tier_db=_w("growth_library.db"),
                              manifest_path=_w("manifest_all.json"),
                              deep_cache_path=_w("deep_read_cache.jsonl"),
                              deep_text_dir=_w("deep_read_texts"),
                              deep_records_path=_w("deep_read_records.json"))
    return kb, records, views, manifest


def main():
    limit = int(os.environ["CS2_LIMIT"]) if "CS2_LIMIT" in os.environ else None
    offset = int(os.environ.get("CS2_OFFSET", "0"))
    qfile = build_cs2_qfile(limit, offset=offset)
    print(f"[cs2] qfile: {qfile} (limit={limit})", flush=True)
    argv = sys.argv[1:]
    sys.argv = ["cs2", "--run", TAG, "--qfile", qfile] + argv

    _orig_build = D.build_tools_ps16

    def build_tools_cs2_wrapped():
        kb, records, views, manifest = build_tools_cs2()
        # F21b projections + F18 resolver（同 multi_ours_run 的包装）
        from multi_ours_run import wrap_f21  # 复用投影包装
        # FULLCHAIN-AUDIT B6：fetch_chunk 的文本查找根——CS2 三文本库
        # （survey/hub 明文 + deep md5），wrap_f21 的 fetch_chunk 依次尝试
        # （原硬编码 Multi corpus/texts 目录，CS2 臂 100% 失败）
        kb._cs2_text_roots = [
            os.path.join(BASE_KB, "survey_texts"),
            os.path.join(BASE_KB, "hub_texts"),
            os.path.join(BASE_KB, "deep_read_texts"),
        ]
        wrap_f21(kb)
        # F18 resolver 需 cards——base_kb 无 cards（survey 路线），
        # 传空（resolver 对空 cards 降级）
        try:
            from f18_resolver import F18Resolver, wrap_f18
            resolver = F18Resolver(kb, manifest, kb.registry, views,
                                   cards=[])
            wrap_f18(kb, resolver, log=F18_LOG)
            print(f"[F18] armed: {len(resolver.pid2entity)} papers",
                  flush=True)
        except Exception as e:
            print(f"[F18] skip ({str(e)[:80]})", flush=True)
        return kb, records, views, manifest

    F18_LOG = []
    D.build_tools_ps16 = build_tools_cs2_wrapped
    D.SKIP_DEINTERNALIZE = True
    try:
        D.main()   # runs stage_answer -> answers_{tag}.json in ARM
    finally:
        _f18p = os.path.join(ARM, f"f18_events_{TAG}.json")
        _prev = []
        if os.path.exists(_f18p):
            try:
                _prev = json.load(open(_f18p, encoding="utf-8"))
            except Exception:
                pass
        json.dump(_prev + F18_LOG, open(_f18p, "w", encoding="utf-8"),
                  ensure_ascii=False)
        D.build_tools_ps16 = _orig_build


if __name__ == "__main__":
    main()
