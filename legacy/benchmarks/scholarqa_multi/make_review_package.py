# -*- coding: utf-8 -*-
"""Assemble the human-review package for the vocabulary gate (user闸门,
prereg S1). Deterministic sampling — zero LLM. Sections:
1. registry: stats + top-mention multi-alias + random + bio/photonics focus
   + collapse groups + flagged items from self-audit
2. vocab: per-dimension batch/cross-merge/compliance stats + family samples
   per domain (bio/photonics weighted — user's named observation point)
3. QC dashboard: guards fired, fallbacks, purity
4. blocklist proposals (if propose_blocklist.py has run)

Usage: python make_review_package.py  -> REVIEW-PACKAGE.md
"""
import json
import os
import random
import sys
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(BASE, "kb")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    reg = json.load(open(os.path.join(KB, "registry.json"), encoding="utf-8"))
    regqc = json.load(open(os.path.join(KB, "registry_qc.json"), encoding="utf-8"))
    vocab = json.load(open(os.path.join(KB, "dim_vocab_v1.json"), encoding="utf-8"))
    vqc = json.load(open(os.path.join(KB, "vocab_qc.json"), encoding="utf-8"))
    man = json.load(open(os.path.join(BASE, "corpus", "manifest.json"), encoding="utf-8"))
    pid_subj = {r["paper_id"]: r["subject"] for r in man}
    ents = reg["entities"]

    def dom(e):
        subs = [pid_subj.get(p) for p in e.get("mention_papers", []) if pid_subj.get(p)]
        return Counter(subs).most_common(1)[0][0] if subs else "?"

    L = []
    L.append("# Multi-431 词表人工审包件（预注册 S1 用户闸门）\n")
    L.append(f"生成：确定性抽样，零 LLM。数据源：registry.json（{len(ents)} 实体）+ "
             f"dim_vocab_v1.json + 两份 QC。\n")

    # ---- section 1: registry ----
    L.append("\n## 1. Registry（实体归一）\n")
    types = Counter(e["entity_type"] for e in ents)
    multi = [e for e in ents if len(e["aliases"]) > 1]
    L.append(f"- 实体 {len(ents)}（{dict(types)}）｜多别名实体 {len(multi)}｜"
             f"surface_index {len(reg['surface_index'])}")
    L.append(f"- 合并账：round1 {regqc.get('block_merges')} + 恢复 "
             f"{(regqc.get('singleton_recovery') or {}).get('recovered')} + 跨块 "
             f"{[r['merges'] for r in regqc.get('cross_rounds', [])]}"
             f"｜守卫拦截：cross {[r.get('guard_rejected_pairs') for r in regqc.get('cross_rounds', [])]}，"
             f"recovery {(regqc.get('singleton_recovery') or {}).get('guard_rejected')}"
             f"｜坍缩组 {regqc.get('entity_id_collapses')}"
             f"｜通道边 {regqc.get('edges')}")
    rng = random.Random(7)
    top = sorted(multi, key=lambda e: -e["mention_count"])[:15]
    rest = [e for e in multi if e not in top]
    rand = rng.sample(rest, min(15, len(rest)))
    bp = [e for e in rest if dom(e) in ("bio", "photonics", "biophysics") and e not in rand]
    bps = rng.sample(bp, min(12, len(bp)))

    def ent_line(e):
        return (f"  - [{dom(e)}|n={e['mention_count']}|{e['entity_type']}] "
                f"**{e['canonical']}** ← {e['aliases'][:6]}"
                + ("…" if len(e["aliases"]) > 6 else ""))
    L.append("\n### 1a. Top-15 高提及合并（错并影响最大，重点读）")
    L += [ent_line(e) for e in top]
    L.append("\n### 1b. 随机 15")
    L += [ent_line(e) for e in rand]
    L.append("\n### 1c. bio/photonics/biophysics 定向 12（用户点名观察点：域覆盖）")
    L += [ent_line(e) for e in bps]
    L.append("\n### 1d. 坍缩组（同 entity_id 多组，卡片 alias 质量问题）")
    for ex in regqc.get("collapse_examples", []):
        L.append(f"  - {ex}")
    L.append("\n### 1e. 自查旗标（Claude 抽读发现，待用户裁定处置）")
    L.append("  1. 'phantom A'/'phantom B' 并入同一实体（论文中两个不同试样——真错误，来源=卡片 alias 塞爆）")
    L.append("  2. 'optical tweezer'（通用技术）被并入某具体论文实体（泛称被狭义吸收）")
    L.append("  3. 'BIC' 同时出现在两个实体的 alias 表（supercavity 与 Topological-BIC）——surface_index 后写覆盖")
    L.append("  4. GPT-3 尺寸变体（13B/175B/2.7B/6.7B）全并入 GPT-3 家族——alias 保留表面名，记录级可分辨，待定夺")
    L.append("  5. DLCZ protocol 分裂成两个实体（漏合，~6% 残留类）")
    L.append("  6. 恢复通道 692 合并（22% 接受率 vs 遗产 8%）——建议抽读下方 variant/subject 家族样本代偿")
    L.append("  7. 'Shokri et al. (2017)' 类引文残渣成为 out_of_corpus 实体（无害，broker 阶段可过滤）")

    # ---- section 2: vocab ----
    L.append("\n## 2. Vocab（维度词表）\n")
    sc = vqc.get("scale") or {}
    for dim in ("subject", "setup", "variant", "hyperparam_item"):
        d = sc.get(dim) or {}
        L.append(f"- {dim}: 候选 {d.get('candidates')} → 批 {d.get('batches')}，跨批合并 {d.get('cross_merges')}")
    L.append(f"- 落盘：subject 家族 {len(vocab['subject'])}｜setup {len(vocab['setup'])}｜"
             f"variant {len(vocab['variant'])}｜hyperparam {len(vocab['hyperparam_items'])}")
    ev = sc.get("compliance_events") or []
    L.append(f"- 合规事件 {len(ev)}：{dict(Counter(e.get('event') for e in ev))}")
    L.append(f"- 确定性清理：{ {k: len(v) for k, v in (vqc.get('cleanup_flags') or {}).items()} }"
             f"｜仲裁队列旗标：family {len((vqc.get('qc') or {}).get('suspect_family_merge', []))}"
             f" / type {len((vqc.get('qc') or {}).get('suspect_entity_type', []))}")

    # family samples per domain — approximate domain via member surfaces' papers?
    # families don't carry papers; sample by family name language + size
    big = sorted(vocab["subject"], key=lambda f: -len(f.get("members", [])))
    L.append("\n### 2a. subject 最大 12 家族（跨域归组质量代表）")
    for f in big[:12]:
        L.append(f"  - **{f.get('family')}**（{len(f.get('members', []))}）: {f.get('members', [])[:6]}")
    L.append("\n### 2b. subject 随机 12 家族")
    for f in rng.sample(vocab["subject"], min(12, len(vocab["subject"]))):
        L.append(f"  - **{f.get('family')}**（{len(f.get('members', []))}）: {f.get('members', [])[:6]}")
    L.append("\n### 2c. setup 随机 10 条（规范名+别名）")
    for e in rng.sample(vocab["setup"], min(10, len(vocab["setup"]))):
        L.append(f"  - {e['canonical']} ← {e.get('aliases', [])[:4]}")
    L.append("\n### 2d. variant 随机 10 条（含 family_hints）")
    for e in rng.sample(vocab["variant"], min(10, len(vocab["variant"]))):
        L.append(f"  - {e['canonical']} ← {e.get('aliases', [])[:3]} | hints {e.get('family_hints', [])[:3]}")
    L.append("\n### 2e. hyperparam 随机 8 条")
    for e in rng.sample(vocab["hyperparam_items"], min(8, len(vocab["hyperparam_items"]))):
        L.append(f"  - {e['canonical']} ← {e.get('aliases', [])[:3]}")
    sfm = (vqc.get("qc") or {}).get("suspect_family_merge", [])
    if sfm:
        L.append(f"\n### 2f. suspect_family_merge 抽样 10/{len(sfm)}（家族名与成员词面不重叠的旗标）")
        for x in rng.sample(sfm, min(10, len(sfm))):
            L.append(f"  - family={x.get('family')!r} member={x.get('member')!r}")

    # ---- section 3: blocklist proposals ----
    bp_path = os.path.join(KB, "blocklist_proposals.json")
    if os.path.exists(bp_path):
        props = json.load(open(bp_path, encoding="utf-8"))
        L.append(f"\n## 3. Blocklist 扩域提案（{len(props)} 条，待用户批）\n")
        by_dom = {}
        for p in props:
            by_dom.setdefault(p.get("domain", "?"), []).append(p)
        for d in sorted(by_dom):
            L.append(f"### [{d}] {len(by_dom[d])} 条")
            for p in by_dom[d][:40]:
                L.append(f"  - {p['name']}（{p.get('reason','')}）")
            if len(by_dom[d]) > 40:
                L.append(f"  - …余 {len(by_dom[d]) - 40} 条见 blocklist_proposals.json")
    else:
        L.append("\n## 3. Blocklist 扩域提案：尚未生成（propose_blocklist.py 待跑）")

    out = os.path.join(BASE, "REVIEW-PACKAGE.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"review package -> {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
