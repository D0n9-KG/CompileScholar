# -*- coding: utf-8 -*-
"""P0-6 failure case 分析: 聚合4seed×4臂漏边, 逐条诊断为什么漏.

对每条 gold 演化边, 统计4seed×4臂里 verdict 分布, 找反复漏的边.
诊断维度:
  - uncovered: lift 没抽到 src/tgt 方法 (抽取召回问题)
  - miss: 方法抽到但没抽边 (judge没判该对 / cluster没配对)
  - type_only: 抽了边但 type 错 (judge type 判错)
  - edge_hit: 命中
对反复漏的边, 关联 lift 产出看根因.
"""
import os, sys, json, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from collections import defaultdict

OUTDIR = os.path.join(os.path.dirname(__file__), "runs", "ARFM2024", "arms_err")
GOLD = os.path.join(os.path.dirname(__file__), "gold", "ARFM2024_gold.json")


def main():
    gold = json.load(open(GOLD, encoding='utf-8'))
    gname = {m['id']: m['name'] for m in gold['methods']}
    galiases = {m['id']: m.get('aliases', []) for m in gold['methods']}
    gedges = gold['evolution_edges']

    # 收集所有 err detail: {edge_key: {arm_seed: verdict}}
    # edge_key = "src->tgt|type"
    edge_verdicts = defaultdict(dict)  # edge_key -> {(arm,seed): verdict}
    arm_seed_missing_methods = defaultdict(set)  # (arm,seed) -> set of missing gold method ids

    files = sorted(glob.glob(os.path.join(OUTDIR, "seed*_en_*_err.json")))
    for f in files:
        base = os.path.basename(f)[:-5]  # seed1_en_T_err
        parts = base.split("_en_")
        if len(parts) != 2: continue
        seed, arm = parts[0], parts[1]  # seed1, T
        e = json.load(open(f, encoding='utf-8'))
        for d in e['detail']:
            ek = f"{d['gold']}|{d['type']}"
            edge_verdicts[ek][(arm, seed)] = d['verdict']
        # missing methods
        for mid in e.get('missing_gold_methods', []):
            arm_seed_missing_methods[(arm, seed)].add(mid)

    print(f"=== 41 gold 演化边 × 4臂 × 4seed 漏边诊断 ===\n")
    # 对每条 gold 边统计
    edge_summary = []
    for ge in gedges:
        src, tgt, typ = ge['src'], ge['tgt'], ge['type']
        ek = f"{src}->{tgt}|{typ}"
        vd = edge_verdicts.get(ek, {})
        # 统计各 verdict 次数 (跨16 runs)
        cnt = defaultdict(int)
        for (a, s), v in vd.items():
            cnt[v] += 1
        total = sum(cnt.values())
        hit_rate = cnt.get('edge_hit', 0) / total if total else 0
        edge_summary.append((src, tgt, typ, cnt, hit_rate, total, vd))

    # 按命中率升序 (最常漏的在前)
    edge_summary.sort(key=lambda x: x[4])
    print(f"{'src->tgt':<12} {'type':<10} {'hit':<4} {'type_only':<9} {'miss':<5} {'uncovered':<10} {'n':<3} hit_rate")
    print("-" * 75)
    for src, tgt, typ, cnt, hr, total, vd in edge_summary:
        print(f"{src+'->'+tgt:<12} {typ:<10} {cnt.get('edge_hit',0):<4} {cnt.get('type_only',0):<9} "
              f"{cnt.get('miss',0):<5} {cnt.get('uncovered',0):<10} {total:<3} {hr:.2f}")

    # 反复漏的边 (hit_rate=0) 逐条根因
    print(f"\n{'='*70}\n=== 反复漏的边 (4seed×4臂从未命中) 根因诊断 ===")
    never_hit = [e for e in edge_summary if e[4] == 0]
    print(f"共 {len(never_hit)}/41 条边从未命中\n")
    # 分类根因
    reason_cnt = defaultdict(int)
    for src, tgt, typ, cnt, hr, total, vd in never_hit:
        # 判断根因
        if cnt.get('uncovered', 0) == total:
            # 全是 uncovered: 方法没抽到
            reason = "uncovered(方法没抽到)"
        elif cnt.get('miss', 0) > 0:
            reason = "miss(方法抽到但没抽边)"
        elif cnt.get('type_only', 0) > 0:
            reason = "type_only(抽了边type错)"
        else:
            reason = "混合"
        reason_cnt[reason] += 1
        print(f"  {src}({gname[src][:20]}) --{typ}--> {tgt}({gname[tgt][:20]}) | {reason} | "
              f"hit{cnt.get('edge_hit',0)} type_only{cnt.get('type_only',0)} miss{cnt.get('miss',0)} uncov{cnt.get('uncovered',0)}")
    print(f"\n根因分布: {dict(reason_cnt)}")

    # uncovered 的边: 哪些方法反复没抽到
    print(f"\n{'='*70}\n=== 反复没抽到的 gold 方法 (uncovered 根因) ===")
    method_miss = defaultdict(int)
    for src, tgt, typ, cnt, hr, total, vd in never_hit:
        if cnt.get('uncovered', 0) == total:
            # 看哪个端点没抽到: 查任意run的 missing_gold_methods
            for (a, s), v in vd.items():
                miss = arm_seed_missing_methods.get((a, s), set())
                if src in miss: method_miss[src] += 1
                if tgt in miss: method_miss[tgt] += 1
    for mid, c in sorted(method_miss.items(), key=lambda x: -x[1])[:15]:
        print(f"  {mid} {gname[mid][:30]} | aliases:{galiases[mid][:3]} | 没抽到{c}次")

    # 命中的边 (供对比)
    print(f"\n{'='*70}\n=== 命中的边 (hit_rate>0, 供对比) ===")
    hit_edges = [e for e in edge_summary if e[4] > 0]
    print(f"共 {len(hit_edges)}/41 条至少1次命中")
    for src, tgt, typ, cnt, hr, total, vd in hit_edges:
        print(f"  {src}({gname[src][:18]}) --{typ}--> {tgt}({gname[tgt][:18]}) | hit_rate={hr:.2f} ({cnt.get('edge_hit',0)}/{total})")

    json.dump({"never_hit": [{"src":s,"tgt":t,"type":ty,"reason":r} for s,t,ty,c,hr,tot,vd in never_hit
                              for r in (["uncovered"] if c.get('uncovered',0)==tot else ["miss" if c.get('miss',0)>0 else "type_only"])],
               "hit_edges": [{"src":s,"tgt":t,"type":ty,"hit_rate":hr} for s,t,ty,c,hr,tot,vd in hit_edges]},
              open(os.path.join(os.path.dirname(__file__), "failure_case_out.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nsaved -> failure_case_out.json")


if __name__ == "__main__":
    main()
