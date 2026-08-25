"""重写验收运行（DECISION-extraction-rewrite.md 预注册）：与探针 Arm B 同口径
（注入 outperforms_on/ablates gold pattern + DQN 真实块），跑在联合抽取管线上。
对照基线 = PROBE_B_PPR_24493BE6E8C2（探针宽松 20%）。

用法: python .research_tmp/acceptance_rewrite_run.py
"""
import os, sys, shutil, time, json
sys.path.insert(0, 'src')

PAPER = 'PPR_24493BE6E8C2'
OUT_BASE = os.path.join('.research_tmp', 'runs', 'kernel_v2')


def inject_gold_patterns(hg):
    from granular_agent.hypergraph_schema import MetaHyperedgePattern
    hg.add_pattern(MetaHyperedgePattern(
        pattern_id='outperforms_on',
        description='[NEW] method A outperforms method B on task/benchmark T, measured by metric M',
        semantic_boundary='quantitative comparison result between two methods on a '
            'shared task with a metric; NOT claim_relation (discourse-level '
            'supports/contrasts), NOT improves (building-on, no numbers)',
        role_slots=[{'role': 'winner', 'type': 'METHOD'},
                    {'role': 'loser', 'type': 'METHOD'},
                    {'role': 'task', 'type': 'THING'},
                    {'role': 'metric', 'type': 'THING'}],
        allowed_qualifiers=['method', 'evidence_strength', 'cited_from'],
        family='claim'), evidence='probe injection', paper_id='PROBE')
    hg.add_pattern(MetaHyperedgePattern(
        pattern_id='ablates',
        description='[NEW] paper ablates component C of method M, observing effect E',
        semantic_boundary='explicit ablation study statement (removing/disabling a '
            'component and measuring the consequence); NOT composed_of (structural '
            'parts), NOT influences (functional dependence)',
        role_slots=[{'role': 'method', 'type': 'METHOD'},
                    {'role': 'component', 'type': 'THING'},
                    {'role': 'effect', 'type': 'RESULT'}],
        allowed_qualifiers=['method', 'evidence_strength', 'cited_from'],
        family='claim'), evidence='probe injection', paper_id='PROBE')
    return hg


def main():
    from granular_agent.agent import GranularFlowAgent
    from granular_agent.hypergraph_schema import seed_meta_hypergraph_general

    hg = seed_meta_hypergraph_general()
    hg = inject_gold_patterns(hg)
    agent = GranularFlowAgent(domain='ml', llms=['DeepSeek-V4-Flash'])
    agent.meta_hg = hg
    agent._initial_seed_pats = set(seed_meta_hypergraph_general().patterns.keys())

    t0 = time.time()
    res = agent.process_paper_via_kernel(PAPER, arm='full')
    dt = time.time() - t0

    src_dir = os.path.join(OUT_BASE, PAPER)
    dst = os.path.join(OUT_BASE, f'REWRITE_{PAPER}')
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    if os.path.isdir(src_dir):
        shutil.move(src_dir, dst)

    # gate 拒绝统计 + 注入 pattern 使用统计
    gate_stats = {}
    injected_kept, injected_dropped = [], []
    for s in res['section_reports']:
        for d in s['extract'].get('dropped_edges', []):
            r = str(d.get('reason', ''))
            if r.startswith('gate:'):
                gate_stats[r] = gate_stats.get(r, 0) + 1
            if d.get('pattern_type') in ('outperforms_on', 'ablates'):
                injected_dropped.append(d)
        for k in s['extract'].get('kept_edges', []):
            if k['pattern_type'] in ('outperforms_on', 'ablates'):
                injected_kept.append(k)
    print(f"REWRITE acceptance run: {res['n_sections']} sections, "
          f"{res['n_concepts']} concepts, {res['n_hyperedges']} hyperedges, "
          f"{dt:.0f}s -> {dst}")
    print(f"gate drops: {gate_stats}")
    print(f"injected pattern kept: {len(injected_kept)}, dropped: {len(injected_dropped)}")


if __name__ == '__main__':
    main()
