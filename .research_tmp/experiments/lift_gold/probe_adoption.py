"""采纳探针 — 运行部分（统计走 probe_stats_from_bundle.py 离线重算）。

Arm B: W1+W2 修复后 + 注入 gold pattern (outperforms_on/ablates) + 真实 MinerU 块
Arm C: W1+W2 修复后, 不注入 (base rate 对照)
输入: PPR_24493BE6E8C2 (DQN Nature 2015, 真实 content_list.json, 零 monkey-patch)

用法 (repo root): python .research_tmp/probe_adoption.py B|C
"""
import os, sys, shutil, time
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


def run(arm: str):
    from granular_agent.agent import GranularFlowAgent
    from granular_agent.hypergraph_schema import seed_meta_hypergraph_general

    hg = seed_meta_hypergraph_general()
    if arm == 'B':
        hg = inject_gold_patterns(hg)
    agent = GranularFlowAgent(domain='ml', llms=['DeepSeek-V4-Flash'])
    agent.meta_hg = hg
    agent._initial_seed_pats = set(seed_meta_hypergraph_general().patterns.keys())

    t0 = time.time()
    res = agent.process_paper_via_kernel(PAPER, arm='full')
    dt = time.time() - t0

    src_dir = os.path.join(OUT_BASE, PAPER)
    dst = os.path.join(OUT_BASE, f'PROBE_{arm}_{PAPER}')
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    if os.path.isdir(src_dir):
        shutil.move(src_dir, dst)
    else:
        os.makedirs(dst, exist_ok=True)

    print(f"PROBE {arm}: sections={res['n_sections']} concepts={res['n_concepts']} "
          f"hyperedges={res['n_hyperedges']} v{res['version_before']}->{res['version_after']} "
          f"({dt:.0f}s) -> {dst}")


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'B')
