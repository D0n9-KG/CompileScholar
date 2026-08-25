"""采纳探针 — 离线统计（从 bundle 重算，不依赖运行进程）。

用法: python .research_tmp/probe_stats_from_bundle.py PROBE_B_PPR_24493BE6E8C2
输出: usage 直方图 + 注入 pattern 全边 dump（供语义审计）+ planner 联动（若有 result.json）
"""
import os, sys, json
from collections import Counter

BASE = os.path.join('.research_tmp', 'runs', 'kernel_v2')
INJECTED = ('outperforms_on', 'ablates')


def main(dirname):
    d = os.path.join(BASE, dirname)
    cg = json.load(open(os.path.join(d, 'concept_graph.json'), encoding='utf-8'))
    dropped = json.load(open(os.path.join(d, 'dropped_edges.json'), encoding='utf-8'))
    concepts = cg['concepts']

    def surface(cid):
        c = concepts.get(cid)
        if not c:
            return cid
        if c.get('canonical_name'):
            return c['canonical_name']
        vs = c.get('surface_variants') or []
        return vs[0]['surface'] if vs else cid

    full_edges = []
    for he in cg['hyperedges']:
        full_edges.append({
            'pattern_type': he.get('pattern_type') or he.get('kind'),
            'nodes': [{'surface': surface(nid), 'role': r}
                      for nid, r in zip(he.get('node_ids', []), he.get('node_roles', []))],
            'evidence': (he.get('provenance') or [{}])[0].get('evidence', ''),
        })
    hist_kept = Counter(e['pattern_type'] for e in full_edges)
    hist_dropped = Counter(e.get('pattern_type', '?') for e in dropped)

    injected_full = [e for e in full_edges if e['pattern_type'] in INJECTED]
    injected_dropped = [e for e in dropped if e.get('pattern_type') in INJECTED]

    out = {
        'bundle': dirname,
        'n_kept_total': len(full_edges),
        'n_dropped_total': len(dropped),
        'hist_kept': dict(hist_kept.most_common()),
        'hist_dropped': dict(hist_dropped.most_common()),
        'injected_usage_kept': {p: hist_kept.get(p, 0) for p in INJECTED},
        'injected_usage_dropped': {p: hist_dropped.get(p, 0) for p in INJECTED},
        'injected_edges_full': injected_full,
        'injected_dropped_detail': injected_dropped,
    }
    with open(os.path.join(d, 'probe_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(d, 'all_kept_edges_full.json'), 'w', encoding='utf-8') as f:
        json.dump(full_edges, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: out[k] for k in
                      ('n_kept_total', 'n_dropped_total', 'hist_kept',
                       'injected_usage_kept', 'injected_usage_dropped')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
