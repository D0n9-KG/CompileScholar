"""重写验收 judge 交叉（DECISION-extraction-rewrite.md 预注册）：
两个 judge（GLM-5-Turbo@paratera + qwen3.5@cst），均 ≠ 抽取模型(V4-Flash) ≠ verifier(deepseek-chat)。
逐边三检（evidence-支撑/槽位绑定/极性方向），输出 per-edge verdict + 汇总。

用法: python .research_tmp/judge_cross.py REWRITE_PPR_24493BE6E8C2
"""
import json, os, sys
sys.path.insert(0, 'src')

BASE = os.path.join('.research_tmp', 'runs', 'kernel_v2')
def _paper_text_for(dirname: str) -> str:
    """Fulltext for a bundle: DQN probe text if it's the DQN paper, else
    reconstruct from MINERU_BASE content_list."""
    if 'PPR_24493BE6E8C2' in dirname:
        return open('.research_tmp/probe_dqn_fulltext.txt', encoding='utf-8').read()
    import re as _re
    m = _re.search(r'(PPR_[A-Z0-9]+)', dirname)
    if not m:
        raise ValueError(f"no PPR id in dirname {dirname}")
    ppr = m.group(1)
    p = f"C:/Users/D0n9/Desktop/science_evo/data/upstream/remote_mineru/mineru_2355/papers/{ppr}/content_list.json"
    cl = json.load(open(p, encoding="utf-8"))
    return "\n\n".join(str(it.get("text", "")).strip() for it in cl
                       if it.get("type") == "text" and it.get("text")
                       and len(it["text"].strip()) >= 10)

_JUDGE_PROMPT = """You are a STRICT semantic judge for knowledge-hypergraph edges extracted from a scientific paper (the source text excerpts are given below).

Edge under judgment:
- pattern_type: {pt}
- pattern meaning: {desc}
- pattern boundary (when it applies): {boundary}
- nodes (role -> surface): {nodes}
- evidence_span (claimed supporting text): {ev}
- surrounding context: ...{ctx_before} [EVIDENCE] {ev} {ctx_after}...

Judge THREE checks independently:
1. EVIDENCE_SUPPORT: does the evidence text STATE this exact relation? (a sentence describing a setup/baseline without a result does NOT support a comparison edge; 'comparable to' does NOT state outperforming)
2. SLOT_BINDING: is each node really what its role claims, per this sentence? (the winner really won; a node bound to a role must be the thing the SENTENCE puts there). IMPORTANT: faithful-to-source beats precision — if the source sentence refers to a vague aggregate ("all previous algorithms", "best existing methods"), an edge binding that aggregate is CORRECT extraction; do not fail it for lacking specificity the source does not provide.
3. POLARITY: does the direction/polarity of the sentence match the roles?

Output JSON (one object):
{{"evidence_support": true/false, "slot_binding": true/false, "polarity_ok": true/false,
 "pass": true/false,  // ALL THREE true
 "reason": "one short sentence"}}"""


def judge_edges(dirname):
    d = os.path.join(BASE, dirname)
    cg = json.load(open(os.path.join(d, 'concept_graph.json'), encoding='utf-8'))
    meta = json.load(open(os.path.join(d, 'meta_snapshot.json'), encoding='utf-8'))
    pats = meta.get('patterns', {})
    concepts = cg['concepts']

    def surface(cid):
        c = concepts.get(cid)
        if not c: return cid
        if c.get('canonical_name'): return c['canonical_name']
        vs = c.get('surface_variants') or []
        return vs[0]['surface'] if vs else cid

    # find context around evidence in the paper text (normalized whitespace)
    import re
    norm_text = re.sub(r'\s+', ' ', _paper_text_for(dirname))

    edges = []
    for he in cg['hyperedges']:
        pt = he.get('pattern_type') or he.get('kind')
        ev = (he.get('provenance') or [{}])[0].get('evidence', '')
        pat = pats.get(pt, {})
        ev_n = re.sub(r'\s+', ' ', ev).strip()
        pos = norm_text.find(ev_n[:80])
        if pos < 0:
            pos = norm_text.find(ev_n[:40]) if len(ev_n) > 40 else -1
        ctx_before = norm_text[max(0, pos - 200):pos] if pos >= 0 else ''
        ctx_after = norm_text[pos + len(ev_n):pos + len(ev_n) + 200] if pos >= 0 else ''
        edges.append({
            'pt': pt,
            'desc': (pat.get('description') or '')[:200],
            'boundary': (pat.get('semantic_boundary') or '')[:300],
            'nodes': [(r, surface(n)) for n, r in zip(he.get('node_ids', []), he.get('node_roles', []))],
            'ev': ev_n[:1200],   # was 400 — truncation hid support sentences from the judge (arbitration found 6/6 long-evidence edges failed)
            'ctx_before': ctx_before[-160:], 'ctx_after': ctx_after[:160],
        })

    from granular_agent.llm_client import call_paratera, call_cst, parse_json_response

    def _one(job):
        i, e, judge, fn, model = job
        prompt = _JUDGE_PROMPT.format(**e)
        raw = fn(prompt, model=model, max_tokens=500)
        obj = parse_json_response(raw) or {}
        return judge, {
            'i': i, 'pt': e['pt'],
            'pass': bool(obj.get('pass')),
            'evidence_support': bool(obj.get('evidence_support')),
            'slot_binding': bool(obj.get('slot_binding')),
            'polarity_ok': bool(obj.get('polarity_ok')),
            'reason': str(obj.get('reason', ''))[:120],
        }

    jobs = [(i, e, judge, fn, model)
            for i, e in enumerate(edges)
            for judge, fn, model in (('GLM-5-Turbo', call_paratera, 'GLM-5-Turbo'),
                                     ('qwen3.5', call_cst, 'qwen3.5'))]
    results = {'GLM-5-Turbo': [], 'qwen3.5': []}
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_one, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            judge, rec = fut.result()
            results[judge].append(rec)
            if n % 20 == 0:
                print(f"  judged {n}/{len(jobs)} calls", flush=True)
    for judge in results:
        results[judge].sort(key=lambda r: r['i'])

    # summary
    summary = {}
    for judge, rs in results.items():
        n = len(rs)
        p = sum(1 for r in rs if r['pass'])
        summary[judge] = {'pass': p, 'total': n, 'rate': round(p / n, 3) if n else None}
    # agreement
    agree = sum(1 for a, b in zip(results['GLM-5-Turbo'], results['qwen3.5'])
                if a['pass'] == b['pass'])
    summary['agreement'] = f"{agree}/{len(edges)}"
    # both-pass rate (strict) and either-pass (lenient)
    both = sum(1 for a, b in zip(results['GLM-5-Turbo'], results['qwen3.5'])
               if a['pass'] and b['pass'])
    either = sum(1 for a, b in zip(results['GLM-5-Turbo'], results['qwen3.5'])
                 if a['pass'] or b['pass'])
    summary['both_pass'] = f"{both}/{len(edges)}"
    summary['either_pass'] = f"{either}/{len(edges)}"

    out = {'summary': summary, 'edges': edges,
           'glm': results['GLM-5-Turbo'], 'qwen': results['qwen3.5']}
    with open(os.path.join(d, 'judge_cross.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    # failures detail for the report
    print("\n=== both-judge failures ===")
    for a, b, e in zip(results['GLM-5-Turbo'], results['qwen3.5'], edges):
        if not (a['pass'] or b['pass']):
            print(f"[{a['i']}] {e['pt']} | {a['reason'][:60]} | {b['reason'][:60]}")


if __name__ == '__main__':
    judge_edges(sys.argv[1] if len(sys.argv) > 1 else 'REWRITE_PPR_24493BE6E8C2')
