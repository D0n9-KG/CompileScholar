# -*- coding: utf-8 -*-
"""P0-3 PSC (Path Semantic Correctness): 对 coverable gold 边用 GLM-5 判语义对齐.

ERR 只判 (src,tgt,type) id 匹配; PSC 加语义层: lift 是否抽了一条真正表达该 gold
演化关系的边 (接受 type 措辞差异如 background vs extends, 判语义). 区分:
  sem_hit  : lift 有边语义等价 gold 关系
  sem_miss : lift 抽了相关边但语义不对
  no_edge  : lift 没抽两端间的任何边
PSC = sem_hit / coverable
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from granular_agent.llm_client import call_paratera, parse_json_response

GOLD = os.path.join(os.path.dirname(__file__), "gold", "ARFM2024_gold.json")

PSC_PROMPT = """判断下面 lift(系统抽取) 的边是否在语义上表达了 gold 演化关系 (接受 type 措辞差异, 判语义).

gold 关系: {g_src_name} --{g_type}--> {g_tgt_name}
gold 证据: {g_ev}

lift 抽到的该对方法间的边 (可能多条, 可能没有):
{lift_edges}

判断: lift 是否至少有一条边在语义上等价于 gold 关系?
- extends/improves/compares/replaces/adapts 之间若语义吻合算等价
- background 一般不算等价 (除非 gold 也是 background)
- 方向相反但语义吻合也算 (如 A改进B vs B被A改进)

输出 JSON: {{"sem_hit": true/false, "reason": "一句话"}}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lift_json")
    ap.add_argument("err_json", help="eval_err_edges.py 产出 (含 alignment + detail)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--gold", default=None, help="gold json 路径(默认ARFM2024, 需与err_json同gold)")
    args = ap.parse_args()

    lift = json.load(open(args.lift_json, encoding='utf-8'))
    err = json.load(open(args.err_json, encoding='utf-8'))
    gold_path = args.gold or GOLD
    gold = json.load(open(gold_path, encoding='utf-8'))
    gname = {m['id']: m['name'] for m in gold['methods']}
    gevidx = {(ge['src'], ge['tgt'], ge['type']): ge.get('evidence', '') for ge in gold['evolution_edges']}

    al = err['alignment']
    l2g = {a['lift_name']: a.get('gold_id') for a in al if a.get('gold_id') and a['gold_id'] != 'null'}
    # lift edges in gold-id space, grouped by (src,tgt) pair (both directions)
    pair_edges = {}
    for r in lift['relations']:
        s = l2g.get(r.get('src')); t = l2g.get(r.get('tgt'))
        if not s or not t:
            continue
        key = tuple(sorted((s, t)))
        pair_edges.setdefault(key, []).append(f"{r.get('src','')[:25]} --{r.get('relation')}--> {r.get('tgt','')[:25]} | {(r.get('rationale') or '')[:100]}")

    detail = []; sem_hit = 0; no_edge = 0; sem_miss = 0
    for d in err['detail']:
        if d['verdict'] == 'uncovered':
            continue
        src, tgt = d['gold'].split('->'); typ = d['type']
        key = tuple(sorted((src, tgt)))
        ledges = pair_edges.get(key, [])
        g_ev = gevidx.get((src, tgt, typ), '')
        if not ledges:
            no_edge += 1; detail.append({"gold": d['gold'], "type": typ, "psc": "no_edge"})
            continue
        resp = call_paratera(PSC_PROMPT.format(
            g_src_name=gname.get(src, src), g_type=typ, g_tgt_name=gname.get(tgt, tgt),
            g_ev=g_ev[:200], lift_edges="\n".join(f"- {e}" for e in ledges)),
            model="GLM-5-Turbo", max_tokens=200, temperature=0.0)
        o = parse_json_response(resp) or {}
        hit = bool(o.get('sem_hit'))
        if hit:
            sem_hit += 1; psc = "sem_hit"
        else:
            sem_miss += 1; psc = "sem_miss"
        detail.append({"gold": d['gold'], "type": typ, "psc": psc, "reason": o.get('reason', '')[:80]})

    coverable = len(detail)
    psc = sem_hit / coverable if coverable else 0.0
    print(f"\n=== PSC (语义对齐, {coverable} coverable 边) ===")
    for d in detail:
        print(f"  {d['psc']:<9} {d['gold']} ({d['type']}) {d.get('reason','')}")
    print(f"\nPSC = sem_hit/coverable = {sem_hit}/{coverable} = {psc:.3f}")
    print(f"  (对比 ERR_cond edge_hit/coverable = {err['counts']['edge_hit']}/{coverable} = {err['err_cond']:.3f}; PSC>=ERR因接受type措辞差异)")

    tag = args.tag or os.path.splitext(os.path.basename(args.lift_json))[0]
    out = {"tag": tag, "coverable": coverable, "sem_hit": sem_hit, "sem_miss": sem_miss,
           "no_edge": no_edge, "psc": psc, "err_cond": err['err_cond'], "detail": detail}
    outp = os.path.join(os.path.dirname(args.lift_json), f"{tag}_psc.json")
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
