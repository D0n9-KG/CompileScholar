# -*- coding: utf-8 -*-
"""诚实 ERR 边级评测 (P0-1, 纠正"3/3类型覆盖"自欺).

口径 (区别于旧 eval_lift_14papers.py 的"类型词出现就算命中"):
  - 方法对齐: lift 方法名 -> gold 53方法(带aliases), LLM 做, null=未抽到
  - 41 gold 演化边逐条判:
      uncovered : lift 没抽到 src 或 tgt 方法 (诚实暴露抽取召回不足)
      miss      : 两端方法都抽到了, 但 lift relations 没有匹配边
      type_only : 有 (src,tgt) 但 type 不对 (方向/type 混对)
      edge_hit  : 精确 (src,tgt,type) 或 (tgt,src,type) 命中
  - ERR_uncond  = edge_hit / 41            (暴露总覆盖)
  - ERR_cond    = edge_hit / coverable      (机制本身准不准)
  - type_only 和 miss 分开报, 不混进 hit

输入: 任意臂的 lift 产出 json (含 methods[].name + relations[].{src,tgt,relation})
输出: 逐条 detail + 汇总数字, 存 *_err.json
"""
import os, sys, json, argparse, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from granular_agent.llm_client import call_paratera, parse_json_response

GOLD = os.path.join(os.path.dirname(__file__), "gold", "ARFM2024_gold.json")

ALIGN_PROMPT = """把下面自动归纳出的【建模方法名】对齐到 gold 方法列表(综述归纳的53方法, 每个带 aliases).
逐个 lift 方法找最匹配的 gold 方法 id (M1-M53), 严格按"是同一个建模方法"判; 若 lift 方法只是 gold 方法的子步骤/参数/不同物理情景但同名本体, 仍算同一方法(用 aliases 区分); 若无任何 gold 方法与之对应写 null.

lift 方法名:
{lift_methods}

gold 方法 (id: name | aliases):
{gold_methods}

输出 JSON: {{"alignment": [{{"lift_name": "...", "gold_id": "M1"或null, "gold_name": "...", "confidence": "high|medium|low|null"}}]}}  顺序与 lift 方法一致.
"""


def align(lift_methods, gold_methods):
    gold_str = "\n".join(f"{m['id']}: {m['name']} | aliases: {', '.join(m.get('aliases', []))}"
                         for m in gold_methods)
    resp = call_paratera(ALIGN_PROMPT.format(
        lift_methods="\n".join(f"- {m}" for m in lift_methods),
        gold_methods=gold_str), model="GLM-5-Turbo", max_tokens=2500, temperature=0.0)
    al = (parse_json_response(resp) or {}).get("alignment", [])
    lift2gold = {}
    for a in al:
        gid = a.get("gold_id")
        if gid and gid != "null":
            lift2gold[a["lift_name"]] = gid
    return al, lift2gold


def _parse_ref(r):
    """gold ref 如 'Jop et al. 2006' -> ('jop', 2006); None if no parse."""
    r = r.replace('é', 'e').replace('ü', 'u').replace('ç', 'c')
    m = re.search(r'([A-Za-z]+)', r); y = re.search(r'(19|20)\d{2}', r)
    if not m or not y:
        return None
    return (m.group(1).lower()[:5], int(y.group(0)))


def fair_recall(gold, lift_cov, papers):
    """公平召回: 14/19篇输入论文涉及的 gold 方法里, lift 抽到多少.
    papers: list of paper name 前缀 (如 'Midi_2004')."""
    inp_set = set(_parse_ref(p.replace('_', ' ')) for p in papers)
    inp_covered, inp_and_lift, lift_miss = 0, 0, []
    for m in gold["methods"]:
        refs = m.get("refs", [])
        in_inp = any(_parse_ref(r) and _parse_ref(r) in inp_set for r in refs)
        in_lift = m["id"] in lift_cov
        if in_inp:
            inp_covered += 1
            if in_lift:
                inp_and_lift += 1
            else:
                lift_miss.append((m["id"], m["name"], [r for r in refs if _parse_ref(r) and _parse_ref(r) in inp_set]))
    return inp_covered, inp_and_lift, lift_miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lift_json", help="lift 产出 json (含 methods[].name, relations[].{src,tgt,relation})")
    ap.add_argument("--tag", default=None, help="输出标签")
    ap.add_argument("--papers", default=None, help="逗号分隔输入论文前缀, 算公平召回(分母=输入涉及的gold方法)")
    ap.add_argument("--gold", default=None, help="gold json 路径(默认ARFM2024)")
    args = ap.parse_args()

    lift = json.load(open(args.lift_json, encoding='utf-8'))
    gold_path = args.gold or GOLD
    gold = json.load(open(gold_path, encoding='utf-8'))
    lift_methods = [m["name"] for m in lift["methods"]]
    lift_rels = lift["relations"]
    gold_edges = gold["evolution_edges"]
    gold_methods = gold["methods"]
    print(f"lift: {len(lift_methods)} methods, {len(lift_rels)} relations")
    print(f"gold: {len(gold_methods)} methods, {len(gold_edges)} evolution edges")

    al, lift2gold = align(lift_methods, gold_methods)
    print(f"\n=== 方法对齐 (lift -> gold) ===")
    for a in al:
        print(f"  {str(a.get('lift_name',''))[:40]} -> {a.get('gold_id')} {str(a.get('gold_name',''))[:25]} ({a.get('confidence')})")
    gold_covered = set(lift2gold.values())
    print(f"\n被 lift 覆盖的 gold 方法: {len(gold_covered)}/53 -> {sorted(gold_covered, key=lambda x:int(x[1:]))}")

    # lift relations -> gold-id space
    lift_gold_rels = set()
    for r in lift_rels:
        s = lift2gold.get(r.get("src", ""))
        t = lift2gold.get(r.get("tgt", ""))
        if s and t:
            lift_gold_rels.add((s, t, r["relation"]))
    print(f"lift relations 映射到 gold-id 空间: {len(lift_gold_rels)} 条 (原 {len(lift_rels)})")

    # 逐条判 41 边
    detail = []
    cnt = {"edge_hit": 0, "type_only": 0, "miss": 0, "uncovered": 0}
    for ge in gold_edges:
        src, tgt, typ = ge["src"], ge["tgt"], ge["type"]
        src_cov, tgt_cov = src in gold_covered, tgt in gold_covered
        if not (src_cov and tgt_cov):
            cnt["uncovered"] += 1
            detail.append({"gold": f"{src}->{tgt}", "type": typ, "verdict": "uncovered",
                           "note": f"缺 {('src' if not src_cov else '')}{(' tgt' if not tgt_cov else '').strip()}"})
            continue
        hit = (src, tgt, typ) in lift_gold_rels or (tgt, src, typ) in lift_gold_rels
        type_match = any(typ == lr[2] for lr in lift_gold_rels)
        if hit:
            cnt["edge_hit"] += 1; v = "edge_hit"
        elif type_match:
            cnt["type_only"] += 1; v = "type_only"
        else:
            cnt["miss"] += 1; v = "miss"
        detail.append({"gold": f"{src}->{tgt}", "type": typ, "verdict": v})

    n_edges = len(gold_edges)
    coverable = n_edges - cnt["uncovered"]
    err_uncond = cnt["edge_hit"] / n_edges
    err_cond = cnt["edge_hit"] / coverable if coverable else 0.0
    print(f"\n{'='*60}\n=== 诚实 ERR (N gold 演化边) ===")
    print(f"  uncovered (方法没抽到):   {cnt['uncovered']}/{n_edges}")
    print(f"  coverable (两端都抽到):   {coverable}/{n_edges}")
    print(f"    └ edge_hit (精确命中):  {cnt['edge_hit']}")
    print(f"    └ type_only (type对边错):{cnt['type_only']}")
    print(f"    └ miss (覆盖但没抽边):   {cnt['miss']}")
    print(f"  ERR_uncond = edge_hit/{n_edges}       = {err_uncond:.3f}")
    print(f"  ERR_cond   = edge_hit/coverable = {err_cond:.3f}")
    print(f"\n  (旧指标 gold_hit='类型词出现'={sum(1 for t in ['extends','improves','compares'] if t in set(r['relation'] for r in lift_rels))}/3 是误导, 已弃用)")

    # uncovered 里哪些 gold 方法没被抽到
    missing_methods = [m['id'] for m in gold_methods if m['id'] not in gold_covered]
    print(f"\n未抽到的 gold 方法 ({len(missing_methods)}/53): {missing_methods[:20]}{' ...' if len(missing_methods)>20 else ''}")

    # 公平召回 (输入论文涉及的 gold 方法里 lift 抽到多少)
    fair = None
    if args.papers:
        papers = [p.strip() for p in args.papers.split(",") if p.strip()]
        inp_cov, inp_lift, lift_miss = fair_recall(gold, gold_covered, papers)
        fair = {"input_methods": inp_cov, "lift_in_input": inp_lift,
                "fair_recall": inp_lift / inp_cov if inp_cov else 0.0,
                "lift_miss_in_input": [(mid, nm[:30], hits) for mid, nm, hits in lift_miss]}
        print(f"\n=== 公平召回 (输入论文涉及的方法为分母) ===")
        print(f"  输入论文涉及的 gold 方法: {inp_cov}/53")
        print(f"  lift 抽到其中: {inp_lift}")
        print(f"  公平召回率 = {inp_lift}/{inp_cov} = {fair['fair_recall']:.3f}")
        print(f"  输入涉及但 lift 漏抽 (真·lift漏抽):")
        for mid, nm, hits in lift_miss:
            print(f"    {mid} {nm} | 输入命中: {hits}")

    tag = args.tag or os.path.splitext(os.path.basename(args.lift_json))[0]
    out = {"tag": tag, "lift_methods": len(lift_methods), "lift_relations": len(lift_rels),
           "alignment": al, "coverable": coverable, "counts": cnt,
           "err_uncond": err_uncond, "err_cond": err_cond, "detail": detail,
           "missing_gold_methods": missing_methods, "fair_recall": fair}
    outp = os.path.join(os.path.dirname(args.lift_json), f"{tag}_err.json")
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
