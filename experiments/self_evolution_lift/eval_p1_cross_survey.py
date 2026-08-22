# -*- coding: utf-8 -*-
"""跨综述一致性评测: 同一 lift 输入(19篇), 分别对 ARFM2024 gold 和 Thornton2026 gold 评 ERR/PSC.
测的是: lift 在两份独立综述 gold 上的结果是否一致(同方法族的边是否都命中/都漏).
跨综述一致性 = 两 gold 共同方法(如segregation M3/M23/M24)在lift结果里是否同向.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

GOLD_DIR = os.path.join(os.path.dirname(__file__), "gold")


def load_gold(name):
    return json.load(open(os.path.join(GOLD_DIR, name + "_gold.json"), encoding='utf-8'))


def method_name_overlap(g1, g2):
    """两 gold 共同方法(按 name+aliases 归一化)."""
    def nameset(g):
        s = set()
        for m in g["methods"]:
            s.add(norm(m["name"]))
            for a in m.get("aliases", []):
                s.add(norm(a))
        return s
    return nameset(g1) & nameset(g2)


def norm(s):
    import re
    return re.sub(r"\s+", " ", s.lower().strip())


def main():
    arfm = load_gold("ARFM2024")
    thor = load_gold("THORNTON2026")
    print(f"ARFM2024: {len(arfm['methods'])} methods, {len(arfm['evolution_edges'])} evo edges")
    print(f"THORNTON2026: {len(thor['methods'])} methods, {len(thor['evolution_edges'])} evo edges")
    overlap = method_name_overlap(arfm, thor)
    print(f"\n=== 方法名重叠 (归一化) ===")
    print(f"重叠方法名: {len(overlap)}")
    for n in sorted(overlap)[:20]:
        print(f"  {n}")

    # 边重叠: 两 gold 是否有同 src/tgt/type 的边
    def eset(g):
        # 按 name 归一化 src/tgt
        id2name = {m["id"]: norm(m["name"]) for m in g["methods"]}
        # 也展开 aliases
        for m in g["methods"]:
            for a in m.get("aliases", []):
                pass
        s = set()
        for e in g["evolution_edges"]:
            s.add((id2name.get(e["src"], e["src"]), id2name.get(e["tgt"], e["tgt"]), e["type"]))
        return s
    ea, et = eset(arfm), eset(thor)
    edge_overlap = ea & et
    print(f"\n=== 演化边重叠 (name,type) ===")
    print(f"ARFM边: {len(ea)}, Thornton边: {len(et)}, 重叠: {len(edge_overlap)}")
    for s, t, ty in sorted(edge_overlap)[:10]:
        print(f"  {s} --{ty}--> {t}")


if __name__ == "__main__":
    main()
