"""B0「直写」：27B 一次性生成 related work。输入=官方 DeepScholar-base 查询模板（摘要+600 词+编号引用）
+ oracle 参考文献编号列表（标题+摘要；缺摘要只给标题）。输出 gen/B0/<gt_dir>.md（正文，剥掉末尾参考文献列表）。"""
import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import llm27

OUT = os.path.join("gen", "B0")
ABS_CAP = 1200  # 每篇参考文献摘要截断（18 篇×~1.2k 字符 ≈ 6k tokens，27B 窗口内）

PROMPT = """{query}

You are given the reference papers that this Related Works section should draw on. Use ONLY these references, cite them by their number in square brackets, and do not invent other references.

Reference papers:
{refs}

Write the Related Works section now (academic prose, organized into thematic paragraphs or subsections, at most 600 words). Do NOT include a reference list at the end."""


def fmt_refs(refs):
    out = []
    for i, r in enumerate(refs, 1):
        line = f"[{i}] {r['title']}"
        if r.get("year"):
            line += f" ({r['year'].split('.')[0]})"
        if r.get("abstract"):
            line += f"\n    Abstract: {r['abstract'][:ABS_CAP].strip()}"
        else:
            line += "\n    (abstract unavailable)"
        out.append(line)
    return "\n".join(out)


def strip_ref_list(text):
    # 去掉模型仍可能追加的参考文献列表（"References"/"Reference List" 标题之后）
    m = re.search(r"^\s*(#+\s*)?(\*\*)?(references|reference list|bibliography)(\*\*)?\s*:?\s*$", text, re.I | re.M)
    return text[:m.start()].rstrip() if m else text.strip()


def one(row):
    t0 = time.time()
    out, usage = llm27.chat(PROMPT.format(query=row["query"], refs=fmt_refs(row["refs"])),
                            max_tokens=3000, temperature=0.3)
    text = strip_ref_list(out)
    open(os.path.join(OUT, row["gt_dir"] + ".md"), "w", encoding="utf-8").write(text)
    return row["gt_dir"], len(text.split()), round(time.time() - t0, 1), getattr(usage, "prompt_tokens", None)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = json.load(open("oracle_inputs.json", encoding="utf-8"))
    ids = set(sys.argv[1:])
    rows = [r for r in rows if (not ids or r["gt_dir"] in ids)
            and not os.path.exists(os.path.join(OUT, r["gt_dir"] + ".md"))]
    print(f"[B0] todo {len(rows)}", flush=True)
    log = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(one, r) for r in rows]):
            try:
                d, w, s, pt = f.result()
                log.append({"gt_dir": d, "words": w, "gen_s": s, "prompt_tokens": pt})
                print(f"  {d} words={w} {s}s prompt_tok={pt}", flush=True)
            except Exception as e:
                print(f"  ERROR {str(e)[:150]}", flush=True)
    old = json.load(open("gen_b0_log.json")) if os.path.exists("gen_b0_log.json") else []
    json.dump(old + log, open("gen_b0_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
