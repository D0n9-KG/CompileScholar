"""B0 的两个机制变体（同输入、同 27B、同长度上限），检验 P5 指出的两个综合层失分点：
  B0+CAT「完整类别枚举」：在 B0 提示上加一条——每个主题段先写出该类方法的完整类别清单（全部子类/代表方法名），
        再展开；并在段尾写出该类方法的共同局限（若参考文献支持）。对应 P5 的 c1 部分枚举(7/37) + c2 共同局限。
  B0+POS「段尾定位」：在 B0 提示上加一条——每段末尾一句说明本文相对该类工作的不同（对应 c1 目标定位）。
输出 p6/gen/<SYS>/<gt_dir>.md；判分沿用 p6/judge_nuggets.py。
用法：python b0_variants.py CAT|POS
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\review_1002\p6")
import llm27  # noqa: E402
from gen_b0 import PROMPT, fmt_refs, strip_ref_list  # noqa: E402

P6 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\review_1002\p6"
EXTRA = {
    "CAT": ("\n\nAdditional requirements: for each thematic paragraph, (a) open with a category-level statement that "
            "names ALL the sub-families or representative approaches of that theme found in the references (complete "
            "enumeration, not just the most prominent ones), and (b) where the references support it, close with the "
            "shared limitation or open problem common to that family of approaches."),
    "POS": ("\n\nAdditional requirement: end each thematic paragraph with one sentence that states how the present paper "
            "differs from or builds on that line of work, based on the paper abstract above."),
}


def one(row, sysname):
    out_dir = os.path.join(P6, "gen", "B0" + sysname)
    t0 = time.time()
    prompt = PROMPT.format(query=row["query"], refs=fmt_refs(row["refs"])) + EXTRA[sysname]
    out, _ = llm27.chat(prompt, max_tokens=3000, temperature=0.3)
    text = strip_ref_list(out)
    open(os.path.join(out_dir, row["gt_dir"] + ".md"), "w", encoding="utf-8").write(text)
    return row["gt_dir"], len(text.split()), round(time.time() - t0, 1)


def main():
    sysname = sys.argv[1]
    out_dir = os.path.join(P6, "gen", "B0" + sysname)
    os.makedirs(out_dir, exist_ok=True)
    rows = json.load(open(os.path.join(P6, "oracle_inputs.json"), encoding="utf-8"))
    rows = [r for r in rows if not os.path.exists(os.path.join(out_dir, r["gt_dir"] + ".md"))]
    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(one, r, sysname) for r in rows]):
            try:
                print(*f.result(), flush=True)
            except Exception as e:
                print("ERR", str(e)[:150], flush=True)


if __name__ == "__main__":
    main()
