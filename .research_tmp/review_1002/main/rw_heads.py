"""人写 related work 的组织轴：小节/段落标题 vs 目标论文（看组织是否围绕目标论文的设计维度）。"""
import csv
import re

csv.field_size_limit(10**9)
D = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/deepscholar/dsb/dataset/"
pc = list(csv.DictReader(open(D + "paper_content.csv", encoding="utf-8")))
BS = chr(92)
pat = re.compile(re.escape(BS) + r"(?:sub)*section\*?\{([^}]*)\}|"
                 + re.escape(BS) + r"paragraph\*?\{([^}]*)\}|"
                 + re.escape(BS) + r"textbf\{([^}]*)\}")
nsec, out = [], []
for r in pc:
    h = [a or b or c for a, b, c in pat.findall(r["related_works_section"])]
    nsec.append(len(h))
    out.append((r["paper_title"][:60].replace("\n", " "), h[:6]))
print("papers", len(pc), "median heads", sorted(nsec)[len(nsec) // 2],
      "zero:", sum(x == 0 for x in nsec))
for t, h in out[:16]:
    print("-", t, "|", h)
