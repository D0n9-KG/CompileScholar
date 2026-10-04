# -*- coding: utf-8 -*-
"""行级 patch：deep_read 通道 3 的独立 import（规避 heredoc 转义）。"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
H = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\_shared\tools\external_tools.py"
lines = open(H, encoding="utf-8").readlines()
for i, l in enumerate(lines):
    if "text = _fetch_text(_aid)" in l:
        indent = " " * 36
        imp = [
            indent + "import sys as _sys\n",
            indent + '_sys.path.insert(\n',
            indent + '    0, "C:/Users/D0n9/Desktop/CompileScholar'
            '/.research_tmp/experiments/benchmarks/cs2/base_kb_build")\n',
            indent + "from fetch_arxiv_html import fetch_text as _ft3\n",
        ]
        lines[i] = l.replace("_fetch_text(_aid)", "_ft3(_aid)")
        lines[i:i] = imp
        break
open(H, "w", encoding="utf-8").writelines(lines)
print("patched")
