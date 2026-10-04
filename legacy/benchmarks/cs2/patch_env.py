# -*- coding: utf-8 -*-
"""_ensure_env 防御性修复：token 总是从 .env 覆盖。"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
H = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\_shared\tools\external_tools.py"
src = open(H, encoding="utf-8").read()
old = (
    'def _ensure_env():\n'
    '    if not os.environ.get("SCIVERSE_API_TOKEN"):\n'
    '        try:\n'
    '            for line in open(os.path.join(\n'
    '                    r"C:\\Users\\D0n9\\Desktop\\CompileScholar", ".env"),\n'
    '                    encoding="utf-8"):\n'
    '                if line.strip().startswith("SCIVERSE_API_TOKEN"):\n'
    '                    _, _, v = line.strip().partition("=")\n'
    '                    os.environ["SCIVERSE_API_TOKEN"] = v.strip()\n'
    '        except Exception:\n'
    '            pass'
)
new = (
    'def _ensure_env():\n'
    '    # .env 是唯一真源（批 8 教训：环境残留的坏 token 会绕过\n'
    '    # not-get 检查——总是覆盖）\n'
    '    try:\n'
    '        for line in open(os.path.join(\n'
    '                r"C:\\Users\\D0n9\\Desktop\\CompileScholar", ".env"),\n'
    '                encoding="utf-8"):\n'
    '            if line.strip().startswith("SCIVERSE_API_TOKEN"):\n'
    '                _, _, v = line.strip().partition("=")\n'
    '                if v.strip():\n'
    '                    os.environ["SCIVERSE_API_TOKEN"] = v.strip()\n'
    '    except Exception:\n'
    '        pass'
)
assert old in src, "old block not found"
src = src.replace(old, new)
open(H, "w", encoding="utf-8").write(src)
print("patched: token 总是从 .env 覆盖")
