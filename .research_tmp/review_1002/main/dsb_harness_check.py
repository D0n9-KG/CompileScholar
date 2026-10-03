"""DSB harness 臂答案开头出现乱码的"Web 工具未授权…先凭知识写"——核查有多少题是闭卷写的。"""
import json

P = "C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/cs2/arm_harness_dsb/answers_harness_dsb.json"
d = json.load(open(P, encoding="utf-8"))


def fix(s):
    for enc in ("gbk", "cp936"):
        try:
            return s.encode("latin-1", errors="strict").decode(enc)
        except Exception:
            pass
    try:
        return s.encode("cp1252", errors="ignore").decode("gbk", errors="ignore")
    except Exception:
        return s


n_closed = 0
for r in d:
    head = (r.get("result") or "")[:300]
    fixed = fix(head)
    closed = any(k in fixed for k in ("未获授权", "未授权", "无法", "凭知识", "凭记忆", "权限")) or \
        any(k in head for k in ("WebSearch/WebFetch",))
    n_closed += closed
print("answers:", len(d), "| first-300-char mentions tool-not-authorized / write-from-knowledge:", n_closed)
print("sample decoded head:", fix((d[0].get("result") or "")[:200]))
print("num_turns dist:", sorted(r.get("num_turns") or 0 for r in d)[::8])
