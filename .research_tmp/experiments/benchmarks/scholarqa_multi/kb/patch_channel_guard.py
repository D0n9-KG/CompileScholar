# -*- coding: utf-8 -*-
"""One-shot: channel-death abort guards.

call_json returns None only when ALL retries failed to produce parseable
JSON (quota exhaustion / dead channel / contract collapse). The existing
`or {}` + _covered_groups fallback then fabricates all-singleton groups —
SILENT registry fragmentation, exactly the pathology the scaling design
exists to prevent. None must abort the stage loudly (stop-and-resume
discipline: rerun is cheap, embeddings cached, determinism preserved).
A dict with PARTIAL groups still legitimately uses the coverage fallback
(truncation salvage — that is what it was designed for).
"""
import io

HELPER = '''
class ChannelDeadError(RuntimeError):
    """LLM channel returned unparseable output on all retries (quota
    exhaustion / dead server / contract collapse). Aborts the stage instead
    of letting coverage-fallback fabricate all-singleton groups (silent
    registry fragmentation). Rerun the stage after switching channel —
    embeddings are cached and every stage is deterministic."""


def _must_json(obj, ctx: str):
    if obj is None:
        raise ChannelDeadError(
            f"{ctx}: call_json returned None on all retries — channel dead or "
            f"output contract violated; aborting to prevent singleton flood")
    return obj

'''

# ---------- registry.py ----------
P = r"C:/Users/D0n9/Desktop/CompileScholar/src/kb_compiler/records/registry.py"
s = io.open(P, encoding="utf-8").read()

# helper after _norm/_eid block
anchor = "def _eid(canonical: str) -> str:\n    return hashlib.md5(_norm(canonical).encode(\"utf-8\")).hexdigest()[:12]\n"
assert anchor in s
s = s.replace(anchor, anchor + HELPER, 1)

# 1. round-1 block call
old = '''        obj = call_json(prompt, model, max_tokens=16000, retries=3)
        local = _covered_groups((obj or {}).get("groups"), len(order),
                                f"registry:b{bi}")'''
new = '''        obj = _must_json(call_json(prompt, model, max_tokens=16000, retries=3),
                         f"registry block {bi}")
        local = _covered_groups(obj.get("groups"), len(order),
                                f"registry:b{bi}")'''
assert old in s, "block call site"
s = s.replace(old, new, 1)

# 2. cross-round call
old = '''            prompt = CROSS_MERGE_PROMPT.replace("{lines}", "\\n".join(lines))
            obj = call_json(prompt, model, max_tokens=16000, retries=3)
            return _covered_groups((obj or {}).get("groups"), len(cb),
                                   f"cross:r{rnd}")'''
new = '''            prompt = CROSS_MERGE_PROMPT.replace("{lines}", "\\n".join(lines))
            obj = _must_json(call_json(prompt, model, max_tokens=16000,
                                       retries=3), f"cross round {rnd}")
            return _covered_groups(obj.get("groups"), len(cb),
                                   f"cross:r{rnd}")'''
assert old in s, "cross call site"
s = s.replace(old, new, 1)

# 3. singleton recovery chunk call
old = '''        obj = call_json(SINGLETON_RECOVERY_PROMPT.replace(
            "{lines}", "\\n".join(lines)), model, max_tokens=6000, retries=3)'''
new = '''        obj = _must_json(call_json(SINGLETON_RECOVERY_PROMPT.replace(
            "{lines}", "\\n".join(lines)), model, max_tokens=6000, retries=3),
            "singleton recovery")'''
assert old in s, "recovery call site"
s = s.replace(old, new, 1)

# 4. vocab one-call
old = '''    obj = call_json(prompt, model, max_tokens=12000, retries=3) or {}
    if dim == "subject":'''
new = '''    obj = _must_json(call_json(prompt, model, max_tokens=12000, retries=3),
                     f"vocab:{dim}:{tag}")
    if dim == "subject":'''
assert old in s, "vocab call site"
s = s.replace(old, new, 1)

# 5. cross-merge entries call
old = '''        prompt = prompt_tmpl.replace("{lines}", "\\n".join(lines))
        obj = call_json(prompt, model, max_tokens=12000, retries=3)
        return _covered_groups((obj or {}).get("groups"), len(b), tag)'''
new = '''        prompt = prompt_tmpl.replace("{lines}", "\\n".join(lines))
        obj = _must_json(call_json(prompt, model, max_tokens=12000, retries=3),
                         f"vocab cross-merge:{tag}")
        return _covered_groups(obj.get("groups"), len(b), tag)'''
assert old in s, "cross-merge-entries call site"
s = s.replace(old, new, 1)

# 6. legacy single-call path (gate-3 A/B uses it; same silent-degradation hole)
old6 = """    prompt = MERGE_PROMPT.replace("{n_papers}", str(len(cards))).replace(
        "{lines}", _merge_lines(keys, mentions))
    obj = call_json(prompt, model, max_tokens=16000, retries=3)
    groups = _covered_groups((obj or {}).get("groups"), len(keys), "registry")"""
new6 = """    prompt = MERGE_PROMPT.replace("{n_papers}", str(len(cards))).replace(
        "{lines}", _merge_lines(keys, mentions))
    obj = _must_json(call_json(prompt, model, max_tokens=16000, retries=3),
                     "registry single-call")
    groups = _covered_groups(obj.get("groups"), len(keys), "registry")"""
assert old6 in s, "single-call site"
s = s.replace(old6, new6, 1)

io.open(P, "w", encoding="utf-8").write(s)
print("registry.py guarded")

# ---------- registry_round2.py ----------
P2 = r"C:/Users/D0n9/Desktop/CompileScholar/src/kb_compiler/records/registry_round2.py"
s2 = io.open(P2, encoding="utf-8").read()
old = '''    obj = call_json(MAP_PROMPT.replace("{registry_lines}", registry_lines)
                    .replace("{queue_lines}", queue_lines),
                    model, max_tokens=12000, retries=3, salvage=True) or {}
    return obj.get("assignments") or []'''
new = '''    from .registry import _must_json
    obj = _must_json(call_json(MAP_PROMPT.replace("{registry_lines}", registry_lines)
                               .replace("{queue_lines}", queue_lines),
                               model, max_tokens=12000, retries=3, salvage=True),
                     "round2 mapping batch")
    return obj.get("assignments") or []'''
assert old in s2, "round2 map batch"
s2 = s2.replace(old, new, 1)
old2 = '''            cobj = call_json(_CONSOL_PROMPT + prop_lines, model,
                             max_tokens=8000, retries=2) or {}'''
new2 = '''            from .registry import _must_json
            cobj = _must_json(call_json(_CONSOL_PROMPT + prop_lines, model,
                                        max_tokens=8000, retries=2),
                              "round2 proposal consolidation")'''
assert old2 in s2, "round2 consolidation"
s2 = s2.replace(old2, new2, 1)
io.open(P2, "w", encoding="utf-8").write(s2)
print("registry_round2.py guarded")
