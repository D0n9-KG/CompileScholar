# -*- coding: utf-8 -*-
"""One-shot patch: parallelize block/batch LLM loops (recovery, vocab,
cross-merge, round2)."""
import io

pr = r"C:/Users/D0n9/Desktop/CompileScholar/src/kb_compiler/records/registry.py"
s = io.open(pr, encoding="utf-8").read()

# ---- 1. _recover_singletons batch loop -> parallel ----
old = '''    merges: dict[int, int] = {}   # singleton entity idx -> target entity idx
    letters = "abcdef"
    for b0 in range(0, len(tasks), batch_size):
        chunk = tasks[b0:b0 + batch_size]
        lines = []
        for pos, (ei, cands) in enumerate(chunk):
            e = entities[ei]
            cpart = " ".join(
                "[{}] {}(别名: {})".format(
                    letters[ci], entities[cj]["canonical"],
                    (", ".join(a for a in entities[cj]["aliases"]
                               if a != entities[cj]["canonical"])[:60] or "-"))
                for ci, cj in enumerate(cands))
            lines.append("[{}] {} | 提及篇数 {} | 候选: {}".format(
                pos, e["canonical"], e["mention_count"], cpart))
        obj = call_json(SINGLETON_RECOVERY_PROMPT.replace(
            "{lines}", "\\n".join(lines)), model, max_tokens=6000, retries=3)
        got = {}
        for a in (obj or {}).get("assignments", []):
            if isinstance(a, dict) and isinstance(a.get("i"), int):
                got.setdefault(a["i"], a.get("match"))
        for pos, (ei, cands) in enumerate(chunk):
            m = got.get(pos)
            if isinstance(m, str) and m in letters[:len(cands)]:
                merges[ei] = cands[letters.index(m)]'''
new = '''    merges: dict[int, int] = {}   # singleton entity idx -> target entity idx
    letters = "abcdef"
    chunks = [tasks[b0:b0 + batch_size] for b0 in range(0, len(tasks), batch_size)]

    def _run_chunk(chunk):
        lines = []
        for pos, (ei, cands) in enumerate(chunk):
            e = entities[ei]
            cpart = " ".join(
                "[{}] {}(别名: {})".format(
                    letters[ci], entities[cj]["canonical"],
                    (", ".join(a for a in entities[cj]["aliases"]
                               if a != entities[cj]["canonical"])[:60] or "-"))
                for ci, cj in enumerate(cands))
            lines.append("[{}] {} | 提及篇数 {} | 候选: {}".format(
                pos, e["canonical"], e["mention_count"], cpart))
        obj = call_json(SINGLETON_RECOVERY_PROMPT.replace(
            "{lines}", "\\n".join(lines)), model, max_tokens=6000, retries=3)
        got = {}
        for a in (obj or {}).get("assignments", []):
            if isinstance(a, dict) and isinstance(a.get("i"), int):
                got.setdefault(a["i"], a.get("match"))
        out = {}
        for pos, (ei, cands) in enumerate(chunk):
            m = got.get(pos)
            if isinstance(m, str) and m in letters[:len(cands)]:
                out[ei] = cands[letters.index(m)]
        return out

    for chunk_merges in par_map(_run_chunk, chunks):
        merges.update(chunk_merges)'''
assert old in s, "recovery loop not found"
s = s.replace(old, new, 1)

# ---- 2. build_vocab batch loop -> parallel ----
old2 = '''        raw = []   # subject: {family, members(idx)}; others: {canonical(idx), members(idx)}
        for bi, batch in enumerate(batches):
            sub = [items[i] for i in batch]
            groups = _vocab_one_call(dim, sub, model, f"{batch_tag}{bi}")
            for g in groups:'''
new2 = '''        raw = []   # subject: {family, members(idx)}; others: {canonical(idx), members(idx)}
        batch_groups = par_map(
            lambda bb: _vocab_one_call(dim, [items[i] for i in bb[1]], model,
                                       f"{batch_tag}{bb[0]}"),
            list(enumerate(batches)))
        for bi, groups in enumerate(batch_groups):
            batch = batches[bi]
            for g in groups:'''
assert old2 in s, "vocab batch loop not found"
s = s.replace(old2, new2, 1)

# ---- 3. _cross_merge_entries block loop -> parallel ----
old3 = '''    out_groups, n_merges = [], 0
    for b in blocks:
        if len(b) == 1:
            out_groups.append(list(b))
            continue
        lines = [entries[ei]["_line"](pos) for pos, ei in enumerate(b)]
        prompt = prompt_tmpl.replace("{lines}", "\\n".join(lines))
        obj = call_json(prompt, model, max_tokens=12000, retries=3)
        local = _covered_groups((obj or {}).get("groups"), len(b), tag)
        for g in local:
            gidx = [b[m] for m in g["members"]]
            n_merges += len(gidx) - 1
            # canonical entry leads the group
            can = b[g["canonical"]] if g.get("canonical") in range(len(b)) else gidx[0]
            out_groups.append([can] + [i for i in gidx if i != can])
    return out_groups, n_merges'''
new3 = '''    call_blocks = [b for b in blocks if len(b) > 1]

    def _run(b):
        lines = [entries[ei]["_line"](pos) for pos, ei in enumerate(b)]
        prompt = prompt_tmpl.replace("{lines}", "\\n".join(lines))
        obj = call_json(prompt, model, max_tokens=12000, retries=3)
        return _covered_groups((obj or {}).get("groups"), len(b), tag)

    results = dict(zip(range(len(call_blocks)), par_map(_run, call_blocks)))
    out_groups, n_merges = [], 0
    ci = 0
    for b in blocks:          # assembly in deterministic block order
        if len(b) == 1:
            out_groups.append(list(b))
            continue
        local = results[ci]
        ci += 1
        for g in local:
            gidx = [b[m] for m in g["members"]]
            n_merges += len(gidx) - 1
            # canonical entry leads the group
            can = b[g["canonical"]] if g.get("canonical") in range(len(b)) else gidx[0]
            out_groups.append([can] + [i for i in gidx if i != can])
    return out_groups, n_merges'''
assert old3 in s, "cross_merge loop not found"
s = s.replace(old3, new3, 1)
io.open(pr, "w", encoding="utf-8").write(s)
print("registry parallelized")

# ---- 4. round2 chunk loop -> parallel ----
p2 = r"C:/Users/D0n9/Desktop/CompileScholar/src/kb_compiler/records/registry_round2.py"
s2 = io.open(p2, encoding="utf-8").read()
old4 = '''    assigns = []
    for bn, (ch, registry_lines) in enumerate(chunks):
        batch = [queue[i] for i in ch]
        got = _map_batch(batch, registry_lines, model)
        by_i_batch = {}
        for a in got:
            if isinstance(a, dict) and isinstance(a.get("i"), int):
                by_i_batch.setdefault(a["i"], a)
        for i in range(len(batch)):
            a = dict(by_i_batch.get(i) or {"action": "new", "canonical": batch[i]["surface"],
                                           "entity_type": "method", "note": "batch_unassigned"})
            a["_qidx"] = ch[i]
            assigns.append(a)
        print(f"  batch {bn + 1}/{len(chunks)}: {len(batch)} surfaces, "
              f"{sum(1 for i in range(len(batch)) if i in by_i_batch)} assigned", flush=True)'''
new4 = '''    from .common import par_map

    def _run_chunk(item):
        bn, (ch, registry_lines) = item
        batch = [queue[i] for i in ch]
        got = _map_batch(batch, registry_lines, model)
        by_i_batch = {}
        for a in got:
            if isinstance(a, dict) and isinstance(a.get("i"), int):
                by_i_batch.setdefault(a["i"], a)
        out = []
        for i in range(len(batch)):
            a = dict(by_i_batch.get(i) or {"action": "new", "canonical": batch[i]["surface"],
                                           "entity_type": "method", "note": "batch_unassigned"})
            a["_qidx"] = ch[i]
            out.append(a)
        n_assigned = sum(1 for i in range(len(batch)) if i in by_i_batch)
        return bn, len(batch), n_assigned, out

    assigns = []
    results = par_map(_run_chunk, list(enumerate(chunks)))
    for bn, nb, n_assigned, out in results:
        assigns.extend(out)
        if (bn + 1) % 10 == 0 or bn == len(chunks) - 1:
            print(f"  batch {bn + 1}/{len(chunks)}: {nb} surfaces, "
                  f"{n_assigned} assigned", flush=True)'''
assert old4 in s2, "round2 chunk loop not found"
s2 = s2.replace(old4, new4, 1)
io.open(p2, "w", encoding="utf-8").write(s2)
print("round2 parallelized")
