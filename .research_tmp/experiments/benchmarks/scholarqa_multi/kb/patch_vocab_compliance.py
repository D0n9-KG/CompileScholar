# -*- coding: utf-8 -*-
"""One-shot: vocab batch compliance retry + smaller batch cap.

Forensics (bio#0 batch, raw captured in forensic_bio0_raw.txt): the model
returned a CONTRACT-PERFECT but LAZY response — {"families": [2 families
covering 5 of 47 items]}. Rule "每个编号必须出现恰好一次" ignored; the other
42 items hit coverage-fallback singletons. Temp-0 retry with the same prompt
reproduces the same laziness, so the retry changes the prompt: a completion
addendum listing exactly the unassigned items (indexed 0..m-1) and asking
for their family assignment ONLY; results map back to original indices and
merge with attempt-1 groups.

Also: VOCAB_MAX_ONE_CALL 600 -> 300 (domain5 batch of 600 lost 453 items —
compliance degrades with batch size; smaller batches = more calls, free).
"""
import io

P = r"C:/Users/D0n9/Desktop/CompileScholar/src/kb_compiler/records/registry.py"
s = io.open(P, encoding="utf-8").read()

old = 'VOCAB_MAX_ONE_CALL = 600   # output-budget bound: ~10-15 tokens/item at 12k cap'
new = ('VOCAB_MAX_ONE_CALL = 300   # compliance bound (09-21 forensics): a 600-item\n'
       '                           # batch lost 453 items to lazy output; 47-item\n'
       '                           # batches ALSO lapse — the completion retry below\n'
       '                           # is the real fix, smaller batches just reduce rate')
assert old in s
s = s.replace(old, new, 1)

old = '''def _vocab_one_call(dim: str, items: list, model: str, tag: str) -> list[dict]:
    """One VOCAB_PROMPT call over `items` [(key, cand)]; returns covered
    groups with local indices (subject: family groups, others: canonical
    groups). Shared by legacy single-call and per-batch scaled paths."""
    extra, shape = DIM_TASKS[dim]
    lines = _lines([(i, c["surface"], len(c["papers"]))
                    for i, (k, c) in enumerate(items)])
    prompt = (VOCAB_PROMPT.replace("{dim_label}", DIM_LABELS[dim])
              .replace("{extra_task}", extra).replace("{out_shape}", shape)
              .replace("{lines}", lines))
    obj = _must_json(call_json(prompt, model, max_tokens=12000, retries=3),
                     f"vocab:{dim}:{tag}")
    if dim == "subject":
        raw = obj.get("families")
        if raw is None:
            # wrong-key tolerance (431-run: model returned "groups" for a
            # families prompt -> 344/384 silent singleton fallback)
            raw = obj.get("groups")
        if raw is None:
            raise ChannelDeadError(
                f"vocab:{dim}:{tag}: response has neither 'families' nor "
                f"'groups' — aborting instead of silent all-singleton fallback")
        groups = [{"canonical": None, "members": g.get("members") or [],
                   "family": g.get("family")} for g in raw]
    else:
        groups = obj.get("groups")
        if groups is None:
            groups = obj.get("families")
        if groups is None:
            raise ChannelDeadError(
                f"vocab:{dim}:{tag}: response has neither 'groups' nor "
                f"'families' — aborting instead of silent all-singleton fallback")
    return _covered_groups(groups, len(items), f"vocab:{dim}:{tag}")'''

new = '''def _vocab_extract(obj: dict, dim: str, ctx: str) -> list:
    """Key-tolerant group extraction (431-run: models cross 'families' and
    'groups' keys); loud abort when neither is present."""
    if dim == "subject":
        raw = obj.get("families")
        if raw is None:
            raw = obj.get("groups")
        if raw is None:
            raise ChannelDeadError(
                f"{ctx}: response has neither 'families' nor 'groups' — "
                f"aborting instead of silent all-singleton fallback")
        return [{"canonical": None, "members": g.get("members") or [],
                 "family": g.get("family")} for g in raw if isinstance(g, dict)]
    groups = obj.get("groups")
    if groups is None:
        groups = obj.get("families")
    if groups is None:
        raise ChannelDeadError(
            f"{ctx}: response has neither 'groups' nor 'families' — "
            f"aborting instead of silent all-singleton fallback")
    return [g for g in groups if isinstance(g, dict)]


def _vocab_one_call(dim: str, items: list, model: str, tag: str,
                    compliance_tries: int = 3) -> list[dict]:
    """One VOCAB_PROMPT call over `items` [(key, cand)]; returns covered
    groups with local indices (subject: family groups, others: canonical
    groups). Shared by legacy single-call and per-batch scaled paths.

    Lazy-output compliance loop (09-21 forensics, bio#0: contract-perfect
    response assigned 5/47 items): if a response leaves >30% of indices
    unassigned, retry asking ONLY for the missing items (completion addendum,
    re-indexed 0..m-1, mapped back). Attempt-1 groups are kept; completion
    groups are appended (cross-merging a late item into an early family is
    lost — acceptable degradation vs singleton flood). Final attempt accepts
    whatever coverage exists (repair tier) with a loud warning."""
    ctx = f"vocab:{dim}:{tag}"
    extra, shape = DIM_TASKS[dim]
    lines = _lines([(i, c["surface"], len(c["papers"]))
                    for i, (k, c) in enumerate(items)])
    prompt = (VOCAB_PROMPT.replace("{dim_label}", DIM_LABELS[dim])
              .replace("{extra_task}", extra).replace("{out_shape}", shape)
              .replace("{lines}", lines))
    accepted = []
    pending_idx = list(range(len(items)))
    for attempt in range(compliance_tries):
        if not pending_idx:
            break
        if attempt == 0:
            cur_prompt = prompt
            sub_items = items
            local_to_orig = {i: i for i in range(len(items))}
        else:
            sub_items = [items[i] for i in pending_idx]
            local_to_orig = {li: oi for li, oi in enumerate(pending_idx)}
            cur_prompt = (prompt + "\\n\\n警告：上一次输出漏掉了大量编号（规则要求"
                          "每个编号恰好出现一次）。本次只需对下面重新编号的候选值"
                          "给出归组（编号 0-" + str(len(sub_items) - 1) + "）。\\n"
                          + _lines([(li, c["surface"], len(c["papers"]))
                                    for li, (k, c) in enumerate(sub_items)])) \\
                if False else (
                    VOCAB_PROMPT.replace("{dim_label}", DIM_LABELS[dim])
                    .replace("{extra_task}", extra)
                    .replace("{out_shape}", shape)
                    .replace("{lines}", _lines(
                        [(li, c["surface"], len(c["papers"]))
                         for li, (k, c) in enumerate(sub_items)]))
                    + f"\\n\\n（第 {attempt + 1} 次补全调用：以下 {len(sub_items)} "
                      f"个候选值在之前的输出中未被分配。请只对这些编号（0-"
                      f"{len(sub_items) - 1}）完成归组，每个编号恰好出现一次。）")
        obj = _must_json(call_json(cur_prompt, model, max_tokens=12000,
                                   retries=3), ctx)
        groups = _vocab_extract(obj, dim, ctx)
        covered_local = set()
        for g in groups:
            mapped = []
            for i in (g.get("members") or []):
                if isinstance(i, str) and i.strip().lstrip("-").isdigit():
                    i = int(i.strip())
                if isinstance(i, int) and 0 <= i < len(sub_items):
                    mapped.append(local_to_orig[i])
                    covered_local.add(i)
            if mapped:
                gg = dict(g)
                gg["members"] = mapped
                if isinstance(g.get("canonical"), int) and g["canonical"] in local_to_orig:
                    gg["canonical"] = local_to_orig[g["canonical"]]
                elif gg.get("canonical") is not None:
                    gg["canonical"] = local_to_orig.get(gg["canonical"], mapped[0])
                accepted.append(gg)
        missing = [i for i in range(len(items))
                   if i not in {m for g in accepted for m in g["members"]}]
        if not missing or len(missing) <= 0.3 * len(items):
            pending_idx = missing
            break
        pending_idx = missing
    if pending_idx and len(pending_idx) > 0.3 * len(items):
        print(f"WARNING: {ctx}: {len(pending_idx)}/{len(items)} items still "
              f"unassigned after {compliance_tries} tries — coverage fallback "
              f"will singleton them", flush=True)
    return _covered_groups(accepted, len(items), ctx)'''
assert old in s, "vocab_one_call site"
s = s.replace(old, new, 1)
io.open(P, "w", encoding="utf-8").write(s)
print("vocab compliance retry installed")
