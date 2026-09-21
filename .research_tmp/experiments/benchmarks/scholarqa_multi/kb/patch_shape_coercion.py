# -*- coding: utf-8 -*-
"""One-shot: output-shape coercion + vocab resume.

431-run forensics: (1) dedup pass crashed on a BARE LIST response (model
emitted [...] not {"groups": [...]}); (2) three subject batches silently
fell back to 344/384, 42/47 singletons — model returned the wrong KEY
("groups" instead of "families"), obj.get("families") -> None ->
_covered_groups(None) fabricated all-singletons. Same silent-degradation
species the channel guard was built for, one level subtler: dict-but-wrong-
shape. Fix: shape-sniffing coercion + loud abort when NEITHER expected key
is present; --reuse-registry so the vocab rerun doesn't redo 1.04M tokens
of merge work.
"""
import io

P = r"C:/Users/D0n9/Desktop/CompileScholar/src/kb_compiler/records/registry.py"
s = io.open(P, encoding="utf-8").read()

# 1. upgrade _must_json -> shape coercion
old = '''def _must_json(obj, ctx: str):
    if obj is None:
        raise ChannelDeadError(
            f"{ctx}: call_json returned None on all retries — channel dead or "
            f"output contract violated; aborting to prevent singleton flood")
    return obj'''
new = '''def _must_json(obj, ctx: str):
    """None -> abort (channel dead). Bare list -> sniff into the expected
    wrapper (431-run: models emit [...] instead of {"groups": [...]}).
    Dict passes through; key-level fallbacks live at the call sites."""
    if obj is None:
        raise ChannelDeadError(
            f"{ctx}: call_json returned None on all retries — channel dead or "
            f"output contract violated; aborting to prevent singleton flood")
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            if any("members" in x for x in obj):
                return {"groups": obj}
            if all("i" in x for x in obj):
                return {"assignments": obj}
            if any("family" in x for x in obj):
                return {"families": obj}
        raise ChannelDeadError(
            f"{ctx}: bare list with unrecognized element shape — aborting")
    if not isinstance(obj, dict):
        raise ChannelDeadError(f"{ctx}: unexpected response type {type(obj).__name__}")
    return obj'''
assert old in s, "must_json site"
s = s.replace(old, new, 1)

# 2. _vocab_one_call: wrong-key tolerance + loud abort when neither key
old = '''    obj = _must_json(call_json(prompt, model, max_tokens=12000, retries=3),
                     f"vocab:{dim}:{tag}")
    if dim == "subject":
        groups = [{"canonical": None, "members": g.get("members") or [],
                   "family": g.get("family")} for g in (obj.get("families") or [])]
    else:
        groups = obj.get("groups")
    return _covered_groups(groups, len(items), f"vocab:{dim}:{tag}")'''
new = '''    obj = _must_json(call_json(prompt, model, max_tokens=12000, retries=3),
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
assert old in s, "vocab one-call site"
s = s.replace(old, new, 1)

# 3. --reuse-registry resume in main()
old = '''    ap.add_argument("--manifest", default="",
                    help="manifest.json for vocab per-domain batching (pid->subject)")
    args = ap.parse_args()'''
new = '''    ap.add_argument("--manifest", default="",
                    help="manifest.json for vocab per-domain batching (pid->subject)")
    ap.add_argument("--reuse-registry", action="store_true",
                    help="skip entity merge; load OUT_DIR/registry.json and "
                         "rebuild only the vocab (crash-resume: the merge is "
                         "the expensive half)")
    args = ap.parse_args()'''
assert old in s, "args site"
s = s.replace(old, new, 1)

old = '''    cards = load_json(args.cards, {})
    mentions = collect_mentions(cards)
    print(f"registry: {len(cards)} cards, {len(mentions)} unique surfaces, "
          f"model={args.model}", flush=True)

    mode = args.merge_mode
    if mode == "auto":
        mode = "blocked" if len(mentions) > MAX_SURFACES_ONE_CALL else "single"
    qc = {"mode": mode, "n_surfaces": len(mentions)}
    if mode == "blocked":
        cache = args.embed_cache or os.path.join(args.out_dir, "embed_cache")
        os.makedirs(cache, exist_ok=True)
        entities, qc = merge_entities_blocked(mentions, cards, args.model,
                                              cache, tau1=args.tau1,
                                              tau2=args.tau2)
    else:
        entities = merge_entities(mentions, cards, args.model)
    surface_index = {}
    for e in entities:
        for a in e["aliases"]:
            surface_index[_norm(a)] = e["entity_id"]'''
new = '''    cards = load_json(args.cards, {})
    cache = args.embed_cache or os.path.join(args.out_dir, "embed_cache")
    if args.reuse_registry:
        registry = load_json(args.out_dir + "/registry.json", None)
        if not registry:
            print("ERROR: --reuse-registry but no registry.json in out-dir",
                  flush=True)
            return
        entities = registry["entities"]
        qc = load_json(args.out_dir + "/registry_qc.json", {}) or {}
        qc["reused"] = True
        mode = qc.get("mode", "blocked")
        print(f"registry: REUSED {len(entities)} entities from "
              f"{args.out_dir}/registry.json (vocab-only rerun)", flush=True)
    else:
        mentions = collect_mentions(cards)
        print(f"registry: {len(cards)} cards, {len(mentions)} unique surfaces, "
              f"model={args.model}", flush=True)
        mode = args.merge_mode
        if mode == "auto":
            mode = "blocked" if len(mentions) > MAX_SURFACES_ONE_CALL else "single"
        qc = {"mode": mode, "n_surfaces": len(mentions)}
        if mode == "blocked":
            os.makedirs(cache, exist_ok=True)
            entities, qc = merge_entities_blocked(mentions, cards, args.model,
                                                  cache, tau1=args.tau1,
                                                  tau2=args.tau2)
        else:
            entities = merge_entities(mentions, cards, args.model)
    surface_index = {}
    for e in entities:
        for a in e["aliases"]:
            surface_index[_norm(a)] = e["entity_id"]'''
assert old in s, "main merge site"
s = s.replace(old, new, 1)

# 4. vocab embed cache dir: reuse `cache` var (was recomputed)
old = '''    cache = args.embed_cache or os.path.join(args.out_dir, "embed_cache")
    vocab = build_vocab(cards, args.model, pid_subject=pid_subject or None,
                        embed_cache_dir=cache if mode == "blocked" else None,
                        tau2=args.tau2)'''
new = '''    vocab = build_vocab(cards, args.model, pid_subject=pid_subject or None,
                        embed_cache_dir=cache if mode == "blocked" else None,
                        tau2=args.tau2)'''
assert old in s, "vocab cache site"
s = s.replace(old, new, 1)

io.open(P, "w", encoding="utf-8").write(s)
print("registry.py patched")
