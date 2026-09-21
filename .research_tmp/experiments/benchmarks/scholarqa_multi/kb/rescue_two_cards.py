# -*- coding: utf-8 -*-
"""Targeted rescue for the 2 CARD FAIL papers (deterministic at temp 0, so
plain retries cannot recover — same input, same wall).

- Language_Models_are_Few-Shot_Learners (GPT-3): ledger shows ctok=8000
  exactly on all 5 attempts = max_tokens truncation (31.4k-token input ->
  huge experimental_matrix). Fix: max_tokens 16000.
- Improving_text_embeddings_with_large_language_models_ (E5): calls succeed
  (ctok~5.6k, complete) but parse fails every time. Fix: capture RAW output
  to file for forensics, then parse_json_response + escape repair tiers.

Both write into cards.json on success (same shape as skeleton output).
Raw outputs saved to kb/rescue_raw_<pid>.txt regardless.
"""
import json
import os
import sys

sys.path.insert(0, r"C:\Users\D0n9\Desktop\CompileScholar\src")
from kb_compiler.records.common import MAX_PAPER_CHARS, load_manifest  # noqa: E402
from kb_compiler.records.skeleton import CARD_PROMPT, _repair_json_escapes  # noqa: E402
from kb_infra.llm import call_local, parse_json_response  # noqa: E402

BASE = r"C:/Users/D0n9/Desktop/CompileScholar/.research_tmp/experiments/benchmarks/scholarqa_multi"
TEXTS = os.path.join(BASE, "corpus", "texts")
CARDS = os.path.join(BASE, "kb", "cards.json")
MANIFEST = os.path.join(BASE, "corpus", "manifest.json")

TARGETS = [
    ("Language_Models_are_Few-Shot_Learners", 16000),
    ("Improving_text_embeddings_with_large_language_models_", 16000),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    man = load_manifest(MANIFEST)
    cards = json.load(open(CARDS, encoding="utf-8"))
    for pid, mtok in TARGETS:
        if pid in cards:
            print(f"[{pid[:40]}] already in cards, skip")
            continue
        text = open(os.path.join(TEXTS, pid + ".md"), encoding="utf-8",
                    errors="replace").read()
        title = (man.get(pid) or {}).get("title") or pid
        prompt = (CARD_PROMPT.replace("{title}", title)
                  .replace("{text}", text[:MAX_PAPER_CHARS]))
        print(f"[{pid[:40]}] calling local, max_tokens={mtok} "
              f"({len(text)} chars text)...", flush=True)
        raw = call_local(prompt, model="Qwen3.8-27B", max_tokens=mtok,
                         temperature=0.0, enable_thinking=False) or ""
        rawp = os.path.join(BASE, "kb", f"rescue_raw_{pid[:40]}.txt")
        open(rawp, "w", encoding="utf-8").write(raw)
        print(f"   raw {len(raw)} chars -> {os.path.basename(rawp)}")
        card = parse_json_response(raw)
        tier = "direct"
        if not isinstance(card, dict):
            fixed, nfix = _repair_json_escapes(raw)
            card = parse_json_response(fixed)
            tier = f"escape-repair({nfix})"
        if not isinstance(card, dict):
            print(f"   STILL UNPARSEABLE ({tier}); raw head: {raw[:200]!r}")
            print(f"   raw tail: {raw[-200:]!r}")
            continue
        card["_paper_id"] = pid
        card["_title_used"] = title
        card["_model"] = "local:Qwen3.8-27B"
        card["_rescue"] = {"max_tokens": mtok, "parse_tier": tier}
        mi = card.get("method_identity") or {}
        print(f"   OK via {tier}: id={mi.get('canonical_name') or '-'} "
              f"matrix={len(card.get('experimental_matrix') or [])} "
              f"figtab={len(card.get('figures_tables') or [])} "
              f"related={len(card.get('related_methods') or [])} "
              f"sections={len(card.get('sections') or [])}")
        cards[pid] = card
        json.dump(cards, open(CARDS, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"cards.json total: {len(cards)}")


if __name__ == "__main__":
    main()
