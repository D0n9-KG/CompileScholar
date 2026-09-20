# -*- coding: utf-8 -*-
"""Formula/notation harvest pass (task #12, 2026-09-20).

Why: mineru-parsed texts carry display LaTeX formulas in 93% of papers
(~34/paper measured on PS-53), but the slot pass emits only ~0.7 notation
records/paper (2% coverage) — notation is routed to narrow section types and
competes with higher-salience kinds. The interpretation substrate that
formula-valued config/result records need (schema v1.3's stated purpose for
notation) is therefore mostly missing.

Design (deterministic sandwich, same as every channel here):
- deterministic: display-formula scanning ($$...$$ with char offsets),
  context windows, batching, and the structural gates below;
- LLM proposes only WHICH symbols are defined and what they mean — one call
  per batch of formulas, each record must quote the formula verbatim
  (char-exact LaTeX, rule 1c) plus the defining sentence;
- gates (no LLM): (G1) symbol must appear (math-stripped) inside the quote;
  (G2) the source formula block must be contained in the quote (whitespace-
  normalized — mineru inserts spaces inside LaTeX); (G3) dedup against
  existing notation records per (paper, symbol); (G4) definition non-empty.
  Gate failures are dropped + logged (visible residue, never silent).

CLI:
  python -m kb_compiler.records.notation_harvest --texts DIR \
      --records records_checked.json --out harvested_notation.json \
      [--provider local] [--model Qwen3.8-27B] [--limit N] [--dry]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kb_infra.llm import call_local, call_paratera, parse_json_response

FORMULA_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE_RE = re.compile(r"\$([^$\n]+?)\$")
CTX_BACK, CTX_FWD = 700, 300   # chars of context around the formula
BATCH_SIZE = 6                 # formulas per LLM call
MIN_FORMULA_CHARS = 8          # skip trivial fragments (dollar-sign noise)
# 6/53 PS-53 papers (NEFTUNE family) emit display math with single-$ delimiters
# (incl. long \begin{array} blocks); only SUBSTANTIAL single-$ spans count —
# bare variables ($x$, $\alpha$) are mention noise, not definition candidates.
MIN_INLINE_CHARS = 40


def _ws_norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _math_strip(s: str) -> str:
    """Strip LaTeX decoration so '$\\alpha$' and 'α'/'\\alpha' compare equal.
    \\prime is preserved as an apostrophe BEFORE command-stripping — otherwise
    $X'$ (the derived quantity) and $X$ (the base) collapse to the same
    identity and the dedup gate eats genuinely distinct symbols (live-run
    lesson: the noised embedding $X'_emb$ was rejected as a dup of $X_emb$)."""
    s = s.replace("\\prime", "'")
    s = re.sub(r"\\[A-Za-z]+", " ", s)
    s = re.sub(r"[${}_^\\]", " ", s)
    return _ws_norm(s)


def scan_formulas(text: str) -> list[dict]:
    """Display formulas with offsets + context windows. Primary form: $$...$$
    blocks. Secondary: substantial single-$ spans (>=MIN_INLINE_CHARS) found
    OUTSIDE $$ regions — the NEFTUNE-family papers emit display math with
    single-$ delimiters, incl. \\begin{array} blocks."""
    out, masked = [], []
    for m in FORMULA_RE.finditer(text):
        f = m.group(1).strip()
        if len(f) < MIN_FORMULA_CHARS:
            continue
        masked.append((m.start(), m.end()))
        cs, ce = max(0, m.start() - CTX_BACK), min(len(text), m.end() + CTX_FWD)
        out.append((m.start(), f, text[cs:ce]))
    # single-$ spans on the text with $$ regions blanked out
    def _is_masked(pos):
        return any(s <= pos < e for s, e in masked)
    for m in INLINE_RE.finditer(text):
        if _is_masked(m.start()):
            continue
        f = m.group(1).strip()
        if len(f) < MIN_INLINE_CHARS:
            continue
        cs, ce = max(0, m.start() - CTX_BACK), min(len(text), m.end() + CTX_FWD)
        out.append((m.start(), f, text[cs:ce]))
    out.sort(key=lambda t: t[0])
    return [{"formula": f, "char_start": s, "context": c, "context_start": 0}
            for s, f, c in out]


HARVEST_PROMPT = """你是科学文献知识编译器的符号收割遍。下面是论文《{title}》中的若干公式块（逐字 LaTeX）及其上下文。为其中【被定义的符号】产出 notation 记录。

什么算"被定义"：(a) 上下文中有明确定义语句（"X 表示/定义为/其中 X 是/where X is/X denotes"类措辞）；或 (b) 公式本身定义了某个量（此时 definition 字段填公式本身）。

纪律：
1. 只收被定义的符号；公式里没被定义的中间变量不收。拿不准就不输出。
2. quote-first：quote 字段必须包含公式块逐字符照抄（保留所有反斜杠宏名、花括号、空格，严禁把 \\alpha 写成 α、严禁重排简化），外加定义语句原句（如上下文有）。
3. symbol 逐字取自公式（保留 $ 定界符，如 "$\\\\alpha$"）。
4. 每个公式块产出 0 到数条记录（每个被定义符号一条）。
5. scope_ref 填该方法/模型表面名（上下文有就填，没有留空）。

输出 JSON：{{"records": [{{"kind":"notation","symbol":"...","quantity":"...","definition":"...","unit":"...或空","scope_ref":"...或空","quote":"...","formula_idx":<编号>}}]}}

公式块列表：
{blocks}"""


def build_prompt(title: str, batch: list[dict]) -> str:
    blocks = []
    for i, b in enumerate(batch):
        blocks.append(f"【公式{i}】(char_start={b['char_start']})\n$$ {b['formula']} $$\n【上下文】{b['context']}")
    return HARVEST_PROMPT.replace("{title}", title or "").replace("{blocks}", "\n\n".join(blocks))


def llm_harvest(prompt: str, model: str, provider: str) -> list[dict] | None:
    if provider == "local":
        # enable_thinking=False is load-bearing (F35 canary lesson: visible
        # reasoning eats the token budget before the JSON).
        raw = call_local(prompt, model=model, max_tokens=3000, enable_thinking=False) or ""
    else:
        raw = call_paratera(prompt, model=model, max_tokens=3000) or ""
    obj = parse_json_response(raw)
    if isinstance(obj, dict):
        obj = obj.get("records")
    if not isinstance(obj, list):
        return None
    return [r for r in obj if isinstance(r, dict)]


def gate(rec: dict, batch: list[dict], existing_symbols: set) -> tuple[bool, str]:
    """Structural validation. Returns (ok, reason)."""
    symbol = str(rec.get("symbol") or "").strip()
    quote = str(rec.get("quote") or "")
    definition = str(rec.get("definition") or "").strip()
    if not symbol or not quote or not definition:
        return False, "G4: missing symbol/quote/definition"
    # G1: symbol (math-stripped) inside the quote
    if _math_strip(symbol) and _math_strip(symbol) not in _math_strip(quote):
        return False, f"G1: symbol '{symbol}' not in quote"
    # G2: the source formula must be contained in the quote (ws-normalized)
    idx = rec.get("formula_idx")
    if not isinstance(idx, int) or not (0 <= idx < len(batch)):
        return False, "G2: bad formula_idx"
    if _ws_norm(batch[idx]["formula"]) not in _ws_norm(quote):
        return False, "G2: source formula not contained in quote (verbatim violation)"
    # G3: dedup
    if _math_strip(symbol).lower() in existing_symbols:
        return False, f"G3: duplicate symbol '{symbol}'"
    return True, ""


def load_texts(texts_dir: str) -> dict:
    tx = {}
    if texts_dir and os.path.isdir(texts_dir):
        for fn in os.listdir(texts_dir):
            if fn.endswith((".md", ".txt")):
                with open(os.path.join(texts_dir, fn), encoding="utf-8", errors="replace") as f:
                    tx[fn.rsplit(".", 1)[0]] = f.read()
    return tx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--records", required=True, help="existing records (dict by pid) for dedup")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default="local", choices=["local", "paratera"])
    ap.add_argument("--model", default="Qwen3.8-27B")
    ap.add_argument("--limit", type=int, default=0, help="cap formulas processed (0=all)")
    ap.add_argument("--dry", action="store_true", help="scan only; report formula surface")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    texts = load_texts(args.texts)
    recs_by_paper = json.load(open(args.records, encoding="utf-8"))

    total_f = 0
    existing = {}
    for pid, pv in recs_by_paper.items():
        recs = pv["records"] if isinstance(pv, dict) and "records" in pv else pv
        existing[pid] = {_math_strip(str(r.get("symbol") or "")).lower()
                         for r in recs if r.get("kind") == "notation" and r.get("symbol")}
    for pid, text in texts.items():
        total_f += len(scan_formulas(text))
    print(f"papers={len(texts)} display_formulas={total_f} "
          f"existing_notation_records={sum(len(v) for v in existing.values())}", flush=True)
    if args.dry:
        return

    harvested = {}   # pid -> [notation records]
    rejected = []
    n_forms = 0
    for pid, text in texts.items():
        formulas = scan_formulas(text)
        if args.limit:
            formulas = formulas[:args.limit]
        title = (recs_by_paper.get(pid) or {})
        title = (title.get("_title") if isinstance(title, dict) else None) or pid
        pid_harvest = []
        pid_symbols = set(existing.get(pid, set()))
        for bstart in range(0, len(formulas), BATCH_SIZE):
            batch = formulas[bstart:bstart + BATCH_SIZE]
            prompt = build_prompt(title, batch)
            props = llm_harvest(prompt, args.model, args.provider)
            if props is None:
                rejected.append({"pid": pid, "batch": bstart, "reason": "llm_unparseable"})
                continue
            for p in props:
                ok, reason = gate(p, batch, pid_symbols)
                if ok:
                    rec = {
                        "kind": "notation",
                        "symbol": str(p.get("symbol")).strip(),
                        "quantity": str(p.get("quantity") or "").strip(),
                        "definition": str(p.get("definition")).strip(),
                        "unit": str(p.get("unit") or "").strip(),
                        "scope_ref": str(p.get("scope_ref") or "").strip(),
                        "quote": str(p.get("quote")),
                        "paper_id": pid,
                        "section": "",
                        "chunk_char_start": batch[int(p.get("formula_idx"))]["char_start"],
                        "provenance": "notation_harvest",
                    }
                    rec["method_ref"] = {"surface": rec["scope_ref"]} if rec["scope_ref"] else None
                    pid_harvest.append(rec)
                    pid_symbols.add(_math_strip(rec["symbol"]).lower())
                else:
                    rejected.append({"pid": pid, "symbol": p.get("symbol"), "reason": reason})
            n_forms += len(batch)
            if n_forms % 30 < BATCH_SIZE:
                print(f"  [{n_forms}/{total_f if not args.limit else args.limit*len(texts)}] "
                      f"harvested={sum(len(v) for v in harvested.values())} "
                      f"rejected={len(rejected)}", flush=True)
        if pid_harvest:
            harvested[pid] = pid_harvest

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(harvested, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(rejected, open(args.out.replace(".json", "_rejected.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    n = sum(len(v) for v in harvested.values())
    print(f"DONE papers={len(harvested)} harvested_notation={n} rejected={len(rejected)} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
