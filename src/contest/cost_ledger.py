# -*- coding: utf-8 -*-
"""Cost ledger for the contest pipeline (2026-08-28) — the efficiency-20%
evidence collector. Every LLM call and retrieval goes through the wrappers
here; at query end the ledger reports calls / latency / est. cost per layer.

Usage:
    from contest.cost_ledger import ledger
    raw = ledger.llm("DeepSeek-V4-Flash", prompt, max_tokens=600)   # wraps call_paratera
    ... ledger.http("s2bulk", url)                                   # wraps urllib
    print(ledger.report())
"""
import json, os, time

_LEDGER_PATH = ".research_tmp/contest_survey/cost_ledger.jsonl"


class CostLedger:
    def __init__(self):
        self.events: list[dict] = []
        self.t0 = time.time()

    # --- wrappers ---
    def llm(self, model: str, prompt: str, max_tokens: int = 600, **kw) -> str | None:
        from granular_agent.llm_client import call_paratera
        t0 = time.time()
        out = call_paratera(prompt, model=model, max_tokens=max_tokens, **kw)
        self._record("llm", model=model, dt=time.time() - t0,
                     in_chars=len(prompt), out_chars=len(out or ""),
                     max_tokens=max_tokens)
        return out

    def http(self, source: str, url: str, timeout: float = 30) -> dict | list | None:
        import urllib.request
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LogicKG-research/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            self._record("http", source=source, dt=time.time() - t0, ok=True)
            return d
        except Exception as e:
            self._record("http", source=source, dt=time.time() - t0, ok=False,
                         err=repr(e)[:80])
            return None

    def _record(self, kind: str, **fields):
        ev = {"kind": kind, "t": round(time.time() - self.t0, 2), **fields}
        self.events.append(ev)
        # incremental disk append (中断安全 — the subagent-discipline lesson)
        try:
            os.makedirs(os.path.dirname(_LEDGER_PATH), exist_ok=True)
            with open(_LEDGER_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # --- reporting ---
    def report(self) -> dict:
        from collections import defaultdict
        by = defaultdict(lambda: {"calls": 0, "time_s": 0.0})
        for ev in self.events:
            k = (ev["kind"], ev.get("model") or ev.get("source") or "?")
            by[k]["calls"] += 1
            by[k]["time_s"] += ev.get("dt", 0)
        # Paratera DeepSeek-V4-Flash list price ≈ $0.28/M in + $0.42/M out (CNY 2/3 per M)
        est_cost = 0.0
        for ev in self.events:
            if ev["kind"] == "llm":
                est_cost += ev.get("in_chars", 0) / 4 / 1e6 * 0.28
                est_cost += ev.get("out_chars", 0) / 4 / 1e6 * 0.42
        return {
            "wall_s": round(time.time() - self.t0, 1),
            "breakdown": {f"{k[0]}:{k[1]}": v for k, v in by.items()},
            "n_events": len(self.events),
            "est_cost_usd": round(est_cost, 6),
        }

    def pretty(self) -> str:
        r = self.report()
        lines = [f"cost: {r['wall_s']}s wall, ~${r['est_cost_usd']:.6f}, "
                 f"{r['n_events']} calls"]
        for k, v in r["breakdown"].items():
            lines.append(f"  {k:<28s} {v['calls']:>4} calls  {v['time_s']:>7.1f}s")
        return "\n".join(lines)


ledger = CostLedger()
