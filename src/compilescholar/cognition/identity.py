# -*- coding: utf-8 -*-
"""Method identity at T (L4, DESIGN-LITERATURE-LAYER N4): a method is anchored to the paper that proposed it.

Evidence (pure function of AsOf(T)):
  self  a paper's own "proposes X" statement (self pass; validated proposal sentence)          -> X anchored to it
  other citation sentences that cite paper P and name it ("LoRA [12]"): meta.name on statements about P -> alias
Resolution of a surface name -> paper: normalized exact match against anchored names and aliases; a name anchored to
more than one paper is ambiguous and never resolved; names flagged generic (transformer, LSTM, ...) are never anchored.
Output is used by lineage (third-party "X improves Y" where Y is only named) and by tools (method -> paper)."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..compile.skeleton.proposes import GENERIC
from .asof import AsOf


def norm_name(s: str) -> str:
    s = re.sub(r"[‐-―]", "-", (s or "").lower())
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = re.sub(r"[^a-z0-9+\-. ]+", " ", s)
    s = re.sub(r"\b(the|a|an|model|method|framework|approach|algorithm|network)s?\b", " ", s)
    return " ".join(s.split()).strip(" .-")


class Identity:
    def __init__(self, view: AsOf):
        self.view = view
        names: dict[str, Counter] = defaultdict(Counter)       # norm name -> paper -> evidence count
        self.proposed: dict[str, list[str]] = defaultdict(list)  # paper -> its own method names
        for s in view.statements(kind="self"):
            nm = (s["meta"] or {}).get("name")
            if s["role"] == "proposes" and nm and not s["meta"].get("generic"):
                n = norm_name(nm)
                if n and n not in GENERIC:
                    names[n][s["about"]] += 3   # self-report weighs more than a single alias mention
                    self.proposed[s["about"]].append(nm)
                    for a in s["meta"].get("aliases") or []:
                        if norm_name(a):
                            names[norm_name(a)][s["about"]] += 2
        for s in view.statements(kind="other"):
            nm = (s["meta"] or {}).get("name")
            if nm and s["about"].startswith("paper:"):
                n = norm_name(nm)
                if n and n not in GENERIC:
                    names[n][s["about"]] += 1
        self.names = names

    def resolve(self, name: str) -> str | None:
        """Surface name -> 'paper:<id>' when one paper clearly owns it (>= 2x the evidence of any other), else None."""
        c = self.names.get(norm_name(name))
        if not c:
            return None
        (best, w), *rest = c.most_common(2) + [(None, 0)]
        second = rest[0][1] if rest else 0
        return best if w >= 2 and w >= 2 * second else None

    def aliases(self, paper: str) -> list[str]:
        return sorted(n for n, c in self.names.items() if self.resolve(n) == paper)
