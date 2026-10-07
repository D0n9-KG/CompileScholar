# -*- coding: utf-8 -*-
"""Method identity at T (phase D reader): the materialised adjudication of cognition.build — deterministic
resolution (a single candidate, or clear evidence dominance: >= 2 and >= 2x the runner-up) plus the LLM
METHOD_ID adjudication of ambiguous names. The method_identity table IS the identity; readers never recompute
the name vocabulary. Adjudicated rows persist across builds (decided once, kept); norm_name — the one
surface -> lookup-key normalisation — is imported from the build so both sides cannot drift.

Identity is not as-of filtered: an adjudication changes only WHICH paper a name resolves to (resolution
quality), never the content or the dates of statements; lineage edges carry their own §2.4 valid_from, so a
name decided today cannot leak future content into an earlier T."""
from __future__ import annotations

from .asof import AsOf
from .build import norm_name


class Identity:
    def __init__(self, view: AsOf):
        self.view = view
        self._map: dict[str, str] = {}
        self._status: dict[str, str] = {}
        if view.cog is not None:
            for n, p, st in view.cog.execute("SELECT name, paper_id, status FROM method_identity"):
                self._status[n] = st
                if p and st in ("ok", "adjudicated"):
                    self._map[n] = p

    def resolve(self, name: str) -> str | None:
        """Surface name -> the bare paper_id that owns it, or None (unowned / ambiguous / unresolved)."""
        return self._map.get(norm_name(name))

    def status(self, name: str) -> str | None:
        """'ok' | 'adjudicated' | 'ambiguous' | 'unresolved' | None (the name is not in the vocabulary)."""
        return self._status.get(norm_name(name))

    def aliases(self, paper_id: str) -> list[str]:
        """The norm names the tables give this paper (deterministic + adjudicated)."""
        return sorted(n for n, p in self._map.items() if p == paper_id)

    def surface_names(self, paper_id: str) -> list[str]:
        """The names as the paper itself writes them (its proposes statements + their aliases) — display."""
        out = []
        for s in self.view.statements(about=paper_id, kind="self"):
            m = s.get("meta") or {}
            if s.get("role") == "proposes" and m.get("name") and not m.get("generic"):
                out.append(m["name"])
                out += [a for a in m.get("aliases") or [] if a]
        return list(dict.fromkeys(out))
