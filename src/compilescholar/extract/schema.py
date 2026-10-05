# -*- coding: utf-8 -*-
"""The one schema of the system (DESIGN-UPGRADE §5, §12a). Both extraction passes write it, the compiler and the tools
read it. Replaces the six relation vocabularies that existed before (INTERNAL-AUDIT §C.2).

Atomic unit: a Statement — who said what, about what, when, in which words.
  speaker   paper that says it (arxiv_id)
  date      speaker's v1_date (YYYY-MM-DD); cognition(T) uses statements with date <= T only
  kind      "self"  : the speaker about its own work
            "other" : the speaker about a cited work (one statement per citation sentence x cited entry)
  about     the object: "paper:<arxiv_id>" | "stub:<norm_title>" | "method:<name>@<paper>" (filled by the compiler)
  role      what the speaker says the object is to it (RELATIONS below); for kind=self: role of the speaker's own
            contribution towards a cited target (target filled) or a free-standing contribution (target None)
  facet     what kind of content the statement carries (FACETS)
  text      one-sentence normalized content (LLM), must be supported by quote
  quote     verbatim sentence from the speaker's text (substring-checked)
  target    for self statements with a relation to a cited work: "paper:<id>" / "stub:<title>"
  group     for other statements: the tuple of objects cited together in the same sentence (co-citation group)
  meta      free dict (e.g. metric names, dataset names, conditions)

Vocabularies (closed; the extractor may only output these):
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# relation of the speaker's work to the object (for kind=self: speaker -> target; for kind=other: the role the cited
# object plays in the speaker's paper). Lineage relations follow Intern-Atlas / IG-Bench (RESEARCH-FIELD-MAP §3);
# "combines" is n-ary (several targets in one statement).
LINEAGE = ("extends", "improves", "replaces", "adapts", "combines")
RELATIONS = LINEAGE + (
    "uses",          # uses as a component / tool / data / metric (not lineage)
    "compares",      # used as a baseline or point of comparison
    "proposes",      # (self only) introduces a new method / dataset / benchmark / metric
    "background",    # general background, motivation, related work mention
    "criticizes",    # points out a weakness / limitation of the object
    "describes",     # non-relational content (a self statement about the speaker's own setting / result / limit)
)
OTHER_RELATIONS = tuple(r for r in RELATIONS if r not in ("proposes", "describes"))
FACETS = (
    "contribution",  # what the work does / proposes
    "method",        # how it works
    "result",        # a reported finding or number
    "limitation",    # a weakness, failure condition, open problem
    "setting",       # task, data, metric, assumptions
    "categorization",  # which family / class the object belongs to ("X is a graph-based method")
)
FUNCTIONS = ("background", "basis", "baseline", "contrast", "data", "tool", "metric")  # citation function (other)


@dataclass
class Statement:
    speaker: str
    date: str
    kind: str
    about: str
    role: str
    facet: str
    text: str
    quote: str
    target: str | None = None
    group: tuple = ()
    function: str | None = None
    meta: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Schema errors (empty list = valid). Quote-in-source is checked by the extractor, which has the source."""
        err = []
        if self.kind not in ("self", "other"):
            err.append(f"kind {self.kind!r}")
        if self.role not in RELATIONS:
            err.append(f"role {self.role!r}")
        if self.facet not in FACETS:
            err.append(f"facet {self.facet!r}")
        if self.kind == "other" and self.role not in OTHER_RELATIONS:
            err.append(f"role {self.role!r} is self-only")
        if self.function is not None and self.function not in FUNCTIONS:
            err.append(f"function {self.function!r}")
        if not self.quote or not self.text:
            err.append("empty text/quote")
        if not self.about.startswith(("paper:", "stub:", "method:")):
            err.append(f"about {self.about!r}")
        if len(self.date) != 10:
            err.append(f"date {self.date!r}")
        return err

    def row(self) -> dict:
        d = asdict(self)
        d["group"] = list(self.group)
        return d


DDL = """CREATE TABLE IF NOT EXISTS statements(
  id INTEGER PRIMARY KEY, speaker TEXT, date TEXT, kind TEXT, about TEXT, role TEXT, facet TEXT, text TEXT,
  quote TEXT, target TEXT, grp TEXT, function TEXT, meta TEXT, model TEXT, prompt_sha TEXT);
CREATE INDEX IF NOT EXISTS ix_st_about ON statements(about, date);
CREATE INDEX IF NOT EXISTS ix_st_speaker ON statements(speaker);
CREATE INDEX IF NOT EXISTS ix_st_target ON statements(target);"""
