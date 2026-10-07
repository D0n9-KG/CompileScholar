# -*- coding: utf-8 -*-
"""The one schema of the system (INTEGRATED-SYSTEM-1005 §7.1, schema v2 — user-approved). Every extraction pass
writes it, the compiler and the tools read it.

Atomic unit: a Statement — who said what, about what, when, in which words, where in the text, with what force.
  speaker   paper_id of the saying paper
  date      the text-version date of the sentence the quote comes from (v2.5: content found in v1 carries the v1
            date, content only a later version has carries that version's date); cognition(T) uses statements
            with date <= T only
  kind      "self"  : the speaker about its own work
            "other" : the speaker about a cited work (one statement per citation sentence x cited entry)
  about     the object: bare paper_id ("arxiv:…" | "doi:…" | "title:…") | "stub:<norm_title>" |
            "method:<name>@<paper>" (filled by the compiler). The pre-C "paper:<arxiv_id>" prefix is abolished
            (citations moved to bare paper_ids in C②; extract follows).
  role      what the speaker says the object is to it (RELATIONS); "combines" is n-ary (several targets)
  facet     what kind of content the statement carries (FACETS)
  text      one-sentence normalized content (LLM), must be supported by quote
  quote     verbatim sentence from the source text — the unified final check (§7.3) locates it through the match
            view and verifies it is a substring; stored quotes stay verbatim
  epistemic the force the paper gives the content: demonstrated | stated | hypothesized | cited
  condition the qualifying fragment from the source ("on ImageNet", "for small batches"), "" when unqualified
  loc       {"unit_id", "sent_id", "char_start", "char_end"} — where in the source document; unit_id/sent_id are
            the pass's anchors, the char offsets are filled by the final check
  target    the other work a relational statement is about ("A extends B") — both kinds may carry it (§7.1)
  group     for other statements: the tuple of objects cited together in the same sentence (co-citation group)
  function  citation function (other statements)
  meta      free dict; facet=config requires meta.config = {item, value, unit, applies_to}
Provenance is written with every row: pass_name/item (the dfc work unit — redoing an item replaces exactly its
rows), run_id, model, prompt_sha. Deterministic products record model as "deterministic:<code-hash>".
The old single-paper `shift` records are retired: reception change is computed by the compile layer across time.

Vocabularies (closed; the extractor may only output these):
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = 2

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
    "config",        # a stated experimental-configuration value: meta.config{item, value, unit, applies_to}
    "absence",       # a missing capability the paper EXPLICITLY writes (never an inference of ours)
    "definition",    # optional: a definition the paper gives for a term
)
FUNCTIONS = ("background", "basis", "baseline", "contrast", "data", "tool", "metric")  # citation function (other)
EPISTEMIC = ("demonstrated", "stated", "hypothesized", "cited")
CONFIG_KEYS = ("item", "value", "unit", "applies_to")
ABOUT_PREFIXES = ("arxiv:", "doi:", "title:", "stub:", "method:")


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
    epistemic: str = "stated"
    condition: str = ""
    loc: dict = field(default_factory=dict)
    target: str | None = None
    group: tuple = ()
    function: str | None = None
    meta: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    pass_name: str = ""        # column `pass` (a Python keyword); the dfc work pass that produced the row
    item: str = ""             # the dfc work item (paper_id) — redoing an item replaces exactly its rows
    run_id: str = ""
    model: str = ""
    prompt_sha: str = ""

    def validate(self) -> list[str]:
        """Schema errors (empty list = valid). Quote-in-source is the final check's job (§7.3), which has the
        source text; this validates the closed vocabularies and the required shapes."""
        err = []
        if self.kind not in ("self", "other"):
            err.append(f"kind {self.kind!r}")
        if self.role not in RELATIONS:
            err.append(f"role {self.role!r}")
        if self.facet not in FACETS:
            err.append(f"facet {self.facet!r}")
        if self.kind == "other" and self.role not in OTHER_RELATIONS:
            err.append(f"role {self.role!r} is self-only")
        if self.epistemic not in EPISTEMIC:
            err.append(f"epistemic {self.epistemic!r}")
        if self.function is not None and self.function not in FUNCTIONS:
            err.append(f"function {self.function!r}")
        if self.facet == "config":
            cfg = self.meta.get("config")
            if not isinstance(cfg, dict) or not cfg.get("item") or cfg.get("value") in (None, ""):
                err.append("config without meta.config{item, value}")
            elif any(k not in cfg for k in CONFIG_KEYS):
                err.append(f"config missing keys of {CONFIG_KEYS}")
        if not self.quote or not self.text:
            err.append("empty text/quote")
        if not self.about.startswith(ABOUT_PREFIXES):
            err.append(f"about {self.about!r}")
        if self.target is not None and not self.target.startswith(ABOUT_PREFIXES):
            err.append(f"target {self.target!r}")
        if len(self.date) != 10:
            err.append(f"date {self.date!r}")
        if not (self.loc.get("unit_id") or self.loc.get("sent_id")):
            err.append("loc without unit_id/sent_id anchor")
        if self.schema_version != SCHEMA_VERSION:
            err.append(f"schema_version {self.schema_version!r}")
        return err

    def row(self) -> dict:
        d = asdict(self)
        d["group"] = list(self.group)
        d["pass"] = d.pop("pass_name")
        return d


# pass / item: the unit of work that produced the row (dfc.store item bookkeeping); redoing an item replaces exactly
# its rows. The unique key makes a duplicated write (two writers, a retried batch) an error instead of a silent copy.
# A pre-C extract.sqlite has the v1 table (no epistemic/condition/loc/schema_version/run_id columns) — the stage
# build refuses it and wants --rebuild, like documents and citations do.
DDL = """CREATE TABLE IF NOT EXISTS statements(
  id INTEGER PRIMARY KEY, speaker TEXT, date TEXT, kind TEXT, about TEXT, role TEXT, facet TEXT, text TEXT,
  quote TEXT, epistemic TEXT, condition TEXT, loc TEXT, target TEXT, grp TEXT, function TEXT, meta TEXT,
  schema_version INT, pass TEXT, item TEXT, run_id TEXT, model TEXT, prompt_sha TEXT);
CREATE INDEX IF NOT EXISTS ix_st_about ON statements(about, date);
CREATE INDEX IF NOT EXISTS ix_st_speaker ON statements(speaker);
CREATE INDEX IF NOT EXISTS ix_st_target ON statements(target);
CREATE INDEX IF NOT EXISTS ix_st_item ON statements(pass, item);
CREATE UNIQUE INDEX IF NOT EXISTS ux_st_natural ON statements(pass, item, speaker, kind, about, role, facet, text,
  quote);"""
