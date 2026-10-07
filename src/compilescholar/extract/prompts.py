# -*- coding: utf-8 -*-
"""The extraction prompts (INTEGRATED-SYSTEM-1005 §7.3; English only — the pre-C Chinese prompts are replaced).
Every prompt sha is stored on the rows it produced; a changed prompt re-opens exactly its own work pass.

Shared output discipline, enforced deterministically after the call (passes.py):
  - every item carries the number `n` of the sentence it comes from; the sentence itself is the quote (a natural
    substring of the source — the unified final check verifies it, one LLM repair, then discard);
  - closed vocabularies (schema.RELATIONS / FACETS / EPISTEMIC / FUNCTIONS) — an item with a value outside them
    is dropped, never remapped;
  - no outside knowledge; nothing the sentence does not say.
"""
from __future__ import annotations

import hashlib

MODEL = "Qwen3.8-27B"

_FACETS = "contribution | method | result | limitation | setting | categorization | config | absence | definition"
_ROLES_SELF = "proposes | uses | extends | improves | replaces | adapts | combines | compares | criticizes | describes"
_EPISTEMIC = "demonstrated | stated | hypothesized | cited"

T1 = """You read the title and the numbered abstract sentences of ONE research paper (its first public version).
Report what the paper says about ITS OWN work. Cite the sentence number every item comes from; the sentence
itself is the evidence — never restate or extend it.

Title: {title}
{sentences}

Return JSON only:
{{"items": [{{"n": <int>, "facet": "<one of: {facets}>", "role": "<one of: {roles}>",
             "epistemic": "<one of: {epistemic}>",
             "condition": "<the qualifying phrase from the sentence (on which data/setting/case), or empty>",
             "text": "<one normalized sentence the item states, fully supported by sentence n>"}}],
 "names": [{{"name": "<the name the paper gives the artefact, acronyms as written>", "aliases": ["..."],
             "artefact": "method | model | dataset | benchmark | metric | system | framework",
             "relation": "proposes | uses", "n": <int>}}]}}

Rules:
- "names"/relation=proposes ONLY for artefacts this paper introduces as new; artefacts it builds on get
  relation=uses. Do not confuse the two.
- facet=absence ONLY where the paper EXPLICITLY writes that something is missing, unavailable, unexplored or
  not addressed — never inferred from silence.
- epistemic: demonstrated = backed by the paper's own experiments/proofs; stated = asserted without evidence in
  this sentence; hypothesized = conjecture/future expectation; cited = attributed to someone else's work.
- 1-8 items, 0-6 names. Empty lists are fine. No outside knowledge."""

T2 = """You read ONE chunk of a research paper's full text. Numbered lines [n] are its sentences; "##" lines are
section headings; unnumbered lines ($$...$$ formulas, table captions) are context only. Extract what the chunk
states about the paper's OWN work: contributions, method components and design choices, results and numbers
stated in prose, limitations, settings (tasks, data, metrics, baselines), categorizations of its own work,
explicitly stated absences, definitions it gives, and stated experimental-configuration values.

Sentences whose main content describes OTHER works (related work, others' methods and findings) are NOT
extracted here — a separate pass owns them. But a sentence about this paper's own relation to another work
("we extend X", "unlike Y we ...") IS extracted: give the relation in "role" and the other work's surface name
in "mentions".

Paper title: {title}
Chunk {chunk_i}/{chunk_n} (sentences numbered per paper):
{text}

Return JSON only:
{{"items": [{{"n": <int>, "facet": "<one of: {facets}>", "role": "<one of: {roles}>",
             "epistemic": "<one of: {epistemic}>",
             "condition": "<the qualifying phrase from the sentence (on which data/setting/case), or empty>",
             "text": "<one normalized sentence the item states, fully supported by sentence n>",
             "mentions": [{{"name": "<surface name of the other work as written>",
                           "relation": "extends | improves | replaces | adapts | combines | uses | compares"}}]}}],
 "config": [{{"item": "<what is configured, e.g. learning rate>", "value": "<as written>", "unit": "<or empty>",
              "applies_to": "<the part of the setup it applies to, or empty>", "n": <int>}}],
 "own_methods": ["<every proper name this paper gives to the method(s)/model(s)/architecture(s)/system(s) it
                  PROPOSES as its own, as written>"]}}

Rules: cite the sentence number each item comes from; the sentence itself is the evidence. Facet=result items
keep the exact number and its condition as the sentence writes them. Facet=absence only for explicitly written
absences — including what the paper explicitly writes that it does NOT do, cover or evaluate ("we do not
evaluate on X"); never inferred from silence. "own_methods" lists proper names only, never descriptions — and
never datasets, benchmarks, test sets, tools, libraries, metrics or prior work the paper builds on (measured
contamination: "UCF101", "Scikit-learn", "bag of words approach", "test set" all showed up and skew the
results pass). Up to 25 items per chunk; empty lists are fine. No outside knowledge."""

RESULTS_AXES = """You read ONE numeric table of a research paper, already repaired into a grid (rows of cells,
spans expanded), with its caption and the sentences around it. Propose the ROLE of each axis. Do not extract any
numbers — the code reads the cells; your roles decide which cells it reads.

Caption: {caption}
Context sentences:
{context}
Grid ({n_rows} rows x {n_cols} cols), first cell of each row shown with its column index:
{grid}

Return JSON only:
{{"usable": <true|false>,
 "why": "<one short phrase when usable=false>",
 "object_axis": "row" | "column",
 "object_index": <int: the column index holding the object/sample/group name in each data row when object_axis
                  is row; or the header row index when object_axis is column>,
 "conditions": [{{"index": <int>, "axis": "row" | "column",
                  "label": "<what this axis conditions on, e.g. dataset, method, hyper-parameter setting>"}}],
 "measures": [{{"index": <int>,
                "metric": "<what this column measures, in the paper's words, INCLUDING the column's header label
                           (on a multi-task table: 'QA RoBERTa-L accuracy', not just 'accuracy')>",
                "unit": "<unit as written, or empty>", "direction": "higher | lower | neutral"}}],
 "skip_rows": [<int>, ...]}}

Rules: "object" = what each data row (or column) is about: a method, model, sample, system or group. "conditions"
= axes that fix the setting a measure was taken under (dataset, split, budget, variant). "measures" = columns
(or rows) whose cells are the reported numbers. skip_rows = data rows that carry no object name (notes, average-
of-nothing, malformed). usable=false when the grid has no identifiable object axis or no measure axis — never
guess. The paper's own methods, for recognising object cells: {own_methods}"""


OTHER = """You read citation sentences from ONE research paper (the citing paper). Each numbered sentence cites
one or more works, listed under the sentence with their keys. For EVERY (sentence, cited work) pair, report what
THIS sentence says about the cited work and how the citing paper's own work relates to it. Use only what the
sentence says; no outside knowledge.

{sentences}

Return JSON only:
{{"pairs": [{{"s": <sentence number>, "key": "<cited key exactly as given>",
  "function": "background | basis | baseline | contrast | data | tool | metric",
  "role": "extends | improves | replaces | adapts | combines | uses | compares | background | criticizes",
  "about": "<one short sentence: what THIS sentence says the cited work is or does, in the sentence's own words;
            null if the sentence says nothing specific (e.g. a bare citation list)>",
  "facet": "<what 'about' describes: contribution | method | result | limitation | setting | categorization>",
  "epistemic": "<one of: {epistemic}>",
  "category": "<the class or family of methods the sentence puts the cited work in, in its words; null if none>",
  "limitation": "<the weakness or shortcoming of the cited work that the sentence states; null if none>",
  "name": "<the name the sentence uses for the cited work's method, model, dataset or benchmark; null if unnamed>",
  "outcome": "<only when the sentence compares results: citing_better | cited_better | mixed; else null>",
  "builds_on": "<surface name of ANOTHER work in the same sentence that the sentence says the cited work extends,
                improves, replaces, adapts or combines (sentence 'FastGF [2] improves GraphFormer [1]' for cited
                work [2] -> 'GraphFormer'); null if none>",
  "builds_on_relation": "extends | improves | replaces | adapts | combines; null when builds_on is null"}}]}}

One object per (sentence, cited work) pair: a sentence citing three works gets three objects. The quote is the
sentence itself — do not restate it."""


REPAIR = """A quality gate rejected the following extracted statements about a research paper. Each case shows
the anchor sentence from the source, the rejected statement, and the violation. Repair each statement so that:
- "quote" is EXACTLY the anchor sentence, copied verbatim (no edits, no truncation);
- every number in "text" appears verbatim in that sentence — never convert, round or compute a number;
- "facet" / "role" / "epistemic" stay inside their enums; "text" stays one normalized sentence fully supported
  by the quote.
If a statement cannot be repaired from its anchor sentence, mark it dropped.

{cases}

Return JSON only:
{{"fixed": [{{"i": <case number>, "drop": false, "text": "...", "quote": "...", "facet": "...", "role": "...",
             "epistemic": "...", "condition": "..."}}]}}
For a statement that cannot be repaired return {{"i": <case number>, "drop": true}}."""


def sha(prompt: str) -> str:
    """The prompt sha stored on its rows: the template plus the vocabularies it formats in — a changed vocabulary
    re-opens the pass exactly like a changed template."""
    return hashlib.sha256((prompt + "|" + _FACETS + "|" + _ROLES_SELF + "|" + _EPISTEMIC).encode()).hexdigest()[:16]


T1_SHA = sha(T1)
T2_SHA = sha(T2)
RESULTS_SHA = sha(RESULTS_AXES)
OTHER_SHA = sha(OTHER)
REPAIR_SHA = sha(REPAIR)
