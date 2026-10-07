# -*- coding: utf-8 -*-
"""The cognition-stage adjudication prompts (§8: rules/statistics recall candidates, the LLM decides; §12 D②:
every adjudication class is piloted on 200 items with the dual-model agreement gate before the full run).
English only. The sha of each prompt is part of its work-pass fingerprints."""
from __future__ import annotations

import hashlib

MODEL = "Qwen3.8-27B"

METHOD_ID = """One method name is claimed by several papers in our library. Decide which paper — if any — OWNED
the name as used in the context sentences below (the owner introduced the method; papers that merely use or
extend it are not the owner).

Name: "{name}"
Candidates:
{candidates}

Context sentences in which the name is mentioned (each prefixed by the mentioning paper's id):
{contexts}

Answer with JSON only:
{{"paper": <candidate number, or 0 when the contexts do not identify one unique owner>,
  "why": "<= 10 words"}}

Rules: choose a candidate only when the contexts (titles, abstracts, years, who cites whom) make it the clear
introducer of the named method; when two papers genuinely use the same name for different methods and the
contexts do not separate them, answer 0. Do not guess."""

CATEGORY_CANON = """Normalize ONE research-topic phrase (extracted from a citation sentence) into its canonical
category name. An UMBRELLA phrase (a broad area containing other topics as subtopics, like "machine learning" or
"NLP") gets "umbrella": true.

Existing canonical names — REUSE one of these when it already fits the phrase:
{existing}

Phrase: "{phrase}"

Answer with JSON only:
{{"canonical": "<canonical phrase, lowercase, <= 6 words>", "umbrella": true|false}}
Rules: keep the most specific standard name of the topic; do not invent topics the phrase does not contain;
English only."""


SHIFT_VERIFY = """A statistical screen flagged a possible RECEPTION SHIFT for one research work: the way other
papers talk about it changed between two time windows. Decide whether the evidence sentences actually show the
shift, or whether the statistics fired on noise / a wording artifact.

Work: {subject}
Claimed shift: {facet} — early window {early_share}, late window {late_share} (statistical p={p}).
{direction}
Early-window sentences (from citing papers):
{early}

Late-window sentences:
{late}

Answer with JSON only:
{{"holds": true|false, "why": "<= 12 words"}}
Rules: "holds" requires the late sentences to genuinely treat the work differently in the claimed way (a real
change of role, category or stance), not just different phrasings of the same treatment. When unsure, false."""


def sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


METHOD_ID_SHA = sha(METHOD_ID)
CATEGORY_CANON_SHA = sha(CATEGORY_CANON)
SHIFT_VERIFY_SHA = sha(SHIFT_VERIFY)
