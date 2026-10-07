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

CATEGORY_CANON = """Normalize research-topic phrases extracted from citation sentences into canonical category
names. Phrases that mean the same topic must map to the same canonical name; an UMBRELLA phrase (a broad area
that contains other listed phrases as subtopics, like "machine learning" or "NLP") gets "umbrella": true.

Phrases (numbered):
{phrases}

Answer with JSON only:
{{"canon": [{{"i": <number>, "canonical": "<canonical phrase, lowercase, <= 6 words>", "umbrella": true|false}}]}}
Rules: keep the most specific standard name of the topic; do not invent topics not present in the phrase;
English only."""


def sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


METHOD_ID_SHA = sha(METHOD_ID)
CATEGORY_CANON_SHA = sha(CATEGORY_CANON)
