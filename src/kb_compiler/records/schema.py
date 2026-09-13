"""Frozen record-layer schema v1.1 (spec §1-§2; v1.0 ratified 2026-09-05,
v1.1 = gold-backtest arbitration RC1-RC8, same day).

Machine-readable form of STAGEB-EXTRACTOR-SPEC-v1.md §1-§2.
Change discipline: concepts here are FROZEN — revisions go through the
arbitration channel (gold-backtest gaps / residual-queue high-frequency items /
smoke failure attribution) and bump SCHEMA_VERSION; no silent edits, no
runtime invention by the extractor (new explicit dimension values go to the
arbitration queue, spec §2). Governance is symmetric (v1.1): every arbitration
agenda pairs the addition list with a USAGE-AUDIT deletion list (fill rate +
downstream reference rate per field).
"""

SCHEMA_VERSION = "1.4"
# v1.4 (finding-epistemic arbitration 2026-09-13, user-approved Option A;
# briefed as "v1.3" in SCHEMA-V13-ARBITRATION-BRIEF.md, which predated the
# notation bump consuming 1.3): finding gains the epistemic TEMPLATE slot
# (enum was already record-legal via COMMON_FIELDS — the gap was prompt-side:
# the finding template offered no epistemic field, so the model stuffed
# provenance into strength; strength='cited' was enum-dropped, 53 records in
# the KB v3 era = largest violation class; manual read confirmed the semantic
# intuition was right, the slot was missing). Postcheck FG7 migrates
# strength='cited' deterministically instead of dropping. Canary v3 adds
# F7 (cited finding survives) / T6 (own-paper finding flipped to cited).
# NO retro-fill: legacy records keep epistemic=None (backward compatible);
# the value materializes only on new extraction.

# ---- record types (spec §1) ----
# v1.3 (granular-pilot arbitration 2026-09-06): +notation — symbol/quantity
# definitions, structural content of formula-heavy domains (~15-entry same-root
# overflow cluster in the 8-paper granular pilot; grounds the interpretation of
# formula-valued config/result records; legacy LogicKG measured this asset
# class as valuable).
RECORD_KINDS = ("result", "config", "lineage", "finding", "absence", "shift",
                "notation")
OVERFLOW_KIND = "overflow"          # residual-queue record (spec §3 Stage 2)
ALL_KINDS = RECORD_KINDS + (OVERFLOW_KIND,)

# ---- closed vocabularies ----
# v1.1 (RC5): +generalizes (theoretical subsumption/unification, T05-m4),
# +concurrent_with (symmetric same-period non-influence edge, C07).
# Negative relations deliberately ABSENT: three backtest batches x 8 cases each
# independently showed zero need — negation is carried by result.delta < 0 /
# finding / absence tri-state.
RELATIONS = ("extends", "improves", "uses", "component_of",
             "replaces", "compares_with", "motivated_by",
             "generalizes", "concurrent_with")
SYMMETRIC_RELATIONS = frozenset({"concurrent_with", "compares_with"})

RESULT_ROLES = ("main_result", "baseline_comparison", "ablation")
# v1.2 (S1+S2 arbitration 2026-09-05): config semantics EXTENDED — item covers
# hyperparameters AND component/architecture parameters AND protocol entries
# (item names registered via Stage 1.5); value may be multi-entry protocol
# text, shape descriptions, or formulas; role gains "protocol" for paper-level
# methodological facts (eval aggregation / trials / reporting conventions).
# Residual-queue signal: ~20 overflow entries had no home; canary F3 (buffer
# 2M config) missed twice under the narrow reading.
CONFIG_ROLES = ("best_reported", "default", "used_in_experiment", "protocol")
ABSENCE_TYPES = ("not_reported", "explicitly_stated", "cannot_tell")
EPISTEMIC = ("stated", "demonstrated", "cited")
FINDING_STRENGTH = ("stated", "demonstrated")
# v1.2 (S3 arbitration): optional claim subgenre — normative recommendations
# (~5 overflow entries) had no explicit home; strength is an evidence axis,
# claim_type is a genre axis. Optional field, NOT in REQUIRED_FIELDS; the
# block-2 compiler may reclassify. Enum list = batch-1 backtest inventory +
# smoke v3 residual evidence.
FINDING_CLAIM_TYPES = ("mechanism", "criticism", "definition", "recommendation",
                       "qualitative_ablation", "observation")
LINEAGE_EVIDENCE = ("explicit_claim", "citation_context")
SHIFT_SOURCES = ("intro_narrative", "related_work", "discussion")
DIRECTIONS = ("higher_better", "lower_better")

# v1.1 (RC8): record-carried quality flags; render layer MUST surface these.
QUALITY_FLAGS = ("suspect_binding", "parse_warning", "figure_only")

# v1.1 (RC2): registry entity granularity — mechanism-level entities
# (target network, epsilon-greedy, n-step returns) and practice-level entities
# ("RL evaluation conventions") are first-class registry citizens; without
# this clause batch-2 temporal E-rate drops 88.4% -> ~50% (measured).
ENTITY_TYPES = ("method", "mechanism", "practice", "out_of_corpus")

# ---- dimension vocabulary (spec §2) ----
# names frozen; explicit-dimension VALUES grow per corpus via Stage 1.5
# registration + arbitration; typed values are open with units.
DIMENSIONS = {
    "subject":    "explicit",  # evaluated object, family->member two levels
    "setup":      "explicit",  # protocol/environment conditions (incl. qualitative hardware)
    "budget":     "typed",     # scale: frames/steps/hours/FLOPs (incl. compute budget)
    "variant":    "explicit",  # method variant / component switch, scoped per method family
    "repeats":    "typed",     # seed count; aggregation single-source-of-truth = measure.aggregation
    "hyperparam": "typed",     # v1.1 (RC3): numeric condition axis {item, value, unit};
                               # item names registered (explicit item + typed value),
                               # scoped per method family; isomorphic to config (item, value)
}

# Cochrane quintuple inside result.measure (spec §1.1)
MEASURE_FIELDS = ("metric", "value", "unit", "direction", "aggregation", "timepoint")

# ---- quote discipline (spec §1.0, ratified F) ----
QUOTE_MAX_WORDS = 40            # prose quotes; table-row quotes = row/col headers + cell
QUOTE_REQUIRED_KINDS = ("result", "config", "lineage", "finding", "shift",
                        "notation", "overflow")
# absence: quote required for explicitly_stated / cannot_tell; not_reported
# carries its burden of proof via `evidence` instead (spec §1.5).

# ---- minimal required fields per kind (Stage 3 completeness check consumes this) ----
# Optional v1.1 fields NOT listed here: finding.target_ref (RC1 join key),
# quality_flag (RC8, common), registry origin_year_cited (RC4, lives on
# registry entities, not records).
REQUIRED_FIELDS = {
    "result":   ("method_ref", "measure", "role", "quote"),
    "config":   ("method_ref", "item", "value", "quote"),
    "lineage":  ("from_method_ref", "relation", "to_method_ref", "evidence_basis", "quote"),
    "finding":  ("claim", "strength", "quote"),
    "absence":  ("subject", "missing", "absence_type", "evidence"),
    "shift":    ("from_state", "to_state", "driver", "quote"),
    "notation": ("symbol", "definition", "quote"),
    "overflow": ("reason", "quote"),
}

# common fields every record carries (spec §1.0)
COMMON_FIELDS = ("id", "paper_id", "kind", "loc", "epistemic", "dims")

# ---- Stage 0 manifest fields (spec §3; RC6 added authors/affiliations;
# v1.2.1 patch added doi/year_source for journal-domain corpora — non-arXiv
# chronology anchor, metadata layer only, record-layer semantics untouched) ----
MANIFEST_FIELDS = ("paper_id", "title", "arxiv_id", "arxiv_year", "venue",
                   "venue_year", "openreview_id", "authors", "affiliations",
                   "doi", "year_source",
                   "source_file", "source_version", "parser", "flags")
