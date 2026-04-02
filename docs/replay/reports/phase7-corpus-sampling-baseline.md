# Phase 7 Corpus Sampling Baseline

Built from the real Phase 7 runtime bundle in `tmp/phase7_corpus_sampling_baseline/`.

## Corpus Root

The baseline was generated from the shared Phase 7 corpus root recorded in the local runtime bundle manifest and summary files.

- Runtime source of truth: `tmp/phase7_corpus_sampling_baseline/bundle_manifest.json`
- Inventory gate: filesystem scan of the shared corpus root, not Neo4j ingestion rows
- Portability rule: this committed report intentionally omits the absolute UNC path so the repo does not hardcode machine-local infrastructure

## Fixed Regression Set

The committed fixed regression set contains exactly `10` papers from `docs/replay/corpus_sampling/phase7-fixed-regression-set.json`.

| ID | Preferred Source | Why It Stays In The Stable Set |
|----|------------------|--------------------------------|
| `1000` | `md` | Healthy markdown-first pair with a matching txt sibling |
| `1001` | `md` | Second healthy markdown-first pair from the same topic slice |
| `1005` | `md` | Healthy markdown-first pair with a longer nested markdown path |
| `1017` | `txt` | Both variants exist but markdown is unreadable, so txt fallback is part of the fixed baseline |
| `1023` | `txt` | Second markdown-broken txt fallback case |
| `1002` | `txt` | Txt-only survivor whose nested markdown directory is missing |
| `1004` | `txt` | Txt-only survivor with a long technical filename and broken markdown path |
| `1007` | `txt` | Txt-only deep-material-network phrasing |
| `1010` | `txt` | Txt-only reinforcement-learning phrasing |
| `1012` | `txt` | Txt-only geometric-deep-learning phrasing |

This mix is intentional rather than “the first ten rows”: it preserves healthy markdown-first pairs, markdown-broken txt fallbacks, and txt-only survivors in one stable reviewable set.

## Random Exploration Batch

The first real random exploration run used seed `7` and selected exactly `5` papers, excluding all fixed-set ids.

| ID | Preferred Source | Corpus Relative Ref |
|----|------------------|---------------------|
| `17` | `txt` | `txt/17_In-situ_characterization_and_quantification_of_melt_pool_variation_under_constant_input_energy_density_in_laser_powder-bed_fusion_additive_manufacturi.txt` |
| `1317` | `txt` | `txt/1317_Physics-Informed_Neural_Networks__A_Deep_Learning_Framework_for_Solving_Forward_and_Inverse_Problems_Involving_Nonlinear_Partial_Differential_Equation.txt` |
| `1844` | `md` | `1844_Deep learning of free boundary and Stefan problems/1844_Deep_learning_of_free_boundary_and_Stefan_problems/1844_Deep_learning_of_free_boundary_and_Stefan_problems.md` |
| `580` | `md` | `580_Extensional Flow Mixer for Polymer Nanocomposites/580_Extensional_Flow_Mixer_for_Polymer_Nanocomposites/580_Extensional_Flow_Mixer_for_Polymer_Nanocomposites.md` |
| `1107` | `txt` | `txt/1107_Preparation_and_thermal_properties_study_of_HNIW_FOX-7_based_high_energy_polymer_bonded_explosive__PBX__with_low_vulnerability_to_thermal_stimulations.txt` |

Runtime summary counts from the same bundle:

- inventory entries: `1756`
- eligible entries: `1755`
- fixed selected: `10`
- random selected: `5`

## Corpus Health Findings

Filesystem inventory produced `1505` corpus-health failures, and those failures were kept separate from the selected-paper lists.

- `931` were `walk_error` records, where the scanner encountered broken or missing nested directories during traversal
- `574` were `unreadable_file` records, where a file path was enumerated but could not be opened
- selected papers such as `1017`, `1023`, `1002`, and `1004` remained eligible through txt fallback or txt-only availability while their broken markdown-side problems stayed in the corpus-health bucket

Representative corpus-health examples from the real bundle:

- `934_Application_of_DFT-based_machine_learning_for_developing_molecular_electrode_materials_in_Li-ion_batteries` surfaced a `walk_error`
- `356_5_6-Fused_Bicyclic_Tetrazolo-Pyridazine_Energetic_Materials.md` surfaced an `unreadable_file`
- `1351_Modular-based_multiscale_modeling_on_viscoelasticity_of_polymer_nanocomposites.md` surfaced an `unreadable_file`, and the matching `images` directory also surfaced a `walk_error`

This is the key Phase 7 outcome: path breakage is now visible as corpus health, not miscounted as model-quality failure.

## Neo4j Metadata Coverage

The runtime bundle recorded `neo4j_lookup_status = ready`, so the CLI did attempt graph enrichment.

- selected papers with Neo4j metadata: `0`
- selected papers without Neo4j metadata: `15`
- interpretation: graph lookup completed, but none of the fixed or random selections matched current Neo4j enrichment rows

Because filesystem inventory is the Phase 7 gate, those `15` selections still entered the bundle. They were recorded under `selected_but_downstream_unavailable` instead of being dropped from eligibility.

## Phase 7 Limits

- This baseline only establishes reproducible sampling and reporting. It does not run `L2` extraction quality checks.
- The random batch is one seed-`7` sample from the current corpus state, not a statement that the full corpus is clean or graph-ready.
- The report does not embed the absolute corpus root, because that machine-specific detail belongs in the local bundle under `tmp/phase7_corpus_sampling_baseline/`.
- Multi-paper packet construction and `L3/L4` validation remain deferred to later phases.
