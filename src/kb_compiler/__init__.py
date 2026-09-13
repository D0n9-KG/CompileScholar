"""kb_compiler: Stage B knowledge-compilation stack (record layer + views).

Spec: .research_tmp/docs_decisions/STAGEB-EXTRACTOR-SPEC-v1.md

Import discipline (machine-gated): this package may import ONLY stdlib,
kb_infra, itself, and the arbitrated third-party whitelist. Importing
granular_agent (frozen legacy stack) is a hard violation — the gate test
fails the build.

Subpackages:
- records/ : record-layer schema + extraction pipeline (Stage 0-3)   [this stage]
- views/   : four-view compiler + typed tools                        [later spec]
"""
