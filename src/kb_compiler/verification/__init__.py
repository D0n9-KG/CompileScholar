"""Verification layer (Stage B blueprint block 4).

Three layers per spec v1.2:
  L1 structural + L2 faithfulness = records/postcheck.py (Stage 3, deterministic)
  L3 cross-record consistency     = this package (deterministic detection +
                                    LLM arbitration QUEUE — detection never
                                    silently rewrites records)
"""
