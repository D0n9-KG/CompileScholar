# -*- coding: utf-8 -*-
"""Moved to compilescholar.documents.tables (10-05 literature-layer upgrade). This shim aliases the module so legacy
imports (kb_compiler.records.table_semantic / table_extract) keep working unchanged."""
import sys as _sys

from compilescholar.documents import tables as _tables

_sys.modules[__name__] = _tables
