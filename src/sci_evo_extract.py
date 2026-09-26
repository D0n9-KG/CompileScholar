# -*- coding: utf-8 -*-
"""兼容垫片：sci-evo-extract 已收编为 src/retrieval/（2026-09-28 合并裁定）。

旧 import（from sci_evo_extract.library.sources import SciverseClient 等）
经此垫片转发到新家 retrieval.*，现有代码零改动。新代码直接 import retrieval。
"""
import sys as _sys

import retrieval as _retr
import retrieval._see_llm as _llm
import retrieval._see_config as _config
import retrieval._see_models as _models
import retrieval._see_io as _io
import retrieval._see_retrieval as _retrieval
import retrieval._see_extraction as _extraction
import retrieval._see_upstream as _upstream
import retrieval._see_llm_cases as _llm_cases

_sys.modules[__name__ + ".library"] = _retr
for _name, _mod in [("llm", _llm), ("config", _config), ("models", _models),
                    ("io", _io), ("retrieval", _retrieval),
                    ("extraction", _extraction), ("upstream", _upstream),
                    ("llm_cases", _llm_cases)]:
    _sys.modules[__name__ + "." + _name] = _mod
