# -*- coding: utf-8 -*-
"""extract: schema validation, self pass quote/proposal checks, other pass vocabulary and lexical-support checks
(LLM replaced by a stub)."""
import json

import pytest

from compilescholar.extract import other_pass as O
from compilescholar.extract import self_pass as S
from compilescholar.extract.schema import Statement

pytestmark = pytest.mark.skip(reason="the pre-C passes write schema v1 (arXiv-id speakers, 'paper:' prefixes); "
                                     "they are rewritten onto schema v2 + registry keys in phase C④ — the v2 "
                                     "schema itself is tested in test_extract_schema.py")

ABSTRACT = ("Fast adversarial training suffers from catastrophic overfitting. We propose SelfFit, a new regularizer "
            "that removes self-information from single-step examples. Our method relies on a small held-out set.")


def test_schema_rejects_bad_vocab_and_self_only_roles():
    good = Statement("2302.1", "2023-02-23", "other", "paper:1706.06083", "uses", "method", "x", "y")
    assert good.validate() == []
    assert Statement("2302.1", "2023-02-23", "other", "paper:1", "proposes", "method", "x", "y").validate()
    assert Statement("2302.1", "2023-02-23", "other", "paper:1", "inspires", "method", "x", "y").validate()
    assert Statement("2302.1", "2023", "self", "paper:1", "proposes", "contribution", "x", "y").validate()


def test_self_pass_keeps_only_verbatim_and_real_proposals():
    out = {"contributions": [{"text": "proposes a regularizer", "quote": "We propose SelfFit, a new regularizer that "
                                                                        "removes self-information from single-step examples."},
                             {"text": "invented", "quote": "This sentence is not in the abstract at all, really."}],
           "proposes": [{"name": "SelfFit", "aliases": [], "artefact": "method",
                         "quote": "We propose SelfFit, a new regularizer that removes self-information from single-step examples."},
                        {"name": "held-out set", "aliases": [], "artefact": "dataset",
                         "quote": "Our method relies on a small held-out set."}],
           "limitations": [], "setting": {"tasks": ["adversarial training"], "datasets": ["ImageNet"], "metrics": []}}
    stmts, st = S.run("2302.1", "2023-02-23", "t", ABSTRACT, None, chat=lambda *a, **k: json.dumps(out))
    names = [s.meta.get("name") for s in stmts if s.meta.get("name")]
    assert names == ["SelfFit"]                     # the reliance sentence is not a proposal
    assert st["quote_rejected"] == 2                 # invented contribution + held-out set
    setting = [s for s in stmts if s.facet == "setting"][0]
    assert setting.meta["tasks"] == ["adversarial training"] and setting.meta["datasets"] == []  # ImageNet not in text
    assert all(not s.validate() for s in stmts)


def test_other_pass_vocab_and_support():
    items = [{"sentence_id": 1, "sentence": "Adversarial training [18] directly augments the data with adversarial "
                                          "examples and is considered one of the most effective defenses.",
              "cited": "paper:1706.06083", "title": "Towards deep learning models resistant", "raw": "", "group": ["paper:1706.06083"]},
             {"sentence_id": 2, "sentence": "Some works [4,14] claim that better initialization helps.",
              "cited": "paper:2001.1", "title": "x", "raw": "", "group": ["paper:2001.1", "paper:2001.2"]},
             {"sentence_id": 3, "sentence": "We follow [7].", "cited": "stub:foo", "title": "foo", "raw": "", "group": ["stub:foo"]}]
    reply = {"pairs": [
        {"id": 1, "function": "background", "relation": "background",
         "about": "augments the data with adversarial examples", "facet": "method",
         "category": "most effective defenses", "limitation": "requires huge compute budgets"},
        {"id": 2, "function": "background", "relation": "background", "about": "better initialization helps",
         "facet": "result", "category": None, "limitation": None},
        {"id": 3, "function": "basis", "relation": "inspires", "about": None, "facet": "method"}]}
    stmts, st = O.run_batch("2302.1", "2023-02-23", items, chat=lambda *a, **k: json.dumps(reply))
    assert st["parsed"] == 2 and st["bad_vocab"] == 1          # 'inspires' is not in the vocabulary
    s1 = [s for s in stmts if s.meta.get("sentence_id") == 1][0]
    assert s1.meta["category"] == "most effective defenses"
    assert s1.meta["limitation"] is None                        # not supported by the sentence
    assert all(s.quote == items[s.meta["sentence_id"] - 1]["sentence"] for s in stmts)
    assert all(not s.validate() for s in stmts)
