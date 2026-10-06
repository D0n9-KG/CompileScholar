# -*- coding: utf-8 -*-
"""documents.versions.delta: re-flowed sentences are not new, edits are revised, new sentences / references / their
citation pairs are the delta."""
from compilescholar.documents.versions import delta


def _doc(sents, entries, cites):
    return {"sentences": [{"sid": f"s{i}", "unit": "u", "text": t} for i, t in enumerate(sents)],
            "entries": entries, "cites": cites}


V1 = _doc(["We propose a graph transformer for molecules.",
           "It outperforms message passing networks on three benchmarks [1].",
           "Training takes two days on one GPU."],
          {"b0": {"raw": "A. B. Message passing. 2017.", "title": "Neural message passing"}},
          [("s1", "b0", 1)])


def test_delta():
    latest = _doc(["We propose a graph transformer for molecules .",                 # re-flowed: same
                   "It outperforms message passing networks on four benchmarks [1].",  # edited: revised
                   "We also prove a convergence bound for the attention layer [2].",   # new, new reference
                   "Training takes two days on one GPU."],
                  {"b0": {"raw": "A. B. Message passing. 2017.", "title": "Neural message passing"},
                   "b1": {"raw": "C. D. Convergence of attention. 2021.", "title": "Convergence of attention"}},
                  [("s1", "b0", 1), ("s2", "b1", 1)])
    d = delta(V1, latest)
    assert [s["text"] for s in d["sentences"]] == ["We also prove a convergence bound for the attention layer [2]."]
    assert [s["text"] for s in d["revised"]] == ["It outperforms message passing networks on four benchmarks [1]."]
    assert d["revised"][0]["v1_text"].endswith("three benchmarks [1].")
    assert list(d["entries"]) == ["b1"]
    assert sorted(d["cites"]) == [("s1", "b0", 1), ("s2", "b1", 1)]


def test_reference_added_in_a_later_version_is_only_in_the_delta():
    """Phase B gate (版本泄漏): a work cited only from v2 on is not part of what v1 says; it enters with the delta and
    therefore with the latest version's date."""
    latest = _doc([s["text"] for s in V1["sentences"]] + ["Recent work [2] extends this to proteins."],
                  {**V1["entries"], "b1": {"raw": "E. F. Protein transformers. 2024.", "title": "Protein transformers"}},
                  V1["cites"] + [("s3", "b1", 1)])
    d = delta(V1, latest)
    assert "b1" in d["entries"] and "b1" not in V1["entries"]
    assert ("s3", "b1", 1) in d["cites"] and all(c[1] != "b1" for c in V1["cites"])


def test_identical_versions_have_no_delta():
    d = delta(V1, V1)
    assert not d["sentences"] and not d["revised"] and not d["entries"] and not d["cites"]
