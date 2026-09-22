# -*- coding: utf-8 -*-
"""Paper-level pipeline tests (user directive 2026-09-22): inline postcheck
inside deep_extract's pool must be byte-equivalent to the batch postcheck
stage, and the stage must skip inline-checked papers (coexistence)."""
import contextlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import kb_compiler.records.deep_extract as de  # noqa: E402
import kb_compiler.records.postcheck as pc  # noqa: E402

TEXT = "\n\n".join(
    "## Section %d\n\n" % i + ("Sentence about method %d with number. " % i) * 40
    for i in range(6))
CARD = {"method_identity": {"canonical_name": "TestMethod"},
        "sections": [["Section %d" % i, "methods"] for i in range(6)]}
REGISTRY = {"entities": [{"entity_id": "e1", "canonical": "TestMethod",
                          "aliases": ["TestMethod"]}],
            "surface_index": {"testmethod": "e1"}}
VOCAB = {"subject": [], "setup": [], "variant": []}


@contextlib.contextmanager
def _fake_calls(fakes):
    """fakes: dict path->fn. Patch call_json in BOTH modules (deep_extract
    imports it into its namespace; postcheck likewise)."""
    orig_de, orig_pc = de.call_json, pc.call_json
    de.call_json = lambda p, m, **k: fakes["_chunk"](p, m, **k)
    pc.call_json = lambda p, m, **k: fakes["_repair"](p, m, **k)
    try:
        yield
    finally:
        de.call_json, pc.call_json = orig_de, orig_pc


def _ok_chunk_records():
    n = [0]

    def _chunk(prompt, model, **kw):
        i = n[0]
        n[0] += 1
        if "absence" in prompt[:300].lower() or "缺失" in prompt[:300]:
            return {"records": []}
        return {"records": [{
            "kind": "result", "method": "TestMethod",
            "quote": "Sentence about method " + str(i),
            "measure": {"metric": "acc", "value": str(i), "unit": "%"},
            "role": "main", "epistemic": "stated"}]}
    return _chunk


def _no_repair(prompt, model, **kw):
    return None  # any repair attempt fails -> drop (records are clean anyway)


def test_inline_equals_batch():
    """Inline path (pool + inline_postcheck) == extract -> batch postcheck."""
    tmp = tempfile.mkdtemp()
    out_a = os.path.join(tmp, "rec_inline.json")
    out_b = os.path.join(tmp, "rec_batch.json")
    manifest = {"pX": {"title": "T"}}
    fake = {"_chunk": _ok_chunk_records(), "_repair": _no_repair}
    with _fake_calls(fake):
        # A: inline
        de.run_pool([("pX", TEXT)], {"pX": CARD}, REGISTRY, VOCAB, "m",
                    manifest, 4, out_a,
                    inline_postcheck={"vocab": VOCAB,
                                      "out_dir": os.path.join(tmp, "pc_a"),
                                      "model": "m", "texts": {"pX": TEXT}})
        # B: batch
        de.run_pool([("pX", TEXT)], {"pX": CARD}, REGISTRY, VOCAB, "m",
                    manifest, 4, out_b)
    pc_a = json.load(open(os.path.join(tmp, "pc_a", "records_checked.json"),
                          encoding="utf-8"))
    checked_b, dropped_b, _w, _s = pc.run_postcheck(
        json.load(open(out_b, encoding="utf-8")), {"pX": TEXT}, VOCAB, "m")
    assert pc_a["pX"]["records"] == checked_b["pX"]["records"], \
        "inline records differ from batch postcheck"
    assert pc_a["pX"]["overflow"] == checked_b["pX"]["overflow"]
    assert pc_a["pX"]["entity_queue"] == checked_b["pX"]["entity_queue"]


def test_stage_skips_inline_checked():
    """postcheck main-style resume: prior records_checked entries carried
    forward, only the remainder re-checked (zero repair calls for skipped)."""
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "rec.json")
    pc_dir = os.path.join(tmp, "pc")
    os.makedirs(pc_dir)
    manifest = {"pX": {"title": "T"}}
    fake = {"_chunk": _ok_chunk_records(), "_repair": _no_repair}
    with _fake_calls(fake):
        de.run_pool([("pX", TEXT)], {"pX": CARD}, REGISTRY, VOCAB, "m",
                    manifest, 4, out,
                    inline_postcheck={"vocab": VOCAB, "out_dir": pc_dir,
                                      "model": "m", "texts": {"pX": TEXT}})
    # stage run with prior: must skip pX entirely — patch check_paper to
    # explode if called for pX
    orig = pc.check_paper
    calls = []

    def spy(pid, *a, **k):
        calls.append(pid)
        return orig(pid, *a, **k)

    pc.check_paper = spy
    try:
        checked, dropped, _w, _s = pc.run_postcheck(
            json.load(open(out, encoding="utf-8")), {"pX": TEXT}, VOCAB, "m",
            skip_pids={"pX"})
        # main() carries the prior forward after the skip (see its
        # checked.update line) — replicate that merge here
        prior = json.load(open(os.path.join(pc_dir, "records_checked.json"),
                               encoding="utf-8"))
        checked.update(prior)
    finally:
        pc.check_paper = orig
    assert calls == [], f"check_paper called for skipped pids: {calls}"
    # resume semantics: carried-forward paper present in merged output
    assert "pX" in checked


if __name__ == "__main__":
    test_inline_equals_batch()
    print("[inline_equals_batch] PASS")
    test_stage_skips_inline_checked()
    print("[stage_skips_inline_checked] PASS")
    print("ALL PASS")
