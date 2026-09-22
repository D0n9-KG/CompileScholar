# -*- coding: utf-8 -*-
"""A2 global chunk-pool tests (no LLM cost): WAL round-trip, chunk-level
resume, repair seeding, and pool-ordering byte-comparability.

The B2 invariance (serial == parallel == pool, byte-for-byte) is the
load-bearing property: the pool must never change output, only wall-clock.
"""
import json
import contextlib
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import kb_compiler.records.deep_extract as deep_extract  # noqa: E402

TEXT = "\n\n".join(
    "## Section %d\n\n" % i + ("Sentence about method %d with number. " % i) * 40
    for i in range(6))
CARD = {"method_identity": {"canonical_name": "TestMethod"},
        "sections": [["Section %d" % i, "methods"] for i in range(6)]}
REGISTRY = {"entities": [{"entity_id": "e1", "canonical": "TestMethod",
                          "aliases": ["TestMethod"]}],
            "surface_index": {"testmethod": "e1"}}
VOCAB = {}


def fake_call_json_factory(delay=0.0, scramble=False):
    """Same fake as the B2 test: one record per section, marker = section id."""
    state = {"n": 0, "order": []}

    def fake(prompt, model, max_tokens=8000, retries=3, salvage=False):
        sec = state["n"]
        state["n"] += 1
        state["order"].append(sec)
        if delay:
            time.sleep(delay if not scramble or sec % 2 == 0 else delay * 3)
        if "absence" in prompt[:300].lower() or "缺失" in prompt[:300]:
            return {"records": []}
        return {"records": [{"kind": "result", "method": "TestMethod",
                             "quote": "Sentence about method " + str(sec),
                             "measure": {"metric": "acc", "value": str(sec),
                                         "unit": "%"},
                             "role": "main", "epistemic": "stated",
                             "marker": str(sec)}]}
    return fake, state


@contextlib.contextmanager
def _with_fake(fake):
    orig = deep_extract.call_json
    deep_extract.call_json = fake
    try:
        yield
    finally:
        deep_extract.call_json = orig


def test_pool_matches_serial_output():
    """Pool path (global executor) must equal extract_paper byte-for-byte."""
    fake1, _ = fake_call_json_factory()
    with _with_fake(fake1):
        _, out_serial = deep_extract.extract_paper("pX", TEXT, CARD, REGISTRY, VOCAB,
                                           "m", "T", 1)
    # fresh fake with the same per-call mapping (content depends on call order
    # per phase, so each phase gets its own counter starting at 0 — the fake's
    # N maps 1:1 to chunk order because both paths process chunks in order)
    fake2, _ = fake_call_json_factory(delay=0.05, scramble=True)
    with _with_fake(fake2):
        ctx = deep_extract.build_paper_tasks("pX", TEXT, CARD, REGISTRY, VOCAB, "T")
        objs = {ch["chunk_id"]: deep_extract._call_chunk(p, "m")
                for ch, p in ctx["chunk_prompts"]}
        abs_obj = deep_extract._call_absence(ctx["absence_prompt"], "m")
        _, out_pool = deep_extract.finalize_paper("pX", ctx, objs, abs_obj,
                                          REGISTRY, CARD, "T")
    # stats identical; records identical (id/order/content)
    assert out_serial["stats"] == out_pool["stats"]
    assert out_serial["records"] == out_pool["records"]
    assert out_serial["entity_queue"] == out_pool["entity_queue"]


def test_wal_roundtrip_and_resume():
    """WAL: written chunks replay; resume only calls MISSING chunks."""
    tmp = tempfile.mkdtemp()
    out_path = os.path.join(tmp, "rec.json")
    wal_path = out_path + ".progress.jsonl"
    fake, state = fake_call_json_factory()
    manifest = {"pX": {"title": "T"}}
    with _with_fake(fake):
        deep_extract.run_pool([("pX", TEXT)], {"pX": CARD}, REGISTRY, VOCAB, "m",
                      manifest, 4, out_path)
    n_first = state["n"]
    assert n_first >= 7  # 6 chunks + absence
    assert os.path.exists(wal_path)
    # everything done -> replay WAL, no new calls
    fake2, state2 = fake_call_json_factory()
    with _with_fake(fake2):
        deep_extract.run_pool([("pX", TEXT)], {"pX": CARD}, REGISTRY, VOCAB, "m",
                      manifest, 4, out_path)
    # finalize re-runs (paper already in results -> not re-todo'd by run_pool?
    # run_pool re-finalizes from WAL: acceptable; the invariant = ZERO new
    # LLM calls on a fully-WAL'd paper
    assert state2["n"] == 0, f"resume made {state2['n']} new calls"


def test_partial_wal_only_calls_holes():
    """A WAL holding 3 of 6 chunks: only the 3 missing chunks get called."""
    tmp = tempfile.mkdtemp()
    out_path = os.path.join(tmp, "rec.json")
    wal_path = out_path + ".progress.jsonl"
    with open(wal_path, "w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({"pid": "pX", "cid": f"pX#c{i}", "ok": True,
                                "obj": {"records": []}}) + "\n")
    fake, state = fake_call_json_factory()
    manifest = {"pX": {"title": "T"}}
    with _with_fake(fake):
        deep_extract.run_pool([("pX", TEXT)], {"pX": CARD}, REGISTRY, VOCAB, "m",
                      manifest, 4, out_path)
    # 6 chunks total, 3 seeded -> 3 chunk calls + 1 absence = 4
    assert state["n"] == 4, f"expected 4 calls (3 holes + absence), got {state['n']}"


def test_repair_seeding_only_calls_holes():
    """--repair semantics: existing records seed the WAL; hole chunks (no
    records under their chunk_id) are the only ones re-called."""
    # build a full result for pX, then knock out records of chunks 4,5
    fake, _ = fake_call_json_factory()
    with _with_fake(fake):
        _, out = deep_extract.extract_paper("pX", TEXT, CARD, REGISTRY, VOCAB, "m", "T", 1)
    out["records"] = [r for r in out["records"]
                      if r["chunk_id"] not in ("pX#c4", "pX#c5")]
    tmp = tempfile.mkdtemp()
    out_path = os.path.join(tmp, "rec.json")
    json.dump({"pX": out}, open(out_path, "w", encoding="utf-8"))
    wal = deep_extract._wal_load(out_path + ".progress.jsonl")
    seeded = deep_extract._seed_wal_from_results({"pX": out}, wal)
    # chunks 0-3 seeded (4 chunks with records); absence seeded (records exist
    # only if absence produced any — fake returns empty, so NOT seeded)
    seeded_cids = sorted(cid for pid, cid in seeded if cid != "absence")
    assert seeded_cids == ["pX#c0", "pX#c1", "pX#c2", "pX#c3"], seeded_cids
    # now run the pool on the WAL: only c4, c5 + absence get called
    fake2, state2 = fake_call_json_factory()
    wal_path = out_path + ".progress.jsonl"
    with open(wal_path, "a", encoding="utf-8") as f:
        for (pid, cid), obj in seeded.items():
            f.write(json.dumps({"pid": pid, "cid": cid, "ok": True,
                                "obj": obj}) + "\n")
    manifest = {"pX": {"title": "T"}}
    with _with_fake(fake2):
        deep_extract.run_pool([("pX", TEXT)], {"pX": CARD}, REGISTRY, VOCAB, "m",
                      manifest, 4, out_path)
    assert state2["n"] == 3, f"expected 3 calls (2 holes + absence), got {state2['n']}"


def test_fully_wald_paper_finalizes_without_calls():
    """Crash between last chunk and finalize: paper fully in WAL but not in
    results — run_pool must finalize it with ZERO new LLM calls."""
    tmp = tempfile.mkdtemp()
    out_path = os.path.join(tmp, "rec.json")
    wal_path = out_path + ".progress.jsonl"
    fake, _ = fake_call_json_factory()
    manifest = {"pX": {"title": "T"}}
    with _with_fake(fake):
        ctx = deep_extract.build_paper_tasks("pX", TEXT, CARD, REGISTRY, VOCAB, "T")
        with open(wal_path, "w", encoding="utf-8") as f:
            for ch, p in ctx["chunk_prompts"]:
                obj = deep_extract._call_chunk(p, "m")
                f.write(json.dumps({"pid": "pX", "cid": ch["chunk_id"],
                                    "ok": True, "obj": obj}) + "\n")
            aobj = deep_extract._call_absence(ctx["absence_prompt"], "m")
            f.write(json.dumps({"pid": "pX", "cid": "absence", "ok": True,
                                "obj": aobj}) + "\n")
    fake2, state2 = fake_call_json_factory()
    with _with_fake(fake2):
        deep_extract.run_pool([("pX", TEXT)], {"pX": CARD}, REGISTRY, VOCAB, "m",
                      manifest, 4, out_path)
    assert state2["n"] == 0
    res = json.load(open(out_path, encoding="utf-8"))
    assert "pX" in res and res["pX"]["stats"]["records"] >= 6


if __name__ == "__main__":
    for fn in [test_pool_matches_serial_output, test_wal_roundtrip_and_resume,
               test_partial_wal_only_calls_holes,
               test_repair_seeding_only_calls_holes,
               test_fully_wald_paper_finalizes_without_calls]:
        fn()
        print(f"[{fn.__name__}] PASS")
    print("ALL PASS")
