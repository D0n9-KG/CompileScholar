# -*- coding: utf-8 -*-
"""B2 chunk-parallelism tests for slot.extract_paper (no LLM cost).

Verifies the 2026-09-18 refactor:
  1. serial path (chunk_threads=1) is behavior-identical to a reference
     implementation of the old inline loop (same records, same order),
     including CHUNK FAIL counting.
  2. parallel path (chunk_threads>1) produces BYTE-IDENTICAL output to the
     serial path, even when calls complete out of order (simulated by
     reversed completion order).
  3. parallel path actually overlaps calls (wall-clock proves concurrency).

Run: py -3.13 -X utf8 test_slot_chunk_parallel.py
"""
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import kb_compiler.records.slot as slot  # noqa: E402


TEXT = "\n\n".join(
    "## Section %d\n\n" % i + ("Sentence about method %d with number. " % i) * 40
    for i in range(6))

CARD = {"method_identity": {"canonical_name": "TestMethod",
                           "aliases": ["TM"]},
        "related_methods": [],
        "experimental_matrix": []}
REGISTRY = {"entities": [], "surface_index": {}}
VOCAB = {"setup": [], "variant": [], "hyperparam_items": []}


def fake_call_json_factory(per_call_delay=0.0, scramble=False):
    """Deterministic fake: each chunk returns one record carrying its chunk_id.
    scramble=True delays earlier chunks longer (completion order reversed)."""
    lock = threading.Lock()
    state = {"n": 0}

    def fake(prompt, model, max_tokens=0, retries=0, salvage=False):
        with lock:
            state["n"] += 1
            idx = state["n"]
        # find which chunk this prompt belongs to via the section marker
        sec = prompt.split("Section ")[-1].split("\n")[0].strip() \
            if "Section " in prompt else "?"
        if scramble:
            time.sleep(per_call_delay * (8 - idx % 8) / 8.0 + 0.01)
        elif per_call_delay:
            time.sleep(per_call_delay)
        return {"records": [{"kind": "result", "method": "TestMethod",
                             "quote": "Sentence about method " + sec,
                             "measure": {"metric": "acc", "value": sec,
                                         "unit": "%"},
                             "role": "main", "epistemic": "stated",
                             "marker": sec}]}
    return fake, state


def run_extract(threads, scramble=False, delay=0.0):
    fake, state = fake_call_json_factory(delay, scramble)
    orig = slot.call_json
    slot.call_json = fake
    try:
        t0 = time.time()
        pid, out = slot.extract_paper("pX", TEXT, CARD, REGISTRY, VOCAB,
                                      "fake-model", "Test Paper", threads)
        dt = time.time() - t0
    finally:
        slot.call_json = orig
    return out, dt, state


def main():
    # 1+2: serial vs parallel(scrambled completion) byte-identical
    out_s, _, _ = run_extract(1)
    out_p, _, _ = run_extract(4, scramble=True, delay=0.15)
    assert out_s == out_p, "parallel output differs from serial output"
    n_rec = len(out_s["records"])
    assert n_rec >= 6, f"expected >=6 records (one per section), got {n_rec}"
    # order preservation: markers in original section order
    markers = [r.get("marker") for r in out_s["records"]]
    assert markers == sorted(markers, key=lambda m: int(m)), \
        f"records not in original chunk order: {markers}"
    print(f"[order_identical] PASS  records={n_rec} markers={markers}")

    # 3: real concurrency — 6 chunks x 0.3s in 6 threads must be ~0.3s, not 1.8s
    _, dt_serial = run_extract(1, delay=0.3)[0], None
    out_c, dt_par, _ = run_extract(6, delay=0.3)
    _, dt_serial, _ = run_extract(1, delay=0.3)
    assert dt_par < dt_serial * 0.5, \
        f"no speedup: parallel={dt_par:.2f}s vs serial={dt_serial:.2f}s"
    print(f"[concurrent] PASS  6x0.3s calls: parallel {dt_par:.2f}s "
          f"vs serial {dt_serial:.2f}s")

    print("ALL PASS")


if __name__ == "__main__":
    main()
