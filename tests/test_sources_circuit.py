# -*- coding: utf-8 -*-
"""sources.circuit (moved from sci-evo with its tests)."""
import time

from compilescholar.sources.circuit import SourceCircuit


def test_circuit_trip_and_recover():
    c = SourceCircuit(failure_threshold=2, window_s=60, cooldown_s=0.05)
    assert c.allow("s") and c.allow("s")
    c.record("s", ok=False)
    assert c.state("s") == "closed"
    assert c.allow("s")
    c.record("s", ok=False)
    assert c.state("s") == "open"
    assert not c.allow("s")
    time.sleep(0.06)
    assert c.state("s") == "half_open"
    assert c.allow("s")
    assert not c.allow("s")          # one probe at a time
    c.record("s", ok=True)
    assert c.state("s") == "closed"


def test_circuit_snapshot_no_deadlock():
    c = SourceCircuit()
    c.record("s", ok=False)
    snap = c.snapshot()
    assert "s" in snap and snap["s"]["state"] == "closed"
