# -*- coding: utf-8 -*-
"""sources.http against a local fake server: Retry-After on 429, retry on 5xx, definitive 404 cached, transient
failures never cached, the circuit fails fast; sources.scihub lookup normalises the DOI and checks the CRC."""
import threading
import zipfile
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from compilescholar.sources import http as H

PLAN: dict[str, list] = {}
HITS: dict[str, int] = {}


class _H(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        HITS[path] = HITS.get(path, 0) + 1
        steps = PLAN.get(path, [(200, {}, b"ok")])
        status, headers, body = steps[min(HITS[path] - 1, len(steps) - 1)]
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("CS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(H, "GAP", {})
    monkeypatch.setattr(H, "CIRCUIT", H.SourceCircuit(failure_threshold=3, window_s=60, cooldown_s=60))
    monkeypatch.setattr(H.time, "sleep", lambda s: None)
    PLAN.clear()
    HITS.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_retry_after_then_success_is_cached(server):
    PLAN["/a"] = [(429, {"Retry-After": "2"}, b""), (200, {}, b'{"x": 1}')]
    r = H.get("t", server + "/a")
    assert r.ok and r.json() == {"x": 1} and HITS["/a"] == 2
    r2 = H.get("t", server + "/a")
    assert r2.from_cache and r2.json() == {"x": 1} and HITS["/a"] == 2


def test_404_is_definitive_and_cached(server):
    PLAN["/missing"] = [(404, {}, b"no")]
    assert H.get("t", server + "/missing").status == 404
    assert H.get("t", server + "/missing").from_cache and HITS["/missing"] == 1


def test_transient_failure_is_raised_and_not_cached(server):
    PLAN["/down"] = [(503, {}, b"")]
    with pytest.raises(H.Transient):
        H.get("t", server + "/down", tries=2)
    PLAN["/down"] = [(200, {}, b"up")]
    HITS.clear()
    assert H.get("t", server + "/down").text() == "up" and HITS["/down"] == 1


def test_circuit_opens_after_repeated_5xx(server):
    PLAN["/bad"] = [(500, {}, b"")]
    with pytest.raises(H.Transient):
        H.get("t2", server + "/bad", tries=3)
    n = HITS["/bad"]
    with pytest.raises(H.Transient, match="circuit open"):
        H.get("t2", server + "/bad", tries=3)
    assert HITS["/bad"] == n


def test_scihub_lookup_and_crc(tmp_path, monkeypatch):
    import sqlite3
    from compilescholar.sources import scihub as S
    pdf = b"%PDF-1.4 test"
    (tmp_path / "arch").mkdir()
    with zipfile.ZipFile(tmp_path / "arch" / "a.zip", "w") as z:
        z.writestr("10.1000/x1.pdf", pdf)
    db = tmp_path / "idx.sqlite"
    c = sqlite3.connect(db)
    c.executescript("CREATE TABLE archives(id INTEGER PRIMARY KEY, rel_path TEXT); CREATE TABLE items(doi_norm TEXT, "
                    "doi_raw TEXT, archive_id INT, inner_path TEXT, size INT, crc INT);")
    c.execute("INSERT INTO archives VALUES (1, 'a.zip')")
    c.execute("INSERT INTO items VALUES ('10.1000/x1', '10.1000/X1', 1, '10.1000/x1.pdf', ?, ?)",
              (len(pdf), zlib.crc32(pdf)))
    c.commit()
    c.close()
    res = {"scihub_index": db, "scihub_archive": tmp_path / "arch"}
    monkeypatch.setattr(S.paths, "resource", lambda k: res[k])
    monkeypatch.setattr(S, "_tl", threading.local())
    hit = S.lookup("https://doi.org/10.1000/X1.")
    assert hit and S.read(hit) == pdf and S.pointer(hit)["channel"] == "scihub_local"
    assert S.lookup("10.1000/nope") is None
    with pytest.raises(ValueError):
        S.read(S.Hit(hit.doi, hit.archive, hit.inner_path, hit.size, hit.crc + 1))
