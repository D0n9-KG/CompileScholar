# -*- coding: utf-8 -*-
"""Read inspect-ai .eval logs (zstd-compressed zip) without zipfile support.

Python 3.13 zipfile rejects compress_type 93 (zstd). This parses local
file headers manually and decompresses with the zstandard package.

Usage:
  from read_eval_log import read_eval_scores
  read_eval_scores('path/to/x.eval')   # -> [{sample scores}, ...]
"""
import glob
import io
import json
import struct
import sys

import zstandard


def _parse(raw: bytes, name_filter=None):
    results = {}
    pos = 0
    while True:
        idx = raw.find(b"PK\x03\x04", pos)
        if idx < 0:
            break
        nlen = struct.unpack("<H", raw[idx+26:idx+28])[0]
        elen = struct.unpack("<H", raw[idx+28:idx+30])[0]
        name = raw[idx+30:idx+30+nlen].decode("utf-8", errors="replace")
        if (not name_filter or name_filter(name)) and name.endswith(".json"):
            method = struct.unpack("<H", raw[idx+8:idx+10])[0]
            csize = struct.unpack("<I", raw[idx+18:idx+22])[0]
            data_start = idx + 30 + nlen + elen
            cdata = raw[data_start:data_start+csize]
            try:
                if method == 93:
                    text = zstandard.ZstdDecompressor().stream_reader(
                        io.BytesIO(cdata)).read()
                elif method == 0:
                    text = cdata
                elif method == 8:
                    import zlib
                    text = zlib.decompress(cdata, -15)
                else:
                    pos = idx + 4
                    continue
                results[name] = json.loads(text)
            except Exception:
                pass
        pos = idx + 4
    return results


def read_eval_scores(eval_path: str) -> list:
    """-> list of per-sample score dicts (from _journal/summaries)."""
    raw = open(eval_path, "rb").read()
    files = _parse(raw, lambda n: "summaries" in n)
    out = []
    for name in sorted(files):
        d = files[name]
        if isinstance(d, list):
            out.extend(x for x in d if isinstance(x, dict))
        elif isinstance(d, dict):
            out.append(d)
    return out


def summarize(eval_path: str) -> dict:
    """Aggregate facet means over samples."""
    samples = read_eval_scores(eval_path)
    acc = {}
    n = 0
    for s in samples:
        sc = s.get("scores") or {}
        for _scorer, obj in sc.items():
            v = obj.get("value") if isinstance(obj, dict) else None
            if isinstance(v, dict):
                for k, x in v.items():
                    if isinstance(x, (int, float)):
                        acc.setdefault(k, []).append(x)
        n += 1
    means = {k: round(sum(v)/len(v), 4) for k, v in acc.items()}
    return {"n_samples": n, "facet_means": means}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) > 1:
        for p in (sys.argv[1:] if len(sys.argv) > 1
                  else sorted(glob.glob("logs/*.eval"))):
            print(p.split("\\")[-1][:50], summarize(p))
