# -*- coding: utf-8 -*-
"""KB v2 记录向量（混合检索的向量腿）。与 HybridIndex.items 同序；float32 落盘 + meta（含 records.json 的 sha16 指纹）。
用法：python embed_kb_v2.py"""
import hashlib
import json
import os
import sys
import time
from array import array

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", "src"))
sys.path.insert(0, SRC)
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",192.168.199.73,localhost,127.0.0.1"
from kb_compiler.retrieve.hybrid import HybridIndex  # noqa: E402
from kb_infra.embedding import embed_local  # noqa: E402

V2 = os.path.join(HERE, "..", "base_kb_v2")


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    P = json.load(open(os.path.join(V2, "papers.json"), encoding="utf-8"))
    R = json.load(open(os.path.join(V2, "records.json"), encoding="utf-8"))
    ix = HybridIndex(R, P)
    texts = [t[:1200] for _, _, t in ix.items]
    out, t0 = [], time.time()
    B = 64
    for i in range(0, len(texts), B):
        for attempt in range(4):
            try:
                out.extend(embed_local(texts[i:i + B], batch_size=B))
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        if (i // B) % 50 == 0:
            print(f"  {i + B}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    dim = len(out[0])
    with open(os.path.join(V2, "record_vecs.f32"), "wb") as fh:
        array("f", (x for v in out for x in v)).tofile(fh)
    json.dump({"n": len(out), "dim": dim, "model": "local-qwen3-embedding-8b",
               "records_sha16": sha16(os.path.join(V2, "records.json")),
               "papers_sha16": sha16(os.path.join(V2, "papers.json"))},
              open(os.path.join(V2, "record_vecs.meta.json"), "w"), indent=1)
    print(f"done n={len(out)} dim={dim} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
