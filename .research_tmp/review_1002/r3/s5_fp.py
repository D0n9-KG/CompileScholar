import sys; sys.stdout.reconfigure(encoding="utf-8")
from kbload import load
kb,records,views,man=load()
n=sum(len(p.get("records",[])) for p in records.values())
print("fingerprint now", kb._emb_fingerprint(), "n", n)
