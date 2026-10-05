# -*- coding: utf-8 -*-
"""Build data/corpus/papers.sqlite from the local OAI snapshot + the OAI-PMH harvest done so far (idempotent; rerun
after the harvest finishes to add the rest)."""
import json
import sys

from compilescholar.corpus import papers


def main():
    stats = papers.build(log=lambda s: print(s, flush=True))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
