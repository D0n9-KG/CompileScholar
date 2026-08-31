# -*- coding: utf-8 -*-
"""Install a sci-evo library paper's MinerU content_list into the MINERU_BASE
layout the kernel pipeline expects (2026-08-28).

MinerU new versions emit BOTH:
  *_content_list.json     — flat [{type, text, text_level, bbox, page_idx}]  <- we use this
  *_content_list_v2.json  — per-page nested {content: {...}}                 <- incompatible
A naive sorted()[-1] picks _v2 (alphabetically last) — explicit filter instead.
"""
import glob, json, os, shutil, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
LIB = "C:/Users/D0n9/Desktop/sci-evo-extract/data/library/papers"
MINERU_BASE = "C:/Users/D0n9/Desktop/science_evo/data/upstream/remote_mineru/mineru_2355/papers"


def install(pid: str) -> bool:
    """Copy the FLAT content_list.json for pid into MINERU_BASE/<pid>/. False if
    the paper has no MinerU output yet or only _v2 exists."""
    srcs = [s for s in glob.glob(f"{LIB}/{pid}/mineru/**/*_content_list.json",
                                 recursive=True)
            if not s.endswith("_v2.json")]
    if not srcs:
        return False
    dst_dir = os.path.join(MINERU_BASE, pid)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(srcs[0], os.path.join(dst_dir, "content_list.json"))
    return True


if __name__ == "__main__":
    for pid in sys.argv[1:]:
        print(f"{pid}: installed={install(pid)}")
