# -*- coding: utf-8 -*-
"""Copy the 5 ML succession papers' MinerU content_list.json into LogicKG's
MINERU_BASE (one dir per PPR id) and sanity-check each (blocks/headings/title)."""
import json, os, shutil, glob

LIB = "C:/Users/D0n9/Desktop/sci-evo-extract/data/library/papers"
BASE = "C:/Users/D0n9/Desktop/science_evo/data/upstream/remote_mineru/mineru_2355/papers"

EXPECT = {
    "PPR_DD48410E18B3": "Double DQN",
    "PPR_11239E342D42": "Dueling",
    "PPR_C944277AF1BE": "Prioritized Experience Replay",
    "PPR_F56A70559364": "A3C",
    "PPR_0A209934694A": "Rainbow",
}
for pid, name in EXPECT.items():
    hits = glob.glob(f"{LIB}/{pid}/mineru/**/vlm/*_content_list.json", recursive=True)
    if not hits:
        print(f"MISSING content_list for {pid} ({name})")
        continue
    src = sorted(hits)[-1]
    cl = json.load(open(src, encoding="utf-8"))
    dst_dir = os.path.join(BASE, pid)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "content_list.json")
    shutil.copyfile(src, dst)
    n_text = sum(1 for it in cl if it.get("type") == "text")
    n_hd = sum(1 for it in cl if it.get("text_level"))
    title = str(cl[0].get("text", ""))[:60]
    print(f"OK {name:28s} {pid} blocks={len(cl):3d} text={n_text:3d} headings={n_hd:2d} | {title}")
