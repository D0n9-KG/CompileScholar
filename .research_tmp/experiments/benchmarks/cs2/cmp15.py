"""同题集官方口径对比（缺答按 0）。用法：python cmp15.py <offset> <limit> file1 file2 ..."""
import json, sys
from cs2_scoring import summarize, split_qids
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
off, lim = int(sys.argv[1]), int(sys.argv[2])
qids = split_qids("dev")[off:off + lim]
for f in sys.argv[3:]:
    s = summarize(f, qids); m = s["mean"]
    print(f"{f.split('/')[-1][:44]:44s} n={s['n']} miss0={s['n_missing_as_zero']} G={m['global']:.3f} IR={m['ingredient_recall']:.3f} AP={m['answer_precision']:.3f} CR={m['citation_recall']:.3f} CP={m['citation_precision']:.3f}")
