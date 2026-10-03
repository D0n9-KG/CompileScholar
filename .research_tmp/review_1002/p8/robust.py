"""Robustness: is the one-hop gain driven by a single near-duplicate seed (companion / earlier version of target)?
Leave-best-seed-out recall, #seeds contributing gold, max seed-target title similarity; plus clean examples."""
import json, sys
from evaluate import G, S, R, Matcher, nbr_rank, recall, boot, pool
from step1_map import sim

arm = sys.argv[1] if len(sys.argv) > 1 else "rec"
full, lbo, ncontrib, maxsim, ex = [], [], [], [], []
for q in G:
    sd = (S.get(q["qid"]) or {}).get(arm) or []
    m, n = Matcher(q), len(q["refs"])
    nb, _ = nbr_rank(sd, 30, m)
    full.append(recall(nb[:100], m, n)[0])
    contrib = []
    for s in sd[:30]:
        h = {m(c) for c in (R.get(s["paperId"]) or {}).get("refs") or []} - {None}
        contrib.append((len(h), s))
    ncontrib.append(sum(1 for c, _ in contrib if c))
    best = max(contrib, key=lambda x: x[0], default=(0, None))[1]
    sd2 = [s for s in sd if s is not best]
    nb2, _ = nbr_rank(sd2, 29, m)
    lbo.append(recall(nb2[:100], m, n)[0])
    maxsim.append(max((sim(s["title"], q["title"]) for s in sd[:30]), default=0))
    semhit = recall(sd[:200], m, n)[1]
    nb_f = nbr_rank(sd, 30, m)
    for c, f in zip(nb_f[0][:100], nb_f[1][:100]):
        i = m(c)
        if i is not None and i not in semhit and f >= 3 and len(ex) < 40:
            cit = [s["title"][:50] for s in sd[:30] if any(m(x) == i for x in (R.get(s["paperId"]) or {}).get("refs") or [])]
            ex.append((q["qid"], q["title"][:55], q["refs"][i]["title"][:80].replace("\r\n", ""), q["refs"][i]["arxiv"], f, cit[:2]))
print(arm, "nbK30@100 full", boot(full))
print(arm, "nbK30@100 leave-best-seed-out", boot(lbo))
print("seeds citing >=1 gold: mean", sum(ncontrib) / len(ncontrib), "min", min(ncontrib))
print("max seed-target title sim: >=0.7 in", sum(1 for x in maxsim if x >= 0.7), "of", len(maxsim))
seen = set()
for e in ex:
    if e[0] not in seen:
        seen.add(e[0]); print(e)
