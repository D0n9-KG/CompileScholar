"""Control: question-independent popularity list (papers most frequently referenced across all 2683 seeds'
reference lists), leave-own-question-out; recall@100 vs one-hop frequency top-100."""
from collections import Counter
from evaluate import G, S, R, Matcher, nbr_rank, recall, boot, key
glob = Counter(); item = {}
for p, x in R.items():
    for c in x["refs"]:
        if c.get("paperId") or c.get("title"):
            glob[key(c)] += 1; item[key(c)] = c
pop, own = [], []
for q in G:
    m, n = Matcher(q), len(q["refs"])
    sd = (S.get(q["qid"]) or {}).get("rec") or []
    mine = Counter(key(c) for s in sd[:30] for c in (R.get(s["paperId"]) or {}).get("refs") or [])
    # subtract this question's own seeds' contribution -> pure prior popularity from other questions
    ranked = sorted(glob, key=lambda k: -(glob[k] - mine.get(k, 0)))
    pop.append(recall([item[k] for k in ranked[:100] if not m.is_target(item[k])], m, n)[0])
    own.append(recall(nbr_rank(sd, 30, m)[0][:100], m, n)[0])
print("global-popularity@100 (leave-own-q-out)", boot(pop))
print("rec one-hop nbK30@100               ", boot(own))
print("delta", boot([a - b for a, b in zip(own, pop)]))
