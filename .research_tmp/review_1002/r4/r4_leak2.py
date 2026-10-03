# -*- coding: utf-8 -*-
"""R4 leak part 2: (i) full-text acquired by earlier runs reused via search_text
anchors; (ii) deep_base citations on papers deep-read by the SAME question's
earlier runs; (iii) post-cutoff years among ours ext citations."""
import json, os, re, glob, collections
CS2 = r"C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2"
ARM = os.path.join(CS2, "arm_ours")
B = ['31a','31b','32a','32b','33a','33b','34a','34b','34c','34d','34e']
idx = {b: i for i, b in enumerate(B)}
added = {}
for f in sorted(glob.glob(os.path.join(ARM, 'batch*_run.log'))):
    b = re.search(r'batch(\w+)_run', f).group(1)
    for line in open(f, encoding='utf-8', errors='replace'):
        m = re.search(r'\[deep_read-index\] (\S+): \+(\d+) chunks', line)
        if m: added.setdefault(m.group(1), b)
anc = re.compile(r'\[([A-Za-z0-9_.:\-]+)#(\d+)\]')
win = pre = 0; runs_win = set(); runs_pre = set()
for b in B:
    for a in json.load(open(os.path.join(ARM, f'answers_pilot_cs2batch{b}.json'), encoding='utf-8')):
        reread = {str((t.get('args') or {}).get('paper_id'))[:40] for t in a['trajectory'] if t.get('tool') == 'deep_read'}
        for pid, _ in anc.findall(a['answer']):
            for pre_, b0 in added.items():
                if pid.startswith(pre_[:44]) or pre_.startswith(pid[:44]):
                    if pid[:40] in reread: continue
                    if b0 in idx and idx[b0] < idx[b]: win += 1; runs_win.add((b, a['id']))
                    elif b0 not in idx: pre += 1; runs_pre.add((b, a['id']))
print(f"(i) fulltext anchors from text acquired in earlier run, not re-read: within 31a-34e window={win} cites/{len(runs_win)} runs; from pre-31a dev runs={pre}/{len(runs_pre)} runs")
R = json.load(open('runs_index.json'))
dr = R['dr']
deep = json.load(open(os.path.join(CS2, 'base_kb', 'deep_read_records.json'), encoding='utf-8'))
rid2pid = {r['id']: p for p, v in deep.items() for r in v['records']}
tot = selfc = 0; runs_self = set()
for b in B:
    for a in json.load(open(os.path.join(ARM, f'answers_pilot_cs2batch{b}.json'), encoding='utf-8')):
        for tok in re.findall(r'\[([0-9a-f]{10,16})\]', a['answer']):
            if tok in rid2pid:
                tot += 1
                if a['id'] in dr.get(rid2pid[tok], []): selfc += 1; runs_self.add((b, a['id']))
print(f"(ii) ours cites to base deep_read records={tot}; on papers deep-read earlier by the SAME question={selfc} ({100*selfc/max(1,tot):.0f}%), in {len(runs_self)} runs")
# (iii) years
J = {}
yrs = collections.Counter()
for b in B:
    for it in json.load(open(os.path.join(CS2, f'judge_input_ours_batch{b}.json'), encoding='utf-8')):
        for s in it['sections']:
            for c in s.get('citations') or []:
                y = (c.get('metadata') or {}).get('year')
                yrs['none' if not y else ('>=2025' if int(y) >= 2025 else '<2025')] += 1
                if y and int(y) >= 2026: yrs['>=2026'] += 1
print('(iii) ours citation years:', dict(yrs))
