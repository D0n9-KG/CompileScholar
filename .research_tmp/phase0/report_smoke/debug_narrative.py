# -*- coding: utf-8 -*-
"""Debug the narrative-compile call directly."""
import sys, json, os
sys.path.insert(0, r'C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\_shared\tools')
sys.path.insert(0, r'C:\Users\D0n9\Desktop\CompileScholar\src')
os.environ['LLM_CALL_LOG'] = r'C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\phase0\report_smoke\llm_debug.jsonl'
os.environ.setdefault('LOCAL_SOCK_TIMEOUT', '900')

from report_adapter import parse_notes, EvidenceStore, NARRATIVE_PROMPT
from kb_infra.llm import call_local

BASE = r'C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\scholarqa_multi'
pilot = json.load(open(BASE + r'\baselines\ours\answers_pilot_multi.json', encoding='utf-8'))
row = next(r for r in pilot if r['id'] == 'norman_bio_1')
records = json.load(open(BASE + r'\kb\postcheck\records_checked.json', encoding='utf-8'))
manifest = json.load(open(BASE + r'\corpus\manifest.json', encoding='utf-8'))
store = EvidenceStore(records, manifest)
claims = parse_notes(row['notes_final'])
usable = [c for c in claims if c['text'] and c['refs'] and not c['invalidated']]
lines = []
for i, c in enumerate(usable, 1):
    evs = []
    for ref in c['refs'][:3]:
        e = store.resolve(ref)
        if e and e.get('quote'):
            evs.append('"' + e['quote'][:220] + '"')
    ev = ('  EVIDENCE: ' + ' | '.join(evs)) if evs else ''
    lines.append(f'[C{i}] {c["text"]}{ev}')
prompt = NARRATIVE_PROMPT.format(
    k=len(usable),
    section_spec='thematic sections with short titles of your choosing',
    word_budget='300-700', question=row['question'],
    claims_block='\n'.join(lines))
print('prompt chars:', len(prompt), '| usable claims:', len(usable))
r = call_local(prompt, max_tokens=4000, temperature=0.3, enable_thinking=False)
print('result None?', r is None)
if r:
    print('--- head ---')
    print(r[:600])
