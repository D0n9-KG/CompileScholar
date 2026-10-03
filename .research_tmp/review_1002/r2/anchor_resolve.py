import json,re,sys
sys.path.insert(0,r'C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\cs2')
sys.path.insert(0,r'C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\_shared\tools')
sys.path.insert(0,r'C:\Users\D0n9\Desktop\CompileScholar\src')
import adapt_batches as A
from report_adapter import EvidenceStore, parse_notes
recs=A._load_records_with_deep()
man=json.load(open(A.BASE/'manifest_all.json',encoding='utf-8'))
snap=json.load(open(A.ARM_OURS/'kb_snapshot_writes'/'manifest_all.json',encoding='utf-8'))
have={r.get('paper_id') for r in man}; man=man+[r for r in snap if r.get('paper_id') not in have]
st=EvidenceStore(recs,man,texts_dir=str(A.BASE/'deep_read_texts'),views_path=str(A.BASE/'views_cs2.json'))
tot={'anchor':0,'anchor_none':0,'all':0,'none':0,'title':0,'abstract':0,'paper_best':0}
for b in ['34a','34b','34c','34d','34e']:
    rows=json.load(open(A.ARM_OURS/f'answers_pilot_cs2batch{b}.json',encoding='utf-8'))
    for r in rows:
        for c in parse_notes(r.get('notes_final') or ''):
            for ref in c['refs']:
                tot['all']+=1
                e=st.resolve(ref)
                if '#' in ref:
                    tot['anchor']+=1
                    if e is None: tot['anchor_none']+=1
                if e is None: tot['none']+=1
                elif e['tier']=='title': tot['title']+=1
                elif (e.get('loc') or {}).get('section')=='abstract': tot['abstract']+=1
                elif (e.get('loc') or {}).get('section')=='paper-level ref': tot['paper_best']+=1
print(tot)
import hashlib,os,collections
cause=collections.Counter(); ex=collections.defaultdict(list)
st2=EvidenceStore(recs,man,texts_dir=str(A.ARM_OURS/'kb_snapshot_writes'/'deep_read_texts'),views_path=str(A.BASE/'views_cs2.json'))
rec2=0
for b in ['34a','34b','34c','34d','34e']:
    rows=json.load(open(A.ARM_OURS/f'answers_pilot_cs2batch{b}.json',encoding='utf-8'))
    for r in rows:
        for c in parse_notes(r.get('notes_final') or ''):
            for ref in c['refs']:
                if '#' not in ref or st.resolve(ref) is not None: continue
                pid,_,cs=ref.partition('#')
                h=hashlib.md5(pid.encode()).hexdigest()+'.txt'
                if pid not in st.by_paper: k='pid_not_in_manifest'
                elif not cs.isdigit(): k='bad_offset'
                elif os.path.exists(A.ARM_OURS/'kb_snapshot_writes'/'deep_read_texts'/h): k='text_only_in_SNAP'
                elif os.path.exists(A.BASE/'survey_texts'/(pid+'.md')) or os.path.exists(A.BASE/'hub_texts'/(pid+'.md')): k='survey_hub_exists?'
                else: k='text_nowhere'
                cause[k]+=1; ex[k].append(ref[:70])
                if st2.resolve(ref) is not None: rec2+=1
print(cause); print('recovered with SNAP texts_dir:',rec2)
for k,v in ex.items(): print(k, v[:3])
print('--- claims fully unresolvable (all refs None) BASE vs SNAP texts_dir')
for b in ['34a','34b','34c','34d','34e']:
    rows=json.load(open(A.ARM_OURS/f'answers_pilot_cs2batch{b}.json',encoding='utf-8'))
    n=n1=n2=0
    for r in rows:
        for c in parse_notes(r.get('notes_final') or ''):
            if not c['refs'] or c['invalidated'] or not c['text']: continue
            n+=1
            if all(st.resolve(x) is None for x in c['refs']): n1+=1
            if all(st2.resolve(x) is None for x in c['refs']): n2+=1
    print(b,'usable',n,'dead_BASE',n1,'dead_SNAP',n2)
print('--- Fix A abstract-tier support heuristic')
import random
random.seed(3)
stop=set('the a an of and or to in for on with by is are was were that this from as at be it its which using based their'.split())
def toks(s): return {t for t in re.findall(r'[a-z0-9]+',(s or '').lower()) if t not in stop and len(t)>2}
rows_all=[]
for b in ['34a','34b','34c','34d','34e']:
    for r in json.load(open(A.ARM_OURS/f'answers_pilot_cs2batch{b}.json',encoding='utf-8')):
        for c in parse_notes(r.get('notes_final') or ''):
            for ref in c['refs']:
                e=st.resolve(ref)
                if e and (e.get('loc') or {}).get('section')=='abstract':
                    ct=toks(c['text']); at=toks(e['quote'])
                    rows_all.append((len(ct&at)/max(1,len(ct)),c['text'][:150],e['quote'][:150]))
ov=[x[0] for x in rows_all]
print('n',len(ov),'claim-token coverage by abstract: <0.3:',sum(o<0.3 for o in ov),'0.3-0.6:',sum(0.3<=o<0.6 for o in ov),'>=0.6:',sum(o>=0.6 for o in ov))
for x in sorted(rows_all)[:4]: print(round(x[0],2),'| CLAIM:',x[1],'| ABS:',x[2])
