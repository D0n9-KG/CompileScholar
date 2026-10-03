import json,sys,random,collections
sys.path.insert(0,r'C:\Users\D0n9\Desktop\CompileScholar\src')
sys.path.insert(0,r'C:\Users\D0n9\Desktop\CompileScholar\.research_tmp\experiments\benchmarks\_shared\tools')
from kb_compiler.records.backflow import build_backflow
from external_tools import _record_mentions
reg=json.load(open('base_kb/registry_v2.json',encoding='utf-8'))
man={r['paper_id']:r for r in json.load(open('base_kb/manifest_all.json',encoding='utf-8'))}
strong_stripped=0; strong_full=[]
for src in ['base_kb/deep_read_records.json','arm_ours/kb_snapshot_writes/deep_read_records.json']:
    d=json.load(open(src,encoding='utf-8'))
    for pid,p in d.items():
        recs=[r for r in p.get('records',[]) if isinstance(r,dict)]
        if not any(r.get('kind')=='lineage' for r in recs): continue
        meta={'title':(man.get(pid) or {}).get('title') or pid,'year':None}
        stripped={'records':[{'id':r.get('id'),'kind':r.get('kind'),'subject':r.get('subject'),'claim':r.get('claim'),'quote':r.get('quote'),'mentions':_record_mentions(r)} for r in recs]}
        full={'records':[dict(r,mentions=_record_mentions(r)) for r in recs]}
        strong_stripped+=sum(1 for e in build_backflow(stripped,meta,reg)['edges'] if e.get('provenance')=='deep')
        for e in build_backflow(full,meta,reg)['edges']:
            if e.get('provenance')=='deep':
                rec=next(r for r in recs if r.get('id')==e['record_id'])
                fr=rec.get('from_method_ref'); to=rec.get('to_method_ref')
                strong_full.append((src.split('/')[0],pid,e['from_name'],e['relation'],e['to_name'],(fr or {}).get('surface') if isinstance(fr,dict) else fr,(to or {}).get('surface') if isinstance(to,dict) else to,e['quote'][:200]))
print('strong edges stripped payload:',strong_stripped,' full payload:',len(strong_full))
print(collections.Counter(x[3] for x in strong_full))
random.seed(7)
for x in random.sample(strong_full,min(20,len(strong_full))):
    print('---'); print('EDGE:',x[2][:50],'--',x[3],'->',x[4][:60]); print('REC from=',x[5],'| to=',x[6]); print('Q:',x[7])
