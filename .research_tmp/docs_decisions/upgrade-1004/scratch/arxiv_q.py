# -*- coding: utf-8 -*-
"""arXiv export API helper (polite: >=3.5s between calls). Usage: python arxiv_q.py ids <id,id,...> | python arxiv_q.py q "<search_query>" [max]"""
import re, sys, time, urllib.parse, urllib.request, html
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def get(params):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    for a in range(3):
        try:
            return op.open(urllib.request.Request(url, headers={"User-Agent": "compilescholar-design-check/0.1"}), timeout=60).read().decode("utf-8", "replace")
        except Exception as e:
            time.sleep(6)
    return ""
def show(t):
    for e in re.findall(r"<entry>(.*?)</entry>", t, re.S):
        i = re.search(r"<id>https?://arxiv.org/abs/([^<]+)</id>", e); ti = re.search(r"<title>(.*?)</title>", e, re.S)
        pu = re.search(r"<published>(\d{4}-\d{2}-\d{2})", e)
        print(f"{i.group(1) if i else '?'} | {pu.group(1) if pu else '?'} | {html.unescape(re.sub(r'\s+',' ',ti.group(1))).strip()[:120] if ti else '?'}")
if sys.argv[1] == "ids":
    show(get({"id_list": sys.argv[2], "max_results": 100}))
else:
    for q in sys.argv[2:]:
        mx = 12
        print("## Q:", q); show(get({"search_query": q, "max_results": mx, "sortBy": "relevance"})); time.sleep(3.6)
