"""Minimal cached Semantic Scholar client (unauthenticated): disk cache, polite pacing, 429 backoff.
Only paper titles / IDs / query strings are sent."""
import hashlib, json, os, random, time, urllib.parse, urllib.request, urllib.error

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)
BASE = "https://api.semanticscholar.org"
STATS = {"net": 0, "ok": 0, "429": 0, "err": 0, "cache": 0}
_last = [0.0]
MIN_GAP = 1.1  # seconds between network requests


def _key(method, url, body):
    return hashlib.sha1(f"{method} {url} {json.dumps(body, sort_keys=True) if body else ''}".encode()).hexdigest()


def call(path, params=None, body=None, max_tries=14):
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    method = "POST" if body is not None else "GET"
    fn = os.path.join(CACHE, "s2_" + _key(method, url, body) + ".json")
    if os.path.exists(fn):
        STATS["cache"] += 1
        return json.load(open(fn, encoding="utf-8"))
    delay = 3.0
    for t in range(max_tries):
        wait = MIN_GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        STATS["net"] += 1
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json", "User-Agent": "p8-pilot"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read().decode("utf-8"))
            STATS["ok"] += 1
            json.dump(out, open(fn, "w", encoding="utf-8"), ensure_ascii=False)
            return out
        except urllib.error.HTTPError as e:
            if e.code == 404:
                STATS["err"] += 1
                out = {"__error__": 404}
                json.dump(out, open(fn, "w", encoding="utf-8"))
                return out
            if e.code == 400:
                STATS["err"] += 1
                return {"__error__": 400, "msg": e.read().decode("utf-8", "ignore")[:300]}
            STATS["429" if e.code == 429 else "err"] += 1
        except Exception:
            STATS["err"] += 1
        time.sleep(delay + random.random() * 2)
        delay = min(delay * 1.6, 60)
    return {"__error__": "gave_up"}
