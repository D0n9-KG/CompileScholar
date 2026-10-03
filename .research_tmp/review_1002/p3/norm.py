"""Step4: 方法/数据集/指标别名归一（规则+小词表）。"""
import re

GENERIC = re.compile(r"^(ours?|proposed|our (model|method)|full( model)?|the proposed( method)?)$", re.I)
M_ALIAS = {"bprmf": "bpr", "bpr": "bpr", "mf": "bpr", "pop": "poprec", "mostpop": "poprec",
           "popularity": "poprec", "poprec": "poprec", "srgnn": "srgnn", "s3rec": "s3rec",
           "s3recmip": "s3rec", "gcsan": "gcsan", "bert4rec": "bert4rec", "sasrec": "sasrec",
           "gru4rec": "gru4rec", "caser": "caser", "fpmc": "fpmc", "narm": "narm", "stamp": "stamp",
           "cl4srec": "cl4srec", "duorec": "duorec", "iclrec": "iclrec", "lightgcn": "lightgcn",
           "tisasrec": "tisasrec", "fmlprec": "fmlprec", "lightsans": "lightsans", "coserec": "coserec",
           "hgn": "hgn", "transrec": "transrec", "fossil": "fossil", "nextitnet": "nextitnet",
           "ngcf": "ngcf", "gcl4sr": "gcl4sr", "unisrec": "unisrec", "recformer": "recformer",
           "mamba4rec": "mamba4rec", "linrec": "linrec", "bsarec": "bsarec", "core": "core"}


def norm_method(s):
    if not s:
        return None
    s0 = re.sub(r"\(.*?\)|\[.*?\]", "", str(s)).strip()
    if GENERIC.match(s0):
        return None
    k = re.sub(r"[^a-z0-9+]", "", s0.lower())
    if not k or len(k) < 2 or str(s).lower().startswith(("w/o", "w/", "wo ", "-")):
        return None
    # 超参/敏感度表的行标签（K=3、d=64、纯数字）不是方法
    if re.fullmatch(r"[a-z]?\d+[a-z]?|[a-z]=?\d+(\.\d+)?", k) or re.search(r"^(k|d|n|l|lambda|alpha)=", s0.lower()):
        return None
    return M_ALIAS.get(k, k)


D_RULES = [("ml1m", r"ml-?1m|movielens-?1m|movielens1m"), ("ml20m", r"ml-?20m|movielens-?20m"),
           ("ml100k", r"ml-?100k"), ("beauty", r"beauty"), ("sports", r"sport"), ("toys", r"toy"),
           ("yelp", r"yelp"), ("steam", r"steam"), ("lastfm", r"last\.?fm"), ("diginetica", r"digi"),
           ("yoochoose", r"yoo"), ("tmall", r"tmall"), ("retailrocket", r"retail"),
           ("games", r"\bgame"), ("electronics", r"electron"), ("clothing", r"cloth"),
           ("home", r"home|tools"), ("cds", r"\bcds?\b|cd ?& ?vinyl"), ("books", r"book"),
           ("kuairec", r"kuai"), ("gowalla", r"gowalla"), ("foursquare", r"foursquare"),
           ("netflix", r"netflix")]


def norm_dataset(s):
    s = str(s or "").lower()
    for k, p in D_RULES:
        if re.search(p, s):
            return k
    k = re.sub(r"[^a-z0-9]", "", s)
    return k or None


def norm_metric(s):
    s = re.sub(r"\s+", "", str(s or "").lower())
    m = re.match(r"(hitratio|hitrate|hit|hr|recall|ndcg|mrr|map|precision|auc|h|r|n|p)@?(\d+)?", s)
    if not m:
        return s or None
    base, k = m.group(1), m.group(2) or ""
    base = {"hit": "hr", "hitratio": "hr", "hitrate": "hr", "h": "hr", "recall": "hr", "r": "hr",
            "n": "ndcg", "p": "precision"}.get(base, base)
    return f"{base}@{k}" if k else base
