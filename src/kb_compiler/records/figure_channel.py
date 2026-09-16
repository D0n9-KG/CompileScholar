# -*- coding: utf-8 -*-
"""Figure channel (2a 识图) — read paper figures/charts via VLM (GLM-4.6V) into
typed KB records, analogous to table_channel for tables.

Why: PaperScope's Reasoning track (figure-table-chart comparison) and AirQA image
questions need figure CONTENT, but the text-only KB has only captions. mineru
already extracts figure images (mineru_or/{pid}/images/*.jpg) + maps them in
content_list.json (type=chart, img_path, chart_caption). The VLM channel
(GLM-4.6V via Paratera) is verified (precheck_2026-09-10/vlm_probe.py: synthetic
ground-truth bars read correctly, real figures described). This module turns each
chart into a structured reading -> records.

Discipline:
- VLM only READS; records are provenance-marked "figure_vlm" (lower trust than
  text-extracted; downstream can weight/filter).
- Fabrication guard: prompt instructs "if you cannot read a value, omit it / say
  so"; values are VLM-reported, never invented by this code. canary (synthetic
  figure with known values) validates accuracy before corpus runs.
- caption is the verbatim quote anchor (provenance), like table_channel.
- Zero gold / question contact (build-side, gold-blind by construction).

Usage:
  python -m kb_compiler.records.figure_channel --mineru MINERU_DIR \
      --pids p1,p2 --out records_figure.json [--model GLM-4.6V] [--limit N] [--dry]
"""
import base64, json, os, re, sys, time, argparse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from kb_infra.llm import ENV, _walled_open, _walled_read, parse_json_response

VLM_MODEL = "GLM-4V-Flash"   # NOT GLM-4.6V: 4.6V is a thinking model whose
# reasoning_tokens eat the budget -> empty content -> all values missed (probe
# grade all-false). GLM-4V-Flash is non-thinking, 1.5s, reads labeled values +
# estimates unlabeled bars correctly (probe grade all-true). See
# precheck_2026-09-10/vlm_probe_report.md.

READ_PROMPT = """You are reading ONE figure from a scientific paper for a knowledge base.

Caption: {caption}

Extract the figure's content faithfully. Reply with ONLY a JSON object:
{{"figure_type": "<bar/line/scatter/heatmap/diagram/architecture/other>",
  "x_axis": "<x-axis label or empty>",
  "y_axis": "<y-axis label or empty>",
  "points": [{{"label": "<series/bar/category name>", "value": <number or null>, "unit": "<or empty>"}}],
  "comparison": "<one sentence: what is being compared and the key difference, or empty>",
  "entities": ["<method/model/dataset names appearing in the figure>"],
  "main_claim": "<one sentence: the takeaway the figure supports, or empty>"}}

Rules: report ONLY values you can actually read. If a value is not printed and
cannot be estimated from the axis, set it to null — NEVER invent numbers. If you
cannot read the figure, return {{"figure_type":"unreadable"}}. Keep labels verbatim."""


def vlm_call(text, img_path, model=VLM_MODEL, max_tokens=700):
    """Verified VLM call (vlm_probe.py pattern): OpenAI-compat multimodal,
    base64 data URI, temp 0. Returns (content, usage) or raises."""
    base = ENV.get("PARATERA_BASE_URL", "").rstrip("/")
    key = ENV.get("PARATERA_API_KEY", "")
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = "png" if img_path.lower().endswith(".png") else "jpeg"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
        ]}],
        "temperature": 0.0, "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    r = _walled_open(req, sock_timeout=60, wall_s=180)
    raw = _walled_read(r, t0, wall_s=180)
    resp = json.loads(raw)
    return resp["choices"][0]["message"]["content"], (resp.get("usage") or {})


def paper_figures(mineru_dir, pid):
    """Yield (img_path, caption, page_idx) for each chart figure of a paper."""
    cl = os.path.join(mineru_dir, pid, "content_list.json")
    if not os.path.exists(cl):
        return
    entries = json.load(open(cl, encoding="utf-8"))
    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("type") != "chart":       # tables handled by table_channel (HTML)
            continue
        ip = e.get("img_path") or ""
        full = os.path.join(mineru_dir, pid, ip)
        if not ip or not os.path.exists(full):
            continue
        cap = ""
        cc = e.get("chart_caption") or e.get("caption") or []
        if isinstance(cc, list) and cc:
            cap = " ".join(str(x) for x in cc)
        elif isinstance(cc, str):
            cap = cc
        yield full, cap.strip(), e.get("page_idx")


def _rid(pid, fp):
    import hashlib
    return hashlib.md5(f"{pid}|figure|{fp}".encode("utf-8")).hexdigest()[:14]


def reading_to_records(pid, reading, caption, img_name, page_idx):
    """Structured VLM reading -> KB records (provenance figure_vlm)."""
    recs = []
    if not isinstance(reading, dict) or reading.get("figure_type") == "unreadable":
        return recs
    ftype = reading.get("figure_type", "")
    entities = [str(x).strip() for x in (reading.get("entities") or []) if str(x).strip()][:8]
    base = {
        "paper_id": pid, "chunk_id": f"{pid}#figure", "section": f"(figure {ftype})",
        "epistemic": "demonstrated", "provenance": "figure_vlm",
        "quote": (caption or f"figure image {img_name}")[:400],
        "figure_meta": {"type": ftype, "image": img_name, "page_idx": page_idx,
                        "x_axis": reading.get("x_axis", ""), "y_axis": reading.get("y_axis", "")},
    }
    # main claim -> finding record
    mc = (reading.get("main_claim") or reading.get("comparison") or "").strip()
    if mc:
        r = dict(base); r["id"] = _rid(pid, f"claim|{mc[:60]}")
        r.update({"kind": "finding", "claim": mc[:300], "claim_type": "observation",
                  "strength": "demonstrated", "target_ref": "",
                  "method_ref": {"surface": entities[0] if entities else None,
                                 "canonical": None, "entity_id": None}})
        recs.append(r)
    # quantitative points -> result records (only with a read value)
    for p in (reading.get("points") or []):
        if not isinstance(p, dict):
            continue
        val = p.get("value")
        lab = str(p.get("label", "")).strip()
        if val is None or not lab:
            continue   # never invent; skip unreadable points
        r = dict(base); r["id"] = _rid(pid, f"pt|{lab}|{val}")
        r.update({"kind": "result",
                  "measure": {"metric": (reading.get("y_axis") or ftype or "value")[:60],
                              "value": str(val), "unit": str(p.get("unit", "")),
                              "direction": "", "aggregation": "", "timepoint": ""},
                  "method_ref": {"surface": lab[:60], "canonical": None, "entity_id": None},
                  "role": "figure_value", "dims": {}})
        recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mineru", required=True)
    ap.add_argument("--pids", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=VLM_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="cap figures (0=all)")
    ap.add_argument("--dry", action="store_true", help="list figures, no VLM calls")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pids = [p for p in args.pids.split(",") if p] or sorted(
        d for d in os.listdir(args.mineru) if os.path.isdir(os.path.join(args.mineru, d)))
    od = os.path.dirname(args.out)
    if od:
        os.makedirs(od, exist_ok=True)
    out = {}
    if os.path.exists(args.out):
        out = json.load(open(args.out, encoding="utf-8"))
    nfig = nrec = nfail = 0
    for pid in pids:
        figs = list(paper_figures(args.mineru, pid))
        if args.dry:
            print(f"[dry] {pid}: {len(figs)} chart figures")
            nfig += len(figs)
            continue
        recs = out.setdefault(pid, [])
        have = {r.get("figure_meta", {}).get("image") for r in recs if isinstance(r, dict)}
        for img, cap, pg in figs:
            if args.limit and nfig >= args.limit:
                break
            img_name = os.path.basename(img)
            if img_name in have:
                continue
            nfig += 1
            try:
                content, usage = vlm_call(READ_PROMPT.format(caption=cap[:400] or "(none)"),
                                          img, model=args.model)
                reading = parse_json_response(content)
                if isinstance(reading, list):
                    reading = next((x for x in reading if isinstance(x, dict)), None)
                rr = reading_to_records(pid, reading, cap, img_name, pg)
                recs.extend(rr)
                nrec += len(rr)
                ftype = (reading or {}).get("figure_type") if isinstance(reading, dict) else None
                print(f"  [{pid}] {img_name[:16]} type={ftype} +{len(rr)} recs", flush=True)
                if not rr:   # debug: why 0 records? dump raw VLM content head
                    print(f"      RAW[{type(reading).__name__}]: {str(content)[:300]}", flush=True)
            except Exception as e:
                nfail += 1
                print(f"  [{pid}] {img_name[:16]} VLM FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)
        out[pid] = recs
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"DONE figures={nfig} records={nrec} fails={nfail} -> {args.out}")


if __name__ == "__main__":
    main()
