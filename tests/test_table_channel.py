# -*- coding: utf-8 -*-
"""Unit tests for kb_compiler.records.table_channel (F24) — covers the three
AirQA-tablecheck fixes (2026-09-13):
  A: cell-level inline citation -> epistemic=cited
  B: full-width group-banner rows (header zone + mid-table) tracked as group
     context; single-token banners are NOT groups (conservative)
  C: single-value rows quotable via value+row-head dual probe (the old
     max(2, 60%) floor silently dropped every single-value row)
plus the preserved PS16 regression guarantees (rowspan mis-quote, fused
cells -> residue).
"""
import sys

sys.path.insert(0, r"C:\Users\D0n9\Desktop\LogicKG\src")
from kb_compiler.records.table_channel import (  # noqa: E402
    extract_tables, _group_row_text, _row_quote)


def _recs(html, pid="t1", heading="## Experiments"):
    text = f"{heading}\n{html}\n"
    recs, res = extract_tables(pid, text)
    return recs, res


# ---------------- fix C: single-value rows ----------------

def test_c_single_value_rows_emitted():
    # e653 pattern: rowhead text col, single-digit col (filtered from quote
    # probes), non-numeric col, one clean numeric col -> 1 filtered val/row.
    # Uses U+00D7 as in the real AirQA cache (ASCII 'x' would parse as a unit
    # suffix — that form fails as honest skip, recorded as an observation).
    html = ("<table>"
            "<tr><td>Image resolution</td><td>blocks</td><td>Field</td><td>params</td></tr>"
            "<tr><td>20×20</td><td>1</td><td>18×18</td><td>360k</td></tr>"
            "<tr><td>40×40</td><td>2</td><td>44×44</td><td>1.8m</td></tr>"
            "<tr><td>80×80</td><td>3</td><td>92×92</td><td>7.6m</td></tr>"
            "</table>")
    recs, _ = _recs(html)
    params = {r["method_ref"]["surface"]: r["measure"]["value"]
              for r in recs if r["measure"]["metric"] == "params"}
    assert params.get("20×20") == "360", params
    assert params.get("40×40") == "1.8", params
    assert params.get("80×80") == "7.6", params
    # quotes must be the row's own segment
    r20 = [r for r in recs if r["method_ref"]["surface"] == "20×20"
           and r["measure"]["metric"] == "params"][0]
    assert "360k" in r20["quote"] and "1.8m" not in r20["quote"]


def test_c_single_blocked_in_fused_damaged_table():
    # lXuC table #6 lesson: header lost its corner cell AND col2 is fused
    # ('2.32 0/22') — column identities are shifted; a lone value cannot
    # self-verify. Whole-table honest skip for the recovery path.
    html = ("<table>"
            "<tr><td>Worst-case log-ppl</td><td>Avg log-ppl</td><td># beating</td></tr>"
            "<tr><td>Baseline (280M)</td><td>2.39</td><td>2.32 0/22</td></tr>"
            "<tr><td>DoReMi (280M)</td><td>2.19</td><td>2.13 22/22</td></tr>"
            "</table>")
    recs, _ = _recs(html)
    assert not any(r["measure"]["value"] == "2.39" for r in recs), recs


def test_c_headless_single_value_still_skipped():
    # rowspan-continuation ambiguity: no row-head probe -> never guess.
    q = _row_quote("<tr><td></td><td>47.4</td></tr>", "", ["47.4"])
    assert q is None


def test_c_same_value_different_heads_disambiguated():
    # LLaMA-4bit lesson preserved under the single-value branch: two rows,
    # same lone value 47.4, different heads -> each quotes its own segment.
    raw = ("<tr><td>Original</td><td>47.4</td></tr>"
           "<tr><td>LLaMA-13B-4bit</td><td>47.4</td></tr>")
    q1 = _row_quote(raw, "Original", ["47.4"])
    q2 = _row_quote(raw, "LLaMA-13B-4bit", ["47.4"])
    assert q1 and "Original" in q1 and "4bit" not in q1
    assert q2 and "4bit" in q2


# ---------------- fix B: group banner rows ----------------

def test_b_midtable_banner_updates_group():
    # BooookScore pattern: leading banner in header zone + mid-table banner.
    html = ("<table>"
            "<tr><td>Model</td><td>Chunk size</td><td>Score</td></tr>"
            '<tr><td colspan="3">Summaries via hierarchical merging</td></tr>'
            "<tr><td>GPT-4</td><td>2048</td><td>89.1</td></tr>"
            '<tr><td colspan="3">Summaries via incremental updating</td></tr>'
            "<tr><td>GPT-4</td><td>2048</td><td>82.5</td></tr>"
            "</table>")
    recs, _ = _recs(html)
    m = {r["measure"]["value"]: r["measure"]["metric"] for r in recs}
    assert m["89.1"] == "Score > Summaries via hierarchical merging", m
    assert m["82.5"] == "Score > Summaries via incremental updating", m
    # both GPT-4 records coexist (distinct fingerprints via metric)
    assert len([r for r in recs if r["method_ref"]["surface"] == "GPT-4"]) >= 4


def test_b_leading_banner_is_group_when_sibling_midtable():
    # s7xWeJ stats-table lesson: leading banner + mid-table sibling banner =
    # section groups (not a caption). SST-2 rows belong to 'Single Sentence
    # Tasks', MNLI rows to 'Sentence Pair Tasks' — no mixed paths.
    html = ("<table>"
            '<tr><td colspan="3">Single Sentence Tasks</td></tr>'
            "<tr><td>Dataset</td><td>|D|</td><td>#Train</td></tr>"
            "<tr><td>SST-2</td><td>2</td><td>67,349</td></tr>"
            '<tr><td colspan="3">Sentence Pair Tasks</td></tr>'
            "<tr><td>MNLI</td><td>3</td><td>392,702</td></tr>"
            "</table>")
    recs, _ = _recs(html)
    m = {(r["method_ref"]["surface"], r["measure"]["value"]): r["measure"]["metric"]
         for r in recs}
    assert m[("SST-2", "67,349")] == "#Train > Single Sentence Tasks", m
    assert m[("MNLI", "392,702")] == "#Train > Sentence Pair Tasks", m
    assert m[("MNLI", "3")] == "|D| > Sentence Pair Tasks", m


def test_b_caption_without_siblings_stays_tier():
    # FhQS shape: caption ABOVE column names, no mid-table banners -> stays
    # a colpath tier (v1 behavior), never a group.
    html = ("<table>"
            '<tr><td colspan="3">F1 score for skeleton variables</td></tr>'
            "<tr><td>Setting</td><td>Ours</td><td>Baseline</td></tr>"
            "<tr><td>Latent+tree</td><td>0.84</td><td>0.36</td></tr>"
            "</table>")
    recs, _ = _recs(html)
    m = {r["measure"]["value"]: r["measure"]["metric"] for r in recs}
    assert m["0.84"] == "F1 score for skeleton variables > Ours", m
    assert m["0.36"] == "F1 score for skeleton variables > Baseline", m


def test_b_banner_with_empty_companion_cell():
    # 22a670fd pattern: <td colspan=10>1-2B Base Models</td><td colspan=7></td>
    row = [("1-2B Base Models", 1, 10), ("", 1, 7)]
    assert _group_row_text(row) == "1-2B Base Models"


def test_b_single_token_banner_not_group():
    # 'ReduceLROnPlateau' full-width cell: config value, not a phrase -> None
    assert _group_row_text([("ReduceLROnPlateau", 1, 4)]) is None
    # numeric-ish banner -> None
    assert _group_row_text([("0.5", 1, 4)]) is None
    # short text -> None
    assert _group_row_text([("Ours", 1, 4)]) is None
    # mixed distinct texts -> None
    assert _group_row_text([("Group A", 1, 2), ("Group B", 1, 2)]) is None
    # narrow span -> None
    assert _group_row_text([("Some phrase here", 1, 2)]) is None


def test_b_banner_only_header_keeps_v1_recall():
    # fragment-table shape (banner but no column-name row — mw1P): keep the
    # v1 tier fallback, values emitted with the banner as the metric.
    html = ("<table>"
            '<tr><td colspan="4">Only a banner phrase here</td></tr>'
            "<tr><td>AlphaNet</td><td>1.5</td><td>2.5</td><td>3.5</td></tr>"
            "</table>")
    recs, _ = _recs(html)
    assert len(recs) == 3, recs
    assert all(r["measure"]["metric"] == "Only a banner phrase here" for r in recs)
    assert {r["measure"]["value"] for r in recs} == {"1.5", "2.5", "3.5"}


# ---------------- fix A: cell-level citation -> cited ----------------

def test_a_cite_cell_epistemic_cited():
    html = ("<table>"
            "<tr><td>Dataset</td><td>Ours</td><td>SOTA</td></tr>"
            "<tr><td>Cornell</td><td>11.56</td><td>12.11 (He et al., 2021)</td></tr>"
            "<tr><td>Daily</td><td>45.90</td><td>42.84 (Chen et al.,2022)</td></tr>"
            "</table>")
    recs, _ = _recs(html)
    sota = [r for r in recs if r["measure"]["metric"] == "SOTA"]
    ours = [r for r in recs if r["measure"]["metric"] == "Ours"]
    assert len(sota) == 2 and all(r["epistemic"] == "cited" for r in sota), sota
    assert {r["measure"]["value"] for r in sota} == {"12.11", "42.84"}
    assert all(r["epistemic"] == "demonstrated" for r in ours)


def test_a_section_cited_default_preserved():
    html = ("<table>"
            "<tr><td>Method</td><td>Acc</td><td>F1</td></tr>"
            "<tr><td>OldNet</td><td>71.3</td><td>68.9</td></tr>"
            "</table>")
    recs, _ = _recs(html, heading="## Related Work")
    assert recs and all(r["epistemic"] == "cited" for r in recs)


# ---------------- preserved PS16 guarantees ----------------

def test_fused_cells_go_to_residue_not_records():
    html = ("<table>"
            "<tr><td>Model</td><td>MMLU</td><td>GSM8K</td></tr>"
            "<tr><td>GoodRow</td><td>88.2</td><td>5.11</td></tr>"
            "<tr><td>BadRow</td><td>91.691.4</td><td>5.31</td></tr>"
            "</table>")
    recs, res = _recs(html)
    assert any(r["method_ref"]["surface"] == "GoodRow" for r in recs)
    assert not any(r["method_ref"]["surface"] == "BadRow" for r in recs)
    assert any("BadRow" in " ".join(x.get("rows", [])) for x in res)


def test_c2_two_col_label_value_emitted():
    # fix C2: [label, value] tables (hyperparameter lists — universal shape)
    # were zeroed by the >=2-value-columns clamp.
    html = ("<table>"
            "<tr><td>Hyperparameter</td><td>Value</td></tr>"
            "<tr><td>learning rate</td><td>0.0003</td></tr>"
            "<tr><td>batch size</td><td>256</td></tr>"
            "</table>")
    recs, _ = _recs(html)
    got = {r["method_ref"]["surface"]: r["measure"]["value"] for r in recs}
    assert got == {"learning rate": "0.0003", "batch size": "256"}, got


def test_c2_three_col_fused_damage_blocks_recovery():
    # 04ed3e06 shape: 'GPT-4 | 4096 580.3 | 566' — official parse fused
    # Chunk+AvgLen into col1. Any fused signature in the value zone turns
    # off single-value recovery for the WHOLE table (lXuC #6 lesson: the
    # parser cannot distinguish benign fusion from header-shifted
    # misalignment, and a lone value cannot self-verify). Honest skip;
    # the fused numbers must never surface as record values either.
    html = ("<table>"
            "<tr><td>Model</td><td>Chunk Size</td><td>Score</td></tr>"
            "<tr><td>GPT-4</td><td>4096 580.3</td><td>566</td></tr>"
            "<tr><td>Claude</td><td>2048 111.2</td><td>570</td></tr>"
            "</table>")
    recs, _ = _recs(html)
    assert recs == [], recs


def test_c3_rejected_dirty_rowhead_still_emitted():
    # C3 (rowhead dirt guard) was implemented and REVERTED: dirty surfaces
    # with correct bindings (pPh9 ED-Pose shape, mineru misalignment
    # signature) are governance debt, not row-drop reasons. The recovery
    # stays bound via the head+value dual probe.
    html = ("<table>"
            "<tr><td>Setting</td><td>Score</td><td>Acc</td></tr>"
            "<tr><td>CleanRow</td><td>1.5</td><td>2.5</td></tr>"
            "<tr><td>TeDaTy |0.669 0.392</td><td>0.474</td><td>0.111</td></tr>"
            "</table>")
    recs, _ = _recs(html)
    surfaces = {r["method_ref"]["surface"] for r in recs}
    assert "CleanRow" in surfaces
    assert any(s.startswith("TeDaTy") for s in surfaces), surfaces


def test_provenance_v2():
    html = ("<table>"
            "<tr><td>Model</td><td>Acc</td><td>F1</td></tr>"
            "<tr><td>Net</td><td>71.3</td><td>70.1</td></tr>"
            "</table>")
    recs, _ = _recs(html)
    assert recs and recs[0]["provenance"] == "table_channel_v2"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
