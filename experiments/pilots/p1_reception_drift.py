# -*- coding: utf-8 -*-
"""Pilot P1 (zero LLM): does the way later papers relate to a cited paper change with the citing year?

Data: Intern-Atlas paper_evolution_edges (HF OpenRaiser/Intern-Atlas, MIT), one row per (citing A, cited B) with
evolution_relation in {extends, improves, replaces, adapts, combines, uses_component, compares}.
For every cited paper B with enough citations, each edge gets a relative age = citing year - B's year.
Question: within the same cited paper, does the share of "lineage" relations (extends/improves/replaces/adapts/combines)
fall and "uses_component" rise as the paper ages, beyond the drift of the whole corpus by citing year?

Controls:
- baseline = share of each relation among all edges in the same citing year (corpus drift);
- per-paper residual = observed indicator - baseline share for that citing year, averaged per age bucket;
- paired within-paper contrast early (age 0-2) vs late (age >= 5), bootstrap over cited papers.
Writes runs/pilot-p1-reception-drift-20261005/{summary.json,trajectories.tsv,examples.tsv}."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "external" / "intern_atlas"
OUT = REPO / "runs" / "pilot-p1-reception-drift-20261005"
COLS = ["paper_a_id", "paper_a_year", "paper_a_venue_canonical", "paper_b_id", "paper_b_title", "paper_b_year",
        "evolution_relation", "paper_b_node_type"]
LINEAGE = {"extends", "improves", "replaces", "adapts", "combines"}
MIN_CITES, MIN_EARLY, MIN_LATE = 30, 5, 5


def load() -> pd.DataFrame:
    parts = [pq.read_table(f, columns=COLS).to_pandas() for f in sorted(SRC.glob("edges-*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    df = df[df["paper_a_year"].between(1990, 2025) & df["paper_b_year"].between(1950, 2025)]
    df["age"] = df["paper_a_year"] - df["paper_b_year"]
    df = df[df["age"] >= 0]
    df["lineage"] = df["evolution_relation"].isin(LINEAGE).astype(float)
    df["component"] = (df["evolution_relation"] == "uses_component").astype(float)
    df["compare"] = (df["evolution_relation"] == "compares").astype(float)
    return df


def boot(x: np.ndarray, B: int = 4000, seed: int = 0) -> list[float]:
    r = np.random.default_rng(seed)
    m = r.choice(x, (B, len(x))).mean(1)
    return [float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def contrast(df: pd.DataFrame, strata: list[str]) -> dict:
    """Within-paper late-minus-early contrast, residualised against the relation shares of the given strata."""
    df = df.copy()
    base = df.groupby(strata)[["lineage", "component", "compare"]].transform("mean")
    for k in ("lineage", "component", "compare"):
        df[k + "_res"] = df[k] - base[k]
    cnt = df.groupby("paper_b_id").size()
    d = df[df["paper_b_id"].isin(cnt[cnt >= MIN_CITES].index)]
    early, late = d[d["age"] <= 2].groupby("paper_b_id"), d[d["age"] >= 5].groupby("paper_b_id")
    e_n, l_n = early.size(), late.size()
    both = e_n[e_n >= MIN_EARLY].index.intersection(l_n[l_n >= MIN_LATE].index)
    out = {"papers": int(len(both))}
    for k in ("lineage", "component", "compare"):
        out[k] = boot((late[k + "_res"].mean().loc[both] - early[k + "_res"].mean().loc[both]).to_numpy())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    files = sorted(p.name for p in SRC.glob("edges-*.parquet"))
    robustness = {
        "baseline_citing_year": contrast(df, ["paper_a_year"]),
        "baseline_citing_year_x_venue": contrast(df, ["paper_a_year", "paper_a_venue_canonical"]),
        "in_corpus_cited_only": contrast(df[~df["paper_b_id"].str.startswith("stub:")], ["paper_a_year"]),
    }
    # corpus drift by citing year
    base = df.groupby("paper_a_year")[["lineage", "component", "compare"]].mean()
    for k in ("lineage", "component", "compare"):
        df[k + "_res"] = df[k] - df["paper_a_year"].map(base[k])
    cnt = df.groupby("paper_b_id").size()
    keep = cnt[cnt >= MIN_CITES].index
    d = df[df["paper_b_id"].isin(keep)].copy()
    d["bucket"] = pd.cut(d["age"], [-1, 2, 4, 9, 200], labels=["0-2", "3-4", "5-9", "10+"])
    traj = d.groupby("bucket", observed=True)[["lineage", "lineage_res", "component", "component_res",
                                               "compare", "compare_res"]].mean()
    traj["n_edges"] = d.groupby("bucket", observed=True).size()
    # within-paper early vs late contrast
    early = d[d["age"] <= 2].groupby("paper_b_id")
    late = d[d["age"] >= 5].groupby("paper_b_id")
    e_n, l_n = early.size(), late.size()
    both = e_n[(e_n >= MIN_EARLY)].index.intersection(l_n[l_n >= MIN_LATE].index)
    res = {}
    for k in ("lineage", "component", "compare"):
        diff = (late[k + "_res"].mean().loc[both] - early[k + "_res"].mean().loc[both]).to_numpy()
        raw = (late[k].mean().loc[both] - early[k].mean().loc[both]).to_numpy()
        res[k] = {"late_minus_early_residual": boot(diff), "late_minus_early_raw": boot(raw),
                  "share_papers_decrease": float((diff < 0).mean())}
    # per-paper examples: largest lineage -> component shift
    e_rel = early["evolution_relation"].agg(lambda s: s.value_counts(normalize=True).round(2).to_dict())
    l_rel = late["evolution_relation"].agg(lambda s: s.value_counts(normalize=True).round(2).to_dict())
    shift = (late["lineage"].mean().loc[both] - early["lineage"].mean().loc[both]).sort_values()
    titles = d.drop_duplicates("paper_b_id").set_index("paper_b_id")[["paper_b_title", "paper_b_year"]]
    ex = pd.DataFrame({"title": titles.loc[shift.index, "paper_b_title"], "year": titles.loc[shift.index, "paper_b_year"],
                       "lineage_shift": shift, "n_early": e_n.loc[shift.index], "n_late": l_n.loc[shift.index],
                       "early": e_rel.loc[shift.index].astype(str), "late": l_rel.loc[shift.index].astype(str)})
    pd.concat([ex.head(15), ex.tail(5)]).to_csv(OUT / "examples.tsv", sep="\t")
    traj.to_csv(OUT / "trajectories.tsv", sep="\t")
    summary = {"source": "HF OpenRaiser/Intern-Atlas paper_evolution_edges (MIT)", "shards": files,
               "edges_used": int(len(df)), "cited_papers_ge_min": int(len(keep)), "min_cites": MIN_CITES,
               "papers_with_early_and_late": int(len(both)),
               "relation_shares_overall": df["evolution_relation"].value_counts(normalize=True).round(4).to_dict(),
               "within_paper_contrast": res, "robustness": robustness,
               "note": "residual = edge indicator minus share of that relation among all edges of the same citing year"}
    json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(traj.round(3).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
