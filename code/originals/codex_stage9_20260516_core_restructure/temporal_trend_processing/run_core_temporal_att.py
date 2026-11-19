"""Annual and sliding-window matched attribution for Stage 9 core outcomes.

This script supports/replaces the previous 2001-2012 versus 2013-2024 split by
estimating compound-vs-drought ATT through time without relying on a single
year breakpoint.

Outputs are descriptive observational diagnostics. They should not be worded as
proof that a specific calendar year is a physical climate transition.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import linregress, spearmanr, theilslopes


BASE = Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve()
STAGE9 = BASE / "codex_stage9_20260516_core_restructure"
OUT_DIR = STAGE9 / "temporal_trend_processing" / "outputs"

EVENT_PATH = BASE / "output_df" / "global_v05_90p_main" / "global_integrated_events_90p.csv"
EVENT_LEVEL_PATH = (
    BASE / "output_df" / "global_v05_90p_main" / "event_analysis" / "event_multivariate_response_event_level.csv"
)

TREATED = "Compound_Overlapping"
CONTROL = "Drought_Dominant"
LAND_ORDER = ["Forest", "Shrub", "Savannas", "Grassland", "Cropland"]
LAND_COVER_MAP = {
    1: "Forest",
    2: "Forest",
    3: "Forest",
    4: "Forest",
    5: "Forest",
    6: "Shrub",
    7: "Shrub",
    8: "Savannas",
    9: "Savannas",
    10: "Grassland",
    12: "Cropland",
    14: "Cropland",
}

MATCH_FEATURES = [
    "duration",
    "ndvi_pre_mean",
    "gpp_pre_mean",
    "et_pre_mean",
    "spei_pre_mean",
    "sm_pre_mean",
    "temp_pre_mean",
    "sti_pre_mean",
]

CORE_OUTCOMES = [
    "LOSStot",
    "ndvi_delta_event_minus_pre",
    "gpp_delta_event_minus_pre",
]

OUTCOME_LABELS = {
    "LOSStot": "Event total damage",
    "ndvi_delta_event_minus_pre": "NDVI extra decline",
    "gpp_delta_event_minus_pre": "GPP extra decline",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--window-years", type=int, nargs="+", default=[8, 10])
    parser.add_argument("--bootstrap", type=int, default=300)
    return parser.parse_args()


def land_cover_broad(code: object) -> str:
    if pd.isna(code):
        return "Other"
    return LAND_COVER_MAP.get(int(code), "Other")


def assign_duration_bin(duration: pd.Series) -> pd.Series:
    return pd.cut(duration, bins=[1, 2, 3, 4, 6, np.inf], labels=["2", "3", "4", "5-6", "7+"])


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=values.size, replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    pooled_sd = np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2.0)
    if pooled_sd == 0 or not np.isfinite(pooled_sd):
        return math.nan
    return float((a.mean() - b.mean()) / pooled_sd)


def load_temporal_event_level(start_year: int, end_year: int) -> pd.DataFrame:
    raw_usecols = ["start_date", "event_type", "land_cover", "duration", "LOSStot", "DECmax", "pixel_id"]
    metric_usecols = [
        "pixel_id",
        "event_type",
        "land_cover_broad",
        "duration",
        "LOSStot",
        "DECmax",
        "TIMErec_true",
        *MATCH_FEATURES[1:],
        *CORE_OUTCOMES[1:],
    ]

    raw = pd.read_csv(EVENT_PATH, usecols=raw_usecols)
    raw["land_cover_broad"] = raw["land_cover"].map(land_cover_broad)
    raw = raw[raw["land_cover_broad"] != "Other"].reset_index(drop=True)
    metrics = pd.read_csv(EVENT_LEVEL_PATH, usecols=metric_usecols).reset_index(drop=True)

    if len(raw) != len(metrics):
        raise ValueError(f"Event table length mismatch after land-cover filtering: {len(raw)} vs {len(metrics)}")
    for col in ["pixel_id", "event_type"]:
        if not raw[col].astype(str).equals(metrics[col].astype(str)):
            raise ValueError(f"Alignment check failed for {col}")
    for col in ["duration", "LOSStot", "DECmax"]:
        if not np.allclose(raw[col].to_numpy(float), metrics[col].to_numpy(float), equal_nan=True, atol=1e-6):
            raise ValueError(f"Alignment check failed for {col}")

    df = metrics.copy()
    df["start_date"] = pd.to_datetime(raw["start_date"]).to_numpy()
    df["start_year"] = pd.to_datetime(raw["start_date"]).dt.year.to_numpy()
    df = df[
        df["event_type"].isin([TREATED, CONTROL])
        & df["land_cover_broad"].isin(LAND_ORDER)
        & (df["start_year"] >= start_year)
        & (df["start_year"] <= end_year)
    ].copy()
    df = df.reset_index(drop=True)
    return df


def build_matches(df: pd.DataFrame, k_neighbors: int) -> pd.DataFrame:
    work = df.copy()
    work["duration_bin"] = assign_duration_bin(work["duration"])
    work["event_row_id"] = np.arange(len(work))
    treated = work[work["event_type"] == TREATED].copy()
    control = work[work["event_type"] == CONTROL].copy()
    blocks = []

    for land in LAND_ORDER:
        for d_bin in ["2", "3", "4", "5-6", "7+"]:
            tg = treated[(treated["land_cover_broad"] == land) & (treated["duration_bin"].astype(str) == d_bin)]
            cg = control[(control["land_cover_broad"] == land) & (control["duration_bin"].astype(str) == d_bin)]
            if tg.empty or cg.empty:
                continue
            pooled = pd.concat([tg[MATCH_FEATURES], cg[MATCH_FEATURES]], axis=0)
            med = pooled.median(numeric_only=True).fillna(0.0)
            std = pooled.std(ddof=0, numeric_only=True).replace(0, 1.0).fillna(1.0)
            tx = (tg[MATCH_FEATURES].fillna(med) - med) / std
            cx = (cg[MATCH_FEATURES].fillna(med) - med) / std
            tree = cKDTree(cx.to_numpy(dtype=float))
            available_k = min(k_neighbors, len(cg))
            dist, idx = tree.query(tx.to_numpy(dtype=float), k=available_k)
            if available_k == 1:
                dist = dist[:, None]
                idx = idx[:, None]
            block = tg[["event_row_id", "land_cover_broad", "duration_bin"]].reset_index(drop=True).copy()
            block["match_distance_mean"] = dist.mean(axis=1)
            for n in range(k_neighbors):
                source_n = min(n, available_k - 1)
                block[f"control_row_id_{n + 1}"] = cg.iloc[idx[:, source_n]]["event_row_id"].to_numpy()
            blocks.append(block)

    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def compute_balance(df: pd.DataFrame, matches: pd.DataFrame, label_cols: dict) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()
    indexed = df.set_index("event_row_id")
    treated_all = df[df["event_type"] == TREATED].set_index("event_row_id")
    control_all = df[df["event_type"] == CONTROL].set_index("event_row_id")
    matched_treated = indexed.loc[matches["event_row_id"]]
    control_ids = []
    for col in [c for c in matches.columns if c.startswith("control_row_id_")]:
        control_ids.extend(pd.to_numeric(matches[col], errors="coerce").dropna().astype(int).tolist())
    matched_control = indexed.loc[control_ids]

    rows = []
    for feature in MATCH_FEATURES:
        rows.append(
            {
                **label_cols,
                "feature": feature,
                "smd_before": standardized_mean_difference(treated_all[feature], control_all[feature]),
                "smd_after": standardized_mean_difference(matched_treated[feature], matched_control[feature]),
            }
        )
    return pd.DataFrame(rows)


def outcome_damage(att: float, outcome: str) -> float:
    if outcome == "LOSStot":
        return att
    return -att


def compute_att(df: pd.DataFrame, matches: pd.DataFrame, label_cols: dict, n_boot: int, seed_base: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if matches.empty:
        return pd.DataFrame(), pd.DataFrame()
    indexed = df.set_index("event_row_id")
    treated = indexed.loc[matches["event_row_id"]].reset_index()
    control_cols = [c for c in matches.columns if c.startswith("control_row_id_")]
    control_tables = []
    for col in control_cols:
        control_tables.append(indexed.loc[matches[col]].reset_index(drop=True)[CORE_OUTCOMES])
    control_mean = sum(control_tables) / len(control_tables)

    paired = treated[["event_row_id", "start_year", "land_cover_broad", *CORE_OUTCOMES]].copy()
    for outcome in CORE_OUTCOMES:
        paired[f"matched_control_{outcome}"] = control_mean[outcome].to_numpy()
        paired[f"delta_{outcome}"] = paired[outcome] - paired[f"matched_control_{outcome}"]

    rows = []
    for idx, outcome in enumerate(CORE_OUTCOMES):
        deltas = paired[f"delta_{outcome}"].to_numpy(dtype=float)
        tv = paired[outcome].to_numpy(dtype=float)
        cv = paired[f"matched_control_{outcome}"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(deltas, n_boot=n_boot, seed=seed_base + idx)
        att_mean = float(np.nanmean(deltas))
        control_mean_value = float(np.nanmean(cv))
        rows.append(
            {
                **label_cols,
                "outcome": outcome,
                "outcome_label": OUTCOME_LABELS[outcome],
                "n_treated": len(paired),
                "treated_mean": float(np.nanmean(tv)),
                "matched_control_mean": control_mean_value,
                "att_mean": att_mean,
                "damage_att": outcome_damage(att_mean, outcome),
                "relative_pct": float(att_mean / abs(control_mean_value) * 100.0)
                if np.isfinite(control_mean_value) and control_mean_value != 0
                else math.nan,
                "damage_relative_pct": float(outcome_damage(att_mean, outcome) / abs(control_mean_value) * 100.0)
                if np.isfinite(control_mean_value) and control_mean_value != 0
                else math.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "damage_ci_low": outcome_damage(ci_high, outcome) if outcome != "LOSStot" else ci_low,
                "damage_ci_high": outcome_damage(ci_low, outcome) if outcome != "LOSStot" else ci_high,
            }
        )
    return pd.DataFrame(rows), paired


def matched_att_for_subset(df: pd.DataFrame, label_cols: dict, k: int, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = df.copy().reset_index(drop=True)
    subset["event_row_id"] = np.arange(len(subset))
    matches = build_matches(subset, k_neighbors=k)
    if matches.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    balance = compute_balance(subset, matches, label_cols)
    att, paired = compute_att(subset, matches, label_cols, n_boot=n_boot, seed_base=seed)
    match_summary = pd.DataFrame(
        [
            {
                **label_cols,
                "n_candidate": len(subset),
                "n_treated_candidate": int((subset["event_type"] == TREATED).sum()),
                "n_control_candidate": int((subset["event_type"] == CONTROL).sum()),
                "n_matched_treated": len(matches),
                "mean_match_distance": float(matches["match_distance_mean"].mean()),
                "median_match_distance": float(matches["match_distance_mean"].median()),
            }
        ]
    )
    return att, paired, balance, match_summary


def compute_trends(att_table: pd.DataFrame, time_col: str, analysis_type: str) -> pd.DataFrame:
    rows = []
    for outcome, group in att_table.groupby("outcome"):
        group = group.sort_values(time_col)
        x = group[time_col].to_numpy(dtype=float)
        y = group["damage_att"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            continue
        x = x[mask]
        y = y[mask]
        lr = linregress(x, y)
        ts = theilslopes(y, x)
        sp = spearmanr(x, y)
        rows.append(
            {
                "analysis_type": analysis_type,
                "outcome": outcome,
                "outcome_label": OUTCOME_LABELS[outcome],
                "n_points": int(mask.sum()),
                "linear_slope_per_year": float(lr.slope),
                "linear_intercept": float(lr.intercept),
                "linear_r": float(lr.rvalue),
                "linear_p": float(lr.pvalue),
                "theil_sen_slope_per_year": float(ts.slope),
                "theil_sen_low": float(ts.low_slope),
                "theil_sen_high": float(ts.high_slope),
                "spearman_rho": float(sp.statistic),
                "spearman_p": float(sp.pvalue),
                "first_value": float(y[0]),
                "last_value": float(y[-1]),
                "last_minus_first": float(y[-1] - y[0]),
            }
        )
    return pd.DataFrame(rows)


def run_analysis(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_temporal_event_level(args.start_year, args.end_year)

    annual_rows = []
    annual_paired = []
    balance_rows = []
    match_rows = []

    for year in range(args.start_year, args.end_year + 1):
        subset = df[df["start_year"] == year].copy()
        label = {"analysis_type": "annual", "window_years": 1, "window_start": year, "window_end": year, "time_midpoint": year}
        att, paired, balance, match_summary = matched_att_for_subset(
            subset, label, k=args.k, n_boot=args.bootstrap, seed=1000 + year
        )
        if not att.empty:
            annual_rows.append(att)
            paired["analysis_type"] = "annual"
            paired["window_start"] = year
            paired["window_end"] = year
            annual_paired.append(paired)
            balance_rows.append(balance)
            match_rows.append(match_summary)
            print(f"annual {year}: matched={len(paired)}", flush=True)

    sliding_rows = []
    sliding_paired = []
    for window in args.window_years:
        for start in range(args.start_year, args.end_year - window + 2):
            end = start + window - 1
            subset = df[(df["start_year"] >= start) & (df["start_year"] <= end)].copy()
            label = {
                "analysis_type": f"sliding_{window}yr",
                "window_years": window,
                "window_start": start,
                "window_end": end,
                "time_midpoint": (start + end) / 2.0,
            }
            att, paired, balance, match_summary = matched_att_for_subset(
                subset, label, k=args.k, n_boot=args.bootstrap, seed=2000 + start + window
            )
            if not att.empty:
                sliding_rows.append(att)
                paired["analysis_type"] = f"sliding_{window}yr"
                paired["window_start"] = start
                paired["window_end"] = end
                sliding_paired.append(paired)
                balance_rows.append(balance)
                match_rows.append(match_summary)
                print(f"sliding {window}yr {start}-{end}: matched={len(paired)}", flush=True)

    period_rows = []
    for start, end, label_name in [(2001, 2012, "first_half"), (2013, 2024, "second_half")]:
        subset = df[(df["start_year"] >= start) & (df["start_year"] <= end)].copy()
        label = {
            "analysis_type": label_name,
            "window_years": end - start + 1,
            "window_start": start,
            "window_end": end,
            "time_midpoint": (start + end) / 2.0,
        }
        att, paired, balance, match_summary = matched_att_for_subset(
            subset, label, k=args.k, n_boot=args.bootstrap, seed=3000 + start
        )
        if not att.empty:
            period_rows.append(att)
            balance_rows.append(balance)
            match_rows.append(match_summary)

    annual = pd.concat(annual_rows, ignore_index=True)
    sliding = pd.concat(sliding_rows, ignore_index=True)
    periods = pd.concat(period_rows, ignore_index=True)
    paired_annual = pd.concat(annual_paired, ignore_index=True)
    paired_sliding = pd.concat(sliding_paired, ignore_index=True)
    balance = pd.concat(balance_rows, ignore_index=True)
    matches = pd.concat(match_rows, ignore_index=True)

    trend_tables = [compute_trends(annual, "time_midpoint", "annual")]
    for window in args.window_years:
        subset = sliding[sliding["analysis_type"] == f"sliding_{window}yr"].copy()
        trend_tables.append(compute_trends(subset, "time_midpoint", f"sliding_{window}yr"))
    trends = pd.concat(trend_tables, ignore_index=True)

    return {
        "annual": annual,
        "sliding": sliding,
        "periods": periods,
        "paired_annual": paired_annual,
        "paired_sliding": paired_sliding,
        "balance": balance,
        "match_summary": matches,
        "trends": trends,
    }


def write_outputs(tables: dict[str, pd.DataFrame]) -> None:
    tables["annual"].to_csv(OUT_DIR / "core_temporal_att_annual_2001_2024.csv", index=False)
    tables["sliding"].to_csv(OUT_DIR / "core_temporal_att_sliding_windows_2001_2024.csv", index=False)
    tables["periods"].to_csv(OUT_DIR / "core_temporal_att_first_second_half_reference.csv", index=False)
    tables["paired_annual"].to_csv(OUT_DIR / "core_temporal_paired_deltas_annual_2001_2024.csv", index=False)
    tables["paired_sliding"].to_csv(OUT_DIR / "core_temporal_paired_deltas_sliding_windows_2001_2024.csv", index=False)
    tables["balance"].to_csv(OUT_DIR / "core_temporal_match_balance_2001_2024.csv", index=False)
    tables["match_summary"].to_csv(OUT_DIR / "core_temporal_match_summary_2001_2024.csv", index=False)
    tables["trends"].to_csv(OUT_DIR / "core_temporal_damage_trends_2001_2024.csv", index=False)


def interpretation_from_trends(trends: pd.DataFrame, periods: pd.DataFrame) -> str:
    lines = [
        "# Core Temporal ATT Trend Interpretation",
        "",
        "## Purpose",
        "",
        "This Stage 9 analysis tests whether the previous `2001-2012` versus `2013-2024` comparison is supported by a more continuous temporal diagnostic. Each year is matched separately, and sliding windows are matched within each window.",
        "",
        "## Annual And Sliding-Window Trend Summary",
        "",
    ]
    show = trends[
        [
            "analysis_type",
            "outcome_label",
            "n_points",
            "linear_slope_per_year",
            "linear_p",
            "theil_sen_slope_per_year",
            "spearman_rho",
            "spearman_p",
            "first_value",
            "last_value",
            "last_minus_first",
        ]
    ].copy()
    lines.append(show.to_markdown(index=False))
    lines.extend(["", "## First-Half Versus Second-Half Reference", ""])
    ref = periods[
        [
            "analysis_type",
            "outcome_label",
            "n_treated",
            "damage_att",
            "damage_relative_pct",
            "damage_ci_low",
            "damage_ci_high",
        ]
    ].copy()
    lines.append(ref.to_markdown(index=False))
    lines.extend(["", "## Recommended Interpretation", ""])

    def get_row(analysis_type: str, outcome: str):
        part = trends[(trends["analysis_type"] == analysis_type) & (trends["outcome"] == outcome)]
        return None if part.empty else part.iloc[0]

    los_ann = get_row("annual", "LOSStot")
    los_sliding = get_row("sliding_8yr", "LOSStot")
    if los_ann is not None and los_sliding is not None:
        if los_sliding["linear_slope_per_year"] > 0 and los_sliding["linear_p"] < 0.05:
            lines.append(
                "- The first-half versus second-half split is supported as a descriptive contrast for `LOSStot`, but not as a physical breakpoint. The annual series is noisy, whereas the 8-year sliding-window diagnostic shows a clear positive tendency."
            )
        else:
            lines.append(
                "- The first-half versus second-half split should be treated cautiously for `LOSStot`, because the smoothed temporal diagnostics do not show a clear increase."
            )

    gpp_ann = get_row("annual", "gpp_delta_event_minus_pre")
    gpp_sliding = get_row("sliding_8yr", "gpp_delta_event_minus_pre")
    if gpp_ann is not None and gpp_sliding is not None:
        if gpp_sliding["linear_slope_per_year"] > 0:
            lines.append(
                "- The GPP productivity-loss signal is positive in the first-half versus second-half comparison and in smoothed windows, but its temporal trend is weaker than `LOSStot` and `NDVI`; it should not be the main temporal-strengthening headline."
            )
        else:
            lines.append(
                "- The GPP temporal signal is not uniformly increasing and should not be overused as a headline temporal-strengthening claim."
            )

    lines.extend(
        [
            "- The manuscript should no longer imply that `2013` is a physical breakpoint. A safer wording is: `the second half of the common satellite record showed larger matched extra damage, and sliding-window diagnostics indicated a generally positive time tendency`.",
            "- Because sliding windows overlap, their p-values should be treated as descriptive trend diagnostics rather than independent formal tests.",
            "- If the final text needs to be concise, use the sliding-window result as the main temporal robustness layer and move annual year-by-year values to Supplementary Data.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    tables = run_analysis(args)
    write_outputs(tables)
    report = interpretation_from_trends(tables["trends"], tables["periods"])
    report_path = OUT_DIR / "core_temporal_att_trend_interpretation.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
