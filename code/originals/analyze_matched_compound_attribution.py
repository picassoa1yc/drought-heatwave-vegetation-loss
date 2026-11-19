from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


BASE_DIR = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/global_v05_90p_main")
EVENT_LEVEL_PATH = BASE_DIR / "event_analysis" / "event_multivariate_response_event_level.csv"
OUTPUT_DIR = BASE_DIR / "matched_attribution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TREATED_LABEL = "Compound_Overlapping"
CONTROL_LABEL = "Drought_Dominant"
LAND_ORDER = ["Forest", "Shrub", "Savannas", "Grassland", "Cropland"]
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
OUTCOME_VARS = [
    "LOSStot",
    "DECmax",
    "TIMErec_true",
    "ndvi_delta_event_minus_pre",
    "sif_delta_event_minus_pre",
    "gpp_delta_event_minus_pre",
    "et_delta_event_minus_pre",
    "spei_delta_event_minus_pre",
    "sm_delta_event_minus_pre",
    "temp_delta_event_minus_pre",
    "sti_delta_event_minus_pre",
]


def assign_duration_bin(duration: pd.Series) -> pd.Series:
    return pd.cut(duration, bins=[1, 2, 3, 4, 6, np.inf], labels=["2", "3", "4", "5-6", "7+"])


def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mean_a = a.mean()
    mean_b = b.mean()
    var_a = a.var(ddof=0)
    var_b = b.var(ddof=0)
    pooled_sd = np.sqrt((var_a + var_b) / 2.0)
    if pooled_sd == 0 or not np.isfinite(pooled_sd):
        return np.nan
    return float((mean_a - mean_b) / pooled_sd)


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 400, seed: int = 42) -> tuple[float, float]:
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = deltas.size
    for i in range(n_boot):
        sample = rng.choice(deltas, size=n, replace=True)
        means[i] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_matches(df: pd.DataFrame, k_neighbors: int = 3) -> pd.DataFrame:
    df = df.copy()
    df["duration_bin"] = assign_duration_bin(df["duration"])
    df["event_row_id"] = np.arange(len(df))

    treated = df[df["event_type"] == TREATED_LABEL].copy()
    control = df[df["event_type"] == CONTROL_LABEL].copy()

    matches: List[pd.DataFrame] = []
    for land_cover in LAND_ORDER:
        for duration_bin in ["2", "3", "4", "5-6", "7+"]:
            treated_group = treated[
                (treated["land_cover_broad"] == land_cover) & (treated["duration_bin"].astype(str) == duration_bin)
            ].copy()
            control_group = control[
                (control["land_cover_broad"] == land_cover) & (control["duration_bin"].astype(str) == duration_bin)
            ].copy()
            if treated_group.empty or control_group.empty:
                continue

            pooled = pd.concat([treated_group[MATCH_FEATURES], control_group[MATCH_FEATURES]], axis=0)
            medians = pooled.median(numeric_only=True)
            stds = pooled.std(ddof=0, numeric_only=True).replace(0, 1.0)

            treated_x = treated_group[MATCH_FEATURES].fillna(medians)
            control_x = control_group[MATCH_FEATURES].fillna(medians)
            treated_x = (treated_x - medians) / stds
            control_x = (control_x - medians) / stds

            tree = cKDTree(control_x.to_numpy(dtype=float))
            k = min(k_neighbors, len(control_group))
            dist, idx = tree.query(treated_x.to_numpy(dtype=float), k=k)
            if k == 1:
                dist = dist[:, None]
                idx = idx[:, None]

            block = treated_group[["event_row_id", "land_cover_broad", "duration_bin"]].reset_index(drop=True).copy()
            block["match_distance_mean"] = dist.mean(axis=1)
            for neighbor in range(k):
                block[f"control_row_id_{neighbor+1}"] = control_group.iloc[idx[:, neighbor]]["event_row_id"].to_numpy()
            matches.append(block)

    match_df = pd.concat(matches, ignore_index=True)
    return match_df


def compute_balance(df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    treated = df[df["event_type"] == TREATED_LABEL].set_index("event_row_id")
    control = df[df["event_type"] == CONTROL_LABEL].set_index("event_row_id")

    matched_control_ids = []
    for col in [c for c in matches.columns if c.startswith("control_row_id_")]:
        matched_control_ids.extend(matches[col].tolist())
    matched_controls = control.loc[matched_control_ids].copy()
    matched_treated = treated.loc[matches["event_row_id"]].copy()

    rows = []
    for feature in MATCH_FEATURES:
        rows.append(
            {
                "feature": feature,
                "smd_before": standardized_mean_difference(
                    treated[feature],
                    control[feature],
                ),
                "smd_after": standardized_mean_difference(
                    matched_treated[feature],
                    matched_controls[feature],
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_att(df: pd.DataFrame, matches: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    treated = df.set_index("event_row_id").loc[matches["event_row_id"]].reset_index()

    control_tables = []
    control_cols = [c for c in matches.columns if c.startswith("control_row_id_")]
    for col in control_cols:
        control_subset = df.set_index("event_row_id").loc[matches[col]].reset_index(drop=True)
        control_tables.append(control_subset[OUTCOME_VARS])
    control_mean = sum(control_tables) / len(control_tables)

    paired = treated[group_cols + OUTCOME_VARS].copy()
    for outcome in OUTCOME_VARS:
        paired[f"matched_control_{outcome}"] = control_mean[outcome].to_numpy()
        paired[f"delta_{outcome}"] = paired[outcome] - paired[f"matched_control_{outcome}"]

    rows: List[Dict[str, object]] = []
    grouped = paired.groupby(group_cols, dropna=False) if group_cols else [(("Global",), paired)]
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_meta = dict(zip(group_cols if group_cols else ["scope"], group_key))
        group_meta["n_treated"] = len(group)
        for outcome in OUTCOME_VARS:
            deltas = group[f"delta_{outcome}"].to_numpy(dtype=float)
            treated_values = group[outcome].to_numpy(dtype=float)
            control_values = group[f"matched_control_{outcome}"].to_numpy(dtype=float)
            treated_mean = float(np.nanmean(treated_values))
            control_mean = float(np.nanmean(control_values))
            pooled_sd = float(
                np.sqrt(
                    (
                        np.nanvar(treated_values, ddof=0)
                        + np.nanvar(control_values, ddof=0)
                    )
                    / 2.0
                )
            )
            ci_low, ci_high = bootstrap_ci(deltas)
            rows.append(
                {
                    **group_meta,
                    "outcome": outcome,
                    "treated_mean": treated_mean,
                    "matched_control_mean": control_mean,
                    "att_mean": float(np.nanmean(deltas)),
                    "att_median": float(np.nanmedian(deltas)),
                    "relative_pct": float(np.nanmean(deltas) / abs(control_mean) * 100.0)
                    if np.isfinite(control_mean) and control_mean != 0
                    else np.nan,
                    "standardized_att": float(np.nanmean(deltas) / pooled_sd)
                    if np.isfinite(pooled_sd) and pooled_sd != 0
                    else np.nan,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def write_report(balance: pd.DataFrame, att_global: pd.DataFrame, att_landcover: pd.DataFrame, matches: pd.DataFrame) -> None:
    key_outcomes = att_global.set_index("outcome").loc[
        [
            "LOSStot",
            "DECmax",
            "TIMErec_true",
            "ndvi_delta_event_minus_pre",
            "sif_delta_event_minus_pre",
            "gpp_delta_event_minus_pre",
            "et_delta_event_minus_pre",
            "spei_delta_event_minus_pre",
            "temp_delta_event_minus_pre",
            "sti_delta_event_minus_pre",
        ]
    ]
    key_view = key_outcomes[
        [
            "treated_mean",
            "matched_control_mean",
            "att_mean",
            "relative_pct",
            "standardized_att",
            "ci_low",
            "ci_high",
        ]
    ]
    lines = [
        "MATCHED ATTRIBUTION SUMMARY",
        f"treated={TREATED_LABEL}",
        f"control={CONTROL_LABEL}",
        f"n_matched_treated={len(matches)}",
        f"mean_match_distance={matches['match_distance_mean'].mean():.4f}",
        "",
        "[balance]",
        balance.to_string(index=False),
        "",
        "[global ATT]",
        key_view.to_string(),
        "",
        "[ATT by land cover: LOSStot]",
        att_landcover[att_landcover["outcome"] == "LOSStot"][
            [
                "land_cover_broad",
                "n_treated",
                "treated_mean",
                "matched_control_mean",
                "att_mean",
                "relative_pct",
                "ci_low",
                "ci_high",
            ]
        ].to_string(index=False),
    ]
    (OUTPUT_DIR / "matched_attribution_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    usecols = ["event_type", "land_cover_broad"] + MATCH_FEATURES + OUTCOME_VARS
    df = pd.read_csv(EVENT_LEVEL_PATH, usecols=usecols)
    df = df[df["event_type"].isin([TREATED_LABEL, CONTROL_LABEL])].copy()
    df = df[df["land_cover_broad"].isin(LAND_ORDER)].copy()
    df["event_row_id"] = np.arange(len(df))

    matches = build_matches(df)
    balance = compute_balance(df, matches)
    att_global = compute_att(df, matches, [])
    att_landcover = compute_att(df, matches, ["land_cover_broad"])

    matches.to_csv(OUTPUT_DIR / "compound_vs_drought_matches.csv", index=False)
    balance.to_csv(OUTPUT_DIR / "compound_vs_drought_balance.csv", index=False)
    att_global.to_csv(OUTPUT_DIR / "compound_vs_drought_att_global.csv", index=False)
    att_landcover.to_csv(OUTPUT_DIR / "compound_vs_drought_att_by_landcover.csv", index=False)
    write_report(balance, att_global, att_landcover, matches)
    print(f"Saved matched attribution outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
