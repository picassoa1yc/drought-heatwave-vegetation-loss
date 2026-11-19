from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

from analyze_matched_compound_attribution import CONTROL_LABEL, LAND_ORDER, MATCH_FEATURES, TREATED_LABEL, compute_att
from analyze_temporal_matched_attribution import bootstrap_mean_difference, build_paired_delta_table, permutation_pvalue
from run_event_threshold_robustness import build_event_level

BASE_DIR = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/global_v05_90p_main")
EVENT_PATH = BASE_DIR / "global_integrated_events_90p.csv"
OUTPUT_DIR = BASE_DIR / "background_regime_attribution"
USED_DATA = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data")
TEMP_PATH = USED_DATA / "tmp_2001-2024.tif"
DROUGHT_MASK_PATH = USED_DATA / "drought_mask_filtered_vegetated_only.tif"
LANDCOVER_PATH = USED_DATA / "landcover_2001-2024_modal.tif"
EXCLUDE_LC = {13, 15, 16}


def assign_duration_bin(duration: pd.Series) -> pd.Series:
    return pd.cut(duration, bins=[1, 2, 3, 4, 6, np.inf], labels=["2", "3", "4", "5-6", "7+"])


def robust_build_matches(df: pd.DataFrame, k_neighbors: int = 3) -> pd.DataFrame:
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

            for feature in MATCH_FEATURES:
                treated_group[feature] = pd.to_numeric(treated_group[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
                control_group[feature] = pd.to_numeric(control_group[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
                if treated_group[feature].notna().sum() == 0:
                    treated_group[feature] = 0.0
                if control_group[feature].notna().sum() == 0:
                    control_group[feature] = 0.0

            pooled = pd.concat([treated_group[MATCH_FEATURES], control_group[MATCH_FEATURES]], axis=0)
            medians = pooled.median(numeric_only=True).fillna(0.0)
            stds = pooled.std(ddof=0, numeric_only=True)
            stds = stds.replace(0, 1.0).fillna(1.0)

            treated_x = ((treated_group[MATCH_FEATURES].fillna(medians) - medians) / stds).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            control_x = ((control_group[MATCH_FEATURES].fillna(medians) - medians) / stds).replace([np.inf, -np.inf], 0.0).fillna(0.0)

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

    return pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()


def build_external_background() -> pd.DataFrame:
    with rasterio.open(TEMP_PATH) as temp_ds, rasterio.open(DROUGHT_MASK_PATH) as drought_ds, rasterio.open(LANDCOVER_PATH) as lc_ds:
        temp = temp_ds.read().astype(np.float32)
        drought = drought_ds.read().astype(np.float32)
        landcover = lc_ds.read(1).astype(np.float32)

    temp[temp <= -9990] = np.nan
    drought[(drought < 0) | (drought > 1)] = np.nan
    lc_int = np.where(np.isfinite(landcover), landcover.astype(int), -9999)
    valid_lc = np.isfinite(landcover) & ~np.isin(lc_int, list(EXCLUDE_LC))

    temp_mean = np.nanmean(temp, axis=0)
    drought_freq = np.nanmean(drought, axis=0)
    valid = valid_lc & np.isfinite(temp_mean) & np.isfinite(drought_freq)

    rows, cols = np.where(valid)
    pixel_id = rows * landcover.shape[1] + cols
    bg = pd.DataFrame(
        {
            "pixel_id": pixel_id.astype(int),
            "temp_bg_external": temp_mean[rows, cols].astype(float),
            "drought_freq_external": drought_freq[rows, cols].astype(float),
        }
    )
    bg["thermal_regime_external"] = pd.qcut(bg["temp_bg_external"], q=3, labels=["Cool_ext", "Warm_ext", "Hot_ext"], duplicates="drop")
    bg["drought_proneness_external"] = pd.qcut(bg["drought_freq_external"], q=3, labels=["Low_ext", "Medium_ext", "High_ext"], duplicates="drop")
    return bg


def compute_period_stratified_att(event_level: pd.DataFrame, group_col: str, preferred_order: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = event_level[event_level["event_type"].isin([TREATED_LABEL, CONTROL_LABEL])].copy()
    df = df[df["land_cover_broad"].isin(LAND_ORDER)].copy()
    df = df[df[group_col].notna()].copy()

    att_frames: List[pd.DataFrame] = []
    paired_store: Dict[tuple[str, str], pd.DataFrame] = {}
    available = [lvl for lvl in preferred_order if lvl in set(df[group_col].astype(str))]

    for period in ["2001-2012", "2013-2024"]:
        for level in available:
            subset = df[(df["period"] == period) & (df[group_col].astype(str) == level)].copy().reset_index(drop=True)
            if subset.empty:
                continue
            subset["event_row_id"] = np.arange(len(subset))
            matches = robust_build_matches(subset)
            if matches.empty:
                continue
            control_cols = [c for c in matches.columns if c.startswith("control_row_id_")]
            first_control = control_cols[0]
            for col in control_cols[1:]:
                matches[col] = matches[col].fillna(matches[first_control])
            matches = matches[matches[first_control].notna()].copy()
            if matches.empty:
                continue
            att = compute_att(subset, matches, [group_col])
            att = att[np.isfinite(att["att_mean"])].copy()
            if att.empty:
                continue
            att["period"] = period
            att_frames.append(att)
            paired = build_paired_delta_table(subset, matches, period)
            paired[group_col] = level
            paired_store[(period, level)] = paired

    att_df = pd.concat(att_frames, ignore_index=True)
    diff_rows: List[Dict[str, float]] = []
    for idx, level in enumerate(available):
        early = paired_store.get(("2001-2012", level))
        late = paired_store.get(("2013-2024", level))
        if early is None or late is None:
            continue
        early_delta = early["delta_LOSStot"].to_numpy(dtype=float)
        late_delta = late["delta_LOSStot"].to_numpy(dtype=float)
        early_mean = float(np.nanmean(early_delta))
        late_mean = float(np.nanmean(late_delta))
        ci_low, ci_high = bootstrap_mean_difference(early_delta, late_delta, seed=1200 + idx)
        diff_rows.append(
            {
                group_col: level,
                "early_att_mean": early_mean,
                "late_att_mean": late_mean,
                "late_minus_early": late_mean - early_mean,
                "relative_change_pct": float((late_mean - early_mean) / abs(early_mean) * 100.0) if np.isfinite(early_mean) and early_mean != 0 else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_permutation": permutation_pvalue(early_delta, late_delta, seed=1500 + idx),
                "n_early": int(len(early_delta)),
                "n_late": int(len(late_delta)),
            }
        )
    return att_df, pd.DataFrame(diff_rows)


def write_report(background: pd.DataFrame, thermal_att: pd.DataFrame, thermal_diff: pd.DataFrame, drought_att: pd.DataFrame, drought_diff: pd.DataFrame) -> None:
    lines = [
        "EXTERNAL BACKGROUND ROBUSTNESS SUMMARY",
        "",
        "[external pixel background distribution]",
        background[["temp_bg_external", "drought_freq_external"]].describe().to_string(),
        "",
        "[external thermal regime ATT by period: LOSStot]",
        thermal_att[thermal_att["outcome"] == "LOSStot"][
            ["thermal_regime_external", "period", "n_treated", "treated_mean", "matched_control_mean", "att_mean", "relative_pct", "ci_low", "ci_high"]
        ].to_string(index=False),
        "",
        "[external thermal regime late minus early]",
        thermal_diff.to_string(index=False),
        "",
        "[external drought-proneness ATT by period: LOSStot]",
        drought_att[drought_att["outcome"] == "LOSStot"][
            ["drought_proneness_external", "period", "n_treated", "treated_mean", "matched_control_mean", "att_mean", "relative_pct", "ci_low", "ci_high"]
        ].to_string(index=False),
        "",
        "[external drought-proneness late minus early]",
        drought_diff.to_string(index=False),
    ]
    (OUTPUT_DIR / "external_background_robustness_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    background = build_external_background()
    background.to_csv(OUTPUT_DIR / "pixel_background_external.csv", index=False)

    event_level = build_event_level(EVENT_PATH)
    event_level = event_level.merge(background, on="pixel_id", how="left")

    thermal_att, thermal_diff = compute_period_stratified_att(event_level, "thermal_regime_external", ["Cool_ext", "Warm_ext", "Hot_ext"])
    drought_att, drought_diff = compute_period_stratified_att(event_level, "drought_proneness_external", ["Low_ext", "Medium_ext", "High_ext"])

    thermal_att.to_csv(OUTPUT_DIR / "external_thermal_regime_att_by_period.csv", index=False)
    thermal_diff.to_csv(OUTPUT_DIR / "external_thermal_regime_period_difference.csv", index=False)
    drought_att.to_csv(OUTPUT_DIR / "external_drought_proneness_att_by_period.csv", index=False)
    drought_diff.to_csv(OUTPUT_DIR / "external_drought_proneness_period_difference.csv", index=False)
    write_report(background, thermal_att, thermal_diff, drought_att, drought_diff)
    print("Saved external background robustness outputs to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
