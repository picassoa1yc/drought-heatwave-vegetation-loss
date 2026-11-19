from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve()
REVISION_ROOT = ROOT / "codex_revision_v4_1_20260715"
OUTPUT_DIR = REVISION_ROOT / "analysis_outputs"
TABLE_DIR = REVISION_ROOT / "table_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

P90_EVENT_PATH = (
    ROOT
    / "output_df"
    / "global_v05_90p_main"
    / "event_analysis"
    / "event_multivariate_response_event_level.csv"
)
P95_EVENT_PATH = (
    ROOT
    / "output_df"
    / "global_v05_95p_robustness"
    / "event_analysis"
    / "event_multivariate_response_event_level.csv"
)
CUMULATIVE_PATH = (
    ROOT
    / "output_df"
    / "global_v05_90p_vpd_main"
    / "event_analysis_anomaly"
    / "event_cumulative_anomaly_event_level.csv"
)
BACKGROUND_PATH = (
    ROOT
    / "output_df"
    / "global_v05_90p_main"
    / "background_regime_attribution"
    / "pixel_background_regimes.csv"
)

TREATED = "Compound_Overlapping"
CONTROL = "Drought_Dominant"
LAND_ORDER = ["Forest", "Shrub", "Savannas", "Grassland", "Cropland"]
PERIOD_ORDER = ["2001-2012", "2013-2024"]
THERMAL_ORDER = ["Cool", "Warm", "Hot"]

MATCH_FEATURES = [
    "duration",
    "ndvi_pre_mean",
    "gpp_pre_mean",
    "et_pre_mean",
    "spei_pre_mean",
    "sm_pre_mean",
    "temp_pre_mean",
]

EVENT_OUTCOMES = [
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
]

FIGURE_EVENT_OUTCOMES = [
    "LOSStot",
    "DECmax",
    "TIMErec_true",
    "ndvi_delta_event_minus_pre",
    "sif_delta_event_minus_pre",
    "gpp_delta_event_minus_pre",
    "et_delta_event_minus_pre",
]

CUM_MATCH_FEATURES = [
    "duration",
    "ndvi_anom_pre_mean",
    "gpp_anom_pre_mean",
    "et_anom_pre_mean",
    "spei_anom_pre_mean",
    "sm_anom_pre_mean",
    "temp_anom_pre_mean",
]

CUM_OUTCOMES = ["ndvi_cum_loss", "sif_cum_loss", "gpp_cum_loss", "et_cum_loss"]

OUTCOME_LABELS = {
    "LOSStot": "Total event loss",
    "DECmax": "Peak NDVI decline",
    "TIMErec_true": "Recovery time",
    "ndvi_delta_event_minus_pre": "NDVI event-window decline",
    "sif_delta_event_minus_pre": "SIF event-window decline",
    "gpp_delta_event_minus_pre": "GPP event-window decline",
    "et_delta_event_minus_pre": "Transpiration event-window decline",
    "ndvi_cum_loss": "Cumulative NDVI anomaly loss",
    "sif_cum_loss": "Cumulative SIF anomaly loss",
    "gpp_cum_loss": "Cumulative GPP anomaly loss",
    "et_cum_loss": "Cumulative transpiration anomaly loss",
}


def assign_duration_bin(duration: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(duration, errors="coerce"),
        bins=[1, 2, 3, 4, 6, np.inf],
        labels=["2", "3", "4", "5-6", ">=7"],
    )


def outcome_orientation(outcome: str) -> float:
    return -1.0 if outcome.endswith("_delta_event_minus_pre") else 1.0


def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float:
    av = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    bv = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(av) == 0 or len(bv) == 0:
        return np.nan
    pooled_sd = np.sqrt((np.var(av, ddof=0) + np.var(bv, ddof=0)) / 2.0)
    if not np.isfinite(pooled_sd) or pooled_sd == 0:
        return np.nan
    return float((np.mean(av) - np.mean(bv)) / pooled_sd)


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 400,
    seed: int = 42,
    batch_size: int = 20,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    for start in range(0, n_boot, batch_size):
        size = min(batch_size, n_boot - start)
        idx = rng.integers(0, len(values), size=(size, len(values)))
        means.append(values[idx].mean(axis=1))
    boot = np.concatenate(means)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def bootstrap_difference_ci(
    early: np.ndarray,
    recent: np.ndarray,
    n_boot: int = 400,
    seed: int = 42,
    batch_size: int = 20,
) -> tuple[float, float]:
    early = np.asarray(early, dtype=float)
    recent = np.asarray(recent, dtype=float)
    early = early[np.isfinite(early)]
    recent = recent[np.isfinite(recent)]
    if len(early) == 0 or len(recent) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs: list[np.ndarray] = []
    for start in range(0, n_boot, batch_size):
        size = min(batch_size, n_boot - start)
        early_idx = rng.integers(0, len(early), size=(size, len(early)))
        recent_idx = rng.integers(0, len(recent), size=(size, len(recent)))
        diffs.append(recent[recent_idx].mean(axis=1) - early[early_idx].mean(axis=1))
    boot = np.concatenate(diffs)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def permutation_pvalue(
    early: np.ndarray,
    recent: np.ndarray,
    n_perm: int = 400,
    seed: int = 42,
) -> float:
    early = np.asarray(early, dtype=float)
    recent = np.asarray(recent, dtype=float)
    early = early[np.isfinite(early)]
    recent = recent[np.isfinite(recent)]
    if len(early) == 0 or len(recent) == 0:
        return np.nan
    observed = abs(float(recent.mean() - early.mean()))
    pooled = np.concatenate([early, recent])
    n_early = len(early)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        permuted = rng.permutation(pooled)
        difference = abs(float(permuted[n_early:].mean() - permuted[:n_early].mean()))
        exceed += int(difference >= observed)
    return float((exceed + 1) / (n_perm + 1))


def prepare_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["event_type"].isin([TREATED, CONTROL])].copy()
    out = out[out["land_cover_broad"].isin(LAND_ORDER)].copy().reset_index(drop=True)
    out["event_row_id"] = np.arange(len(out), dtype=int)
    out["duration_bin"] = assign_duration_bin(out["duration"])
    return out


def build_matches(
    df: pd.DataFrame,
    features: list[str],
    k_neighbors: int = 3,
) -> pd.DataFrame:
    treated = df[df["event_type"].eq(TREATED)].copy()
    controls = df[df["event_type"].eq(CONTROL)].copy()
    blocks: list[pd.DataFrame] = []
    for land_cover in LAND_ORDER:
        for duration_bin in ["2", "3", "4", "5-6", ">=7"]:
            tg = treated[
                treated["land_cover_broad"].eq(land_cover)
                & treated["duration_bin"].astype(str).eq(duration_bin)
            ].copy()
            cg = controls[
                controls["land_cover_broad"].eq(land_cover)
                & controls["duration_bin"].astype(str).eq(duration_bin)
            ].copy()
            if tg.empty or cg.empty:
                continue
            pooled = pd.concat([tg[features], cg[features]], axis=0)
            medians = pooled.median(numeric_only=True)
            scales = pooled.std(ddof=0, numeric_only=True).replace(0, 1.0).fillna(1.0)
            tx = (tg[features].fillna(medians) - medians) / scales
            cx = (cg[features].fillna(medians) - medians) / scales
            tree = cKDTree(cx.to_numpy(dtype=float))
            k = min(k_neighbors, len(cg))
            distances, indices = tree.query(tx.to_numpy(dtype=float), k=k)
            if k == 1:
                distances = distances[:, None]
                indices = indices[:, None]
            block = tg[["event_row_id", "land_cover_broad", "duration_bin"]].reset_index(drop=True)
            block["match_distance_mean"] = distances.mean(axis=1)
            for neighbour in range(k):
                block[f"control_row_id_{neighbour + 1}"] = (
                    cg.iloc[indices[:, neighbour]]["event_row_id"].to_numpy(dtype=int)
                )
            blocks.append(block)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True, sort=False)


def build_paired_table(
    df: pd.DataFrame,
    matches: pd.DataFrame,
    outcomes: list[str],
    metadata: Iterable[str] = (),
) -> pd.DataFrame:
    indexed = df.set_index("event_row_id", drop=False)
    treated = indexed.loc[matches["event_row_id"].astype(int)].reset_index(drop=True)
    control_columns = [c for c in matches.columns if c.startswith("control_row_id_")]
    stacks: list[np.ndarray] = []
    for column in control_columns:
        ids = pd.to_numeric(matches[column], errors="coerce")
        block = np.full((len(matches), len(outcomes)), np.nan, dtype=float)
        valid = ids.notna().to_numpy()
        if valid.any():
            block[valid, :] = indexed.reindex(ids[valid].astype(int))[outcomes].to_numpy(dtype=float)
        stacks.append(block)
    control_mean = np.nanmean(np.stack(stacks, axis=2), axis=2)
    keep = list(dict.fromkeys([*metadata, *outcomes]))
    paired = treated[keep].copy().reset_index(drop=True)
    for index, outcome in enumerate(outcomes):
        paired[f"matched_control_{outcome}"] = control_mean[:, index]
        paired[f"delta_{outcome}"] = paired[outcome] - control_mean[:, index]
    return paired


def summarize_paired(
    paired: pd.DataFrame,
    outcomes: list[str],
    seed_offset: int = 0,
) -> pd.DataFrame:
    rows = []
    for index, outcome in enumerate(outcomes):
        treated = pd.to_numeric(paired[outcome], errors="coerce").to_numpy(dtype=float)
        control = pd.to_numeric(paired[f"matched_control_{outcome}"], errors="coerce").to_numpy(dtype=float)
        delta = pd.to_numeric(paired[f"delta_{outcome}"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(delta) & np.isfinite(treated) & np.isfinite(control)
        treated = treated[valid]
        control = control[valid]
        delta = delta[valid]
        pooled_sd = float(np.sqrt((np.var(treated, ddof=0) + np.var(control, ddof=0)) / 2.0))
        ci_low, ci_high = bootstrap_mean_ci(delta, seed=42 + seed_offset + index)
        treated_mean = float(np.mean(treated))
        control_mean = float(np.mean(control))
        att = float(np.mean(delta))
        orientation = outcome_orientation(outcome)
        oriented_ci = sorted([orientation * ci_low, orientation * ci_high])
        rows.append(
            {
                "outcome": outcome,
                "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
                "n_treated": int(len(delta)),
                "treated_mean": treated_mean,
                "matched_control_mean": control_mean,
                "att_mean": att,
                "att_median": float(np.median(delta)),
                "relative_pct": float(att / abs(control_mean) * 100.0) if control_mean != 0 else np.nan,
                "pooled_sd": pooled_sd,
                "standardized_att": float(att / pooled_sd) if pooled_sd > 0 else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "orientation": orientation,
                "additional_effect": orientation * att,
                "additional_relative_pct": orientation * float(att / abs(control_mean) * 100.0)
                if control_mean != 0
                else np.nan,
                "additional_ci_low": oriented_ci[0],
                "additional_ci_high": oriented_ci[1],
            }
        )
    return pd.DataFrame(rows)


def compute_balance(df: pd.DataFrame, matches: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    treated_all = df[df["event_type"].eq(TREATED)].copy()
    control_all = df[df["event_type"].eq(CONTROL)].copy()
    indexed = df.set_index("event_row_id", drop=False)
    matched_treated = indexed.loc[matches["event_row_id"].astype(int)].copy()
    control_ids = []
    for column in [c for c in matches.columns if c.startswith("control_row_id_")]:
        control_ids.extend(pd.to_numeric(matches[column], errors="coerce").dropna().astype(int).tolist())
    matched_controls = indexed.loc[control_ids].copy()
    rows = []
    for feature in features:
        rows.append(
            {
                "feature": feature,
                "smd_before": standardized_mean_difference(treated_all[feature], control_all[feature]),
                "smd_after": standardized_mean_difference(matched_treated[feature], matched_controls[feature]),
            }
        )
    return pd.DataFrame(rows)


def attach_fixed_standardization(
    summary: pd.DataFrame,
    reference_sd: dict[str, float],
    prefix: str = "fixed",
) -> pd.DataFrame:
    out = summary.copy()
    out["reference_pooled_sd"] = out["outcome"].map(reference_sd)
    out[f"{prefix}_standardized_effect"] = out["additional_effect"] / out["reference_pooled_sd"]
    out[f"{prefix}_standardized_ci_low"] = out["additional_ci_low"] / out["reference_pooled_sd"]
    out[f"{prefix}_standardized_ci_high"] = out["additional_ci_high"] / out["reference_pooled_sd"]
    return out


def read_p90_event_table() -> pd.DataFrame:
    usecols = [
        "pixel_id",
        "event_type",
        "land_cover_broad",
        *MATCH_FEATURES,
        *EVENT_OUTCOMES,
    ]
    event = pd.read_csv(P90_EVENT_PATH, usecols=list(dict.fromkeys(usecols)))
    meta = pd.read_csv(
        CUMULATIVE_PATH,
        usecols=["event_type", "land_cover_broad", "duration", "start_date", "start_year", "period", "LOSStot"],
    )
    if len(event) != len(meta):
        raise ValueError(f"P90 event table ({len(event)}) and temporal metadata ({len(meta)}) do not align.")
    for column in ["event_type", "land_cover_broad"]:
        if not event[column].astype(str).reset_index(drop=True).equals(meta[column].astype(str).reset_index(drop=True)):
            raise ValueError(f"P90 alignment failed for {column}.")
    for column in ["duration", "LOSStot"]:
        if not np.allclose(
            pd.to_numeric(event[column], errors="coerce"),
            pd.to_numeric(meta[column], errors="coerce"),
            equal_nan=True,
            atol=1e-8,
        ):
            raise ValueError(f"P90 alignment failed for {column}.")
    event["start_date"] = meta["start_date"].to_numpy()
    event["start_year"] = meta["start_year"].to_numpy()
    event["period"] = meta["period"].to_numpy()
    return prepare_analysis_frame(event)


def read_p95_event_table() -> pd.DataFrame:
    usecols = [
        "pixel_id",
        "event_type",
        "land_cover_broad",
        "start_date",
        "start_year",
        "period",
        *MATCH_FEATURES,
        *EVENT_OUTCOMES,
    ]
    event = pd.read_csv(P95_EVENT_PATH, usecols=list(dict.fromkeys(usecols)))
    return prepare_analysis_frame(event)


def read_cumulative_table() -> pd.DataFrame:
    usecols = [
        "event_type",
        "land_cover_broad",
        "start_date",
        "start_year",
        "period",
        *CUM_MATCH_FEATURES,
        *CUM_OUTCOMES,
    ]
    event = pd.read_csv(CUMULATIVE_PATH, usecols=list(dict.fromkeys(usecols)))
    return prepare_analysis_frame(event)


def run_period_analysis(
    df: pd.DataFrame,
    features: list[str],
    outcomes: list[str],
    reference_sd: dict[str, float],
    stem: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    summaries = []
    paired_by_period: dict[str, pd.DataFrame] = {}
    for period_index, period in enumerate(PERIOD_ORDER):
        subset = df[df["period"].eq(period)].copy().reset_index(drop=True)
        subset["event_row_id"] = np.arange(len(subset), dtype=int)
        subset["duration_bin"] = assign_duration_bin(subset["duration"])
        matches = build_matches(subset, features, k_neighbors=3)
        paired = build_paired_table(subset, matches, outcomes, metadata=["start_year"])
        summary = summarize_paired(paired, outcomes, seed_offset=100 * (period_index + 1))
        summary = attach_fixed_standardization(summary, reference_sd)
        summary["period"] = period
        summaries.append(summary)
        paired_by_period[period] = paired
        matches.to_csv(OUTPUT_DIR / f"{stem}_matches_{period}.csv", index=False)
        paired.to_csv(OUTPUT_DIR / f"{stem}_paired_{period}.csv", index=False)
    by_period = pd.concat(summaries, ignore_index=True)

    difference_rows = []
    for index, outcome in enumerate(outcomes):
        early = paired_by_period[PERIOD_ORDER[0]][f"delta_{outcome}"].to_numpy(dtype=float)
        recent = paired_by_period[PERIOD_ORDER[1]][f"delta_{outcome}"].to_numpy(dtype=float)
        early = early[np.isfinite(early)]
        recent = recent[np.isfinite(recent)]
        early_att = float(np.mean(early))
        recent_att = float(np.mean(recent))
        raw_ci_low, raw_ci_high = bootstrap_difference_ci(early, recent, seed=700 + index)
        orientation = outcome_orientation(outcome)
        oriented_early = orientation * early_att
        oriented_recent = orientation * recent_att
        oriented_diff = oriented_recent - oriented_early
        oriented_ci = sorted([orientation * raw_ci_low, orientation * raw_ci_high])
        sd = float(reference_sd[outcome])
        difference_rows.append(
            {
                "outcome": outcome,
                "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
                "early_att_mean": early_att,
                "recent_att_mean": recent_att,
                "recent_minus_early": recent_att - early_att,
                "additional_effect_early": oriented_early,
                "additional_effect_recent": oriented_recent,
                "additional_effect_recent_minus_early": oriented_diff,
                "relative_change_pct": float(oriented_diff / abs(oriented_early) * 100.0)
                if oriented_early != 0
                else np.nan,
                "ci_low": raw_ci_low,
                "ci_high": raw_ci_high,
                "additional_ci_low": oriented_ci[0],
                "additional_ci_high": oriented_ci[1],
                "p_permutation": permutation_pvalue(early, recent, seed=900 + index),
                "n_early": int(len(early)),
                "n_recent": int(len(recent)),
                "reference_pooled_sd": sd,
                "standardized_effect_early": oriented_early / sd,
                "standardized_effect_recent": oriented_recent / sd,
                "standardized_effect_recent_minus_early": oriented_diff / sd,
                "standardized_ci_low": oriented_ci[0] / sd,
                "standardized_ci_high": oriented_ci[1] / sd,
            }
        )
    differences = pd.DataFrame(difference_rows)
    by_period.to_csv(OUTPUT_DIR / f"{stem}_att_by_period_sti_free.csv", index=False)
    differences.to_csv(OUTPUT_DIR / f"{stem}_period_difference_sti_free.csv", index=False)
    return by_period, differences, paired_by_period


def run_matching_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_matches: pd.DataFrame | None = None
    for k in [1, 3, 5]:
        matches = build_matches(df, MATCH_FEATURES, k_neighbors=k)
        if k == 3:
            base_matches = matches.copy()
        paired = build_paired_table(df, matches, ["LOSStot"])
        summary = summarize_paired(paired, ["LOSStot"], seed_offset=1000 + k).iloc[0]
        balance = compute_balance(df, matches, MATCH_FEATURES)
        rows.append(
            {
                "matching_scheme": f"k={k}",
                "matched_treated_events": int(len(matches)),
                "att_losstot": float(summary["att_mean"]),
                "relative_change_pct": float(summary["additional_relative_pct"]),
                "ci_low": float(summary["additional_ci_low"]),
                "ci_high": float(summary["additional_ci_high"]),
                "max_abs_smd_after": float(balance["smd_after"].abs().max()),
            }
        )
    if base_matches is None:
        raise RuntimeError("k=3 matches were not generated.")
    cutoff = float(base_matches["match_distance_mean"].quantile(0.90))
    trimmed = base_matches[base_matches["match_distance_mean"].le(cutoff)].copy()
    paired = build_paired_table(df, trimmed, ["LOSStot"])
    summary = summarize_paired(paired, ["LOSStot"], seed_offset=1100).iloc[0]
    balance = compute_balance(df, trimmed, MATCH_FEATURES)
    rows.append(
        {
            "matching_scheme": "k=3, trim worst 10%",
            "matched_treated_events": int(len(trimmed)),
            "att_losstot": float(summary["att_mean"]),
            "relative_change_pct": float(summary["additional_relative_pct"]),
            "ci_low": float(summary["additional_ci_low"]),
            "ci_high": float(summary["additional_ci_high"]),
            "max_abs_smd_after": float(balance["smd_after"].abs().max()),
        }
    )
    return pd.DataFrame(rows)


def run_stratified_period_analysis(
    df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    stem: str,
) -> pd.DataFrame:
    rows = []
    for group_index, group in enumerate(group_order):
        paired_by_period: dict[str, pd.DataFrame] = {}
        summaries: dict[str, pd.Series] = {}
        for period_index, period in enumerate(PERIOD_ORDER):
            subset = df[df[group_col].astype(str).eq(group) & df["period"].eq(period)].copy().reset_index(drop=True)
            subset["event_row_id"] = np.arange(len(subset), dtype=int)
            subset["duration_bin"] = assign_duration_bin(subset["duration"])
            matches = build_matches(subset, MATCH_FEATURES, k_neighbors=3)
            if matches.empty:
                continue
            paired = build_paired_table(subset, matches, ["LOSStot"])
            paired_by_period[period] = paired
            summaries[period] = summarize_paired(
                paired,
                ["LOSStot"],
                seed_offset=1400 + group_index * 20 + period_index,
            ).iloc[0]
        if not all(period in paired_by_period for period in PERIOD_ORDER):
            continue
        early_delta = paired_by_period[PERIOD_ORDER[0]]["delta_LOSStot"].to_numpy(dtype=float)
        recent_delta = paired_by_period[PERIOD_ORDER[1]]["delta_LOSStot"].to_numpy(dtype=float)
        early_delta = early_delta[np.isfinite(early_delta)]
        recent_delta = recent_delta[np.isfinite(recent_delta)]
        early_att = float(early_delta.mean())
        recent_att = float(recent_delta.mean())
        ci_low, ci_high = bootstrap_difference_ci(
            early_delta,
            recent_delta,
            seed=1600 + group_index,
        )
        rows.append(
            {
                "group_dimension": group_col,
                "group": group,
                "early_att_mean": early_att,
                "recent_att_mean": recent_att,
                "recent_minus_early": recent_att - early_att,
                "relative_change_pct": float((recent_att - early_att) / abs(early_att) * 100.0)
                if early_att != 0
                else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_permutation": permutation_pvalue(
                    early_delta,
                    recent_delta,
                    seed=1800 + group_index,
                ),
                "n_early": int(len(early_delta)),
                "n_recent": int(len(recent_delta)),
                "early_treated_mean": float(summaries[PERIOD_ORDER[0]]["treated_mean"]),
                "early_control_mean": float(summaries[PERIOD_ORDER[0]]["matched_control_mean"]),
                "recent_treated_mean": float(summaries[PERIOD_ORDER[1]]["treated_mean"]),
                "recent_control_mean": float(summaries[PERIOD_ORDER[1]]["matched_control_mean"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / f"{stem}_period_difference_sti_free.csv", index=False)
    return out


def build_tables(
    main_summary: pd.DataFrame,
    balance: pd.DataFrame,
    sensitivity: pd.DataFrame,
    event_period_diff: pd.DataFrame,
    cumulative_period_diff: pd.DataFrame,
    landcover: pd.DataFrame,
    thermal: pd.DataFrame,
) -> None:
    table1 = main_summary[main_summary["outcome"].isin(FIGURE_EVENT_OUTCOMES)].copy()
    table1 = table1[
        [
            "outcome_label",
            "treated_mean",
            "matched_control_mean",
            "additional_effect",
            "additional_relative_pct",
            "additional_ci_low",
            "additional_ci_high",
            "fixed_standardized_effect",
            "fixed_standardized_ci_low",
            "fixed_standardized_ci_high",
            "n_treated",
        ]
    ].rename(
        columns={
            "outcome_label": "Outcome",
            "treated_mean": "Treated mean",
            "matched_control_mean": "Matched control mean",
            "additional_effect": "Additional effect",
            "additional_relative_pct": "Relative change (%)",
            "additional_ci_low": "CI low",
            "additional_ci_high": "CI high",
            "fixed_standardized_effect": "Standardized additional effect",
            "fixed_standardized_ci_low": "Standardized CI low",
            "fixed_standardized_ci_high": "Standardized CI high",
            "n_treated": "Matched compound events",
        }
    )

    table2 = balance.rename(
        columns={"feature": "Feature", "smd_before": "SMD before", "smd_after": "SMD after"}
    )
    table3 = sensitivity.rename(
        columns={
            "matching_scheme": "Matching scheme",
            "matched_treated_events": "Matched compound events",
            "att_losstot": "Additional total event loss",
            "relative_change_pct": "Relative change (%)",
            "ci_low": "CI low",
            "ci_high": "CI high",
            "max_abs_smd_after": "Maximum absolute SMD after matching",
        }
    )

    event_rows = event_period_diff[event_period_diff["outcome"].isin(["LOSStot", "DECmax", "TIMErec_true"])].copy()
    cumulative_rows = cumulative_period_diff.copy()
    table4 = pd.concat([event_rows, cumulative_rows], ignore_index=True, sort=False)
    table4 = table4[
        [
            "outcome_label",
            "additional_effect_early",
            "additional_effect_recent",
            "additional_effect_recent_minus_early",
            "relative_change_pct",
            "additional_ci_low",
            "additional_ci_high",
            "p_permutation",
            "standardized_effect_early",
            "standardized_effect_recent",
        ]
    ].rename(
        columns={
            "outcome_label": "Outcome",
            "additional_effect_early": "2001-2012 additional effect",
            "additional_effect_recent": "2013-2024 additional effect",
            "additional_effect_recent_minus_early": "Recent minus early",
            "relative_change_pct": "Relative change (%)",
            "additional_ci_low": "CI low",
            "additional_ci_high": "CI high",
            "p_permutation": "Permutation P",
            "standardized_effect_early": "2001-2012 standardized effect",
            "standardized_effect_recent": "2013-2024 standardized effect",
        }
    )

    table5 = pd.concat(
        [
            landcover.assign(**{"Group dimension": "Ecosystem"}),
            thermal.assign(**{"Group dimension": "Thermal background"}),
        ],
        ignore_index=True,
        sort=False,
    )
    table5 = table5[
        [
            "Group dimension",
            "group",
            "early_att_mean",
            "recent_att_mean",
            "recent_minus_early",
            "relative_change_pct",
            "ci_low",
            "ci_high",
            "p_permutation",
            "n_early",
            "n_recent",
        ]
    ].rename(
        columns={
            "group": "Group",
            "early_att_mean": "2001-2012 additional total event loss",
            "recent_att_mean": "2013-2024 additional total event loss",
            "recent_minus_early": "Recent minus early",
            "relative_change_pct": "Relative change (%)",
            "ci_low": "CI low",
            "ci_high": "CI high",
            "p_permutation": "Permutation P",
            "n_early": "Matched compound events 2001-2012",
            "n_recent": "Matched compound events 2013-2024",
        }
    )

    for number, table in [(1, table1), (2, table2), (3, table3), (4, table4), (5, table5)]:
        table.to_csv(TABLE_DIR / f"Supplementary_Table_{number}_sti_free.csv", index=False)


def main() -> None:
    print("Loading P90 event-level data...")
    p90 = read_p90_event_table()
    print(f"P90 eligible events: {len(p90):,}")
    p90_matches = build_matches(p90, MATCH_FEATURES, k_neighbors=3)
    p90_paired = build_paired_table(p90, p90_matches, EVENT_OUTCOMES)
    p90_summary = summarize_paired(p90_paired, EVENT_OUTCOMES)
    event_reference_sd = p90_summary.set_index("outcome")["pooled_sd"].to_dict()
    p90_summary = attach_fixed_standardization(p90_summary, event_reference_sd)
    p90_balance = compute_balance(p90, p90_matches, MATCH_FEATURES)
    p90_matches.to_csv(OUTPUT_DIR / "p90_matches_sti_free.csv", index=False)
    p90_paired.to_csv(OUTPUT_DIR / "p90_paired_sti_free.csv", index=False)
    p90_summary.to_csv(OUTPUT_DIR / "p90_global_att_sti_free.csv", index=False)
    p90_balance.to_csv(OUTPUT_DIR / "p90_balance_sti_free.csv", index=False)

    print("Loading P95 event-level data...")
    p95 = read_p95_event_table()
    p95_matches = build_matches(p95, MATCH_FEATURES, k_neighbors=3)
    p95_paired = build_paired_table(p95, p95_matches, EVENT_OUTCOMES)
    p95_summary = summarize_paired(p95_paired, EVENT_OUTCOMES, seed_offset=50)
    p95_summary = attach_fixed_standardization(p95_summary, event_reference_sd)
    p95_matches.to_csv(OUTPUT_DIR / "p95_matches_sti_free.csv", index=False)
    p95_summary.to_csv(OUTPUT_DIR / "p95_global_att_sti_free.csv", index=False)

    threshold = p90_summary.merge(
        p95_summary,
        on=["outcome", "outcome_label"],
        suffixes=("_p90", "_p95"),
    )
    threshold.to_csv(OUTPUT_DIR / "heat_threshold_p90_p95_sti_free.csv", index=False)

    print("Running temporal event-metric matching...")
    event_by_period, event_period_diff, _ = run_period_analysis(
        p90,
        MATCH_FEATURES,
        EVENT_OUTCOMES,
        event_reference_sd,
        stem="event",
    )

    print("Running cumulative anomaly-loss matching...")
    cumulative = read_cumulative_table()
    cumulative_matches = build_matches(cumulative, CUM_MATCH_FEATURES, k_neighbors=3)
    cumulative_paired = build_paired_table(cumulative, cumulative_matches, CUM_OUTCOMES)
    cumulative_summary = summarize_paired(cumulative_paired, CUM_OUTCOMES, seed_offset=200)
    cumulative_reference_sd = cumulative_summary.set_index("outcome")["pooled_sd"].to_dict()
    cumulative_summary = attach_fixed_standardization(cumulative_summary, cumulative_reference_sd)
    cumulative_matches.to_csv(OUTPUT_DIR / "cumulative_matches_sti_free.csv", index=False)
    cumulative_summary.to_csv(OUTPUT_DIR / "cumulative_global_att_sti_free.csv", index=False)
    cumulative_by_period, cumulative_period_diff, _ = run_period_analysis(
        cumulative,
        CUM_MATCH_FEATURES,
        CUM_OUTCOMES,
        cumulative_reference_sd,
        stem="cumulative",
    )

    print("Running matching sensitivity...")
    sensitivity = run_matching_sensitivity(p90)
    sensitivity.to_csv(OUTPUT_DIR / "matching_sensitivity_sti_free.csv", index=False)

    print("Running ecosystem and thermal-background temporal analyses...")
    landcover = run_stratified_period_analysis(
        p90,
        "land_cover_broad",
        LAND_ORDER,
        stem="landcover",
    )
    background = pd.read_csv(BACKGROUND_PATH, usecols=["pixel_id", "thermal_regime"])
    p90_thermal = p90.merge(background, on="pixel_id", how="left", validate="many_to_one")
    thermal = run_stratified_period_analysis(
        p90_thermal[p90_thermal["thermal_regime"].notna()].copy(),
        "thermal_regime",
        THERMAL_ORDER,
        stem="thermal_background",
    )

    build_tables(
        p90_summary,
        p90_balance,
        sensitivity,
        event_period_diff,
        cumulative_period_diff,
        landcover,
        thermal,
    )

    reference = {
        "event_reference_pooled_sd": event_reference_sd,
        "cumulative_reference_pooled_sd": cumulative_reference_sd,
        "standardization_definition": (
            "Outcome-specific pooled standard deviation of treated and matched-control values "
            "from the full-record P90 STI-free main design; signs are oriented so positive values "
            "always indicate greater loss, deeper decline, or longer recovery under compound events."
        ),
    }
    (OUTPUT_DIR / "standardized_effect_reference.json").write_text(
        json.dumps(reference, indent=2),
        encoding="utf-8",
    )

    headline = p90_summary.set_index("outcome").loc["LOSStot"]
    hot = thermal.set_index("group").loc["Hot"]
    report = {
        "p90_matched_compound_events": int(len(p90_matches)),
        "p90_losstot_att": float(headline["att_mean"]),
        "p90_losstot_relative_pct": float(headline["additional_relative_pct"]),
        "p90_losstot_standardized_effect": float(headline["fixed_standardized_effect"]),
        "max_abs_smd_after": float(p90_balance["smd_after"].abs().max()),
        "hot_early_att": float(hot["early_att_mean"]),
        "hot_recent_att": float(hot["recent_att_mean"]),
        "hot_recent_minus_early": float(hot["recent_minus_early"]),
        "hot_permutation_p": float(hot["p_permutation"]),
    }
    (OUTPUT_DIR / "recalculation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
