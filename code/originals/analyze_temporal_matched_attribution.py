from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_matched_compound_attribution import (
    CONTROL_LABEL,
    LAND_ORDER,
    OUTCOME_VARS,
    TREATED_LABEL,
    bootstrap_ci,
    build_matches,
    compute_att,
    compute_balance,
)


BASE_DIR = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/global_v05_90p_main")
EVENT_LEVEL_PATH = BASE_DIR / "event_analysis" / "event_multivariate_response_event_level.csv"
EVENT_PATH = BASE_DIR / "global_integrated_events_90p.csv"
OUTPUT_DIR = BASE_DIR / "temporal_attribution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = BASE_DIR / "figures_multivariate"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def land_cover_broad(code: object) -> str:
    mapping = {
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
    if pd.isna(code):
        return "Other"
    return mapping.get(int(code), "Other")


def build_temporal_event_level() -> pd.DataFrame:
    event_level = pd.read_csv(EVENT_LEVEL_PATH)
    meta = pd.read_csv(EVENT_PATH, usecols=["start_date", "land_cover", "event_type", "duration", "LOSStot"])
    meta["land_cover_broad"] = meta["land_cover"].map(land_cover_broad)
    meta = meta[meta["land_cover_broad"] != "Other"].copy().reset_index(drop=True)

    if len(meta) != len(event_level):
        raise ValueError("Event-level table length does not match filtered event metadata.")

    category_cols = ["event_type", "land_cover_broad"]
    for col in category_cols:
        left = meta[col].astype(str).reset_index(drop=True)
        right = event_level[col].astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f"Alignment check failed for column: {col}")

    numeric_cols = ["duration", "LOSStot"]
    for col in numeric_cols:
        left = pd.to_numeric(meta[col], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(event_level[col], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(left, right, equal_nan=True, atol=1e-6):
            raise ValueError(f"Alignment check failed for numeric column: {col}")

    event_level = event_level.copy()
    event_level["start_date"] = pd.to_datetime(meta["start_date"]).to_numpy()
    event_level["start_year"] = pd.to_datetime(meta["start_date"]).dt.year.to_numpy()
    event_level["period"] = np.where(event_level["start_year"] <= 2012, "2001-2012", "2013-2024")
    return event_level


def compute_yearly_att(df: pd.DataFrame, matches: pd.DataFrame, outcome: str = "LOSStot") -> pd.DataFrame:
    treated = df.set_index("event_row_id").loc[matches["event_row_id"]].reset_index()
    control_cols = [c for c in matches.columns if c.startswith("control_row_id_")]
    control_tables = []
    for col in control_cols:
        control_subset = df.set_index("event_row_id").loc[matches[col]].reset_index(drop=True)
        control_tables.append(control_subset[[outcome]])
    control_mean = sum(control_tables) / len(control_tables)

    paired = treated[["start_year", outcome]].copy()
    paired["matched_control"] = control_mean[outcome].to_numpy()
    paired["delta"] = paired[outcome] - paired["matched_control"]

    rows: List[Dict[str, float]] = []
    for year, group in paired.groupby("start_year", sort=True):
        deltas = group["delta"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(deltas, n_boot=300, seed=int(year))
        rows.append(
            {
                "start_year": int(year),
                "n_treated": len(group),
                "att_mean": float(np.nanmean(deltas)),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def build_paired_delta_table(df: pd.DataFrame, matches: pd.DataFrame, period: str) -> pd.DataFrame:
    treated = df.set_index("event_row_id").loc[matches["event_row_id"]].reset_index()
    control_cols = [c for c in matches.columns if c.startswith("control_row_id_")]
    control_tables = []
    for col in control_cols:
        control_subset = df.set_index("event_row_id").loc[matches[col]].reset_index(drop=True)
        control_tables.append(control_subset[OUTCOME_VARS])
    control_mean = sum(control_tables) / len(control_tables)

    paired = treated[["start_year"] + OUTCOME_VARS].copy()
    paired["period"] = period
    for outcome in OUTCOME_VARS:
        paired[f"matched_control_{outcome}"] = control_mean[outcome].to_numpy()
        paired[f"delta_{outcome}"] = paired[outcome] - paired[f"matched_control_{outcome}"]
    return paired


def bootstrap_mean_difference(a: np.ndarray, b: np.ndarray, n_boot: int = 500, seed: int = 42) -> tuple[float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample_a = rng.choice(a, size=a.size, replace=True)
        sample_b = rng.choice(b, size=b.size, replace=True)
        diffs[i] = sample_b.mean() - sample_a.mean()
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 400, seed: int = 42) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    observed = abs(b.mean() - a.mean())
    combined = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        permuted = rng.permutation(combined)
        diff = abs(permuted[n_a:].mean() - permuted[:n_a].mean())
        if diff >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


def compare_periods(paired_by_period: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    early = paired_by_period["2001-2012"]
    late = paired_by_period["2013-2024"]
    rows: List[Dict[str, float]] = []
    for idx, outcome in enumerate(OUTCOME_VARS):
        early_delta = early[f"delta_{outcome}"].to_numpy(dtype=float)
        late_delta = late[f"delta_{outcome}"].to_numpy(dtype=float)
        early_mean = float(np.nanmean(early_delta))
        late_mean = float(np.nanmean(late_delta))
        diff = late_mean - early_mean
        ci_low, ci_high = bootstrap_mean_difference(early_delta, late_delta, seed=101 + idx)
        rows.append(
            {
                "outcome": outcome,
                "early_att_mean": early_mean,
                "late_att_mean": late_mean,
                "late_minus_early": diff,
                "relative_change_pct": float(diff / abs(early_mean) * 100.0)
                if np.isfinite(early_mean) and early_mean != 0
                else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_permutation": permutation_pvalue(early_delta, late_delta, seed=202 + idx),
            }
        )
    return pd.DataFrame(rows)


def make_figure(period_att: pd.DataFrame, yearly_att: pd.DataFrame) -> Path:
    key = period_att[period_att["outcome"].isin(["LOSStot", "gpp_delta_event_minus_pre", "temp_delta_event_minus_pre"])].copy()
    key["label"] = key["outcome"].map(
        {
            "LOSStot": "LOSStot",
            "gpp_delta_event_minus_pre": "GPP shift",
            "temp_delta_event_minus_pre": "Temp shift",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    order = ["2001-2012", "2013-2024"]
    x = np.arange(len(order))
    width = 0.22
    colors = {"LOSStot": "#b2182b", "gpp_delta_event_minus_pre": "#2166ac", "temp_delta_event_minus_pre": "#fdae61"}
    for i, outcome in enumerate(["LOSStot", "gpp_delta_event_minus_pre", "temp_delta_event_minus_pre"]):
        subset = key[key["outcome"] == outcome].set_index("period").loc[order].reset_index()
        axes[0].bar(x + (i - 1) * width, subset["att_mean"], width=width, color=colors[outcome], label=subset["label"].iloc[0])
        axes[0].errorbar(
            x + (i - 1) * width,
            subset["att_mean"],
            yerr=[subset["att_mean"] - subset["ci_low"], subset["ci_high"] - subset["att_mean"]],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3,
        )
    axes[0].axhline(0, color="0.3", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order)
    axes[0].set_title("Matched attribution in early vs late periods", fontsize=12, fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", linestyle="--", linewidth=0.4, color="0.85")

    axes[1].plot(yearly_att["start_year"], yearly_att["att_mean"], color="#542788", linewidth=2.2)
    axes[1].fill_between(yearly_att["start_year"], yearly_att["ci_low"], yearly_att["ci_high"], color="#542788", alpha=0.18)
    axes[1].axhline(0, color="0.3", linewidth=1)
    axes[1].set_title("Annual attributable extra LOSStot", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Event start year")
    axes[1].set_ylabel("ATT for LOSStot")
    axes[1].grid(True, linestyle="--", linewidth=0.4, color="0.85")

    fig.suptitle("Temporal strengthening of compound-event attribution", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / "temporal_matched_attribution.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def write_report(
    balance_by_period: Dict[str, pd.DataFrame],
    att_period: pd.DataFrame,
    yearly_att: pd.DataFrame,
    period_difference: pd.DataFrame,
) -> None:
    lines = ["TEMPORAL MATCHED ATTRIBUTION SUMMARY", ""]
    for period, balance in balance_by_period.items():
        lines.append(f"[balance: {period}]")
        lines.append(balance.to_string(index=False))
        lines.append("")

    lines.append("[ATT by period]")
    lines.append(att_period.to_string(index=False))
    lines.append("")
    lines.append("[late minus early ATT difference]")
    lines.append(period_difference.to_string(index=False))
    lines.append("")
    lines.append("[annual LOSStot ATT]")
    lines.append(yearly_att.to_string(index=False))
    (OUTPUT_DIR / "temporal_matched_attribution_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = build_temporal_event_level()
    df = df[df["event_type"].isin([TREATED_LABEL, CONTROL_LABEL])].copy()
    df = df[df["land_cover_broad"].isin(LAND_ORDER)].copy()
    df["event_row_id"] = np.arange(len(df))

    balance_by_period: Dict[str, pd.DataFrame] = {}
    period_att_frames = []
    yearly_frames = []
    paired_by_period: Dict[str, pd.DataFrame] = {}
    for period in ["2001-2012", "2013-2024"]:
        subset = df[df["period"] == period].copy().reset_index(drop=True)
        subset["event_row_id"] = np.arange(len(subset))
        matches = build_matches(subset)
        balance = compute_balance(subset, matches)
        att = compute_att(subset, matches, [])
        att["period"] = period
        yearly = compute_yearly_att(subset, matches, outcome="LOSStot")
        yearly["period"] = period
        paired = build_paired_delta_table(subset, matches, period)

        matches.to_csv(OUTPUT_DIR / f"matches_{period}.csv", index=False)
        balance.to_csv(OUTPUT_DIR / f"balance_{period}.csv", index=False)
        att.to_csv(OUTPUT_DIR / f"att_global_{period}.csv", index=False)
        yearly.to_csv(OUTPUT_DIR / f"yearly_losstot_att_{period}.csv", index=False)
        paired.to_csv(OUTPUT_DIR / f"paired_deltas_{period}.csv", index=False)

        balance_by_period[period] = balance
        period_att_frames.append(att)
        yearly_frames.append(yearly)
        paired_by_period[period] = paired

    att_period = pd.concat(period_att_frames, ignore_index=True)
    yearly_att = pd.concat(yearly_frames, ignore_index=True)
    period_difference = compare_periods(paired_by_period)
    att_period.to_csv(OUTPUT_DIR / "att_global_by_period.csv", index=False)
    yearly_att.to_csv(OUTPUT_DIR / "yearly_losstot_att_all.csv", index=False)
    period_difference.to_csv(OUTPUT_DIR / "att_period_difference.csv", index=False)
    write_report(balance_by_period, att_period, yearly_att, period_difference)
    fig_path = make_figure(att_period, yearly_att)
    print(f"Saved temporal attribution outputs to {OUTPUT_DIR}")
    print(fig_path)


if __name__ == "__main__":
    main()
