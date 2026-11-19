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
    TREATED_LABEL,
    build_matches,
    compute_att,
)
from analyze_temporal_matched_attribution import (
    bootstrap_mean_difference,
    build_paired_delta_table,
    permutation_pvalue,
)
from run_event_threshold_robustness import build_event_level


BASE_DIR = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/global_v05_90p_main")
EVENT_PATH = BASE_DIR / "global_integrated_events_90p.csv"
PANEL_PATH = BASE_DIR / "global_pixel_month_panel_90p.csv"
OUTPUT_DIR = BASE_DIR / "background_regime_attribution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_pixel_background(panel_path: Path) -> pd.DataFrame:
    grouped_chunks: List[pd.DataFrame] = []
    usecols = ["pixel_id", "temp", "drought", "ndvi_clim_month", "growing_season_flag"]
    for chunk in pd.read_csv(panel_path, usecols=usecols, chunksize=500000):
        chunk = chunk[chunk["growing_season_flag"] == 1].copy()
        grouped = chunk.groupby("pixel_id", sort=False).agg(
            temp_sum=("temp", "sum"),
            drought_sum=("drought", "sum"),
            ndvi_clim_sum=("ndvi_clim_month", "sum"),
            n_obs=("pixel_id", "size"),
        )
        grouped_chunks.append(grouped)

    background = pd.concat(grouped_chunks).groupby(level=0).sum()
    background["temp_bg_mean"] = background["temp_sum"] / background["n_obs"]
    background["drought_freq_bg"] = background["drought_sum"] / background["n_obs"]
    background["ndvi_clim_bg_mean"] = background["ndvi_clim_sum"] / background["n_obs"]
    background = background.reset_index()

    background["thermal_regime"] = pd.qcut(
        background["temp_bg_mean"],
        q=3,
        labels=["Cool", "Warm", "Hot"],
        duplicates="drop",
    )
    background["drought_proneness"] = pd.qcut(
        background["drought_freq_bg"],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop",
    )
    background["greenness_regime"] = pd.qcut(
        background["ndvi_clim_bg_mean"],
        q=3,
        labels=["Sparse", "Intermediate", "Dense"],
        duplicates="drop",
    )
    return background[
        [
            "pixel_id",
            "temp_bg_mean",
            "drought_freq_bg",
            "ndvi_clim_bg_mean",
            "thermal_regime",
            "drought_proneness",
            "greenness_regime",
        ]
    ]


def compute_period_stratified_att(event_level: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = event_level[event_level["event_type"].isin([TREATED_LABEL, CONTROL_LABEL])].copy()
    df = df[df["land_cover_broad"].isin(LAND_ORDER)].copy()
    df = df[df[group_col].notna()].copy()

    att_frames: List[pd.DataFrame] = []
    paired_store: Dict[tuple[str, str], pd.DataFrame] = {}

    ordered_labels = [str(v) for v in df[group_col].dropna().unique()]
    preferred_orders = {
        "thermal_regime": ["Cool", "Warm", "Hot"],
        "drought_proneness": ["Low", "Medium", "High"],
        "greenness_regime": ["Sparse", "Intermediate", "Dense"],
    }
    level_order = [lvl for lvl in preferred_orders.get(group_col, ordered_labels) if lvl in ordered_labels]

    for period in ["2001-2012", "2013-2024"]:
        for level in level_order:
            subset = df[(df["period"] == period) & (df[group_col].astype(str) == level)].copy().reset_index(drop=True)
            if subset.empty:
                continue
            subset["event_row_id"] = np.arange(len(subset))
            matches = build_matches(subset)
            if matches.empty:
                continue
            control_cols = [c for c in matches.columns if c.startswith("control_row_id_")]
            if not control_cols:
                continue
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
    for idx, level in enumerate(level_order):
        early = paired_store.get(("2001-2012", level))
        late = paired_store.get(("2013-2024", level))
        if early is None or late is None:
            continue
        early_delta = early["delta_LOSStot"].to_numpy(dtype=float)
        late_delta = late["delta_LOSStot"].to_numpy(dtype=float)
        early_mean = float(np.nanmean(early_delta))
        late_mean = float(np.nanmean(late_delta))
        ci_low, ci_high = bootstrap_mean_difference(early_delta, late_delta, seed=500 + idx)
        diff_rows.append(
            {
                group_col: level,
                "early_att_mean": early_mean,
                "late_att_mean": late_mean,
                "late_minus_early": late_mean - early_mean,
                "relative_change_pct": float((late_mean - early_mean) / abs(early_mean) * 100.0)
                if np.isfinite(early_mean) and early_mean != 0
                else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_permutation": permutation_pvalue(early_delta, late_delta, seed=800 + idx),
                "n_early": int(len(early_delta)),
                "n_late": int(len(late_delta)),
            }
        )

    return att_df, pd.DataFrame(diff_rows)


def make_figure(thermal_diff: pd.DataFrame, drought_diff: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    for ax, df, label_col, title in [
        (axes[0], thermal_diff, "thermal_regime", "Thermal background"),
        (axes[1], drought_diff, "drought_proneness", "Drought-proneness background"),
    ]:
        x = np.arange(len(df))
        colors = ["#b2182b" if (p < 0.05 and d > 0) else "#f4a582" for p, d in zip(df["p_permutation"], df["late_minus_early"])]
        ax.bar(x, df["late_minus_early"], color=colors, width=0.62)
        ax.errorbar(
            x,
            df["late_minus_early"],
            yerr=[df["late_minus_early"] - df["ci_low"], df["ci_high"] - df["late_minus_early"]],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=3,
        )
        ax.axhline(0, color="0.3", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(df[label_col].astype(str))
        ax.set_ylabel("Late minus early ATT for LOSStot")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, color="0.85")
        for xi, row in df.reset_index(drop=True).iterrows():
            if row["p_permutation"] < 0.05:
                ax.text(xi, row["late_minus_early"] + (0.003 if row["late_minus_early"] >= 0 else -0.003), "*", ha="center", va="bottom" if row["late_minus_early"] >= 0 else "top", fontsize=13, fontweight="bold")

    fig.suptitle("Background climate regimes of temporal strengthening", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUTPUT_DIR / "background_regime_temporal_strengthening.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def write_report(thermal_att: pd.DataFrame, thermal_diff: pd.DataFrame, drought_att: pd.DataFrame, drought_diff: pd.DataFrame, background: pd.DataFrame) -> None:
    lines = [
        "BACKGROUND REGIME ATTRIBUTION SUMMARY",
        "",
        "[pixel background distribution]",
        background[["temp_bg_mean", "drought_freq_bg", "ndvi_clim_bg_mean"]].describe().to_string(),
        "",
        "[thermal regime ATT by period]",
        thermal_att[thermal_att["outcome"] == "LOSStot"][
            ["thermal_regime", "period", "n_treated", "treated_mean", "matched_control_mean", "att_mean", "relative_pct", "ci_low", "ci_high"]
        ].to_string(index=False),
        "",
        "[thermal regime late minus early]",
        thermal_diff.to_string(index=False),
        "",
        "[drought-proneness ATT by period]",
        drought_att[drought_att["outcome"] == "LOSStot"][
            ["drought_proneness", "period", "n_treated", "treated_mean", "matched_control_mean", "att_mean", "relative_pct", "ci_low", "ci_high"]
        ].to_string(index=False),
        "",
        "[drought-proneness late minus early]",
        drought_diff.to_string(index=False),
    ]
    (OUTPUT_DIR / "background_regime_attribution_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    background = build_pixel_background(PANEL_PATH)
    background.to_csv(OUTPUT_DIR / "pixel_background_regimes.csv", index=False)

    event_level = build_event_level(EVENT_PATH)
    event_level = event_level.merge(background, on="pixel_id", how="left")

    thermal_att, thermal_diff = compute_period_stratified_att(event_level, "thermal_regime")
    drought_att, drought_diff = compute_period_stratified_att(event_level, "drought_proneness")

    thermal_att.to_csv(OUTPUT_DIR / "thermal_regime_att_by_period.csv", index=False)
    thermal_diff.to_csv(OUTPUT_DIR / "thermal_regime_period_difference.csv", index=False)
    drought_att.to_csv(OUTPUT_DIR / "drought_proneness_att_by_period.csv", index=False)
    drought_diff.to_csv(OUTPUT_DIR / "drought_proneness_period_difference.csv", index=False)

    fig_path = make_figure(thermal_diff, drought_diff)
    write_report(thermal_att, thermal_diff, drought_att, drought_diff, background)
    print(f"Saved background regime attribution outputs to {OUTPUT_DIR}")
    print(fig_path)


if __name__ == "__main__":
    main()
