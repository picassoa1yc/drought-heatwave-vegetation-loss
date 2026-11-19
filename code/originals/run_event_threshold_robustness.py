from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from analyze_matched_compound_attribution import (
    CONTROL_LABEL,
    LAND_ORDER,
    OUTCOME_VARS,
    TREATED_LABEL,
    build_matches,
    compute_att,
    compute_balance,
)
from analyze_multivariate_event_responses import (
    ALL_VARS,
    build_event_level_metrics,
    build_variable_highlight_table,
    summarize_group,
)
from analyze_temporal_matched_attribution import (
    build_paired_delta_table,
    compare_periods,
    compute_yearly_att,
)


def build_event_level(event_path: Path) -> pd.DataFrame:
    usecols = [
        "pixel_id",
        "start_date",
        "event_type",
        "land_cover",
        "duration",
        "LOSStot",
        "DECmax",
        "TIMErec_true",
        "compound_share",
    ]
    for variable in ALL_VARS:
        usecols.extend([f"pre_{variable}_path", f"{variable}_path", f"post_{variable}_path"])

    frames: List[pd.DataFrame] = []
    meta_frames: List[pd.DataFrame] = []
    for chunk in pd.read_csv(event_path, usecols=usecols, chunksize=25000):
        meta_frames.append(chunk[["start_date"]].copy())
        frames.append(build_event_level_metrics(chunk))

    event_level = pd.concat(frames, ignore_index=True)
    meta = pd.concat(meta_frames, ignore_index=True)
    event_level["start_date"] = pd.to_datetime(meta["start_date"])
    event_level["start_year"] = event_level["start_date"].dt.year
    event_level["period"] = np.where(event_level["start_year"] <= 2012, "2001-2012", "2013-2024")
    event_level = event_level[event_level["land_cover_broad"] != "Other"].copy()
    return event_level


def write_multivariate_outputs(event_level: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_by_type = summarize_group(event_level, ["event_type"]).sort_values("LOSStot_mean", ascending=False)
    summary_by_type_landcover = summarize_group(event_level, ["event_type", "land_cover_broad"]).sort_values(
        ["event_type", "LOSStot_mean"], ascending=[True, False]
    )
    summary_by_landcover = summarize_group(event_level, ["land_cover_broad"]).sort_values(
        "LOSStot_mean", ascending=False
    )
    variable_highlights = build_variable_highlight_table(summary_by_type)

    event_level.to_csv(output_dir / "event_multivariate_response_event_level.csv", index=False)
    summary_by_type.to_csv(output_dir / "event_multivariate_response_by_type.csv", index=False)
    summary_by_type_landcover.to_csv(output_dir / "event_multivariate_response_by_type_landcover.csv", index=False)
    summary_by_landcover.to_csv(output_dir / "event_multivariate_response_by_landcover.csv", index=False)
    variable_highlights.to_csv(output_dir / "event_multivariate_response_highlights.csv", index=False)

    report_lines = [
        "MULTIVARIATE EVENT RESPONSE SUMMARY",
        "",
        "[top event types by LOSStot]",
        summary_by_type[
            [
                "event_type",
                "LOSStot_mean",
                "DECmax_mean",
                "TIMErec_true_mean",
                "ndvi_delta_event_minus_pre_mean",
                "sif_delta_event_minus_pre_mean",
                "gpp_delta_event_minus_pre_mean",
                "et_delta_event_minus_pre_mean",
                "spei_delta_event_minus_pre_mean",
                "temp_delta_event_minus_pre_mean",
                "sti_delta_event_minus_pre_mean",
                "sm_delta_event_minus_pre_mean",
            ]
        ].to_string(index=False),
        "",
        "[land cover means]",
        summary_by_landcover[
            [
                "land_cover_broad",
                "LOSStot_mean",
                "DECmax_mean",
                "TIMErec_true_mean",
                "ndvi_delta_event_minus_pre_mean",
                "sif_delta_event_minus_pre_mean",
                "gpp_delta_event_minus_pre_mean",
                "et_delta_event_minus_pre_mean",
            ]
        ].to_string(index=False),
    ]
    (output_dir / "event_multivariate_response_report.txt").write_text("\n".join(report_lines), encoding="utf-8")


def write_matched_outputs(event_level: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = event_level[event_level["event_type"].isin([TREATED_LABEL, CONTROL_LABEL])].copy()
    df = df[df["land_cover_broad"].isin(LAND_ORDER)].copy()
    df["event_row_id"] = np.arange(len(df))

    matches = build_matches(df)
    balance = compute_balance(df, matches)
    att_global = compute_att(df, matches, [])
    att_landcover = compute_att(df, matches, ["land_cover_broad"])

    matches.to_csv(output_dir / "compound_vs_drought_matches.csv", index=False)
    balance.to_csv(output_dir / "compound_vs_drought_balance.csv", index=False)
    att_global.to_csv(output_dir / "compound_vs_drought_att_global.csv", index=False)
    att_landcover.to_csv(output_dir / "compound_vs_drought_att_by_landcover.csv", index=False)

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
        key_outcomes[
            [
                "treated_mean",
                "matched_control_mean",
                "att_mean",
                "relative_pct",
                "standardized_att",
                "ci_low",
                "ci_high",
            ]
        ].to_string(),
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
    (output_dir / "matched_attribution_report.txt").write_text("\n".join(lines), encoding="utf-8")


def write_temporal_outputs(event_level: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = event_level[event_level["event_type"].isin([TREATED_LABEL, CONTROL_LABEL])].copy()
    df = df[df["land_cover_broad"].isin(LAND_ORDER)].copy()

    balance_by_period: Dict[str, pd.DataFrame] = {}
    period_att_frames: List[pd.DataFrame] = []
    yearly_frames: List[pd.DataFrame] = []
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

        matches.to_csv(output_dir / f"matches_{period}.csv", index=False)
        balance.to_csv(output_dir / f"balance_{period}.csv", index=False)
        att.to_csv(output_dir / f"att_global_{period}.csv", index=False)
        yearly.to_csv(output_dir / f"yearly_losstot_att_{period}.csv", index=False)
        paired.to_csv(output_dir / f"paired_deltas_{period}.csv", index=False)

        balance_by_period[period] = balance
        period_att_frames.append(att)
        yearly_frames.append(yearly)
        paired_by_period[period] = paired

    att_period = pd.concat(period_att_frames, ignore_index=True)
    yearly_att = pd.concat(yearly_frames, ignore_index=True)
    period_difference = compare_periods(paired_by_period)

    att_period.to_csv(output_dir / "att_global_by_period.csv", index=False)
    yearly_att.to_csv(output_dir / "yearly_losstot_att_all.csv", index=False)
    period_difference.to_csv(output_dir / "att_period_difference.csv", index=False)

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
    (output_dir / "temporal_matched_attribution_report.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run event-based robustness analyses for a chosen heat threshold.")
    parser.add_argument("--event-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_level = build_event_level(args.event_path)

    event_analysis_dir = args.output_dir / "event_analysis"
    matched_dir = args.output_dir / "matched_attribution"
    temporal_dir = args.output_dir / "temporal_attribution"

    write_multivariate_outputs(event_level, event_analysis_dir)
    write_matched_outputs(event_level, matched_dir)
    write_temporal_outputs(event_level, temporal_dir)
    print(f"Saved robustness outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
