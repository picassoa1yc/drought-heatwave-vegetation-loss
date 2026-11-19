from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/global_v05_90p_main")
EVENT_PATH = BASE_DIR / "global_integrated_events_90p.csv"
LOCAL_PATH = BASE_DIR / "global_local_coefficients_90p_ndvi_min60.csv"
OUTPUT_DIR = BASE_DIR / "event_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def classify_amplification_zone(df: pd.DataFrame) -> pd.Series:
    sig = df["heat_amplifies_drought_sig"].fillna(False)
    neg = df["beta_interaction"].lt(0)
    pos = df["beta_interaction"].ge(0)
    return np.select(
        [sig, neg, pos],
        ["Sig_Aggravation", "Negative_Nonsig", "Nonnegative_Interaction"],
        default="No_Local_Model",
    )


def summarize_events(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    def q25(x: pd.Series) -> float:
        return float(x.quantile(0.25))

    def q75(x: pd.Series) -> float:
        return float(x.quantile(0.75))

    grouped = df.groupby(group_cols, dropna=False, sort=False)
    summary = grouped.agg(
        n_events=("pixel_id", "size"),
        n_pixels=("pixel_id", "nunique"),
        duration_mean=("duration", "mean"),
        duration_median=("duration", "median"),
        duration_q25=("duration", q25),
        duration_q75=("duration", q75),
        losstot_mean=("LOSStot", "mean"),
        losstot_median=("LOSStot", "median"),
        losstot_q25=("LOSStot", q25),
        losstot_q75=("LOSStot", q75),
        decmax_mean=("DECmax", "mean"),
        decmax_median=("DECmax", "median"),
        timerec_mean=("TIMErec_true", "mean"),
        timerec_median=("TIMErec_true", "median"),
        compound_share_mean=("compound_share", "mean"),
        compound_share_median=("compound_share", "median"),
        drought_months_mean=("n_drought_event", "mean"),
        heat_months_mean=("n_heat_event", "mean"),
        compound_months_mean=("n_compound_event", "mean"),
        beta_interaction_mean=("beta_interaction", "mean"),
        beta_interaction_median=("beta_interaction", "median"),
    ).reset_index()
    return summary


def build_type_share_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["amplification_zone", "event_type"], dropna=False)
        .size()
        .rename("n_events")
        .reset_index()
    )
    totals = counts.groupby("amplification_zone")["n_events"].transform("sum")
    counts["share_within_zone"] = counts["n_events"] / totals
    return counts.sort_values(["amplification_zone", "n_events"], ascending=[True, False])


def build_contrast_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = ["losstot_mean", "decmax_mean", "timerec_mean", "duration_mean"]
    if group_cols:
        summary = summarize_events(df, group_cols + ["event_type"])
        pivot = summary.pivot(index=group_cols, columns="event_type", values=metrics)
    else:
        summary = summarize_events(df.assign(_global_key="Global"), ["_global_key", "event_type"])
        pivot = summary.pivot(index="_global_key", columns="event_type", values=metrics)
    pivot.columns = [f"{metric}_{event_type}" for metric, event_type in pivot.columns]
    pivot = pivot.reset_index()
    if "_global_key" in pivot.columns:
        pivot = pivot.rename(columns={"_global_key": "scope"})

    if "losstot_mean_Compound_Overlapping" in pivot.columns and "losstot_mean_Drought_Dominant" in pivot.columns:
        pivot["delta_losstot_compound_minus_drought"] = (
            pivot["losstot_mean_Compound_Overlapping"] - pivot["losstot_mean_Drought_Dominant"]
        )
    if "losstot_mean_Compound_Overlapping" in pivot.columns and "losstot_mean_Heat_Dominant" in pivot.columns:
        pivot["delta_losstot_compound_minus_heat"] = (
            pivot["losstot_mean_Compound_Overlapping"] - pivot["losstot_mean_Heat_Dominant"]
        )
    if "timerec_mean_Compound_Overlapping" in pivot.columns and "timerec_mean_Drought_Dominant" in pivot.columns:
        pivot["delta_timerec_compound_minus_drought"] = (
            pivot["timerec_mean_Compound_Overlapping"] - pivot["timerec_mean_Drought_Dominant"]
        )
    if "timerec_mean_Compound_Overlapping" in pivot.columns and "timerec_mean_Heat_Dominant" in pivot.columns:
        pivot["delta_timerec_compound_minus_heat"] = (
            pivot["timerec_mean_Compound_Overlapping"] - pivot["timerec_mean_Heat_Dominant"]
        )
    return pivot


def main() -> None:
    local = pd.read_csv(
        LOCAL_PATH,
        usecols=[
            "pixel_id",
            "beta_interaction",
            "p_interaction",
            "heat_amplifies_drought",
            "heat_amplifies_drought_sig",
            "n_obs",
            "r_squared",
        ],
    )

    events = pd.read_csv(
        EVENT_PATH,
        usecols=[
            "pixel_id",
            "row",
            "col",
            "land_cover",
            "event_type",
            "duration",
            "LOSStot",
            "DECmax",
            "TIMErec_true",
            "compound_share",
            "n_drought_event",
            "n_heat_event",
            "n_compound_event",
            "stress_lead_months",
        ],
    )

    analysis = events.merge(local, on="pixel_id", how="left")
    analysis["amplification_zone"] = classify_amplification_zone(analysis)
    analysis["has_compound_month"] = analysis["n_compound_event"].fillna(0).gt(0)
    analysis["has_heat_month"] = analysis["n_heat_event"].fillna(0).gt(0)
    analysis["has_drought_month"] = analysis["n_drought_event"].fillna(0).gt(0)

    overall = pd.DataFrame(
        {
            "n_events": [len(analysis)],
            "n_pixels": [analysis["pixel_id"].nunique()],
            "losstot_mean": [analysis["LOSStot"].mean()],
            "losstot_median": [analysis["LOSStot"].median()],
            "decmax_mean": [analysis["DECmax"].mean()],
            "timerec_median": [analysis["TIMErec_true"].median()],
            "compound_event_share": [analysis["has_compound_month"].mean()],
            "sig_aggravation_event_share": [(analysis["amplification_zone"] == "Sig_Aggravation").mean()],
            "no_local_model_event_share": [(analysis["amplification_zone"] == "No_Local_Model").mean()],
        }
    )

    by_type = summarize_events(analysis, ["event_type"]).sort_values("n_events", ascending=False)
    by_zone = summarize_events(analysis, ["amplification_zone"]).sort_values("n_events", ascending=False)
    by_type_zone = summarize_events(analysis, ["amplification_zone", "event_type"]).sort_values(
        ["amplification_zone", "n_events"], ascending=[True, False]
    )
    type_share = build_type_share_table(analysis)
    contrast_global = build_contrast_table(analysis, [])
    contrast_by_zone = build_contrast_table(analysis, ["amplification_zone"])

    lead_summary = (
        analysis.groupby(["event_type", "stress_lead_months"], dropna=False)
        .size()
        .rename("n_events")
        .reset_index()
        .sort_values(["event_type", "stress_lead_months"])
    )

    overall.to_csv(OUTPUT_DIR / "event_overall_summary.csv", index=False)
    by_type.to_csv(OUTPUT_DIR / "event_summary_by_type.csv", index=False)
    by_zone.to_csv(OUTPUT_DIR / "event_summary_by_amplification_zone.csv", index=False)
    by_type_zone.to_csv(OUTPUT_DIR / "event_summary_by_type_and_zone.csv", index=False)
    type_share.to_csv(OUTPUT_DIR / "event_type_share_within_zone.csv", index=False)
    contrast_global.to_csv(OUTPUT_DIR / "event_contrast_global.csv", index=False)
    contrast_by_zone.to_csv(OUTPUT_DIR / "event_contrast_by_amplification_zone.csv", index=False)
    lead_summary.to_csv(OUTPUT_DIR / "event_stress_lead_summary.csv", index=False)
    analysis.to_csv(OUTPUT_DIR / "event_table_with_amplification_zone.csv", index=False)

    report_lines = [
        "GLOBAL EVENT ANALYSIS SUMMARY",
        f"event_path={EVENT_PATH}",
        f"local_path={LOCAL_PATH}",
        f"merged_events={len(analysis)}",
        f"merged_pixels={analysis['pixel_id'].nunique()}",
        "",
        "[overall]",
        overall.to_string(index=False),
        "",
        "[top event types]",
        by_type.head(6).to_string(index=False),
        "",
        "[amplification zones]",
        by_zone.to_string(index=False),
        "",
        "[compound contrasts]",
        contrast_by_zone.to_string(index=False),
    ]
    (OUTPUT_DIR / "event_analysis_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved event analysis outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
