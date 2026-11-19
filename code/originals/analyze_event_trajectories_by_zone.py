from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_event_trajectories import parse_path_values
from analyze_global_events import classify_amplification_zone


BASE_DIR = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/global_v05_90p_main")
EVENT_PATH = BASE_DIR / "global_integrated_events_90p.csv"
LOCAL_PATH = BASE_DIR / "global_local_coefficients_90p_ndvi_min60.csv"
OUTPUT_DIR = BASE_DIR / "event_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIABLE_CONFIG = {
    "ndvi": ("pre_ndvi_path", "ndvi_path", "post_ndvi_path"),
    "spei": ("pre_spei_path", "spei_path", "post_spei_path"),
    "temp": ("pre_temp_path", "temp_path", "post_temp_path"),
}

FOCUSED_EVENT_TYPES = ["Drought_Dominant", "Compound_Overlapping"]
FOCUSED_ZONES = ["Sig_Aggravation", "Negative_Nonsig", "Nonnegative_Interaction"]


def process_chunk(chunk: pd.DataFrame, stats: Dict[Tuple[str, str, str, str, int], list[float]]) -> None:
    for row in chunk.itertuples(index=False):
        event_type = row.event_type
        zone = row.amplification_zone
        if event_type not in FOCUSED_EVENT_TYPES or zone not in FOCUSED_ZONES:
            continue

        for variable, columns in VARIABLE_CONFIG.items():
            pre_values = parse_path_values(getattr(row, columns[0]))
            event_values = parse_path_values(getattr(row, columns[1]))
            post_values = parse_path_values(getattr(row, columns[2]))

            if pre_values:
                pre_len = len(pre_values)
                for idx, value in enumerate(pre_values, start=1):
                    if np.isfinite(value):
                        key = (event_type, zone, variable, "pre", -pre_len + idx - 1)
                        state = stats[key]
                        state[0] += float(value)
                        state[1] += 1.0
                        state[2] += float(value) * float(value)

            if event_values:
                for idx, value in enumerate(event_values, start=1):
                    if np.isfinite(value):
                        key = (event_type, zone, variable, "event", idx)
                        state = stats[key]
                        state[0] += float(value)
                        state[1] += 1.0
                        state[2] += float(value) * float(value)

            if post_values:
                for idx, value in enumerate(post_values, start=1):
                    if np.isfinite(value):
                        key = (event_type, zone, variable, "post", idx)
                        state = stats[key]
                        state[0] += float(value)
                        state[1] += 1.0
                        state[2] += float(value) * float(value)


def build_summary() -> pd.DataFrame:
    local = pd.read_csv(
        LOCAL_PATH,
        usecols=["pixel_id", "beta_interaction", "heat_amplifies_drought_sig"],
    )
    local["amplification_zone"] = classify_amplification_zone(local)

    usecols = ["pixel_id", "event_type"]
    for cols in VARIABLE_CONFIG.values():
        usecols.extend(cols)
    events = pd.read_csv(EVENT_PATH, usecols=usecols)
    merged = events.merge(local[["pixel_id", "amplification_zone"]], on="pixel_id", how="left")
    merged["amplification_zone"] = merged["amplification_zone"].fillna("No_Local_Model")

    stats: Dict[Tuple[str, str, str, str, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    chunk_size = 25000
    for start in range(0, len(merged), chunk_size):
        process_chunk(merged.iloc[start : start + chunk_size], stats)

    rows = []
    for (event_type, zone, variable, stage, stage_month), (value_sum, count, value_sq_sum) in stats.items():
        if count <= 0:
            continue
        mean = value_sum / count
        variance = max(value_sq_sum / count - mean * mean, 0.0)
        std = variance**0.5
        se = std / (count**0.5)
        rows.append(
            {
                "event_type": event_type,
                "amplification_zone": zone,
                "variable": variable,
                "stage": stage,
                "stage_month": stage_month,
                "n_obs": int(count),
                "mean_value": mean,
                "std_value": std,
                "se_value": se,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["event_type", "amplification_zone", "variable", "stage", "stage_month"]
    ).reset_index(drop=True)


def filter_robust_months_by_zone(summary: pd.DataFrame) -> pd.DataFrame:
    filtered_parts = []
    for _, group in summary.groupby(["event_type", "amplification_zone", "variable", "stage"], sort=False):
        max_n = float(group["n_obs"].max())
        min_n = max(50.0, max_n * 0.01)
        filtered_parts.append(group[group["n_obs"] >= min_n])
    return pd.concat(filtered_parts, ignore_index=True)


def write_report(summary: pd.DataFrame) -> Path:
    report_path = OUTPUT_DIR / "event_trajectory_by_zone_report.txt"
    lines = ["EVENT TRAJECTORY BY AMPLIFICATION ZONE", f"event_path={EVENT_PATH}", f"local_path={LOCAL_PATH}", ""]
    for event_type in FOCUSED_EVENT_TYPES:
        lines.append(f"[{event_type}]")
        for zone in FOCUSED_ZONES:
            ndvi_event = summary[
                (summary["event_type"] == event_type)
                & (summary["amplification_zone"] == zone)
                & (summary["variable"] == "ndvi")
                & (summary["stage"] == "event")
                & (summary["stage_month"] <= 6)
            ]
            spei_pre = summary[
                (summary["event_type"] == event_type)
                & (summary["amplification_zone"] == zone)
                & (summary["variable"] == "spei")
                & (summary["stage"] == "pre")
                & (summary["stage_month"] == -1)
            ]
            if ndvi_event.empty:
                continue
            min_row = ndvi_event.loc[ndvi_event["mean_value"].idxmin()]
            lines.append(
                f"{zone}: ndvi_min={min_row['mean_value']:.4f} at event_month={int(min_row['stage_month'])}, n={int(min_row['n_obs'])}"
            )
            if not spei_pre.empty:
                row = spei_pre.iloc[0]
                lines.append(f"{zone}: spei_pre_minus1={row['mean_value']:.4f}, n={int(row['n_obs'])}")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def make_plot(summary: pd.DataFrame) -> Path:
    color_map = {
        "Sig_Aggravation": "#ca0020",
        "Negative_Nonsig": "#0571b0",
        "Nonnegative_Interaction": "#7f7f7f",
    }
    variable_titles = {"ndvi": "NDVI anomaly", "spei": "SPEI", "temp": "Temperature"}
    stage_titles = {"pre": "Pre-event", "event": "Event", "post": "Post-event"}

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey="row")
    for row_idx, variable in enumerate(["ndvi", "spei", "temp"]):
        for col_idx, stage in enumerate(["pre", "event", "post"]):
            ax = axes[row_idx, col_idx]
            for zone in FOCUSED_ZONES:
                subset = summary[
                    (summary["event_type"] == "Compound_Overlapping")
                    & (summary["amplification_zone"] == zone)
                    & (summary["variable"] == variable)
                    & (summary["stage"] == stage)
                ].sort_values("stage_month")
                if subset.empty:
                    continue
                ax.plot(
                    subset["stage_month"],
                    subset["mean_value"],
                    color=color_map[zone],
                    linewidth=2,
                    label=zone,
                )
            ax.axhline(0, color="0.75", linewidth=0.8, linestyle="--")
            ax.grid(True, linestyle="--", linewidth=0.4, color="0.85")
            if row_idx == 0:
                ax.set_title(stage_titles[stage], fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(variable_titles[variable], fontsize=11)
            if row_idx == 2:
                if stage == "pre":
                    ax.set_xlabel("Months before event start")
                elif stage == "event":
                    ax.set_xlabel("Event month")
                else:
                    ax.set_xlabel("Months after event end")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Compound event trajectories across amplification zones", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    output_path = OUTPUT_DIR / "event_trajectory_by_zone_preview.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    summary = build_summary()
    summary.to_csv(OUTPUT_DIR / "event_trajectory_by_zone_summary.csv", index=False)
    robust = filter_robust_months_by_zone(summary)
    robust.to_csv(OUTPUT_DIR / "event_trajectory_by_zone_summary_robust.csv", index=False)
    report_path = write_report(robust)
    plot_path = make_plot(robust)
    print(f"Saved zone trajectory summary to {OUTPUT_DIR / 'event_trajectory_by_zone_summary.csv'}")
    print(f"Saved robust zone trajectory summary to {OUTPUT_DIR / 'event_trajectory_by_zone_summary_robust.csv'}")
    print(f"Saved zone trajectory report to {report_path}")
    print(f"Saved zone trajectory preview to {plot_path}")


if __name__ == "__main__":
    main()
