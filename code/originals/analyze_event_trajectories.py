from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/global_v05_90p_main")
EVENT_PATH = BASE_DIR / "global_integrated_events_90p.csv"
OUTPUT_DIR = BASE_DIR / "event_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIABLE_CONFIG = {
    "ndvi": ("pre_ndvi_path", "ndvi_path", "post_ndvi_path"),
    "spei": ("pre_spei_path", "spei_path", "post_spei_path"),
    "temp": ("pre_temp_path", "temp_path", "post_temp_path"),
}

FOCUSED_EVENT_TYPES = [
    "Drought_Dominant",
    "Heat_Dominant",
    "Compound_Overlapping",
    "Sequential_Drought_Heat",
]


def parse_path_values(value: object) -> list[float]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    values: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token or token.lower() == "nan":
            values.append(np.nan)
        else:
            try:
                values.append(float(token))
            except ValueError:
                values.append(np.nan)
    return values


def update_stats(
    stats: Dict[Tuple[str, str, str, int], list[float]],
    event_type: str,
    variable: str,
    stage: str,
    stage_month: int,
    values: Iterable[float],
) -> None:
    for offset, value in enumerate(values, start=1):
        if not np.isfinite(value):
            continue
        if stage == "pre":
            month_index = -len(list(values)) + offset - 1
        elif stage == "event":
            month_index = offset
        else:
            month_index = offset
        key = (event_type, variable, stage, month_index)
        state = stats[key]
        state[0] += float(value)
        state[1] += 1.0
        state[2] += float(value) * float(value)


def process_chunk(chunk: pd.DataFrame, stats: Dict[Tuple[str, str, str, int], list[float]]) -> None:
    for row in chunk.itertuples(index=False):
        event_type = row.event_type
        for variable, columns in VARIABLE_CONFIG.items():
            pre_values = parse_path_values(getattr(row, columns[0]))
            event_values = parse_path_values(getattr(row, columns[1]))
            post_values = parse_path_values(getattr(row, columns[2]))

            if pre_values:
                pre_len = len(pre_values)
                for idx, value in enumerate(pre_values, start=1):
                    if np.isfinite(value):
                        key = (event_type, variable, "pre", -pre_len + idx - 1)
                        state = stats[key]
                        state[0] += float(value)
                        state[1] += 1.0
                        state[2] += float(value) * float(value)

            if event_values:
                for idx, value in enumerate(event_values, start=1):
                    if np.isfinite(value):
                        key = (event_type, variable, "event", idx)
                        state = stats[key]
                        state[0] += float(value)
                        state[1] += 1.0
                        state[2] += float(value) * float(value)

            if post_values:
                for idx, value in enumerate(post_values, start=1):
                    if np.isfinite(value):
                        key = (event_type, variable, "post", idx)
                        state = stats[key]
                        state[0] += float(value)
                        state[1] += 1.0
                        state[2] += float(value) * float(value)


def build_summary() -> pd.DataFrame:
    usecols = ["event_type"]
    for columns in VARIABLE_CONFIG.values():
        usecols.extend(columns)

    stats: Dict[Tuple[str, str, str, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for chunk in pd.read_csv(EVENT_PATH, usecols=usecols, chunksize=25000):
        process_chunk(chunk, stats)

    rows = []
    for (event_type, variable, stage, stage_month), (value_sum, count, value_sq_sum) in stats.items():
        if count <= 0:
            continue
        mean = value_sum / count
        variance = max(value_sq_sum / count - mean * mean, 0.0)
        std = variance**0.5
        se = std / (count**0.5)
        rows.append(
            {
                "event_type": event_type,
                "variable": variable,
                "stage": stage,
                "stage_month": stage_month,
                "n_obs": int(count),
                "mean_value": mean,
                "std_value": std,
                "se_value": se,
            }
        )

    summary = pd.DataFrame(rows).sort_values(["variable", "event_type", "stage", "stage_month"]).reset_index(drop=True)
    return summary


def filter_robust_months(summary: pd.DataFrame) -> pd.DataFrame:
    filtered_parts = []
    for _, group in summary.groupby(["event_type", "variable", "stage"], sort=False):
        max_n = float(group["n_obs"].max())
        min_n = max(100.0, max_n * 0.01)
        filtered_parts.append(group[group["n_obs"] >= min_n])
    return pd.concat(filtered_parts, ignore_index=True)


def write_report(summary: pd.DataFrame) -> Path:
    report_path = OUTPUT_DIR / "event_trajectory_report.txt"
    lines = ["GLOBAL EVENT TRAJECTORY SUMMARY", f"event_path={EVENT_PATH}", ""]
    for event_type in FOCUSED_EVENT_TYPES:
        lines.append(f"[{event_type}]")
        ndvi_event = summary[
            (summary["event_type"] == event_type)
            & (summary["variable"] == "ndvi")
            & (summary["stage"] == "event")
        ]
        spei_pre = summary[
            (summary["event_type"] == event_type)
            & (summary["variable"] == "spei")
            & (summary["stage"] == "pre")
            & (summary["stage_month"] == -1)
        ]
        temp_event = summary[
            (summary["event_type"] == event_type)
            & (summary["variable"] == "temp")
            & (summary["stage"] == "event")
        ]
        if not ndvi_event.empty:
            base_n = float(ndvi_event["n_obs"].max())
            min_n = max(100.0, base_n * 0.01)
            ndvi_event = ndvi_event[ndvi_event["n_obs"] >= min_n]
        if not temp_event.empty:
            base_n = float(temp_event["n_obs"].max())
            min_n = max(100.0, base_n * 0.01)
            temp_event = temp_event[temp_event["n_obs"] >= min_n]
        if not ndvi_event.empty:
            min_row = ndvi_event.loc[ndvi_event["mean_value"].idxmin()]
            lines.append(
                f"ndvi_event_min_mean={min_row['mean_value']:.4f} at event_month={int(min_row['stage_month'])}, n={int(min_row['n_obs'])}"
            )
        if not spei_pre.empty:
            row = spei_pre.iloc[0]
            lines.append(f"spei_pre_month_minus1_mean={row['mean_value']:.4f}, n={int(row['n_obs'])}")
        if not temp_event.empty:
            max_row = temp_event.loc[temp_event["mean_value"].idxmax()]
            lines.append(
                f"temp_event_max_mean={max_row['mean_value']:.4f} at event_month={int(max_row['stage_month'])}, n={int(max_row['n_obs'])}"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def make_plot(summary: pd.DataFrame) -> Path:
    plot_df = summary[
        summary["event_type"].isin(FOCUSED_EVENT_TYPES)
        & summary["variable"].isin(["ndvi", "spei", "temp"])
    ].copy()

    color_map = {
        "Drought_Dominant": "#b35806",
        "Heat_Dominant": "#d7301f",
        "Compound_Overlapping": "#542788",
        "Sequential_Drought_Heat": "#1b7837",
    }
    variable_titles = {
        "ndvi": "NDVI anomaly",
        "spei": "SPEI",
        "temp": "Temperature",
    }
    stage_titles = {
        "pre": "Pre-event",
        "event": "Event",
        "post": "Post-event",
    }

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey="row")
    stage_order = ["pre", "event", "post"]
    variable_order = ["ndvi", "spei", "temp"]

    for row_idx, variable in enumerate(variable_order):
        for col_idx, stage in enumerate(stage_order):
            ax = axes[row_idx, col_idx]
            subset = plot_df[(plot_df["variable"] == variable) & (plot_df["stage"] == stage)]
            for event_type in FOCUSED_EVENT_TYPES:
                series = subset[subset["event_type"] == event_type].sort_values("stage_month")
                if series.empty:
                    continue
                ax.plot(
                    series["stage_month"],
                    series["mean_value"],
                    color=color_map[event_type],
                    linewidth=2,
                    label=event_type,
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
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Aligned trajectories of key event types", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    output_path = OUTPUT_DIR / "event_trajectory_preview.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    summary = build_summary()
    summary.to_csv(OUTPUT_DIR / "event_trajectory_summary.csv", index=False)
    focused = summary[summary["event_type"].isin(FOCUSED_EVENT_TYPES)].copy()
    focused.to_csv(OUTPUT_DIR / "event_trajectory_summary_focused.csv", index=False)
    robust = filter_robust_months(summary)
    robust.to_csv(OUTPUT_DIR / "event_trajectory_summary_robust.csv", index=False)
    robust_focused = robust[robust["event_type"].isin(FOCUSED_EVENT_TYPES)].copy()
    robust_focused.to_csv(OUTPUT_DIR / "event_trajectory_summary_focused_robust.csv", index=False)
    report_path = write_report(robust)
    plot_path = make_plot(robust)
    print(f"Saved trajectory summary to {OUTPUT_DIR / 'event_trajectory_summary.csv'}")
    print(f"Saved focused trajectory summary to {OUTPUT_DIR / 'event_trajectory_summary_focused.csv'}")
    print(f"Saved robust trajectory summary to {OUTPUT_DIR / 'event_trajectory_summary_robust.csv'}")
    print(f"Saved robust focused trajectory summary to {OUTPUT_DIR / 'event_trajectory_summary_focused_robust.csv'}")
    print(f"Saved trajectory report to {report_path}")
    print(f"Saved trajectory preview plot to {plot_path}")


if __name__ == "__main__":
    main()
