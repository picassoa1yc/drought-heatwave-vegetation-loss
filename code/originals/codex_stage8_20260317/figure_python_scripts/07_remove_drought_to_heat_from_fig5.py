from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image


PROJECT_ROOT = Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve()
INPUT_BASE = (
    PROJECT_ROOT
    / "codex_stage9_20260516_core_restructure"
    / "main_figures_python"
    / "outputs"
    / "Figure5_event_dynamics_spatial_temporal_nature_revised"
)
OUTPUT_BASE = INPUT_BASE.with_name(f"{INPUT_BASE.name}_three_event_types")
SOURCE_DATA = (
    PROJECT_ROOT
    / "codex_stage8_20260317"
    / "figure_r_package"
    / "data"
    / "functional_trajectory_summary_stage2_robust.csv"
)
SOURCE_DIR = (
    PROJECT_ROOT
    / "codex_stage9_20260516_core_restructure"
    / "main_figures_python"
    / "source_data"
)

PANEL_B_PASTE_BOX = (1240, 0, 4800, 1360)
PANEL_DPI = 300

EVENT_ORDER = ["Compound_Overlapping", "Drought_Dominant", "Heat_Dominant"]
EVENT_LABELS = {
    "Compound_Overlapping": "Compound",
    "Drought_Dominant": "Drought",
    "Heat_Dominant": "Heat",
}
EVENT_COLORS = {
    "Compound_Overlapping": "#542788",
    "Drought_Dominant": "#B35806",
    "Heat_Dominant": "#D7301F",
}
VARIABLE_ORDER = ["et", "gpp", "sif"]
VARIABLE_LABELS = {
    "et": "ET anomaly",
    "gpp": "GPP anomaly",
    "sif": "SIF anomaly",
}
STAGE_BACKGROUNDS = [
    (-3.5, -0.5, "#E9F0F7"),
    (-0.5, 5.5, "#FCE8DC"),
    (5.5, 8.5, "#EEF4E3"),
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 9.5,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def prepare_panel_b_data() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_DATA)
    frame["stage_month"] = pd.to_numeric(frame["stage_month"], errors="coerce")
    frame["mean_value"] = pd.to_numeric(frame["mean_value"], errors="coerce")
    frame["n_obs"] = pd.to_numeric(frame["n_obs"], errors="coerce")
    frame = frame[
        frame["event_type"].isin(EVENT_ORDER)
        & frame["variable"].isin(VARIABLE_ORDER)
        & (
            ((frame["stage"] == "pre") & (frame["stage_month"] <= 3))
            | ((frame["stage"] == "event") & (frame["stage_month"] <= 6))
            | ((frame["stage"] == "post") & (frame["stage_month"] <= 3))
        )
    ].copy()
    frame["relative_month"] = np.select(
        [
            frame["stage"].eq("pre"),
            frame["stage"].eq("event"),
            frame["stage"].eq("post"),
        ],
        [
            frame["stage_month"],
            frame["stage_month"] - 1,
            6 + frame["stage_month"] - 1,
        ],
        default=np.nan,
    )
    frame["event_type_label"] = frame["event_type"].map(EVENT_LABELS)
    frame["variable_label"] = frame["variable"].map(VARIABLE_LABELS)
    return frame.sort_values(["variable", "event_type", "relative_month"])


def format_panel_axis(ax: plt.Axes, variable: str) -> None:
    for xmin, xmax, color in STAGE_BACKGROUNDS:
        ax.axvspan(xmin, xmax, color=color, alpha=0.25, linewidth=0, zorder=0)
    ax.axhline(0, color="#D9D9D9", lw=0.8, zorder=1)
    ax.axvline(0, color="#7A7A7A", lw=0.75, ls=(0, (4, 4)), zorder=1)
    ax.set_xlim(-3.5, 8.5)
    ax.set_xticks([-3, -2, -1, 0, 1, 3, 5, 7, 8])
    ax.grid(axis="y", color="#F0F0F0", lw=0.55, zorder=0)
    ax.tick_params(labelsize=8.8)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
        label.set_color("black")
    ax.set_title(
        VARIABLE_LABELS[variable],
        loc="left",
        fontsize=10.8,
        fontweight="bold",
        pad=7,
    )

    if variable == "et":
        ax.set_ylim(-5.15, 1.90)
        ax.set_yticks([-4, -2, 0])
    elif variable == "gpp":
        ax.set_ylim(-19.5, 3.65)
        ax.set_yticks([-15, -10, -5, 0])
    else:
        ax.set_ylim(-0.0275, 0.0060)
        ax.set_yticks([-0.02, -0.01, 0.00])
        ax.set_yticklabels(["-0.02", "-0.01", "0.00"])


def render_panel_b(frame: pd.DataFrame, size_px: tuple[int, int]) -> Image.Image:
    width_px, height_px = size_px
    fig = plt.figure(
        figsize=(width_px / PANEL_DPI, height_px / PANEL_DPI),
        dpi=PANEL_DPI,
        facecolor="white",
    )

    axes = [
        fig.add_axes([0.030, 0.156, 0.281, 0.692]),
        fig.add_axes([0.360, 0.156, 0.281, 0.692]),
        fig.add_axes([0.701, 0.156, 0.281, 0.692]),
    ]
    for ax, variable in zip(axes, VARIABLE_ORDER):
        format_panel_axis(ax, variable)
        subset = frame[frame["variable"].eq(variable)]
        for event_type in EVENT_ORDER:
            line = subset[subset["event_type"].eq(event_type)]
            ax.plot(
                line["relative_month"],
                line["mean_value"],
                color=EVENT_COLORS[event_type],
                lw=1.8,
                alpha=0.9,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3,
            )

    handles = [
        Line2D(
            [0],
            [0],
            color=EVENT_COLORS[event_type],
            lw=2.2,
            solid_capstyle="round",
            label=EVENT_LABELS[event_type],
        )
        for event_type in EVENT_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.975),
        ncol=3,
        frameon=False,
        fontsize=10.2,
        handlelength=1.25,
        handletextpad=0.55,
        columnspacing=1.05,
    )
    fig.text(
        0.020,
        0.965,
        "(b)",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        family="serif",
    )
    fig.text(
        0.515,
        0.065,
        "Relative event month (0 = event onset)",
        ha="center",
        va="center",
        fontsize=11.2,
    )

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=PANEL_DPI, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def write_source_data(frame: pd.DataFrame) -> Path:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    output = SOURCE_DIR / "Figure5_panel_b_three_event_types_source_data.csv"
    frame.to_csv(output, index=False)
    return output


def save_outputs(image: Image.Image) -> list[Path]:
    outputs = [
        OUTPUT_BASE.with_suffix(".png"),
        OUTPUT_BASE.with_suffix(".tiff"),
        OUTPUT_BASE.with_suffix(".pdf"),
    ]
    image.save(outputs[0], dpi=(300, 300))
    image.save(outputs[1], compression="tiff_lzw", dpi=(300, 300))
    image.save(outputs[2], resolution=300)
    return outputs


def main() -> None:
    apply_style()
    base = Image.open(INPUT_BASE.with_suffix(".tiff")).convert("RGB")
    if base.size != (4800, 4275):
        raise ValueError(f"Unexpected Figure 5 size: {base.size}")

    frame = prepare_panel_b_data()
    left, top, right, bottom = PANEL_B_PASTE_BOX
    panel_b = render_panel_b(frame, (right - left, bottom - top))
    base.paste(panel_b, (left, top))

    outputs = save_outputs(base)
    source_data = write_source_data(frame)
    qa = """# Figure 5 panel-b event-type revision

- Panels a and b now contain the same event types: Compound, Drought and Heat.
- Panel b was redrawn from the original robust functional-trajectory source table after excluding Sequential_Drought_Heat.
- The R workflow's month mapping, stage backgrounds, event colors and free-y ranges were retained.
- Panels a and c-f were preserved from the accepted composite without changes.
"""
    qa_path = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.name}_qa.md")
    qa_path.write_text(qa, encoding="utf-8")

    for path in [*outputs, source_data, qa_path]:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
