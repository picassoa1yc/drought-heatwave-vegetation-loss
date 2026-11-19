from __future__ import annotations

import os
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject


PROJECT_ROOT = Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve()
STAGE8_ROOT = PROJECT_ROOT / "codex_stage8_20260317"
STAGE9_ROOT = PROJECT_ROOT / "codex_stage9_20260516_core_restructure"

ORIGINAL_PDF = STAGE8_ROOT / "figure_r_package" / "outputs" / "Figure4_dynamics_selectivity_R.pdf"
TEMPORAL_TABLE = (
    STAGE9_ROOT
    / "supplementary_tables_package"
    / "tables"
    / "Supplementary_Table_S7_temporal_diagnostics.csv"
)
OUTPUT_DIR = STAGE8_ROOT / "figure_r_package" / "outputs"
SOURCE_DIR = STAGE8_ROOT / "figure_python_scripts" / "source_data"

BOTTOM_PDF = OUTPUT_DIR / "Figure4_sliding_windows_bottom_strip.pdf"
COMBINED_PDF = OUTPUT_DIR / "Figure4_dynamics_selectivity_R_with_sliding_windows.pdf"
COMBINED_PNG = OUTPUT_DIR / "Figure4_dynamics_selectivity_R_with_sliding_windows.png"
COMBINED_TIFF = OUTPUT_DIR / "Figure4_dynamics_selectivity_R_with_sliding_windows.tiff"
SOURCE_CSV = SOURCE_DIR / "Figure4_dynamics_selectivity_R_with_sliding_windows_bottom_source.csv"

OUTCOMES = ["LOSStot", "ndvi_delta_event_minus_pre", "gpp_delta_event_minus_pre"]
OUTCOME_LABEL = {
    "LOSStot": "LOSStot",
    "ndvi_delta_event_minus_pre": "NDVI",
    "gpp_delta_event_minus_pre": "GPP",
}
OUTCOME_COLOR = {
    "LOSStot": "#B64342",
    "ndvi_delta_event_minus_pre": "#42949E",
    "gpp_delta_event_minus_pre": "#0F4D92",
}
PANEL_LABELS = ["d", "e", "f"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12.2,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
        }
    )


def midpoint(df: pd.DataFrame) -> pd.Series:
    return (df["window_start"].astype(float) + df["window_end"].astype(float)) / 2.0


def set_zoomed_ylim(ax: plt.Axes, *series: pd.Series | np.ndarray) -> None:
    values = np.concatenate([np.asarray(s, dtype=float) for s in series])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    ymin = float(values.min())
    ymax = float(values.max())
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0) * 0.10
    pad = span * 0.18
    ax.set_ylim(ymin - pad, ymax + pad)


def read_temporal_data() -> pd.DataFrame:
    temporal = pd.read_csv(TEMPORAL_TABLE)
    temporal = temporal[
        (temporal["panel"] == "C_sliding_windows")
        & temporal["outcome"].isin(OUTCOMES)
        & temporal["analysis_type"].isin(["sliding_8yr", "sliding_10yr"])
    ].copy()
    return temporal


def make_bottom_strip(width_pt: float, height_in: float = 3.25) -> pd.DataFrame:
    apply_style()
    temporal = read_temporal_data()
    width_in = width_pt / 72.0
    fig, axes = plt.subplots(1, 3, figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.070, right=0.985, top=0.80, bottom=0.22, wspace=0.30)

    for i, (ax, outcome, panel_label) in enumerate(zip(axes, OUTCOMES, PANEL_LABELS)):
        s8 = temporal[
            (temporal["outcome"] == outcome) & (temporal["analysis_type"] == "sliding_8yr")
        ].copy()
        s10 = temporal[
            (temporal["outcome"] == outcome) & (temporal["analysis_type"] == "sliding_10yr")
        ].copy()
        color = OUTCOME_COLOR[outcome]
        x8 = midpoint(s8).to_numpy(dtype=float)
        ax.axhline(0, color="#777777", lw=0.6, zorder=0)
        ax.fill_between(
            x8,
            s8["damage_ci_low"].to_numpy(dtype=float),
            s8["damage_ci_high"].to_numpy(dtype=float),
            color=color,
            alpha=0.17,
            linewidth=0,
        )
        ax.plot(x8, s8["damage_att"], color=color, lw=1.45, marker="o", ms=3.6)
        ax.plot(midpoint(s10), s10["damage_att"], color="#4D4D4D", lw=1.25, ls="--")
        set_zoomed_ylim(
            ax,
            s8["damage_att"],
            s8["damage_ci_low"],
            s8["damage_ci_high"],
            s10["damage_att"],
        )
        ax.text(
            -0.17,
            1.10,
            f"({panel_label})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=18,
            fontfamily="Times New Roman",
            fontweight="bold",
            color="#272727",
        )
        ax.set_title(OUTCOME_LABEL[outcome], loc="left", fontweight="bold", pad=2)
        ax.set_xlim(2004, 2021)
        ax.set_xticks([2005, 2010, 2015, 2020])
        ax.set_xlabel("Year")
        ax.set_ylabel("ATT")
        ax.grid(axis="y", color="#EFEFEF", lw=0.45)
        if i == 0:
            handles = [
                Line2D([0], [0], color="#272727", marker="o", lw=1.45, ms=3.6, label="8-year"),
                Line2D([0], [0], color="#4D4D4D", lw=1.25, ls="--", label="10-year"),
            ]
            ax.legend(handles=handles, loc="upper left", borderaxespad=0.0, handlelength=1.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(BOTTOM_PDF, facecolor="white")
    plt.close(fig)
    return temporal


def append_pdf(bottom_height_pt: float) -> None:
    reader = PdfReader(str(ORIGINAL_PDF))
    bottom_reader = PdfReader(str(BOTTOM_PDF))
    original_page = reader.pages[0]
    bottom_page = bottom_reader.pages[0]

    width = float(original_page.mediabox.width)
    original_height = float(original_page.mediabox.height)
    total_height = original_height + bottom_height_pt

    new_page = PageObject.create_blank_page(width=width, height=total_height)
    new_page.merge_page(bottom_page)
    new_page.merge_transformed_page(
        original_page,
        Transformation().translate(tx=0, ty=bottom_height_pt),
        expand=False,
    )

    writer = PdfWriter()
    writer.add_page(new_page)
    with COMBINED_PDF.open("wb") as f:
        writer.write(f)


def add_upper_panel_labels() -> None:
    doc = fitz.open(COMBINED_PDF)
    page = doc[0]
    labels = [
        ("(a)", 30, 30),
        ("(b)", 315, 30),
        ("(c)", 38, 315),
    ]
    for text, x, y in labels:
        page.insert_text(
            fitz.Point(x, y),
            text,
            fontsize=20,
            fontname="Times-Bold",
            color=(0.15, 0.15, 0.15),
        )
    doc.saveIncr()
    doc.close()


def render_preview() -> None:
    doc = fitz.open(COMBINED_PDF)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    pix.save(COMBINED_PNG)
    pix_tiff = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False)
    pix_tiff.pil_save(COMBINED_TIFF, dpi=(300, 300), compression="tiff_lzw")
    doc.close()


def main() -> None:
    original_reader = PdfReader(str(ORIGINAL_PDF))
    width_pt = float(original_reader.pages[0].mediabox.width)
    bottom_height_in = 3.25
    bottom_height_pt = bottom_height_in * 72.0

    temporal = make_bottom_strip(width_pt=width_pt, height_in=bottom_height_in)
    append_pdf(bottom_height_pt=bottom_height_pt)
    add_upper_panel_labels()
    render_preview()

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    temporal.to_csv(SOURCE_CSV, index=False)

    print("Saved:")
    print(COMBINED_PDF)
    print(COMBINED_PNG)
    print(COMBINED_TIFF)
    print(BOTTOM_PDF)
    print("Source data:")
    print(SOURCE_CSV)


if __name__ == "__main__":
    main()
