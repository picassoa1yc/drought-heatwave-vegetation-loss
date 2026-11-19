from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
STAGE9_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = STAGE9_ROOT.parent
OUTPUT_DIR = PACKAGE_ROOT / "outputs"
SUPP_TABLE_DIR = STAGE9_ROOT / "supplementary_tables_package" / "tables"
STAGE8_DATA_DIR = PROJECT_ROOT / "codex_stage8_20260317" / "figure_r_package" / "data"

PALETTE = {
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "compound": "#D98EA6",
    "compound_soft": "#E4CCD8",
    "neutral_black": "#272727",
    "neutral_dark": "#4D4D4D",
    "neutral_mid": "#767676",
    "neutral_light": "#D8D8D8",
    "signal_blue": "#0F4D92",
    "signal_teal": "#42949E",
    "accent_red": "#B64342",
    "accent_orange": "#E28E2C",
}

LANDCOVER_ORDER = ["Forest", "Shrub", "Savannas", "Grassland", "Cropland"]
EVENT_ORDER = [
    "Mild_Reference",
    "Heat_Dominant",
    "Drought_Dominant",
    "Sequential_Heat_Drought",
    "Sequential_Drought_Heat",
    "Compound_Overlapping",
]
EVENT_LABELS = {
    "Mild_Reference": "Mild",
    "Heat_Dominant": "Heat",
    "Drought_Dominant": "Drought",
    "Sequential_Heat_Drought": "Heat -> drought",
    "Sequential_Drought_Heat": "Drought -> heat",
    "Compound_Overlapping": "Compound",
}
OUTCOME_LABELS = {
    "LOSStot": "Total damage",
    "DECmax": "Peak decline",
    "TIMErec_true": "Recovery time",
    "TIMErec": "Recovery time",
    "ndvi_delta_event_minus_pre": "NDVI decline",
    "gpp_delta_event_minus_pre": "GPP decline",
}


def apply_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
        }
    )


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=PALETTE["neutral_black"],
    )


def save_figure(fig: plt.Figure, base: Path, dpi_tiff: int = 600, dpi_png: int = 300) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".tiff", {"dpi": dpi_tiff}),
        (".png", {"dpi": dpi_png}),
    ]:
        path = base.with_suffix(suffix)
        fig.savefig(path, facecolor="white", **kwargs)
        saved.append(path)
    return saved


def read_supp_table(name: str) -> pd.DataFrame:
    return pd.read_csv(SUPP_TABLE_DIR / name)


def read_stage8_data(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(STAGE8_DATA_DIR / name, **kwargs)


def ordered_categories(values: pd.Series, order: list[str]) -> pd.Categorical:
    return pd.Categorical(values, categories=order, ordered=True)


def grid_to_matrix(df: pd.DataFrame, value_col: str, nrows: int = 360, ncols: int = 720) -> np.ndarray:
    matrix = np.full((nrows, ncols), np.nan, dtype=float)
    rows = df["row"].to_numpy(dtype=int)
    cols = df["col"].to_numpy(dtype=int)
    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    ok = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols) & np.isfinite(vals)
    matrix[rows[ok], cols[ok]] = vals[ok]
    return matrix


def style_geo_axis(ax: plt.Axes) -> None:
    import cartopy.crs as ccrs

    ax.set_facecolor("white")
    ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())
    ax.coastlines(resolution="110m", linewidth=0.25, color=PALETTE["neutral_dark"])
    ax.set_anchor("N")
    ax.set_xticks([])
    ax.set_yticks([])
    if hasattr(ax, "outline_patch"):
        ax.outline_patch.set_linewidth(0.35)
        ax.outline_patch.set_edgecolor(PALETTE["neutral_light"])
    elif "geo" in ax.spines:
        ax.spines["geo"].set_linewidth(0.35)
        ax.spines["geo"].set_edgecolor(PALETTE["neutral_light"])


def draw_global_raster(ax: plt.Axes, matrix: np.ndarray, cmap, norm=None):
    import cartopy.crs as ccrs

    im = ax.imshow(
        matrix,
        origin="upper",
        extent=(-180, 180, -90, 90),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        rasterized=True,
    )
    style_geo_axis(ax)
    return im


def compact_colorbar(fig: plt.Figure, ax: plt.Axes, mappable, label: str, ticks=None) -> None:
    cbar = fig.colorbar(mappable, ax=ax, orientation="horizontal", fraction=0.052, pad=0.035, ticks=ticks)
    cbar.outline.set_linewidth(0.35)
    cbar.ax.tick_params(labelsize=5.4, width=0.4, length=2)
    cbar.set_label(label, fontsize=5.8, labelpad=1.5)


def write_source(df: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def print_saved(saved: list[Path], source_paths: list[Path] | None = None) -> None:
    print("Saved:")
    for path in saved:
        print(path)
    if source_paths:
        for path in source_paths:
            print(path)
