from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image


PROJECT_ROOT = Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve()
STAGE8_ROOT = PROJECT_ROOT / "codex_stage8_20260317"
BASE_FIGURE = (
    STAGE8_ROOT
    / "figure_r_package"
    / "outputs"
    / "Figure4_dynamics_selectivity_R_with_sliding_windows.tiff"
)
LOCAL_COEFFICIENTS = (
    PROJECT_ROOT / "output_df" / "global_v05_90p_main" / "global_local_coefficients_90p_ndvi_min60.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "codex_stage9_20260516_core_restructure"
    / "main_figures_python"
    / "outputs"
)
SOURCE_DIR = (
    PROJECT_ROOT
    / "codex_stage9_20260516_core_restructure"
    / "main_figures_python"
    / "source_data"
)
OUT_BASE = OUTPUT_DIR / "Figure5_event_dynamics_spatial_temporal_nature_revised"

# Start below the trajectory-panel x-axis labels so panel c cannot cover them.
PASTE_BOX = (0, 1360, 4800, 3290)
PANEL_DPI = 300


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 16,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def add_lonlat(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lon"] = -179.75 + out["col"].astype(float) * 0.5
    out["lat"] = 89.75 - out["row"].astype(float) * 0.5
    return out


def grid_to_matrix(df: pd.DataFrame, value_col: str) -> np.ndarray:
    matrix = np.full((360, 720), np.nan, dtype=float)
    rows = df["row"].to_numpy(dtype=int)
    cols = df["col"].to_numpy(dtype=int)
    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    ok = (rows >= 0) & (rows < 360) & (cols >= 0) & (cols < 720) & np.isfinite(vals)
    matrix[rows[ok], cols[ok]] = vals[ok]
    return matrix


def load_local_interaction() -> pd.DataFrame:
    local = pd.read_csv(
        LOCAL_COEFFICIENTS,
        usecols=["row", "col", "beta_interaction", "p_interaction", "heat_amplifies_drought_sig"],
    )
    local["beta_interaction"] = pd.to_numeric(local["beta_interaction"], errors="coerce")
    local["p_interaction"] = pd.to_numeric(local["p_interaction"], errors="coerce")
    local["heat_amplifies_drought_sig"] = (
        local["heat_amplifies_drought_sig"].astype(str).str.lower().eq("true")
    )
    return add_lonlat(local)


def write_c_source(local: pd.DataFrame) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    out = local.copy()
    valid = out["beta_interaction"].notna()
    out["interaction_class"] = np.select(
        [
            (out["beta_interaction"] < 0) & (out["p_interaction"] < 0.05),
            out["beta_interaction"] < 0,
            out["beta_interaction"] >= 0,
        ],
        ["significant_negative", "negative_non_significant", "non_negative"],
        default="missing",
    )
    pct = (
        out.loc[valid, "interaction_class"]
        .value_counts(normalize=True)
        .rename_axis("interaction_class")
        .reset_index(name="fraction")
    )
    pct["percent"] = pct["fraction"] * 100
    sign_values, _ = interaction_pie_values(local)
    sign_pct = pd.DataFrame(
        {
            "interaction_sign": ["negative_beta", "non_negative_beta"],
            "percent": sign_values,
        }
    )
    out.to_csv(SOURCE_DIR / "Figure4_R_tracks_new_panel_c_local_interaction.csv", index=False)
    pct.to_csv(SOURCE_DIR / "Figure4_R_tracks_new_panel_c_interaction_class_percent.csv", index=False)
    sign_pct.to_csv(SOURCE_DIR / "Figure4_R_tracks_new_panel_c_sign_percent.csv", index=False)


def interaction_pie_values(local: pd.DataFrame) -> tuple[list[float], list[str]]:
    valid = local["beta_interaction"].notna()
    neg = float((local.loc[valid, "beta_interaction"] < 0).mean() * 100.0)
    non_neg = 100.0 - neg
    return [neg, non_neg], [f"Negative {neg:.1f}%", f"Positive {non_neg:.1f}%"]


def make_panel_c(local: pd.DataFrame, size_px: tuple[int, int]) -> Image.Image:
    width_px, height_px = size_px
    fig_w = width_px / PANEL_DPI
    fig_h = height_px / PANEL_DPI
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=PANEL_DPI)
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([0.045, 0.235, 0.91, 0.635], projection=ccrs.PlateCarree())
    matrix = np.clip(grid_to_matrix(local, "beta_interaction"), -1.5, 1.5)
    cmap = LinearSegmentedColormap.from_list("interaction", ["#2166AC", "#F7F7F7", "#B2182B"])
    cmap.set_bad("#FFFFFF")
    im = ax.imshow(
        matrix,
        origin="upper",
        extent=(-180, 180, -90, 90),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5),
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())
    try:
        ax.coastlines(resolution="110m", linewidth=0.42, color="#4D4D4D")
    except Exception:
        pass
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    if hasattr(ax, "outline_patch"):
        ax.outline_patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    sig = local[(local["beta_interaction"] < 0) & (local["p_interaction"] < 0.05)].copy()
    ax.scatter(
        sig["lon"],
        sig["lat"],
        s=3.0,
        marker="o",
        c="#333333",
        alpha=0.52,
        transform=ccrs.PlateCarree(),
        rasterized=True,
        linewidths=0,
        zorder=5,
    )
    pie_ax = fig.add_axes([0.105, 0.060, 0.185, 0.175])
    pie_ax.set_aspect("equal")
    pie_values, _ = interaction_pie_values(local)
    pie_ax.pie(
        pie_values,
        startangle=90,
        counterclock=False,
        colors=["#2166AC", "#B2182B"],
        autopct="%1.1f%%",
        pctdistance=0.54,
        wedgeprops={"edgecolor": "white", "linewidth": 1.1},
        textprops={"fontsize": 12.0, "color": "white", "fontweight": "bold"},
    )
    pie_ax.set_axis_off()

    sig_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="#333333",
        markeredgewidth=0,
        markersize=5.0,
        alpha=0.58,
        label=r"Significant negative interaction ($\beta_{D\times H}<0$, p < 0.05)",
    )
    legend = fig.legend(
        handles=[sig_handle],
        loc="lower right",
        bbox_to_anchor=(0.915, 0.120),
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#D8D8D8",
        fontsize=12,
        handlelength=1.0,
        borderpad=0.35,
    )
    legend.get_frame().set_linewidth(0.5)

    cax = fig.add_axes([0.365, 0.100, 0.270, 0.027])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=12, width=0.55, length=3.0, pad=2)
    cbar.ax.set_title(r"Drought x heat interaction coefficient ($\beta_{D\times H}$)", fontsize=15, pad=7)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    panel = Image.fromarray(rgba).convert("RGB")
    plt.close(fig)
    return panel


def save_outputs(composite: Image.Image) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    composite.save(OUT_BASE.with_suffix(".png"), dpi=(300, 300))
    composite.save(OUT_BASE.with_suffix(".tiff"), compression="tiff_lzw", dpi=(300, 300))
    composite.save(OUT_BASE.with_suffix(".pdf"), resolution=300)


def main() -> None:
    apply_style()
    base = Image.open(BASE_FIGURE).convert("RGB")
    if base.size != (4800, 4275):
        raise ValueError(f"Unexpected base figure size: {base.size}")
    left, top, right, bottom = PASTE_BOX
    local = load_local_interaction()
    panel_c = make_panel_c(local, (right - left, bottom - top))
    base.paste(panel_c, PASTE_BOX)
    save_outputs(base)
    write_c_source(local)
    print(f"Saved {OUT_BASE.with_suffix('.png')}")
    print(f"Saved {OUT_BASE.with_suffix('.tiff')}")
    print(f"Saved {OUT_BASE.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
