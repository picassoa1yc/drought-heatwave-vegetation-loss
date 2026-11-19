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
INPUT_BASE = (
    PROJECT_ROOT
    / "codex_stage9_20260516_core_restructure"
    / "main_figures_python"
    / "outputs"
    / "Figure5_event_dynamics_spatial_temporal_nature_revised_three_event_types"
)
LOCAL_COEFFICIENTS = (
    PROJECT_ROOT / "output_df" / "global_v05_90p_main" / "global_local_coefficients_90p_ndvi_min60.csv"
)
OUTPUT_BASE = INPUT_BASE.with_name(f"{INPUT_BASE.name}_spatial_enlarged")
SOURCE_DIR = (
    PROJECT_ROOT
    / "codex_stage9_20260516_core_restructure"
    / "main_figures_python"
    / "source_data"
)

# This row begins below the trajectory-panel x-axis labels and ends above panels d-f.
PASTE_BOX = (0, 1360, 4800, 3290)
PANEL_DPI = 300


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 15,
            "axes.linewidth": 0.7,
            "legend.frameon": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
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


def interaction_pie_values(local: pd.DataFrame) -> list[float]:
    valid = local["beta_interaction"].notna()
    negative = float((local.loc[valid, "beta_interaction"] < 0).mean() * 100.0)
    return [negative, 100.0 - negative]


def write_source_data(local: pd.DataFrame) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    out = local.copy()
    out["significant_negative"] = (
        (out["beta_interaction"] < 0) & (out["p_interaction"] < 0.05)
    )
    out["interaction_class"] = np.select(
        [
            out["significant_negative"],
            out["beta_interaction"] < 0,
            out["beta_interaction"] >= 0,
        ],
        ["significant_negative", "negative_non_significant", "non_negative"],
        default="missing",
    )
    out.to_csv(
        SOURCE_DIR / "Figure5_spatial_enlarged_local_interaction_source_data.csv",
        index=False,
    )


def make_spatial_panel(local: pd.DataFrame, size_px: tuple[int, int]) -> Image.Image:
    width_px, height_px = size_px
    fig = plt.figure(
        figsize=(width_px / PANEL_DPI, height_px / PANEL_DPI),
        dpi=PANEL_DPI,
        facecolor="white",
    )

    # The geographic panel is the hero panel and uses nearly the full row.
    ax = fig.add_axes([0.010, 0.070, 0.980, 0.900], projection=ccrs.PlateCarree())
    matrix = np.clip(grid_to_matrix(local, "beta_interaction"), -1.5, 1.5)
    cmap = LinearSegmentedColormap.from_list(
        "interaction",
        ["#2166AC", "#F7F7F7", "#B2182B"],
    )
    cmap.set_bad("#FFFFFF")
    image = ax.imshow(
        matrix,
        origin="upper",
        extent=(-180, 180, -90, 90),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5),
        interpolation="nearest",
        rasterized=True,
        zorder=1,
    )
    ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())
    try:
        ax.coastlines(resolution="110m", linewidth=0.48, color="#4D4D4D", zorder=4)
    except Exception:
        pass
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    if hasattr(ax, "outline_patch"):
        ax.outline_patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # High-contrast open symbols remain visible over both blue and pale map cells.
    significant = local[(local["beta_interaction"] < 0) & (local["p_interaction"] < 0.05)]
    ax.scatter(
        significant["lon"],
        significant["lat"],
        s=8.0,
        marker="o",
        facecolors=(1.0, 1.0, 1.0, 0.92),
        edgecolors="#111111",
        linewidths=0.38,
        transform=ccrs.PlateCarree(),
        rasterized=True,
        zorder=5,
    )

    # The sign summary sits over the southeastern Pacific, away from land.
    pie_ax = fig.add_axes([0.025, 0.005, 0.220, 0.220], facecolor="white")
    pie_values = interaction_pie_values(local)
    pie_ax.pie(
        pie_values,
        startangle=90,
        counterclock=False,
        colors=["#2166AC", "#B2182B"],
        autopct="%1.1f%%",
        pctdistance=0.48,
        wedgeprops={"edgecolor": "white", "linewidth": 1.1},
        textprops={"fontsize": 13.0, "color": "white", "fontweight": "bold"},
    )
    pie_ax.set_axis_off()

    colorbar_ax = fig.add_axes([0.340, 0.040, 0.320, 0.026], facecolor="white")
    colorbar = fig.colorbar(
        image,
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=[-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5],
    )
    colorbar.outline.set_linewidth(0.65)
    colorbar.ax.tick_params(labelsize=12, width=0.6, length=3.2, pad=2)
    colorbar.ax.set_title(
        r"Drought $\times$ heat interaction coefficient ($\beta_{D\times H}$)",
        fontsize=14.5,
        pad=5,
    )

    significance_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor="#111111",
        markeredgewidth=0.8,
        markersize=6.5,
        label=r"Significant negative interaction ($\beta_{D\times H}<0$, p < 0.05)",
    )
    legend = fig.legend(
        handles=[significance_handle],
        loc="lower right",
        bbox_to_anchor=(0.965, 0.012),
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#CFCFCF",
        fontsize=11.5,
        handlelength=1.0,
        borderpad=0.4,
    )
    legend.get_frame().set_linewidth(0.55)

    fig.canvas.draw()
    panel = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return panel


def save_outputs(composite: Image.Image) -> None:
    OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    composite.save(OUTPUT_BASE.with_suffix(".png"), dpi=(300, 300))
    composite.save(OUTPUT_BASE.with_suffix(".tiff"), compression="tiff_lzw", dpi=(300, 300))
    composite.save(OUTPUT_BASE.with_suffix(".pdf"), resolution=300)


def write_qa(local: pd.DataFrame) -> None:
    valid = local["beta_interaction"].notna()
    significant = (
        (local.loc[valid, "beta_interaction"] < 0)
        & (local.loc[valid, "p_interaction"] < 0.05)
    )
    qa = OUTPUT_BASE.with_name(f"{OUTPUT_BASE.name}_qa").with_suffix(".md")
    qa.write_text(
        "\n".join(
            [
                "# Figure 5 spatial-panel revision QA",
                "",
                "- Core conclusion: local drought x heat effects are geographically heterogeneous,",
                "  and significantly negative interactions must remain legible at final figure scale.",
                "- Layout: the spatial panel uses the full middle-row width.",
                "- Significance encoding: high-contrast white-filled, black-edged circles replace grey points.",
                "- Coefficient integrity: the original beta values and -1.5 to 1.5 color scale are unchanged.",
                f"- Valid fitted pixels: {int(valid.sum())}.",
                f"- Significant negative pixels: {int(significant.sum())} ({100 * significant.mean():.2f}%).",
                "- Panels a, b and d-f are preserved from the preceding three-event-type figure.",
                "- Exported as PNG, TIFF and PDF at 300 dpi.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    apply_style()
    input_file = INPUT_BASE.with_suffix(".tiff")
    composite = Image.open(input_file).convert("RGB")
    if composite.size != (4800, 4275):
        raise ValueError(f"Unexpected base figure size: {composite.size}")

    left, top, right, bottom = PASTE_BOX
    local = load_local_interaction()
    panel = make_spatial_panel(local, (right - left, bottom - top))
    composite.paste(panel, PASTE_BOX)
    save_outputs(composite)
    write_source_data(local)
    write_qa(local)

    for suffix in (".png", ".tiff", ".pdf"):
        print(f"Saved {OUTPUT_BASE.with_suffix(suffix)}")


if __name__ == "__main__":
    main()
