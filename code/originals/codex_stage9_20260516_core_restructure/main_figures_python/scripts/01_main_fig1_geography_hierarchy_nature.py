from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from fig_common import (
    EVENT_LABELS,
    EVENT_ORDER,
    OUTPUT_DIR,
    PALETTE,
    apply_nature_style,
    draw_global_raster,
    grid_to_matrix,
    print_saved,
    read_stage8_data,
    save_figure,
    write_source,
)


def add_lonlat(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lon"] = -179.75 + out["col"].astype(float) * 0.5
    out["lat"] = 89.75 - out["row"].astype(float) * 0.5
    return out


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontfamily="Times New Roman",
        fontweight="bold",
        color=PALETTE["neutral_black"],
    )


def remove_map_frame(ax: plt.Axes) -> None:
    if hasattr(ax, "outline_patch"):
        ax.outline_patch.set_visible(False)
    if "geo" in ax.spines:
        ax.spines["geo"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0)


def block_colorbar(fig: plt.Figure, ax: plt.Axes, mappable, label: str, boundaries: list[float], tick_labels: list[str]) -> None:
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="horizontal",
        fraction=0.052,
        pad=0.035,
        boundaries=boundaries,
        ticks=boundaries,
        spacing="uniform",
        drawedges=True,
    )
    cbar.ax.set_xticklabels(tick_labels)
    cbar.outline.set_linewidth(0.35)
    if cbar.dividers is not None:
        cbar.dividers.set_linewidth(0.35)
        cbar.dividers.set_color("white")
    cbar.ax.tick_params(labelsize=4.8, width=0.4, length=2.0, pad=1.4)
    labels = cbar.ax.get_xticklabels()
    if labels:
        labels[0].set_ha("left")
        labels[-1].set_ha("right")
    cbar.set_label(label, fontsize=5.8, labelpad=1.5)


def plot_event_hierarchy(ax: plt.Axes, event_summary: pd.DataFrame) -> None:
    df = event_summary[event_summary["event_type"].isin(EVENT_ORDER)].copy()
    df["event_type"] = pd.Categorical(df["event_type"], categories=EVENT_ORDER, ordered=True)
    df = df.sort_values("event_type")
    compact_labels = {
        **EVENT_LABELS,
        "Sequential_Heat_Drought": "Heat-drought",
        "Sequential_Drought_Heat": "Drought-heat",
    }
    df["event_label"] = df["event_type"].map(compact_labels)

    y = np.arange(len(df))[::-1]
    ax.barh(
        y,
        df["losstot_mean"],
        height=0.62,
        color="#FFFFFF",
        edgecolor="#BEBEBE",
        linewidth=0.45,
    )
    cap_half_height = 0.13
    for yi, (_, row) in zip(y, df.iterrows()):
        ax.hlines(yi, row["losstot_q25"], row["losstot_q75"], color=PALETTE["neutral_black"], lw=0.9)
        ax.vlines(
            [row["losstot_q25"], row["losstot_q75"]],
            yi - cap_half_height,
            yi + cap_half_height,
            color=PALETTE["neutral_black"],
            lw=0.9,
        )
        ax.plot(row["losstot_median"], yi, "o", ms=3.0, color="white", mec=PALETTE["neutral_black"], mew=0.35)
        label_x = max(row["losstot_mean"], row["losstot_q75"]) + 0.016
        ax.text(label_x, yi + 0.16, f"{row['losstot_mean']:.3f}", ha="left", va="bottom", fontsize=5.6)
        ax.text(label_x, yi - 0.20, f"n={int(row['n_events']):,}", ha="left", va="bottom", fontsize=5.1, color=PALETTE["neutral_dark"])

    ax.set_yticks(y)
    ax.set_yticklabels(df["event_label"])
    ax.set_xlabel("Mean LOSStot")
    ax.set_xlim(0, 0.505)
    ax.set_ylim(-0.60, len(df) - 0.18)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.tick_params(axis="y", length=0)
    add_panel_label(ax, "d", x=-0.07, y=1.02)


def main() -> None:
    apply_nature_style()

    spatial_cols = [
        "row",
        "col",
        "event_count",
        "mean_losstot",
        "delta_losstot_compound_minus_drought",
        "drought_count",
        "compound_count",
    ]
    spatial = read_stage8_data("spatial_event_summary_by_pixel.csv", usecols=spatial_cols)
    event_summary = read_stage8_data("event_summary_by_type.csv")

    spatial_source = add_lonlat(spatial)
    spatial_source["local_contrast_estimable"] = spatial_source["delta_losstot_compound_minus_drought"].notna()
    spatial_source["not_estimable_reason"] = np.where(
        spatial_source["local_contrast_estimable"],
        "",
        "fewer than 3 compound or drought-dominant events",
    )
    source_map = write_source(spatial_source, "Figure1_geography_hierarchy_nature_spatial_source_data.csv")
    source_summary = write_source(
        event_summary[event_summary["event_type"].isin(EVENT_ORDER)].copy(),
        "Figure1_geography_hierarchy_nature_event_summary_source_data.csv",
    )

    event_count = grid_to_matrix(spatial, "event_count")
    mean_loss = np.clip(grid_to_matrix(spatial, "mean_losstot"), 0, 0.75)
    delta_raw = grid_to_matrix(spatial, "delta_losstot_compound_minus_drought")
    delta_loss = np.clip(delta_raw, -0.35, 0.35)
    not_estimable = np.where(np.isfinite(event_count) & ~np.isfinite(delta_raw), 1.0, np.nan)

    freq_bounds = [0.5, 4.5, 10.5, 16.5, 20.5, 32.5]
    loss_bounds = [0, 0.05, 0.10, 0.20, 0.35, 0.75]
    delta_bounds = [-0.35, -0.15, -0.05, 0.05, 0.15, 0.35]
    cmap_freq = ListedColormap(["#FFF7BC", "#FEC44F", "#FC8D59", "#E34A33", "#B30000"])
    cmap_loss = ListedColormap(["#D9F0D3", "#ADDD8E", "#FFFFBF", "#FDAE61", "#B64342"])
    cmap_delta = ListedColormap(["#2166AC", "#67A9CF", "#F6E6A8", "#EF8A62", "#B2182B"])
    cmap_not_estimable = ListedColormap(["#D6D6D6"])
    for cmap in [cmap_freq, cmap_loss]:
        cmap.set_bad("#FFFFFF")
    cmap_delta.set_bad((1, 1, 1, 0))
    cmap_not_estimable.set_bad((1, 1, 1, 0))
    norm_freq = BoundaryNorm(freq_bounds, cmap_freq.N, clip=True)
    norm_loss = BoundaryNorm(loss_bounds, cmap_loss.N, clip=True)
    norm_delta = BoundaryNorm(delta_bounds, cmap_delta.N, clip=True)

    fig = plt.figure(figsize=(9.45, 5.55))
    ax_a = fig.add_axes([0.016, 0.585, 0.474, 0.362], projection=ccrs.PlateCarree())
    ax_b = fig.add_axes([0.508, 0.585, 0.474, 0.362], projection=ccrs.PlateCarree())
    ax_c = fig.add_axes([0.016, 0.105, 0.474, 0.362], projection=ccrs.PlateCarree())
    ax_d = fig.add_axes([0.604, 0.125, 0.368, 0.328])

    im_a = draw_global_raster(ax_a, event_count, cmap_freq, norm_freq)
    remove_map_frame(ax_a)
    add_panel_label(ax_a, "a", x=0.005, y=1.02)
    block_colorbar(fig, ax_a, im_a, "Event count", freq_bounds, ["1", "4", "10", "16", "20", ">=21"])

    im_b = draw_global_raster(ax_b, mean_loss, cmap_loss, norm_loss)
    remove_map_frame(ax_b)
    add_panel_label(ax_b, "b", x=0.005, y=1.02)
    block_colorbar(fig, ax_b, im_b, "Mean LOSStot", loss_bounds, ["0", "0.05", "0.10", "0.20", "0.35", ">=0.35"])

    draw_global_raster(ax_c, not_estimable, cmap_not_estimable, BoundaryNorm([0.5, 1.5], cmap_not_estimable.N))
    im_c = draw_global_raster(ax_c, delta_loss, cmap_delta, norm_delta)
    remove_map_frame(ax_c)
    add_panel_label(ax_c, "c", x=0.005, y=1.02)
    legend = ax_c.legend(
        handles=[Patch(facecolor="#D6D6D6", edgecolor="none", label="Not estimable")],
        loc="lower left",
        bbox_to_anchor=(0.010, 0.030),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor=PALETTE["neutral_light"],
        fontsize=5.4,
        borderpad=0.28,
        handlelength=1.2,
    )
    legend.get_frame().set_linewidth(0.35)
    block_colorbar(
        fig,
        ax_c,
        im_c,
        "Compound - drought LOSStot",
        delta_bounds,
        ["-0.35", "-0.15", "-0.05", "0.05", "0.15", "0.35"],
    )

    plot_event_hierarchy(ax_d, event_summary)

    saved = save_figure(fig, OUTPUT_DIR / "Figure1_geography_hierarchy_nature")
    plt.close(fig)
    print_saved(saved, [source_map, source_summary])


if __name__ == "__main__":
    main()
