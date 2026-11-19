import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable

from fig_common import (
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


EVENT_COLUMNS = [
    ("Compound_Overlapping", "compound dry-hot events"),
    ("Drought_Dominant", "drought-only events"),
    ("Heat_Dominant", "heat-only events"),
]


def add_lonlat(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lon"] = -179.75 + out["col"].astype(float) * 0.5
    out["lat"] = 89.75 - out["row"].astype(float) * 0.5
    return out


def add_panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        0.004,
        1.02,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontfamily="Times New Roman",
        fontweight="bold",
        color=PALETTE["neutral_black"],
    )
    ax.text(
        0.055,
        1.025,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=PALETTE["neutral_black"],
    )


def remove_map_frame(ax: plt.Axes) -> None:
    if hasattr(ax, "outline_patch"):
        ax.outline_patch.set_visible(False)
    if "geo" in ax.spines:
        ax.spines["geo"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0)


def add_shared_colorbar(fig: plt.Figure, cmap: ListedColormap, norm: BoundaryNorm) -> None:
    cax = fig.add_axes([0.315, 0.030, 0.370, 0.020])
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        mappable,
        cax=cax,
        orientation="horizontal",
        boundaries=[-0.5, 0.5, 2.5, 5.5, 10.5, 25.5],
        ticks=[-0.5, 0.5, 2.5, 5.5, 10.5],
        spacing="uniform",
        drawedges=True,
    )
    cbar.ax.set_xticklabels(["0", "1", "3", "6", ">10"])
    cbar.outline.set_linewidth(0.35)
    if cbar.dividers is not None:
        cbar.dividers.set_linewidth(0.35)
        cbar.dividers.set_color("white")
    cbar.ax.tick_params(labelsize=5.5, width=0.4, length=2.0, pad=1.8)
    cbar.ax.get_xticklabels()[0].set_ha("left")
    cbar.ax.get_xticklabels()[-1].set_ha("left")
    cbar.set_label("Event count", fontsize=6.0, labelpad=1.6)


def main() -> None:
    apply_nature_style()
    cols = ["row", "col", "event_count"] + [name for name, _ in EVENT_COLUMNS]
    spatial = read_stage8_data("spatial_event_summary_by_pixel.csv", usecols=cols)
    spatial_source = add_lonlat(spatial)
    source_path = write_source(
        spatial_source,
        "Supplementary_event_pathway_count_maps_source_data.csv",
    )

    bounds = [-0.5, 0.5, 2.5, 5.5, 10.5, 25.5]
    cmap = ListedColormap(["#EEEEEE", "#FFF7BC", "#FEC44F", "#FC8D59", "#B30000"])
    cmap.set_bad("#FFFFFF")
    norm = BoundaryNorm(bounds, cmap.N, clip=True)

    fig = plt.figure(figsize=(7.35, 8.35))
    gs = fig.add_gridspec(
        3,
        1,
        left=0.025,
        right=0.985,
        top=0.975,
        bottom=0.085,
        hspace=0.075,
    )
    axes = [fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree()) for i in range(3)]

    for idx, (ax, (column, title)) in enumerate(zip(axes, EVENT_COLUMNS)):
        matrix = grid_to_matrix(spatial, column)
        im = draw_global_raster(ax, matrix, cmap, norm)
        remove_map_frame(ax)
        add_panel_label(ax, chr(ord("a") + idx), title)

    add_shared_colorbar(fig, cmap, norm)
    saved = save_figure(fig, OUTPUT_DIR / "Supplementary_event_pathway_count_maps")
    plt.close(fig)
    print_saved(saved, [source_path])


if __name__ == "__main__":
    main()
