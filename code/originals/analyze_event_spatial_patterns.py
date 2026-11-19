from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm


BASE_DIR = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/global_v05_90p_main")
EVENT_PATH = BASE_DIR / "global_integrated_events_90p.csv"
SPATIAL_DIR = BASE_DIR / "spatial_event_maps"
SPATIAL_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_PATH = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/landcover_2001-2024_modal.tif")

EVENT_TYPE_CODE = {
    "Mild_Reference": 1,
    "Drought_Dominant": 2,
    "Heat_Dominant": 3,
    "Compound_Overlapping": 4,
    "Sequential_Drought_Heat": 5,
    "Sequential_Heat_Drought": 6,
}
EVENT_TYPE_LABELS = {v: k for k, v in EVENT_TYPE_CODE.items()}


def ensure_summary_table() -> pd.DataFrame:
    usecols = ["row", "col", "event_type", "LOSStot", "TIMErec_true"]
    df = pd.read_csv(EVENT_PATH, usecols=usecols)

    grouped = df.groupby(["row", "col"], sort=False)
    summary = grouped.agg(
        event_count=("event_type", "size"),
        mean_losstot=("LOSStot", "mean"),
        mean_timerec=("TIMErec_true", "mean"),
    ).reset_index()

    counts = (
        df.assign(dummy=1)
        .pivot_table(
            index=["row", "col"],
            columns="event_type",
            values="dummy",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    summary = summary.merge(counts, on=["row", "col"], how="left")

    for event_type in EVENT_TYPE_CODE:
        if event_type not in summary.columns:
            summary[event_type] = 0

    summary["compound_fraction"] = summary["Compound_Overlapping"] / summary["event_count"]
    summary["dominant_event_type"] = (
        summary[list(EVENT_TYPE_CODE.keys())]
        .idxmax(axis=1)
        .map(EVENT_TYPE_CODE)
        .astype(np.int16)
    )

    drought_mask = df["event_type"] == "Drought_Dominant"
    compound_mask = df["event_type"] == "Compound_Overlapping"

    drought_mean = (
        df.loc[drought_mask]
        .groupby(["row", "col"], sort=False)["LOSStot"]
        .mean()
        .rename("mean_losstot_drought")
        .reset_index()
    )
    compound_mean = (
        df.loc[compound_mask]
        .groupby(["row", "col"], sort=False)["LOSStot"]
        .mean()
        .rename("mean_losstot_compound")
        .reset_index()
    )
    drought_count = (
        df.loc[drought_mask]
        .groupby(["row", "col"], sort=False)
        .size()
        .rename("drought_count")
        .reset_index()
    )
    compound_count = (
        df.loc[compound_mask]
        .groupby(["row", "col"], sort=False)
        .size()
        .rename("compound_count")
        .reset_index()
    )

    summary = summary.merge(drought_mean, on=["row", "col"], how="left")
    summary = summary.merge(compound_mean, on=["row", "col"], how="left")
    summary = summary.merge(drought_count, on=["row", "col"], how="left")
    summary = summary.merge(compound_count, on=["row", "col"], how="left")
    summary[["drought_count", "compound_count"]] = summary[["drought_count", "compound_count"]].fillna(0)

    valid_delta = (summary["drought_count"] >= 3) & (summary["compound_count"] >= 3)
    summary["delta_losstot_compound_minus_drought"] = np.where(
        valid_delta,
        summary["mean_losstot_compound"] - summary["mean_losstot_drought"],
        np.nan,
    )
    return summary


def build_raster_arrays(summary: pd.DataFrame) -> tuple[dict[str, np.ndarray], rasterio.profiles.Profile, rasterio.Affine]:
    with rasterio.open(TEMPLATE_PATH) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width
        transform = src.transform

    metrics = {
        "event_count": np.full((height, width), np.nan, dtype=np.float32),
        "mean_losstot": np.full((height, width), np.nan, dtype=np.float32),
        "mean_timerec": np.full((height, width), np.nan, dtype=np.float32),
        "compound_fraction": np.full((height, width), np.nan, dtype=np.float32),
        "mean_losstot_compound": np.full((height, width), np.nan, dtype=np.float32),
        "mean_losstot_drought": np.full((height, width), np.nan, dtype=np.float32),
        "delta_losstot_compound_minus_drought": np.full((height, width), np.nan, dtype=np.float32),
        "dominant_event_type": np.full((height, width), 255, dtype=np.uint8),
    }

    rows = summary["row"].astype(int).to_numpy()
    cols = summary["col"].astype(int).to_numpy()
    metrics["event_count"][rows, cols] = summary["event_count"].astype(float).to_numpy()
    metrics["mean_losstot"][rows, cols] = summary["mean_losstot"].astype(float).to_numpy()
    metrics["mean_timerec"][rows, cols] = summary["mean_timerec"].astype(float).to_numpy()
    metrics["compound_fraction"][rows, cols] = summary["compound_fraction"].astype(float).to_numpy()
    metrics["mean_losstot_compound"][rows, cols] = summary["mean_losstot_compound"].astype(float).to_numpy()
    metrics["mean_losstot_drought"][rows, cols] = summary["mean_losstot_drought"].astype(float).to_numpy()
    metrics["delta_losstot_compound_minus_drought"][rows, cols] = summary[
        "delta_losstot_compound_minus_drought"
    ].astype(float).to_numpy()
    metrics["dominant_event_type"][rows, cols] = summary["dominant_event_type"].astype(np.uint8).to_numpy()
    return metrics, profile, transform


def save_rasters(metrics: dict[str, np.ndarray], profile: rasterio.profiles.Profile) -> None:
    for name, array in metrics.items():
        out_path = SPATIAL_DIR / f"{name}.tif"
        out_profile = profile.copy()
        if name == "dominant_event_type":
            out_profile.update(dtype=rasterio.uint8, count=1, nodata=255, compress="lzw")
        else:
            out_profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0, compress="lzw")
            array = np.where(np.isfinite(array), array, -9999.0).astype(np.float32)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(array, 1)


def raster_extent(transform: rasterio.Affine, shape: tuple[int, int]) -> list[float]:
    height, width = shape
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + transform.a * width
    ymin = ymax + transform.e * height
    return [xmin, xmax, ymin, ymax]


def setup_geo_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 91, 30))
    ax.grid(True, linestyle="--", linewidth=0.4, color="0.84")
    ax.set_xlabel("")
    ax.set_ylabel("")


def save_previews(metrics: dict[str, np.ndarray], transform: rasterio.Affine) -> None:
    extent = raster_extent(transform, next(iter(metrics.values())).shape)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    count_arr = np.where(np.isfinite(metrics["event_count"]), metrics["event_count"], np.nan)
    im0 = axes[0, 0].imshow(count_arr, extent=extent, origin="upper", cmap="YlOrBr", interpolation="nearest")
    setup_geo_axis(axes[0, 0])
    axes[0, 0].set_title("Event frequency")
    plt.colorbar(im0, ax=axes[0, 0], orientation="horizontal", pad=0.08, shrink=0.8)

    loss_arr = np.where(np.isfinite(metrics["mean_losstot"]), metrics["mean_losstot"], np.nan)
    im1 = axes[0, 1].imshow(loss_arr, extent=extent, origin="upper", cmap="YlGnBu", interpolation="nearest")
    setup_geo_axis(axes[0, 1])
    axes[0, 1].set_title("Mean LOSStot")
    plt.colorbar(im1, ax=axes[0, 1], orientation="horizontal", pad=0.08, shrink=0.8)

    delta_arr = np.where(np.isfinite(metrics["delta_losstot_compound_minus_drought"]), metrics["delta_losstot_compound_minus_drought"], np.nan)
    limit = np.nanquantile(np.abs(delta_arr), 0.98)
    im2 = axes[1, 0].imshow(
        delta_arr,
        extent=extent,
        origin="upper",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        interpolation="nearest",
    )
    setup_geo_axis(axes[1, 0])
    axes[1, 0].set_title("Compound minus drought LOSStot")
    plt.colorbar(im2, ax=axes[1, 0], orientation="horizontal", pad=0.08, shrink=0.8)

    dom_arr = metrics["dominant_event_type"].astype(float)
    dom_arr[dom_arr == 255] = np.nan
    cmap = ListedColormap(["#d9d9d9", "#f0e442", "#1b9e77", "#d95f02", "#7570b3", "#66a61e"])
    norm = BoundaryNorm(np.arange(0.5, 7.5, 1), cmap.N)
    im3 = axes[1, 1].imshow(dom_arr, extent=extent, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
    setup_geo_axis(axes[1, 1])
    axes[1, 1].set_title("Predominant event type")
    cbar = plt.colorbar(im3, ax=axes[1, 1], orientation="horizontal", pad=0.08, shrink=0.9, ticks=np.arange(1, 7))
    cbar.ax.set_xticklabels(
        [
            "Reference",
            "Drought",
            "Heat",
            "Compound",
            "D→H",
            "H→D",
        ]
    )

    fig.suptitle("Global spatial patterns of vegetation negative-anomaly events", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(SPATIAL_DIR / "spatial_event_maps_preview.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    report = [
        "SPATIAL EVENT MAP SUMMARY",
        f"event_path={EVENT_PATH}",
        f"n_pixels={len(summary)}",
        f"mean_event_count={summary['event_count'].mean():.3f}",
        f"mean_losstot={summary['mean_losstot'].mean():.4f}",
        f"mean_timerec={summary['mean_timerec'].mean():.4f}",
        f"mean_compound_fraction={summary['compound_fraction'].mean():.4f}",
        f"valid_delta_pixels={summary['delta_losstot_compound_minus_drought'].notna().sum()}",
        "",
        "[predominant event type counts]",
        summary["dominant_event_type"].map(EVENT_TYPE_LABELS).value_counts().to_string(),
    ]
    (SPATIAL_DIR / "spatial_event_maps_report.txt").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    summary = ensure_summary_table()
    summary.to_csv(SPATIAL_DIR / "spatial_event_summary_by_pixel.csv", index=False)
    metrics, profile, transform = build_raster_arrays(summary)
    save_rasters(metrics, profile)
    save_previews(metrics, transform)
    write_report(summary)
    print(f"Saved spatial event map outputs to {SPATIAL_DIR}")


if __name__ == "__main__":
    main()
