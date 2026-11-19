from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve()
REVISION_ROOT = ROOT / "codex_revision_v4_1_20260715"
FIGURE_DIR = REVISION_ROOT / "figure_outputs"
TABLE_DIR = REVISION_ROOT / "table_outputs"
ANALYSIS_DIR = REVISION_ROOT / "analysis_outputs"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

PANEL_PATH = ROOT / "codex_stage8_20260317" / "reports" / "fluxnet_validation_panel_v2_stage8.csv"
FIGURE_BASENAME = "Supplementary_Figure_S3_FLUXNET2015_merged"

THERMAL_ORDER = ["All", "Cool", "Warm", "Hot"]
THERMAL_COLORS = {"All": "#333333", "Cool": "#4C78A8", "Warm": "#E39C37", "Hot": "#B64342"}
STATE_ORDER = ["Reference", "DroughtOnly_LowVPD", "DroughtPlusHighVPD"]
STATE_LABEL = {
    "Reference": "Reference",
    "DroughtOnly_LowVPD": "Low-VPD drought",
    "DroughtPlusHighVPD": "High-VPD drought",
}
STATE_COLORS = {
    "Reference": "#BDBDBD",
    "DroughtOnly_LowVPD": "#42949E",
    "DroughtPlusHighVPD": "#B64342",
}
LAND_ORDER = ["Forest", "Cropland", "Grassland", "Shrub", "Savanna"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 5.9,
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


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.11, y: float = 1.04) -> None:
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
        color="#272727",
    )


def bootstrap(values: np.ndarray, seed: int, n_boot: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, n_boot, 250):
        size = min(250, n_boot - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means.append(values[indices].mean(axis=1))
    boot = np.concatenate(means)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def build_site_contrasts(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    site_meta = (
        panel[["site_id", "lat", "lon", "land_cover_broad", "thermal_background"]]
        .drop_duplicates("site_id")
        .reset_index(drop=True)
    )
    focus = panel[panel["validation_state_v2"].isin(["DroughtOnly_LowVPD", "DroughtPlusHighVPD"])].copy()
    metrics = ["GPP_MEAN_VUT_REF_anom", "LE_F_MDS_anom", "VPD_F_anom"]
    means = (
        focus.groupby(
            ["site_id", "lat", "lon", "land_cover_broad", "thermal_background", "validation_state_v2"],
            observed=True,
            dropna=False,
        )[metrics]
        .mean()
        .reset_index()
    )
    wide = means.pivot(
        index=["site_id", "lat", "lon", "land_cover_broad", "thermal_background"],
        columns="validation_state_v2",
        values=metrics,
    )
    records = wide.index.to_frame(index=False)
    for metric, short in [
        ("GPP_MEAN_VUT_REF_anom", "gpp"),
        ("LE_F_MDS_anom", "le"),
        ("VPD_F_anom", "vpd"),
    ]:
        records[f"{short}_low_vpd_drought"] = wide[(metric, "DroughtOnly_LowVPD")].to_numpy()
        records[f"{short}_high_vpd_drought"] = wide[(metric, "DroughtPlusHighVPD")].to_numpy()
        records[f"{short}_contrast"] = records[f"{short}_high_vpd_drought"] - records[f"{short}_low_vpd_drought"]
    return records, site_meta


def contrast_summary(site: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_meta = [
        ("gpp_contrast", "GPP", "umol CO2 m-2 s-1"),
        ("le_contrast", "Latent heat flux", "W m-2"),
        ("vpd_contrast", "VPD", "hPa"),
    ]
    for metric_index, (column, label, unit) in enumerate(metric_meta):
        for group_index, group in enumerate(THERMAL_ORDER):
            values = site[column] if group == "All" else site.loc[site["thermal_background"].eq(group), column]
            values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
            ci_low, ci_high = bootstrap(values, seed=1000 + metric_index * 20 + group_index)
            rows.append(
                {
                    "panel": "A_paired_high_minus_low_VPD_drought",
                    "metric": label,
                    "unit": unit,
                    "group": group,
                    "n_sites": int(len(values)),
                    "n_months": np.nan,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "proportion_negative": float(np.mean(values < 0)),
                    "proportion_positive": float(np.mean(values > 0)),
                }
            )
    return pd.DataFrame(rows)


def state_context(panel: pd.DataFrame) -> pd.DataFrame:
    metric_meta = [
        ("GPP_MEAN_VUT_REF_anom", "GPP", "umol CO2 m-2 s-1"),
        ("LE_F_MDS_anom", "Latent heat flux", "W m-2"),
        ("VPD_F_anom", "VPD", "hPa"),
    ]
    site_state = (
        panel[panel["validation_state_v2"].isin(STATE_ORDER)]
        .groupby(["site_id", "validation_state_v2"], observed=True)[[x[0] for x in metric_meta]]
        .mean()
        .reset_index()
    )
    month_counts = (
        panel[panel["validation_state_v2"].isin(STATE_ORDER)]
        .groupby("validation_state_v2", observed=True)
        .size()
        .to_dict()
    )
    rows = []
    for state_index, state in enumerate(STATE_ORDER):
        state_df = site_state[site_state["validation_state_v2"].eq(state)]
        for metric_index, (column, label, unit) in enumerate(metric_meta):
            values = pd.to_numeric(state_df[column], errors="coerce").dropna().to_numpy(dtype=float)
            ci_low, ci_high = bootstrap(values, seed=2000 + state_index * 20 + metric_index)
            rows.append(
                {
                    "panel": "B_site_state_context",
                    "metric": label,
                    "unit": unit,
                    "group": STATE_LABEL[state],
                    "state_code": state,
                    "n_sites": int(len(values)),
                    "n_months": int(month_counts.get(state, 0)),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "proportion_negative": float(np.mean(values < 0)),
                    "proportion_positive": float(np.mean(values > 0)),
                }
            )
    return pd.DataFrame(rows)


def biome_summary(site: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, group in enumerate(LAND_ORDER):
        values = pd.to_numeric(
            site.loc[site["land_cover_broad"].eq(group), "gpp_contrast"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        ci_low, ci_high = bootstrap(values, seed=3000 + index)
        rows.append(
            {
                "panel": "C_paired_GPP_by_ecosystem",
                "metric": "GPP",
                "unit": "umol CO2 m-2 s-1",
                "group": group,
                "n_sites": int(len(values)),
                "n_months": np.nan,
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "proportion_negative": float(np.mean(values < 0)),
                "proportion_positive": float(np.mean(values > 0)),
            }
        )
    return pd.DataFrame(rows)


def plot_map(fig: plt.Figure, spec, site: pd.DataFrame, site_meta: pd.DataFrame) -> plt.Axes:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax = fig.add_subplot(spec, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#F4F4F1", edgecolor="none", zorder=0)
    ax.coastlines(resolution="110m", linewidth=0.35, color="#666666")
    ax.scatter(
        site_meta["lon"],
        site_meta["lat"],
        s=9,
        color="#CFCFCF",
        edgecolor="white",
        linewidth=0.2,
        transform=ccrs.PlateCarree(),
        zorder=1,
    )
    mapped = site.dropna(subset=["gpp_contrast"]).copy()
    vmax = max(float(np.nanquantile(np.abs(mapped["gpp_contrast"]), 0.95)), 0.25)
    scatter = ax.scatter(
        mapped["lon"],
        mapped["lat"],
        c=mapped["gpp_contrast"],
        s=20,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        edgecolor="#333333",
        linewidth=0.25,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    colorbar = fig.colorbar(scatter, ax=ax, orientation="horizontal", fraction=0.048, pad=0.035)
    colorbar.outline.set_linewidth(0.4)
    colorbar.ax.tick_params(labelsize=5.6, width=0.4, length=2)
    colorbar.set_label("GPP difference: high-VPD minus low-VPD drought (umol CO2 m-2 s-1)", fontsize=6.2, labelpad=2)
    add_panel_label(ax, "a", x=-0.025, y=1.01)
    return ax


def plot_forest(
    ax: plt.Axes,
    site: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
    metric_label: str,
    xlabel: str,
    panel_label: str,
) -> None:
    rng = np.random.default_rng(4000 + ord(panel_label))
    y = np.arange(len(THERMAL_ORDER))
    ax.axvline(0, color="#777777", lw=0.7, ls=(0, (2.5, 2.5)), zorder=0)
    for yi, group in zip(y, THERMAL_ORDER):
        values = site[metric] if group == "All" else site.loc[site["thermal_background"].eq(group), metric]
        values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(values, yi + jitter, s=8, color=THERMAL_COLORS[group], alpha=0.22, edgecolor="none", zorder=1)
        row = summary[(summary["metric"].eq(metric_label)) & summary["group"].eq(group)].iloc[0]
        mean = float(row["mean"])
        ax.errorbar(
            mean,
            yi,
            xerr=[[mean - float(row["ci_low"])], [float(row["ci_high"]) - mean]],
            fmt="o",
            ms=4.6,
            color=THERMAL_COLORS[group],
            ecolor=THERMAL_COLORS[group],
            elinewidth=1.15,
            capsize=2.4,
            markeredgecolor="black",
            markeredgewidth=0.3,
            zorder=3,
        )
    labels = []
    for group in THERMAL_ORDER:
        row = summary[(summary["metric"].eq(metric_label)) & summary["group"].eq(group)].iloc[0]
        labels.append(f"{group} (n={int(row['n_sites'])})")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(y) - 0.55, -0.55)
    ax.set_title(metric_label, loc="left", fontweight="bold", pad=2)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#ECECEC", lw=0.42)
    ax.tick_params(axis="y", length=0)
    add_panel_label(ax, panel_label)


def plot_state_context(fig: plt.Figure, spec, summary: pd.DataFrame) -> list[plt.Axes]:
    sub = spec.subgridspec(1, 3, wspace=0.42)
    axes = [fig.add_subplot(sub[0, index]) for index in range(3)]
    metric_meta = [
        ("GPP", "GPP anomaly"),
        ("Latent heat flux", "Latent heat flux anomaly"),
        ("VPD", "VPD anomaly"),
    ]
    x = np.arange(len(STATE_ORDER))
    colors = [STATE_COLORS[state] for state in STATE_ORDER]
    for index, (ax, (metric, title)) in enumerate(zip(axes, metric_meta)):
        frame = summary[summary["metric"].eq(metric)].set_index("state_code").loc[STATE_ORDER].reset_index()
        means = frame["mean"].to_numpy(dtype=float)
        low = means - frame["ci_low"].to_numpy(dtype=float)
        high = frame["ci_high"].to_numpy(dtype=float) - means
        ax.bar(x, means, width=0.65, color=colors, edgecolor="#333333", linewidth=0.4, zorder=2)
        ax.errorbar(x, means, yerr=[low, high], fmt="none", ecolor="#333333", elinewidth=0.7, capsize=2.0, zorder=3)
        ax.axhline(0, color="#777777", lw=0.6, zorder=0)
        ax.set_title(title, loc="left", fontweight="bold", pad=2)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color="#ECECEC", lw=0.4)
        ax.set_axisbelow(True)
        if index == 0:
            add_panel_label(ax, "e", x=-0.25, y=1.06)
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=STATE_COLORS[state],
            markeredgecolor="#333333",
            markeredgewidth=0.4,
            markersize=5,
            label=STATE_LABEL[state],
        )
        for state in STATE_ORDER
    ]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, columnspacing=0.7, handletextpad=0.3)
    return axes


def plot_biome(ax: plt.Axes, site: pd.DataFrame, summary: pd.DataFrame) -> None:
    frame = summary.set_index("group").loc[[x for x in LAND_ORDER if x in summary["group"].values]].reset_index()
    y = np.arange(len(frame))
    ax.axvline(0, color="#777777", lw=0.7, ls=(0, (2.5, 2.5)), zorder=0)
    rng = np.random.default_rng(5000)
    for yi, row in frame.iterrows():
        values = pd.to_numeric(
            site.loc[site["land_cover_broad"].eq(row["group"]), "gpp_contrast"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        ax.scatter(values, yi + rng.uniform(-0.13, 0.13, len(values)), s=8, color="#7A9E68", alpha=0.24, edgecolor="none")
        mean = float(row["mean"])
        ax.errorbar(
            mean,
            yi,
            xerr=[[mean - float(row["ci_low"])], [float(row["ci_high"]) - mean]],
            fmt="o",
            ms=4.6,
            color="#4F7A3E",
            ecolor="#4F7A3E",
            elinewidth=1.15,
            capsize=2.4,
            markeredgecolor="black",
            markeredgewidth=0.3,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{row['group']} (n={int(row['n_sites'])})" for _, row in frame.iterrows()])
    ax.set_ylim(len(y) - 0.55, -0.55)
    ax.set_title("Paired GPP contrast by ecosystem", loc="left", fontweight="bold", pad=2)
    ax.set_xlabel("High-VPD minus low-VPD drought (umol CO2 m-2 s-1)")
    ax.grid(axis="x", color="#ECECEC", lw=0.42)
    ax.tick_params(axis="y", length=0)
    add_panel_label(ax, "f", x=-0.18, y=1.04)


def save_figure(fig: plt.Figure, base: Path) -> None:
    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".tiff", {"dpi": 600}),
        (".png", {"dpi": 300}),
    ]:
        fig.savefig(base.with_suffix(suffix), facecolor="white", **kwargs)


def main() -> None:
    apply_style()
    panel = pd.read_csv(PANEL_PATH)
    site, site_meta = build_site_contrasts(panel)
    paired = contrast_summary(site)
    state = state_context(panel)
    biome = biome_summary(site)
    table = pd.concat([paired, state, biome], ignore_index=True, sort=False)

    site.to_csv(ANALYSIS_DIR / "fluxnet_paired_site_contrasts.csv", index=False)
    site_meta.to_csv(ANALYSIS_DIR / "fluxnet_eligible_site_metadata.csv", index=False)
    table.to_csv(TABLE_DIR / "Supplementary_Table_6_FLUXNET2015_revised.csv", index=False)

    fig = plt.figure(figsize=(8.7, 10.6))
    grid = fig.add_gridspec(
        3,
        3,
        left=0.075,
        right=0.985,
        top=0.975,
        bottom=0.065,
        height_ratios=[2.25, 1.0, 1.05],
        wspace=0.42,
        hspace=0.36,
    )
    plot_map(fig, grid[0, :], site, site_meta)
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2])
    plot_forest(ax_b, site, paired, "gpp_contrast", "GPP", "High-VPD minus low-VPD drought", "b")
    plot_forest(ax_c, site, paired, "le_contrast", "Latent heat flux", "High-VPD minus low-VPD drought", "c")
    plot_forest(ax_d, site, paired, "vpd_contrast", "VPD", "High-VPD minus low-VPD drought", "d")
    lower = grid[2, :].subgridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.32)
    plot_state_context(fig, lower[0, 0], state)
    ax_f = fig.add_subplot(lower[0, 1])
    plot_biome(ax_f, site, biome)

    save_figure(fig, FIGURE_DIR / FIGURE_BASENAME)
    plt.close(fig)

    all_gpp = paired[(paired["metric"].eq("GPP")) & paired["group"].eq("All")].iloc[0]
    hot_gpp = paired[(paired["metric"].eq("GPP")) & paired["group"].eq("Hot")].iloc[0]
    qa = f"""# Merged Supplementary Figure S3 QA

- 100 sites meet the tower-data eligibility criteria; 88 contain both low- and high-VPD drought months.
- Panel a maps the paired-site GPP contrast and is the dominant panel.
- Panels b-d report within-site high-minus-low VPD drought contrasts by thermal background.
- Panel e retains the unique overall-state context from former Fig. S4a and labels the water-flux variable as latent heat flux.
- Panel f retains the ecosystem information from former Fig. S4c using paired site contrasts.
- Former Fig. S4b is redundant with panels b-d; former Fig. S4d is descriptive and is not retained.
- Overall paired GPP contrast: {all_gpp['mean']:.3f}, 95% CI [{all_gpp['ci_low']:.3f}, {all_gpp['ci_high']:.3f}], n={int(all_gpp['n_sites'])}.
- Hot-background paired GPP contrast: {hot_gpp['mean']:.3f}, 95% CI [{hot_gpp['ci_low']:.3f}, {hot_gpp['ci_high']:.3f}], n={int(hot_gpp['n_sites'])}.
"""
    (FIGURE_DIR / f"{FIGURE_BASENAME}_qa.md").write_text(qa, encoding="utf-8")
    print(FIGURE_DIR / f"{FIGURE_BASENAME}.png")
    print(TABLE_DIR / "Supplementary_Table_6_FLUXNET2015_revised.csv")


if __name__ == "__main__":
    main()
