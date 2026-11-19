from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from fig_common import (
    OUTPUT_DIR,
    PALETTE,
    apply_nature_style,
    print_saved,
    read_stage8_data,
    save_figure,
    write_source,
)


FIGURE_BASENAME = "Figure4_climate_ecosystem_selectivity_nature"

PERIOD_COLORS_THERMAL = {"2001-2012": "#7370B5", "2013-2024": "#D95F02"}
PERIOD_COLORS_BIOME = {"2001-2012": "#9BBE7A", "2013-2024": "#D95F02"}

ZONE_LABEL = {
    "Sig_Aggravation": "Significant negative",
    "Negative_Nonsig": "Negative, non-significant",
    "Nonnegative_Interaction": "Positive",
}
ZONE_ORDER = ["Sig_Aggravation", "Negative_Nonsig", "Nonnegative_Interaction"]
ZONE_COLORS = {
    "Sig_Aggravation": "#2166AC",
    "Negative_Nonsig": "#92C5DE",
    "Nonnegative_Interaction": "#B2182B",
}

METRIC_COLUMNS = {
    "NDVI": "ndvi_anom_delta_event_minus_pre_mean",
    "SIF": "sif_anom_delta_event_minus_pre_mean",
    "GPP": "gpp_anom_delta_event_minus_pre_mean",
    "ET": "et_anom_delta_event_minus_pre_mean",
    "SPEI": "spei_anom_delta_event_minus_pre_mean",
    "SM": "sm_anom_delta_event_minus_pre_mean",
    "Temp": "temp_anom_delta_event_minus_pre_mean",
    "VPD": "vpd_anom_delta_event_minus_pre_mean",
}
STRESS_POSITIVE = {"Temp", "VPD"}
LANDCOVER_HEATMAP_ORDER = ["Forest", "Grassland", "Savannas", "Cropland", "Shrub"]


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        fontfamily="Times New Roman",
        fontweight="bold",
        color=PALETTE["neutral_black"],
    )


def format_delta(value: float, p_value: float) -> str:
    star = "*" if p_value < 0.05 else ""
    return f"{value:+.3f}{star}"


def plot_period_dumbbell(
    ax: plt.Axes,
    df: pd.DataFrame,
    group_col: str,
    order: list[str],
    colors: dict[str, str],
    xlim: tuple[float, float],
    xticks: list[float],
    xlabel: str,
    panel_label: str,
) -> None:
    plot_df = df.set_index(group_col).loc[order].reset_index()
    y = np.arange(len(plot_df))
    span = xlim[1] - xlim[0]

    ax.axvline(0, color="#9A9A9A", lw=0.75, ls=(0, (2.8, 2.8)), zorder=0)
    ax.grid(axis="x", color="#EAEAEA", lw=0.55)
    ax.set_axisbelow(True)

    for yi, (_, row) in zip(y, plot_df.iterrows()):
        early = float(row["early_att_mean"])
        late = float(row["late_att_mean"])
        delta = float(row["late_minus_early"])
        p_value = float(row["p_permutation"])
        ax.plot([early, late], [yi, yi], color="#CFCFCF", lw=1.7, zorder=1)
        ax.scatter(
            early,
            yi,
            s=34,
            color=colors["2001-2012"],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.scatter(
            late,
            yi,
            s=34,
            color=colors["2013-2024"],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )

        left, right = sorted([early, late])
        close_pair = abs(late - early) < 0.075 * span
        left_label = f"{early:.3f}" if early >= 0 else f"{early:.3f}"
        late_label = f"{late:.3f}" if late >= 0 else f"{late:.3f}"
        early_ha = "right" if early <= late else "left"
        late_ha = "left" if late >= early else "right"
        early_dx = -0.012 * span if early <= late else 0.012 * span
        late_dx = 0.012 * span if late >= early else -0.012 * span
        early_y = yi
        late_y = yi
        if early < 0:
            early_ha = "left"
            early_dx = 0.014 * span
        if late < 0:
            late_ha = "left"
            late_dx = 0.014 * span
        if close_pair:
            early_ha = "center"
            late_ha = "center"
            early_dx = 0.0
            late_dx = 0.0
            early_y = yi + 0.22
            late_y = yi - 0.22
        if not close_pair:
            ax.text(
                early + early_dx,
                early_y,
                left_label,
                ha=early_ha,
                va="center",
                fontsize=6.0,
                fontweight="bold",
                color=colors["2001-2012"],
            )
            ax.text(
                late + late_dx,
                late_y,
                late_label,
                ha=late_ha,
                va="center",
                fontsize=6.0,
                fontweight="bold",
                color=colors["2013-2024"],
            )

        label_x = (left + right) / 2
        if close_pair:
            label_x = min(right + 0.050 * span, xlim[1] - 0.055 * span)
        ax.text(
            label_x,
            yi - 0.18,
            format_delta(delta, p_value),
            ha="center",
            va="center",
            fontsize=6.1,
            fontweight="bold",
            color=PALETTE["neutral_dark"],
        )

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontweight="bold")
    ax.set_ylim(len(order) - 0.55, -0.55)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors["2001-2012"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=5.1,
            label="2001-2012",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors["2013-2024"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=5.1,
            label="2013-2024",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.12),
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    add_panel_label(ax, panel_label, x=-0.095, y=1.08)


def prepare_heatmap() -> pd.DataFrame:
    raw = read_stage8_data("event_multivariate_response_anomaly_by_type_landcover.csv")
    compound = raw[raw["event_type"] == "Compound_Overlapping"].copy()
    records = []
    for label, col in METRIC_COLUMNS.items():
        tmp = compound[["land_cover_broad", col]].copy()
        tmp = tmp.rename(columns={col: "value"})
        tmp["metric"] = label
        tmp["intensity"] = tmp["value"] if label in STRESS_POSITIVE else -tmp["value"]
        mean = tmp["intensity"].mean()
        std = tmp["intensity"].std(ddof=1)
        tmp["z"] = (tmp["intensity"] - mean) / std if std > 0 else 0.0
        records.append(tmp[["land_cover_broad", "metric", "z"]])
    return pd.concat(records, ignore_index=True)


def plot_heatmap(ax: plt.Axes, heat_df: pd.DataFrame) -> None:
    heat = (
        heat_df.pivot(index="land_cover_broad", columns="metric", values="z")
        .loc[LANDCOVER_HEATMAP_ORDER, list(METRIC_COLUMNS.keys())]
        .astype(float)
    )
    matrix = heat.to_numpy()
    cmap = LinearSegmentedColormap.from_list("damage", ["#2C7BB6", "#F7F7F7", "#D7191C"])
    im = ax.imshow(matrix, cmap=cmap, norm=Normalize(vmin=-1.5, vmax=1.5), aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(list(METRIC_COLUMNS.keys()))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(LANDCOVER_HEATMAP_ORDER, fontweight="bold")
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if abs(value) >= 1.15 else "black"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=5.7, fontweight="bold", color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Damage / stress intensity", fontstyle="italic", color=PALETTE["neutral_dark"], labelpad=2)
    cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal", fraction=0.075, pad=0.14, ticks=[-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(labelsize=5.2, width=0.35, length=2)
    cbar.set_label("Relative damage / stress intensity (Z-score)", fontsize=5.8, fontweight="bold", labelpad=2)
    add_panel_label(ax, "c", x=-0.13, y=1.05)


def prepare_trajectory() -> pd.DataFrame:
    df = read_stage8_data("event_trajectory_by_zone_vpd_summary_robust.csv")
    df = df[df["event_type"] == "Compound_Overlapping"].copy()
    df["stage_month"] = pd.to_numeric(df["stage_month"], errors="coerce")
    df["mean_value"] = pd.to_numeric(df["mean_value"], errors="coerce")
    df["n_obs"] = pd.to_numeric(df["n_obs"], errors="coerce")
    df = df[
        ((df["stage"] == "pre") & df["stage_month"].between(-6, -1))
        | ((df["stage"] == "event") & df["stage_month"].between(1, 6))
    ].copy()

    baseline_rows = []
    for (zone, variable), sub in df[df["stage"] == "pre"].groupby(
        ["amplification_zone", "variable"], observed=True
    ):
        valid = sub[["mean_value", "n_obs"]].dropna()
        baseline = np.average(valid["mean_value"], weights=valid["n_obs"])
        baseline_rows.append(
            {"amplification_zone": zone, "variable": variable, "pre_event_baseline": baseline}
        )
    baseline = pd.DataFrame(baseline_rows)
    df = df.merge(baseline, on=["amplification_zone", "variable"], how="left")
    df["change_from_pre"] = df["mean_value"] - df["pre_event_baseline"]
    df["rel_month"] = np.where(df["stage"].eq("pre"), df["stage_month"], df["stage_month"] - 1)
    return df.sort_values(["variable", "amplification_zone", "rel_month"])


def plot_trajectories(fig: plt.Figure, spec, traj: pd.DataFrame) -> list[plt.Axes]:
    sub = spec.subgridspec(3, 1, hspace=0.32)
    variables = [("spei", "SPEI"), ("temp", "Temperature"), ("vpd", "VPD")]
    axes = [fig.add_subplot(sub[i, 0]) for i in range(3)]
    for ax, (var, title) in zip(axes, variables):
        ax.axvspan(-6.5, -0.5, color="#EEF2F6", zorder=0)
        ax.axvspan(-0.5, 5.5, color="#FAEFE8", zorder=0)
        ax.axhline(0, color=PALETTE["neutral_mid"], lw=0.55, zorder=0)
        ax.axvline(0, color=PALETTE["neutral_mid"], lw=0.65, ls=(0, (3, 3)))
        for zone in ZONE_ORDER:
            d = traj[(traj["variable"] == var) & (traj["amplification_zone"] == zone)].sort_values("rel_month")
            ax.plot(d["rel_month"], d["change_from_pre"], color=ZONE_COLORS[zone], lw=1.35, label=ZONE_LABEL[zone])
        ax.set_title(f"{title}: change from pre-event mean", loc="left", fontsize=6.4, fontweight="bold", pad=1.5)
        ax.grid(axis="y", color="#EAEAEA", lw=0.4)
        ax.set_xlim(-6.4, 5.4)
        ax.tick_params(axis="y", labelsize=5.7)
        if var == "spei":
            ax.set_ylim(-0.95, 0.45)
            ax.set_yticks([-0.8, -0.4, 0.0, 0.4])
        elif var == "temp":
            ax.set_ylim(-6.0, 10.5)
            ax.set_yticks([-5, 0, 5, 10])
        else:
            ax.set_ylim(-0.25, 0.68)
            ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6])
        if ax is not axes[-1]:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xticks([-6, -3, 0, 3, 5])
            ax.set_xlabel("Relative event month (0 = event onset)")
    handles = [Line2D([0], [0], color=ZONE_COLORS[z], lw=1.35, label=ZONE_LABEL[z]) for z in ZONE_ORDER]
    axes[0].legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.00, 1.14),
        ncol=3,
        title="Local drought x heat coefficient",
        title_fontsize=5.5,
        handlelength=1.6,
        columnspacing=0.62,
        borderaxespad=0.0,
        fontsize=5.1,
    )
    add_panel_label(axes[0], "d", x=-0.105, y=1.32)
    return axes


def write_qa_report() -> None:
    qa_text = """# Figure 4 climate and ecosystem selectivity QA

Core claim: late-period strengthening is concentrated by climate background and ecosystem type, and compound-event multivariate responses differ across land-cover classes.

Panel map:
- a: thermal-background late-minus-early matched LOSStot, moved before ecosystems to match manuscript logic.
- b: ecosystem late-minus-early matched LOSStot.
- c: compound-event multivariate damage/stress intensity by land-cover type.

Review risks checked:
- Significance is shown only as an asterisk appended to the late-minus-early label in panels a and b.
- The former VPD trajectory panel has been transferred to the standalone atmospheric-demand figure.
"""
    (OUTPUT_DIR / f"{FIGURE_BASENAME}_qa.md").write_text(qa_text, encoding="utf-8")


def main() -> None:
    apply_nature_style()
    plt.rcParams.update({"axes.facecolor": "white", "savefig.facecolor": "white"})

    biome = read_stage8_data("att_period_difference_by_landcover.csv")
    biome = biome[biome["outcome"] == "LOSStot"].copy()
    for col in ["early_att_mean", "late_att_mean", "late_minus_early", "p_permutation"]:
        biome[col] = pd.to_numeric(biome[col], errors="coerce")
    biome_order = biome.sort_values("late_minus_early", ascending=False)["land_cover_broad"].tolist()

    thermal = read_stage8_data("external_thermal_regime_period_difference.csv")
    thermal["thermal_regime"] = thermal["thermal_regime_external"].map(
        {"Cool_ext": "Cool", "Warm_ext": "Warm", "Hot_ext": "Hot"}
    )
    for col in ["early_att_mean", "late_att_mean", "late_minus_early", "p_permutation"]:
        thermal[col] = pd.to_numeric(thermal[col], errors="coerce")

    heatmap = prepare_heatmap()
    source = pd.concat(
        [
            thermal.assign(panel="a_thermal_background"),
            biome.assign(panel="b_ecosystem"),
            heatmap.assign(panel="c_multivariate_heatmap"),
        ],
        ignore_index=True,
        sort=False,
    )
    source_path = write_source(source, f"{FIGURE_BASENAME}_source_data.csv")

    fig = plt.figure(figsize=(7.35, 5.45))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.985,
        top=0.94,
        bottom=0.105,
        width_ratios=[1.0, 1.05],
        height_ratios=[0.82, 1.05],
        wspace=0.30,
        hspace=0.50,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    bottom = gs[1, :].subgridspec(1, 3, width_ratios=[0.12, 0.76, 0.12], wspace=0)
    ax_c = fig.add_subplot(bottom[0, 1])

    plot_period_dumbbell(
        ax_a,
        thermal,
        group_col="thermal_regime",
        order=["Hot", "Warm", "Cool"],
        colors=PERIOD_COLORS_THERMAL,
        xlim=(-0.005, 0.088),
        xticks=[0, 0.02, 0.04, 0.06, 0.08],
        xlabel="Matched additional total event loss",
        panel_label="a",
    )
    plot_period_dumbbell(
        ax_b,
        biome,
        group_col="land_cover_broad",
        order=biome_order,
        colors=PERIOD_COLORS_BIOME,
        xlim=(-0.011, 0.166),
        xticks=[0, 0.05, 0.10, 0.15],
        xlabel="Matched additional total event loss",
        panel_label="b",
    )
    plot_heatmap(ax_c, heatmap)

    saved = save_figure(fig, OUTPUT_DIR / FIGURE_BASENAME)
    plt.close(fig)
    write_qa_report()
    print_saved(saved, [source_path, OUTPUT_DIR / f"{FIGURE_BASENAME}_qa.md"])


if __name__ == "__main__":
    main()
