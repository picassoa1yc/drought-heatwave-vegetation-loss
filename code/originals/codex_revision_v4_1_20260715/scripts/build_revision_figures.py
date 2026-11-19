from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve()
REVISION_ROOT = ROOT / "codex_revision_v4_1_20260715"
ANALYSIS_DIR = REVISION_ROOT / "analysis_outputs"
FIGURE_DIR = REVISION_ROOT / "figure_outputs"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MAIN_SCRIPT_DIR = ROOT / "codex_stage9_20260516_core_restructure" / "main_figures_python" / "scripts"
SUPP_DATA_DIR = ROOT / "codex_stage8_20260317" / "figure_r_package" / "data"
CUMULATIVE_EVENT_PATH = (
    ROOT
    / "output_df"
    / "global_v05_90p_vpd_main"
    / "event_analysis_anomaly"
    / "event_cumulative_anomaly_event_level.csv"
)

EVENT_METRIC_OUTCOMES = ["LOSStot", "DECmax", "TIMErec_true"]
CUM_OUTCOMES = ["ndvi_cum_loss", "sif_cum_loss", "gpp_cum_loss", "et_cum_loss"]
FIG2_OUTCOMES = EVENT_METRIC_OUTCOMES + CUM_OUTCOMES
FIG2_LABEL = {
    "LOSStot": "LOSStot",
    "DECmax": "DECmax",
    "TIMErec_true": "TIMErec",
    "ndvi_cum_loss": "Cumulative NDVI loss",
    "sif_cum_loss": "Cumulative SIF loss",
    "gpp_cum_loss": "Cumulative GPP loss",
    "et_cum_loss": "Cumulative ET loss",
}
SHARED_XLABEL = "Additional impact of compound vs. matched drought events (%)"
SHARED_XLIM = (-1.5, 32.0)
SHARED_XTICKS = [0, 5, 10, 15, 20, 25, 30]

PALETTE = {
    "black": "#272727",
    "gray": "#777777",
    "light": "#D8D8D8",
    "main": "#B64342",
    "main_soft": "#D98EA6",
    "teal": "#42949E",
    "orange": "#E28E2C",
    "early": "#484878",
}


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
            "axes.titlesize": 7.6,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.0,
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


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.04) -> None:
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
        color=PALETTE["black"],
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".tiff", {"dpi": 600}),
        (".png", {"dpi": 300}),
    ]:
        fig.savefig(base.with_suffix(suffix), facecolor="white", **kwargs)


def ordered(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    return (
        df.assign(_order=pd.Categorical(df["outcome"], categories=order, ordered=True))
        .sort_values("_order")
        .drop(columns="_order")
    )


def style_effect_axis(ax: plt.Axes) -> None:
    ax.axvline(0, color=PALETTE["gray"], lw=0.65, ls=(0, (2.4, 2.4)), zorder=0)
    ax.grid(axis="x", color="#ECECEC", lw=0.45)
    ax.set_axisbelow(True)
    ax.set_xlim(*SHARED_XLIM)
    ax.set_xticks(SHARED_XTICKS)
    ax.set_xlabel(SHARED_XLABEL)
    ax.tick_params(axis="y", length=0)


def relative_ci(frame: pd.DataFrame, suffix: str = "") -> tuple[np.ndarray, np.ndarray]:
    denominator = frame[f"matched_control_mean{suffix}"].abs().to_numpy(dtype=float)
    low = frame[f"additional_ci_low{suffix}"].to_numpy(dtype=float) / denominator * 100.0
    high = frame[f"additional_ci_high{suffix}"].to_numpy(dtype=float) / denominator * 100.0
    return low, high


def plot_fig2_a(ax: plt.Axes, main: pd.DataFrame) -> None:
    frame = ordered(main[main["outcome"].isin(FIG2_OUTCOMES)].copy(), FIG2_OUTCOMES)
    y = np.arange(len(frame))
    values = frame["additional_relative_pct"].to_numpy(dtype=float)
    ci_low, ci_high = relative_ci(frame)
    low = values - ci_low
    high = ci_high - values
    ax.barh(
        y,
        values,
        height=0.58,
        color="#D9D9D9",
        edgecolor="#333333",
        linewidth=0.4,
        zorder=2,
    )
    ax.errorbar(values, y, xerr=[low, high], fmt="none", ecolor="#333333", elinewidth=0.75, capsize=2.0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["outcome"].map(FIG2_LABEL))
    ax.set_ylim(len(frame) - 0.55, -0.55)
    style_effect_axis(ax)
    for yi, value, upper in zip(y, values, ci_high):
        ax.text(
            upper + 0.45,
            yi,
            f"{value:.1f}%",
            ha="left",
            va="center",
            fontsize=5.8,
        )
    att_column_x = 30.2
    ax.text(
        att_column_x,
        1.012,
        "ATT",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.2,
        fontweight="bold",
    )
    for yi, outcome, att in zip(
        y,
        frame["outcome"],
        frame["additional_effect"].to_numpy(dtype=float),
    ):
        digits = 2 if outcome in {"gpp_cum_loss", "et_cum_loss"} else 3
        ax.text(
            att_column_x,
            yi,
            f"{att:.{digits}f}",
            ha="center",
            va="center",
            fontsize=5.8,
        )
    add_panel_label(ax, "a")


def plot_fig2_b(ax: plt.Axes, threshold: pd.DataFrame) -> None:
    frame = ordered(threshold[threshold["outcome"].isin(FIG2_OUTCOMES)].copy(), FIG2_OUTCOMES)
    y = np.arange(len(frame))
    p90 = frame["additional_relative_pct_p90"].to_numpy(dtype=float)
    p95 = frame["additional_relative_pct_p95"].to_numpy(dtype=float)
    for yi, left, right in zip(y, p90, p95):
        ax.plot([left, right], [yi, yi], color=PALETTE["light"], lw=1.2, zorder=1)
    ax.scatter(p90, y, s=27, color=PALETTE["teal"], edgecolor="black", linewidth=0.35, label="Heat P90", zorder=3)
    ax.scatter(p95, y, s=34, marker="^", color=PALETTE["orange"], edgecolor="black", linewidth=0.35, label="Heat P95", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["outcome"].map(FIG2_LABEL))
    ax.set_ylim(len(frame) - 0.55, -0.55)
    style_effect_axis(ax)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), handletextpad=0.35, borderaxespad=0)
    add_panel_label(ax, "b")


def plot_fig2_c(
    ax: plt.Axes,
    event_by_period: pd.DataFrame,
    event_difference: pd.DataFrame,
    full_record_main: pd.DataFrame,
) -> None:
    def select_period(frame: pd.DataFrame, outcomes: list[str], period: str) -> pd.DataFrame:
        return ordered(
            frame[frame["period"].eq(period) & frame["outcome"].isin(outcomes)].copy(),
            outcomes,
        )

    early = select_period(event_by_period, FIG2_OUTCOMES, "2001-2012")
    recent = select_period(event_by_period, FIG2_OUTCOMES, "2013-2024")
    differences = ordered(
        event_difference[event_difference["outcome"].isin(FIG2_OUTCOMES)].copy(),
        FIG2_OUTCOMES,
    )
    full_record_denominator = (
        full_record_main.set_index("outcome")["matched_control_mean"].abs().to_dict()
    )
    for frame in (early, recent):
        denominator = frame["outcome"].map(full_record_denominator).to_numpy(dtype=float)
        frame["plot_relative_pct"] = frame["additional_effect"].to_numpy(dtype=float) / denominator * 100.0
        frame["plot_ci_low_pct"] = frame["additional_ci_low"].to_numpy(dtype=float) / denominator * 100.0
        frame["plot_ci_high_pct"] = frame["additional_ci_high"].to_numpy(dtype=float) / denominator * 100.0
    y = np.array([0, 1, 2, 4, 5, 6, 7], dtype=float)
    early_values = early["plot_relative_pct"].to_numpy(dtype=float)
    recent_values = recent["plot_relative_pct"].to_numpy(dtype=float)
    early_ci_low = early["plot_ci_low_pct"].to_numpy(dtype=float)
    early_ci_high = early["plot_ci_high_pct"].to_numpy(dtype=float)
    recent_ci_low = recent["plot_ci_low_pct"].to_numpy(dtype=float)
    recent_ci_high = recent["plot_ci_high_pct"].to_numpy(dtype=float)
    for yi, left, right in zip(y, early_values, recent_values):
        ax.plot([left, right], [yi, yi], color=PALETTE["light"], lw=1.25, zorder=1)
    ax.errorbar(
        early_values,
        y,
        xerr=[early_values - early_ci_low, early_ci_high - early_values],
        fmt="o",
        ms=4.2,
        color=PALETTE["early"],
        ecolor=PALETTE["early"],
        elinewidth=0.65,
        capsize=1.7,
        markeredgecolor="black",
        markeredgewidth=0.3,
        zorder=3,
        label="2001-2012",
    )
    ax.errorbar(
        recent_values,
        y,
        xerr=[recent_values - recent_ci_low, recent_ci_high - recent_values],
        fmt="o",
        ms=4.5,
        color=PALETTE["orange"],
        ecolor=PALETTE["orange"],
        elinewidth=0.65,
        capsize=1.7,
        markeredgecolor="black",
        markeredgewidth=0.3,
        zorder=3,
        label="2013-2024",
    )
    for yi, early_value, recent_value in zip(y, early_values, recent_values):
        ax.text(early_value, yi + 0.27, f"{early_value:.1f}%", ha="center", va="center", fontsize=5.1, color=PALETTE["early"])
        ax.text(recent_value, yi - 0.27, f"{recent_value:.1f}%", ha="center", va="center", fontsize=5.1, color=PALETTE["orange"])
    for yi, outcome, early_value, recent_value, (_, row) in zip(
        y,
        early["outcome"],
        early_values,
        recent_values,
        differences.iterrows(),
    ):
        marker = "*" if float(row["p_permutation"]) < 0.05 else "ns"
        ax.text(
            30.25,
            yi,
            f"{recent_value - early_value:+.1f}%",
            ha="right",
            va="center",
            fontsize=5.7,
        )
        ax.text(31.3, yi, marker, ha="right", va="center", fontsize=6.0, fontweight="bold" if marker == "*" else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(early["outcome"].map(FIG2_LABEL))
    ax.set_ylim(7.55, -0.65)
    style_effect_axis(ax)
    ax.axhline(3.0, color="#D6D6D6", lw=0.6)
    ax.text(-1.35, -0.43, "Event metrics", ha="left", va="bottom", fontsize=6.2, fontweight="bold")
    ax.text(-1.35, 3.55, "Cumulative anomaly loss", ha="left", va="bottom", fontsize=6.2, fontweight="bold")
    ax.text(30.25, -0.43, "Change", ha="right", va="bottom", fontsize=6.2, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.10), ncol=2, handletextpad=0.35, columnspacing=0.9)
    add_panel_label(ax, "c", x=-0.065, y=1.08)


def build_figure2() -> None:
    event_main = pd.read_csv(ANALYSIS_DIR / "p90_global_att_sti_free.csv")
    cumulative_main = pd.read_csv(ANALYSIS_DIR / "cumulative_global_att_sti_free.csv")
    main = pd.concat(
        [
            event_main[event_main["outcome"].isin(EVENT_METRIC_OUTCOMES)],
            cumulative_main[cumulative_main["outcome"].isin(CUM_OUTCOMES)],
        ],
        ignore_index=True,
        sort=False,
    )

    event_threshold = pd.read_csv(ANALYSIS_DIR / "heat_threshold_p90_p95_sti_free.csv")
    cumulative_threshold = pd.read_csv(ANALYSIS_DIR / "cumulative_heat_threshold_p90_p95_sti_free.csv")
    threshold = pd.concat(
        [
            event_threshold[event_threshold["outcome"].isin(EVENT_METRIC_OUTCOMES)],
            cumulative_threshold[cumulative_threshold["outcome"].isin(CUM_OUTCOMES)],
        ],
        ignore_index=True,
        sort=False,
    )

    event_by_period_all = pd.read_csv(ANALYSIS_DIR / "event_att_by_period_sti_free.csv")
    cumulative_by_period = pd.read_csv(ANALYSIS_DIR / "cumulative_att_by_period_sti_free.csv")
    event_by_period = pd.concat(
        [
            event_by_period_all[event_by_period_all["outcome"].isin(EVENT_METRIC_OUTCOMES)],
            cumulative_by_period[cumulative_by_period["outcome"].isin(CUM_OUTCOMES)],
        ],
        ignore_index=True,
        sort=False,
    )

    event_difference_all = pd.read_csv(ANALYSIS_DIR / "event_period_difference_sti_free.csv")
    cumulative_difference = pd.read_csv(ANALYSIS_DIR / "cumulative_period_difference_sti_free.csv")
    event_difference = pd.concat(
        [
            event_difference_all[event_difference_all["outcome"].isin(EVENT_METRIC_OUTCOMES)],
            cumulative_difference[cumulative_difference["outcome"].isin(CUM_OUTCOMES)],
        ],
        ignore_index=True,
        sort=False,
    )
    full_record_denominator = main.set_index("outcome")["matched_control_mean"].abs().to_dict()
    period_source = event_by_period.copy()
    period_source["figure_fixed_reference_control_mean"] = period_source["outcome"].map(full_record_denominator)
    period_source["figure_additional_relative_pct"] = (
        period_source["additional_effect"] / period_source["figure_fixed_reference_control_mean"] * 100.0
    )
    period_source["figure_additional_ci_low_pct"] = (
        period_source["additional_ci_low"] / period_source["figure_fixed_reference_control_mean"] * 100.0
    )
    period_source["figure_additional_ci_high_pct"] = (
        period_source["additional_ci_high"] / period_source["figure_fixed_reference_control_mean"] * 100.0
    )
    period_pct = period_source.pivot(
        index="outcome",
        columns="period",
        values="figure_additional_relative_pct",
    )
    figure_recent_minus_early_pct = (
        period_pct["2013-2024"] - period_pct["2001-2012"]
    ).to_dict()
    event_difference["figure_recent_minus_early_pct"] = event_difference["outcome"].map(
        figure_recent_minus_early_pct
    )
    source = pd.concat(
        [
            main.assign(panel="a_main_P90"),
            threshold.assign(panel="b_P90_P95_sensitivity"),
            period_source.assign(panel="c_early_recent_estimates"),
            event_difference.assign(panel="c_early_recent_difference"),
        ],
        ignore_index=True,
        sort=False,
    )
    source.to_csv(FIGURE_DIR / "Figure2_relative_additional_effect_source_data.csv", index=False)

    fig = plt.figure(figsize=(7.35, 6.45))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.135,
        right=0.985,
        top=0.955,
        bottom=0.085,
        height_ratios=[0.94, 1.08],
        wspace=0.38,
        hspace=0.46,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    plot_fig2_a(ax_a, main)
    plot_fig2_b(ax_b, threshold)
    plot_fig2_c(ax_c, event_by_period, event_difference, main)
    save_figure(fig, FIGURE_DIR / "Figure2_relative_additional_effect_sti_free")
    plt.close(fig)

    qa = """# Figure 2 relative-effect QA

- All three panels use the same relative-effect scale, limits and tick positions.
- The plotted metric is the direction-oriented ATT divided by an absolute matched drought-control mean and multiplied by 100.
- Signs are oriented so positive values always mean greater loss, deeper decline or longer recovery under compound dry-hot events.
- Panel a shows the main P90 matched result; panel b shows P90/P95 threshold sensitivity; panel c shows early/recent estimates.
- Panels a-c use the same seven outcomes: three event metrics and four cumulative anomaly-loss metrics.
- Panel a uses one neutral gray fill for every bar because bar color does not encode an additional variable.
- Panel b includes newly matched P95 cumulative NDVI, SIF, GPP and ET anomaly-loss estimates rather than reusing P90 values.
- Panel c uses the full-record P90 matched drought-control mean as a fixed outcome-specific denominator for both periods, so early-recent differences are not affected by period-specific denominator shifts.
- Panel c reports the plotted recent-period percentage minus the plotted early-period percentage. The resulting percentage-scale difference is shown for every outcome, including recovery time.
- The P95 cumulative matching audit returned a maximum absolute post-match SMD of 0.111; this is retained as a transparent residual-balance diagnostic.
- ET is used consistently in the figure labels and denotes transpiration.
- Panel a labels the raw ATT in each outcome's original units; complete means and confidence intervals remain in Supplementary Tables 1 and 4.
"""
    (FIGURE_DIR / "Figure2_relative_additional_effect_sti_free_qa.md").write_text(qa, encoding="utf-8")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_sti_free_panel_model(ax: plt.Axes, frame: pd.DataFrame, module) -> None:
    offsets = {"Base model": -0.12, "+ VPD model": 0.12}
    colors = {"Base model": "#7884B4", "+ VPD model": "#E28E2C"}
    y_map = {term: index for index, term in enumerate(module.TERM_ORDER)}
    ax.axvline(0, color=module.PALETTE["neutral_mid"], lw=0.65, ls=(0, (2.3, 2.3)), zorder=0)
    ax.grid(axis="x", color="#ECECEC", lw=0.45)
    for _, row in frame.iterrows():
        y = y_map[row["term"]] + offsets[row["model"]]
        coefficient = float(row["coef"])
        ax.plot([0, coefficient], [y, y], color=colors[row["model"]], lw=0.9, alpha=0.35, zorder=1)
        ax.errorbar(
            coefficient,
            y,
            xerr=[[coefficient - float(row["ci_low"])], [float(row["ci_high"]) - coefficient]],
            fmt="o",
            ms=4.1,
            color=colors[row["model"]],
            ecolor=colors[row["model"]],
            elinewidth=0.85,
            capsize=2.0,
            markeredgecolor="black",
            markeredgewidth=0.3,
            zorder=3,
        )
        offset = -0.005 if coefficient < 0 else 0.005
        ax.text(
            coefficient + offset,
            y,
            f"{coefficient:.3f}",
            ha="right" if coefficient < 0 else "left",
            va="center",
            fontsize=5.3,
        )
    ax.set_yticks(np.arange(len(module.TERM_ORDER)))
    ax.set_yticklabels([module.TERM_LABEL[term] for term in module.TERM_ORDER])
    ax.set_ylim(len(module.TERM_ORDER) - 0.5, -0.5)
    ax.set_xlim(-0.17, 0.11)
    ax.set_xticks([-0.15, -0.10, -0.05, 0.00, 0.05, 0.10])
    ax.set_xlabel("Standardized coefficient (95% CI)")
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D([0], [0], marker="o", color=colors["Base model"], lw=1, markersize=4.2, label="Base"),
        Line2D([0], [0], marker="o", color=colors["+ VPD model"], lw=1, markersize=4.2, label="+ VPD"),
    ]
    ax.legend(handles=handles, loc="lower left", handlelength=1.1, borderaxespad=0)
    module.add_panel_label(ax, "a")


def build_figure3() -> None:
    sys.path.insert(0, str(MAIN_SCRIPT_DIR))
    import fig_common

    fig_common.OUTPUT_DIR = FIGURE_DIR
    module = load_module("figure3_vpd_revision", MAIN_SCRIPT_DIR / "03_main_fig3_vpd_evidence_nature.py")
    module.OUTPUT_DIR = FIGURE_DIR
    module.BASE_MODEL_PATH = ANALYSIS_DIR / "pooled_panel_base_sti_free.csv"
    module.VPD_MODEL_PATH = ANALYSIS_DIR / "pooled_panel_vpd_sti_free.csv"
    module.FIGURE_BASENAME = "Figure3_atmospheric_demand_evidence_sti_free"
    module.plot_model = lambda ax, frame: plot_sti_free_panel_model(ax, frame, module)
    module.main()


def plot_selectivity_heatmap(ax: plt.Axes, heat_df: pd.DataFrame) -> None:
    metric_order = ["NDVI", "SIF", "GPP", "ET", "SPEI", "SM", "Temp", "VPD"]
    land_order = ["Forest", "Grassland", "Savannas", "Cropland", "Shrub"]
    heat = (
        heat_df.pivot(index="land_cover_broad", columns="metric", values="z")
        .loc[land_order, metric_order]
        .astype(float)
    )
    matrix = heat.to_numpy()
    cmap = LinearSegmentedColormap.from_list("loss_stress", ["#2C7BB6", "#F7F7F7", "#D7191C"])
    image = ax.imshow(matrix, cmap=cmap, norm=Normalize(vmin=-1.5, vmax=1.5), aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(metric_order)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(land_order, fontweight="bold")
    ax.tick_params(length=0)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=5.7,
                fontweight="bold",
                color="white" if abs(value) >= 1.15 else "black",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.figure.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.075,
        pad=0.14,
        ticks=[-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5],
    )
    colorbar.outline.set_linewidth(0.4)
    colorbar.ax.tick_params(labelsize=5.2, width=0.35, length=2)
    colorbar.set_label("Relative loss / stress intensity (Z-score)", fontsize=5.8, fontweight="bold", labelpad=2)
    add_panel_label(ax, "c", x=-0.13, y=1.05)


def build_figure4() -> None:
    sys.path.insert(0, str(MAIN_SCRIPT_DIR))
    import fig_common

    fig_common.OUTPUT_DIR = FIGURE_DIR
    module = load_module(
        "figure4_selectivity_revision",
        MAIN_SCRIPT_DIR / "03_main_fig3_reordered_biomes_backgrounds_nature.py",
    )
    module.OUTPUT_DIR = FIGURE_DIR
    module.FIGURE_BASENAME = "Figure4_climate_ecosystem_selectivity_sti_free"
    module.apply_nature_style()
    plt.rcParams.update({"axes.facecolor": "white", "savefig.facecolor": "white"})

    thermal = pd.read_csv(ANALYSIS_DIR / "thermal_background_period_difference_sti_free.csv").rename(
        columns={
            "group": "thermal_regime",
            "recent_att_mean": "late_att_mean",
            "recent_minus_early": "late_minus_early",
        }
    )
    biome = pd.read_csv(ANALYSIS_DIR / "landcover_period_difference_sti_free.csv").rename(
        columns={
            "group": "land_cover_broad",
            "recent_att_mean": "late_att_mean",
            "recent_minus_early": "late_minus_early",
        }
    )
    biome_order = biome.sort_values("late_minus_early", ascending=False)["land_cover_broad"].tolist()
    heatmap = module.prepare_heatmap()
    pd.concat(
        [
            thermal.assign(panel="a_thermal_background"),
            biome.assign(panel="b_ecosystem"),
            heatmap.assign(panel="c_multivariate_heatmap"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(FIGURE_DIR / "Figure4_climate_ecosystem_selectivity_sti_free_source_data.csv", index=False)

    fig = plt.figure(figsize=(7.35, 5.45))
    grid = fig.add_gridspec(
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
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    lower = grid[1, :].subgridspec(1, 3, width_ratios=[0.12, 0.76, 0.12], wspace=0)
    ax_c = fig.add_subplot(lower[0, 1])
    module.plot_period_dumbbell(
        ax_a,
        thermal,
        group_col="thermal_regime",
        order=["Hot", "Warm", "Cool"],
        colors=module.PERIOD_COLORS_THERMAL,
        xlim=(-0.005, 0.088),
        xticks=[0, 0.02, 0.04, 0.06, 0.08],
        xlabel="Matched additional total event loss",
        panel_label="a",
    )
    module.plot_period_dumbbell(
        ax_b,
        biome,
        group_col="land_cover_broad",
        order=biome_order,
        colors=module.PERIOD_COLORS_BIOME,
        xlim=(-0.015, 0.165),
        xticks=[0, 0.05, 0.10, 0.15],
        xlabel="Matched additional total event loss",
        panel_label="b",
    )
    plot_selectivity_heatmap(ax_c, heatmap)
    module.save_figure(fig, FIGURE_DIR / module.FIGURE_BASENAME)
    plt.close(fig)
    qa = """# Figure 4 STI-free consistency QA

- Panels a and b and Supplementary Table 5 are generated from the same STI-free matching outputs.
- Thermal background is defined once from the 2001-2024 mean growing-season temperature of each pixel and divided into pixel terciles.
- Panel c contains NDVI, SIF, GPP, transpiration, SPEI, root-zone soil moisture, temperature and VPD; STI is absent.
- Asterisks denote permutation P < 0.05 for the recent-minus-early difference.
"""
    (FIGURE_DIR / "Figure4_climate_ecosystem_selectivity_sti_free_qa.md").write_text(qa, encoding="utf-8")


def finalize_heatmap_long(frame: pd.DataFrame, group_order: list[str], metric_order: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["group"] = pd.Categorical(out["group"], categories=group_order, ordered=True)
    out["metric"] = pd.Categorical(out["metric"], categories=metric_order, ordered=True)
    out = out.sort_values(["group", "metric"])
    out["stress_z"] = out.groupby("metric", observed=False)["stress_value"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=1) if values.std(ddof=1) > 0 else 0.0
    )
    return out


def prepare_s2_event_heatmap(event_order: list[str], metrics: list[str]) -> pd.DataFrame:
    labels = {"ndvi": "NDVI", "sif": "SIF", "gpp": "GPP", "et": "ET", "spei": "SPEI", "sm": "SM", "temp": "Temp", "vpd": "VPD"}
    cumulative_columns = {
        "ndvi": "ndvi_cum_loss",
        "sif": "sif_cum_loss",
        "gpp": "gpp_cum_loss",
        "et": "et_cum_loss",
    }
    cumulative = pd.read_csv(
        CUMULATIVE_EVENT_PATH,
        usecols=["event_type", *cumulative_columns.values()],
    )
    cumulative = cumulative.groupby("event_type", observed=True)[list(cumulative_columns.values())].mean()
    climate = pd.read_csv(SUPP_DATA_DIR / "event_multivariate_response_anomaly_by_type.csv").set_index("event_type")
    rows = []
    for group in event_order:
        for metric in metrics:
            if metric in cumulative_columns:
                raw_value = float(cumulative.loc[group, cumulative_columns[metric]])
                stress_value = raw_value
                definition = "Cumulative negative anomaly loss during the event"
            else:
                raw_value = float(pd.to_numeric(climate.loc[group, f"{metric}_anom_event_mean_mean"], errors="coerce"))
                stress_value = -raw_value if metric in {"spei", "sm"} else raw_value
                definition = "Mean event-period anomaly, oriented so positive denotes stronger stress"
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "metric_label": labels[metric],
                    "raw_value": raw_value,
                    "stress_value": stress_value,
                    "value_definition": definition,
                }
            )
    return finalize_heatmap_long(pd.DataFrame(rows), event_order, metrics)


def plot_s2_heatmap(ax: plt.Axes, frame: pd.DataFrame, groups: list[str], metrics: list[str], ylabel: str, norm) -> plt.Axes:
    labels = {"ndvi": "NDVI", "sif": "SIF", "gpp": "GPP", "et": "ET", "spei": "SPEI", "sm": "SM", "temp": "Temp", "vpd": "VPD"}
    heat = frame.pivot(index="group_label", columns="metric_label", values="stress_z").loc[groups, [labels[x] for x in metrics]]
    cmap = LinearSegmentedColormap.from_list("stress", ["#2C7BB6", "#F7F7F7", "#D7191C"])
    image = ax.imshow(heat.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([labels[x] for x in metrics])
    ax.xaxis.tick_top()
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_ylabel(ylabel)
    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            z = float(heat.iloc[row, col])
            ax.text(col, row, f"{z:.1f}", ha="center", va="center", fontsize=5.2, fontweight="bold", color="white" if abs(z) >= 1.15 else "black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def build_supplementary_s2() -> None:
    event_order = ["Mild_Reference", "Heat_Dominant", "Drought_Dominant", "Sequential_Heat_Drought", "Sequential_Drought_Heat", "Compound_Overlapping"]
    event_label = {"Mild_Reference": "Mild", "Heat_Dominant": "Heat", "Drought_Dominant": "Drought", "Sequential_Heat_Drought": "Heat -> drought", "Sequential_Drought_Heat": "Drought -> heat", "Compound_Overlapping": "Compound"}
    metrics = ["ndvi", "sif", "gpp", "et", "spei", "sm", "temp", "vpd"]
    event = prepare_s2_event_heatmap(event_order, metrics)
    event["group_label"] = event["group"].astype(str).map(event_label)
    event.assign(panel="event_type").to_csv(
        FIGURE_DIR / "Supplementary_Figure_S2_anomaly_multivariate_sti_free_source_data.csv",
        index=False,
    )
    norm = Normalize(vmin=-1.5, vmax=1.5)
    fig = plt.figure(figsize=(8.25, 3.25))
    grid = fig.add_gridspec(1, 1, left=0.15, right=0.985, top=0.84, bottom=0.27)
    ax = fig.add_subplot(grid[0, 0])
    image = plot_s2_heatmap(ax, event, [event_label[x] for x in event_order], metrics, "Event type", norm)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.10, pad=0.16, ticks=[-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
    colorbar.outline.set_linewidth(0.35)
    colorbar.ax.tick_params(labelsize=5.7, width=0.4, length=2)
    colorbar.set_label("Relative loss / stress intensity (Z-score)", fontsize=6.0, fontweight="bold", labelpad=1.5)
    save_figure(fig, FIGURE_DIR / "Supplementary_Figure_S2_anomaly_multivariate_sti_free")
    plt.close(fig)
    qa = """# Supplementary Figure S2 heatmap QA

- The heatmap uses cumulative anomaly losses for NDVI, SIF, GPP and ET; hydroclimatic cells use oriented event-period anomalies.
- The duplicated ecosystem heatmap was removed because the corresponding compound-event land-cover evidence is already presented in Figure 4c.
- Positive Z-scores (red) indicate relatively greater loss or stress within a variable; negative Z-scores (blue) indicate relatively lower loss or stress.
- Mild events are the lowest cumulative vegetation-loss group, whereas compound dry-hot events are the highest across the four vegetation variables.
"""
    (FIGURE_DIR / "Supplementary_Figure_S2_anomaly_multivariate_sti_free_qa.md").write_text(qa, encoding="utf-8")


def main() -> None:
    apply_style()
    build_figure2()
    build_figure3()
    build_figure4()
    build_supplementary_s2()
    print(FIGURE_DIR)


if __name__ == "__main__":
    main()
