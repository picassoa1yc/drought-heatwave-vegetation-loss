from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from fig_common import OUTPUT_DIR, PALETTE, apply_nature_style, print_saved, read_stage8_data, save_figure, write_source


FIGURE_BASENAME = "Figure3_atmospheric_demand_evidence_nature"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL_PATH = PROJECT_ROOT / "output_df" / "global_v05_90p_main" / "global_model_coefficients_90p_ndvi.csv"
VPD_MODEL_PATH = PROJECT_ROOT / "output_df" / "global_v05_90p_vpd_main" / "global_model_coefficients_90p_vpd_ndvi_full.csv"
EVENT_METRICS_PATH = (
    PROJECT_ROOT
    / "output_df"
    / "global_v05_90p_vpd_main"
    / "event_analysis_vpd"
    / "vpd_event_level_metrics.csv"
)

TERM_ORDER = ["vpd", "drought_x_heat", "drought", "heat"]
TERM_LABEL = {"vpd": "VPD", "drought_x_heat": "Drought x heat", "drought": "Drought", "heat": "Heat"}
GROUP_ORDER = ["Drought_LowVPD", "Drought_HighVPD", "Compound_HotDry"]
GROUP_LABEL = {
    "Drought_LowVPD": "Low-VPD drought",
    "Drought_HighVPD": "High-VPD drought",
    "Compound_HotDry": "Compound dry-hot",
}
GROUP_COLORS = {
    "Drought_LowVPD": "#42949E",
    "Drought_HighVPD": "#E28E2C",
    "Compound_HotDry": "#B64342",
}
ZONE_ORDER = ["Sig_Aggravation", "Negative_Nonsig", "Nonnegative_Interaction"]
ZONE_LABEL = {
    "Sig_Aggravation": "Significant negative",
    "Negative_Nonsig": "Negative, non-significant",
    "Nonnegative_Interaction": "Positive",
}
ZONE_COLORS = {
    "Sig_Aggravation": "#2166AC",
    "Negative_Nonsig": "#92C5DE",
    "Nonnegative_Interaction": "#B2182B",
}


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.05) -> None:
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


def read_model_coefficients() -> pd.DataFrame:
    rows = []
    for model, path in [("Base model", BASE_MODEL_PATH), ("+ VPD model", VPD_MODEL_PATH)]:
        df = pd.read_csv(path)
        df = df[df["term"].isin(TERM_ORDER)].copy()
        df["model"] = model
        df["ci_low"] = df["coef"] - 1.96 * df["std_err"]
        df["ci_high"] = df["coef"] + 1.96 * df["std_err"]
        rows.append(df)
    return pd.concat(rows, ignore_index=True, sort=False)


def compute_event_statistics() -> pd.DataFrame:
    cols = ["event_type", "LOSStot", "gpp_event_minus_pre", "vpd_event_excess"]
    df = pd.read_csv(EVENT_METRICS_PATH, usecols=cols)
    drought_median = df.loc[df["event_type"].eq("Drought_Dominant"), "vpd_event_excess"].median()
    df["event_group"] = np.select(
        [
            df["event_type"].eq("Compound_Overlapping"),
            df["event_type"].eq("Drought_Dominant") & df["vpd_event_excess"].le(drought_median),
            df["event_type"].eq("Drought_Dominant") & df["vpd_event_excess"].gt(drought_median),
        ],
        ["Compound_HotDry", "Drought_LowVPD", "Drought_HighVPD"],
        default="Other",
    )
    df = df[df["event_group"].isin(GROUP_ORDER)].copy()
    df["gpp_decline"] = -df["gpp_event_minus_pre"]
    rows = []
    for group, group_df in df.groupby("event_group", observed=True):
        for metric, source in [
            ("Total event loss", "LOSStot"),
            ("GPP loss", "gpp_decline"),
            ("VPD anomaly", "vpd_event_excess"),
        ]:
            values = pd.to_numeric(group_df[source], errors="coerce").dropna()
            mean = float(values.mean())
            se = float(values.std(ddof=1) / np.sqrt(len(values)))
            rows.append(
                {
                    "event_group": group,
                    "metric": metric,
                    "mean": mean,
                    "se": se,
                    "ci_low": mean - 1.96 * se,
                    "ci_high": mean + 1.96 * se,
                    "n": int(len(values)),
                    "drought_vpd_anomaly_median": float(drought_median),
                }
            )
    return pd.DataFrame(rows)


def build_event_contrasts() -> pd.DataFrame:
    vpd = read_stage8_data("vpd_trajectory_summary_robust.csv")
    vpd = vpd[vpd["variable"].isin(["ndvi", "gpp", "vpd"])].copy()
    functional = read_stage8_data("functional_trajectory_summary_stage2_robust.csv")
    functional = functional[functional["variable"].eq("sif")].copy()
    df = pd.concat([vpd, functional], ignore_index=True, sort=False)
    df = df[df["event_type"].isin(["Drought_Dominant", "Compound_Overlapping"])].copy()
    for col in ["stage_month", "mean_value", "se_value", "n_obs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[
        ((df["stage"] == "pre") & df["stage_month"].between(-6, -1))
        | ((df["stage"] == "event") & df["stage_month"].between(1, 10))
    ].copy()
    df["relative_month"] = np.where(df["stage"].eq("pre"), df["stage_month"], df["stage_month"] - 1)
    index = ["variable", "stage", "stage_month", "relative_month"]
    means = df.pivot_table(index=index, columns="event_type", values="mean_value", aggfunc="first")
    ses = df.pivot_table(index=index, columns="event_type", values="se_value", aggfunc="first")
    ns = df.pivot_table(index=index, columns="event_type", values="n_obs", aggfunc="first")
    out = means.reset_index()
    out["difference"] = means["Compound_Overlapping"].to_numpy() - means["Drought_Dominant"].to_numpy()
    out["difference_se"] = np.sqrt(
        np.square(ses["Compound_Overlapping"].to_numpy()) + np.square(ses["Drought_Dominant"].to_numpy())
    )
    out["ci_low"] = out["difference"] - 1.96 * out["difference_se"]
    out["ci_high"] = out["difference"] + 1.96 * out["difference_se"]
    out["n_compound"] = ns["Compound_Overlapping"].to_numpy(dtype=float)
    out["n_drought"] = ns["Drought_Dominant"].to_numpy(dtype=float)
    out["unit"] = out["variable"].map(
        {
            "ndvi": "unitless",
            "sif": "W m-2 um-1 sr-1",
            "gpp": "g C m-2 month-1",
            "vpd": "kPa",
        }
    )
    return out.sort_values(["variable", "relative_month"])


def build_local_climate_trajectories() -> pd.DataFrame:
    df = read_stage8_data("event_trajectory_by_zone_vpd_summary_robust.csv")
    df = df[
        df["event_type"].eq("Compound_Overlapping")
        & df["variable"].isin(["spei", "temp", "vpd"])
    ].copy()
    for col in ["stage_month", "mean_value", "n_obs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[
        ((df["stage"] == "pre") & df["stage_month"].between(-6, -1))
        | ((df["stage"] == "event") & df["stage_month"].between(1, 6))
    ].copy()
    baselines = []
    for (zone, variable), sub in df[df["stage"].eq("pre")].groupby(["amplification_zone", "variable"], observed=True):
        valid = sub[["mean_value", "n_obs"]].dropna()
        baselines.append(
            {
                "amplification_zone": zone,
                "variable": variable,
                "pre_event_baseline": np.average(valid["mean_value"], weights=valid["n_obs"]),
            }
        )
    df = df.merge(pd.DataFrame(baselines), on=["amplification_zone", "variable"], how="left")
    df["change_from_pre"] = df["mean_value"] - df["pre_event_baseline"]
    df["relative_month"] = np.where(df["stage"].eq("pre"), df["stage_month"], df["stage_month"] - 1)
    return df.sort_values(["variable", "amplification_zone", "relative_month"])


def plot_model(ax: plt.Axes, df: pd.DataFrame) -> None:
    offsets = {"Base model": -0.12, "+ VPD model": 0.12}
    colors = {"Base model": "#7884B4", "+ VPD model": "#E28E2C"}
    y_map = {term: i for i, term in enumerate(TERM_ORDER)}
    ax.axvline(0, color=PALETTE["neutral_mid"], lw=0.65, ls=(0, (2.3, 2.3)), zorder=0)
    ax.grid(axis="x", color="#ECECEC", lw=0.45)
    for _, row in df.iterrows():
        y = y_map[row["term"]] + offsets[row["model"]]
        coef = float(row["coef"])
        ax.plot(
            [0, coef],
            [y, y],
            color=colors[row["model"]],
            lw=0.9,
            alpha=0.35,
            solid_capstyle="round",
            zorder=1,
        )
        ax.errorbar(
            coef,
            y,
            xerr=[[coef - row["ci_low"]], [row["ci_high"] - coef]],
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
        ax.text(coef + (-0.005 if coef < 0 else 0.004), y, f"{coef:.3f}", ha="right" if coef < 0 else "left", va="center", fontsize=5.3)
    ax.set_yticks(np.arange(len(TERM_ORDER)))
    ax.set_yticklabels([TERM_LABEL[x] for x in TERM_ORDER])
    ax.set_ylim(len(TERM_ORDER) - 0.5, -0.5)
    ax.set_xlim(-0.165, 0.025)
    ax.set_xticks([-0.15, -0.10, -0.05, 0])
    ax.set_xlabel("Standardized coefficient (95% CI)")
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D([0], [0], marker="o", color=colors["Base model"], lw=1, markersize=4.2, label="Base"),
        Line2D([0], [0], marker="o", color=colors["+ VPD model"], lw=1, markersize=4.2, label="+ VPD"),
    ]
    ax.legend(handles=handles, loc="lower left", handlelength=1.1, borderaxespad=0)
    add_panel_label(ax, "a")


def plot_event_summary(fig: plt.Figure, spec, df: pd.DataFrame) -> list[plt.Axes]:
    sub = spec.subgridspec(1, 3, wspace=0.52)
    axes = [fig.add_subplot(sub[0, i]) for i in range(3)]
    metrics = [
        ("Total event loss", "Total event loss", "Mean loss"),
        ("GPP loss", "GPP", "Mean loss"),
        ("VPD anomaly", "VPD", "Mean anomaly (kPa)"),
    ]
    x = np.arange(len(GROUP_ORDER))
    colors = [GROUP_COLORS[g] for g in GROUP_ORDER]
    for i, (ax, (metric, title, ylabel)) in enumerate(zip(axes, metrics)):
        d = df[df["metric"].eq(metric)].set_index("event_group").reindex(GROUP_ORDER)
        values = d["mean"].to_numpy(dtype=float)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#EFEFEF", lw=0.35, zorder=0)
        ax.axhline(0, color=PALETTE["neutral_mid"], lw=0.55, zorder=1)
        ax.bar(
            x,
            values,
            width=0.62,
            color=colors,
            edgecolor="#333333",
            linewidth=0.35,
            zorder=2,
        )
        ax.errorbar(
            x,
            values,
            yerr=[values - d["ci_low"].to_numpy(), d["ci_high"].to_numpy() - values],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.75,
            capsize=2,
            zorder=3,
        )
        ax.set_title(title, fontsize=6.8, pad=2)
        ax.set_ylabel(ylabel, fontsize=5.8, labelpad=2)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        for xi, value in zip(x, values):
            pad = max(np.ptp(values) * 0.035, abs(value) * 0.025, 0.005)
            ax.text(xi, value + pad, f"{value:.2f}", ha="center", va="bottom", fontsize=5.1)
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=GROUP_COLORS[g],
            markeredgecolor="#333333",
            markeredgewidth=0.35,
            markersize=5,
            label=GROUP_LABEL[g],
        )
        for g in GROUP_ORDER
    ]
    axes[1].legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        handletextpad=0.3,
        columnspacing=0.65,
        borderaxespad=0,
        fontsize=5.2,
    )
    add_panel_label(axes[0], "b", x=-0.30, y=1.08)
    return axes


def plot_event_contrasts(fig: plt.Figure, spec, df: pd.DataFrame) -> list[plt.Axes]:
    sub = spec.subgridspec(1, 4, wspace=0.50)
    variables = [
        ("ndvi", "NDVI", "Difference", "#2166AC"),
        ("sif", "SIF", r"Difference (W m$^{-2}$ $\mu$m$^{-1}$ sr$^{-1}$)", "#4393C3"),
        ("gpp", "GPP", r"Difference (g C m$^{-2}$ month$^{-1}$)", "#053061"),
        ("vpd", "VPD", "Difference (kPa)", "#D6604D"),
    ]
    axes = [fig.add_subplot(sub[0, i]) for i in range(4)]
    for i, (ax, (variable, title, ylabel, color)) in enumerate(zip(axes, variables)):
        d = df[df["variable"].eq(variable)].sort_values("relative_month")
        x = d["relative_month"].to_numpy(dtype=float)
        y = d["difference"].to_numpy(dtype=float)
        ax.axvspan(-6.5, -0.5, color="#EEF2F6", zorder=0)
        ax.axvspan(-0.5, 9.5, color="#FAEFE8", zorder=0)
        ax.axhline(0, color=PALETTE["neutral_mid"], lw=0.6)
        ax.axvline(0, color=PALETTE["neutral_mid"], lw=0.65, ls=(0, (2.3, 2.3)))
        ax.fill_between(x, d["ci_low"], d["ci_high"], color=color, alpha=0.18, linewidth=0)
        ax.plot(x, y, color=color, lw=1.35, marker="o", ms=2.4)
        ax.set_title(title, loc="left", fontsize=6.8, fontweight="bold", pad=1.5)
        ax.set_ylabel(ylabel, fontsize=5.2, labelpad=1.2)
        ax.set_xlim(-6.3, 9.3)
        ax.set_xticks([-6, 0, 3, 6, 9])
        ax.set_xlabel("Relative month")
        ax.grid(axis="y", color="#ECECEC", lw=0.4)
        if i == 0:
            add_panel_label(ax, "c", x=-0.34, y=1.13)
    axes[1].text(
        1.05,
        1.14,
        "Compound dry-hot minus drought-only events",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=6.6,
        fontweight="bold",
        color=PALETTE["neutral_dark"],
    )
    return axes


def plot_local_climate(fig: plt.Figure, spec, df: pd.DataFrame) -> list[plt.Axes]:
    sub = spec.subgridspec(1, 3, wspace=0.34)
    variables = [
        ("spei", "SPEI", "Change from pre-event mean"),
        ("temp", "Temperature", "Change (degrees C)"),
        ("vpd", "VPD", "Change (kPa)"),
    ]
    axes = [fig.add_subplot(sub[0, i]) for i in range(3)]
    for i, (ax, (variable, title, ylabel)) in enumerate(zip(axes, variables)):
        ax.axvspan(-6.5, -0.5, color="#EEF2F6", zorder=0)
        ax.axvspan(-0.5, 5.5, color="#FAEFE8", zorder=0)
        ax.axhline(0, color=PALETTE["neutral_mid"], lw=0.6)
        ax.axvline(0, color=PALETTE["neutral_mid"], lw=0.65, ls=(0, (2.3, 2.3)))
        for zone in ZONE_ORDER:
            d = df[(df["variable"].eq(variable)) & (df["amplification_zone"].eq(zone))]
            ax.plot(d["relative_month"], d["change_from_pre"], color=ZONE_COLORS[zone], lw=1.3)
        ax.set_title(title, loc="left", fontsize=6.8, fontweight="bold", pad=1.5)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-6.3, 5.3)
        ax.set_xticks([-6, -3, 0, 3, 5])
        ax.set_xlabel("Relative month")
        ax.grid(axis="y", color="#ECECEC", lw=0.4)
        if i == 0:
            add_panel_label(ax, "d", x=-0.23, y=1.22)
    handles = [Line2D([0], [0], color=ZONE_COLORS[z], lw=1.4, label=ZONE_LABEL[z]) for z in ZONE_ORDER]
    axes[1].legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.11),
        ncol=3,
        title="Local drought x heat coefficient",
        title_fontsize=5.8,
        handlelength=1.6,
        columnspacing=0.75,
        borderaxespad=0,
        fontsize=5.2,
    )
    return axes


def write_qa_report() -> Path:
    path = OUTPUT_DIR / f"{FIGURE_BASENAME}_qa.md"
    text = """# Figure 3 atmospheric-demand evidence QA

Core claim: higher atmospheric moisture demand accompanies stronger vegetation suppression during drought but does not fully account for the non-additive drought-by-heat response.

Panel map:
- a: global panel-model coefficients with robust 95% confidence intervals before and after adding VPD.
- b: descriptive low-VPD drought, high-VPD drought and compound dry-hot event summaries.
- c: compound-minus-drought event-aligned differences in NDVI, SIF, GPP and VPD.
- d: compound-event SPEI, temperature and VPD changes grouped by local drought-by-heat coefficient class.

Panel c retains native units: NDVI is unitless, SIF is W m-2 um-1 sr-1, GPP is g C m-2 month-1 and VPD is kPa.

Panel b and panel c are descriptive rather than matched ATT analyses. FLUXNET remains a separate supplementary validation figure to avoid overloading the main figure.
"""
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    apply_nature_style()
    model = read_model_coefficients()
    event = compute_event_statistics()
    contrasts = build_event_contrasts()
    local = build_local_climate_trajectories()

    source = pd.concat(
        [
            model.assign(panel="a_panel_model"),
            event.assign(panel="b_event_vpd_stratification"),
            contrasts.assign(panel="c_event_aligned_contrasts"),
            local.assign(panel="d_local_climate_trajectories"),
        ],
        ignore_index=True,
        sort=False,
    )
    source_path = write_source(source, f"{FIGURE_BASENAME}_source_data.csv")

    fig = plt.figure(figsize=(7.35, 7.65))
    gs = fig.add_gridspec(
        3,
        2,
        left=0.105,
        right=0.985,
        top=0.955,
        bottom=0.075,
        width_ratios=[0.92, 1.08],
        height_ratios=[0.88, 1.04, 0.98],
        wspace=0.35,
        hspace=0.62,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    plot_model(ax_a, model)
    plot_event_summary(fig, gs[0, 1], event)
    plot_event_contrasts(fig, gs[1, :], contrasts)
    plot_local_climate(fig, gs[2, :], local)

    qa_path = write_qa_report()
    saved = save_figure(fig, OUTPUT_DIR / FIGURE_BASENAME)
    plt.close(fig)
    print_saved(saved, [source_path, qa_path])


if __name__ == "__main__":
    main()
