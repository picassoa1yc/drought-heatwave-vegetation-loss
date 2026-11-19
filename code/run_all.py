"""Run the manuscript analyses from prepared inputs."""
from pathlib import Path
import argparse
import ast
import importlib.util
import json
import os
import platform
import sys
import time
import types

import numpy as np
import pandas as pd
import bottleneck

CODE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("CAPSULE_DATA", CODE.parent / "data")).resolve()
RESULTS = Path(os.environ.get("CAPSULE_RESULTS", CODE.parent / "results")).resolve()
INPUT = DATA / "input"
REFERENCE = DATA / "reference"
ANALYSIS = RESULTS / "analysis_outputs"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
ORIGINAL = CODE / "originals"
REV = "codex_revision_v4_1_20260715/scripts"
FIG = "codex_stage9_20260516_core_restructure/main_figures_python/scripts"

MATCHING_INPUTS = (
    "p90_events.csv.gz", "p95_events.csv.gz", "cumulative_events.csv.gz",
    "pixel_background_regimes.csv",
)
REFERENCE_TABLES = (
    "p90_global_att_sti_free.csv", "p95_global_att_sti_free.csv", "p90_balance_sti_free.csv",
    "cumulative_global_att_sti_free.csv", "p95_cumulative_global_att_sti_free.csv",
    "matching_sensitivity_sti_free.csv", "event_period_difference_sti_free.csv",
    "cumulative_period_difference_sti_free.csv", "landcover_period_difference_sti_free.csv",
    "thermal_background_period_difference_sti_free.csv", "fluxnet_paired_site_contrasts.csv",
)
POOLED_TABLES = ("pooled_panel_base_sti_free.csv", "pooled_panel_vpd_sti_free.csv")
PLOT_TABLES = (
    "p90_global_att_sti_free.csv", "cumulative_global_att_sti_free.csv",
    "heat_threshold_p90_p95_sti_free.csv", "cumulative_heat_threshold_p90_p95_sti_free.csv",
    "event_att_by_period_sti_free.csv", "cumulative_att_by_period_sti_free.csv",
    "event_period_difference_sti_free.csv", "cumulative_period_difference_sti_free.csv",
    "thermal_background_period_difference_sti_free.csv", "landcover_period_difference_sti_free.csv",
)
FIGURE_INPUTS = (
    "spatial_event_summary_by_pixel.csv", "event_summary_by_type.csv",
    "vpd_trajectory_summary_robust.csv", "functional_trajectory_summary_stage2_robust.csv",
    "event_trajectory_by_zone_vpd_summary_robust.csv",
    "event_multivariate_response_anomaly_by_type_landcover.csv",
)


def check_inputs(mode, no_figures):
    required = set()
    has_figures = mode == "figures" or (mode in {"review", "full"} and not no_figures)
    if mode in {"review", "full"}:
        required.update(INPUT / name for name in MATCHING_INPUTS)
    if mode in {"review", "full", "fluxnet"}:
        required.add(INPUT / "fluxnet_monthly_panel.csv")
    if mode in {"full", "pooled"}:
        required.add(INPUT / "pixel_month_panel.csv.gz")
    if mode in {"review", "full", "verify"}:
        required.update(REFERENCE / name for name in REFERENCE_TABLES)
    if has_figures:
        required.add(INPUT / "vpd_events.csv.gz")
        required.update(DATA / "figure_inputs" / name for name in FIGURE_INPUTS)
    if (has_figures and not all((ANALYSIS / name).is_file() for name in POOLED_TABLES)) or mode == "full":
        required.update(REFERENCE / name for name in POOLED_TABLES)
    if mode == "figures":
        required.update(ANALYSIS / name for name in PLOT_TABLES)
    if mode == "verify":
        required.update(ANALYSIS / name for name in (*REFERENCE_TABLES, "fluxnet_baseline_normalized_percentages.csv"))
        required.update(REFERENCE / name for name in POOLED_TABLES if (ANALYSIS / name).is_file())
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        raise SystemExit("Required inputs are missing:\n  " + "\n  ".join(missing) + "\nSee docs/inputs.md.")
    return len(required)


def load_original(name, relative, overrides=None):
    """Override only module-level I/O constants, preserving scientific functions."""
    path = ORIGINAL / relative
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    overrides = overrides or {}
    seen = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in overrides:
                node.value = ast.Subscript(value=ast.Name(id="__path_overrides__", ctx=ast.Load()),
                                          slice=ast.Constant(key), ctx=ast.Load())
                seen.add(key)
    missing = set(overrides) - seen
    if missing:
        raise ValueError(f"Unrecognized module I/O overrides: {relative}: {missing}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__["__path_overrides__"] = overrides
    sys.modules[name] = module
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


def prepare():
    for p in [RESULTS, ANALYSIS, TABLES, FIGURES]:
        p.mkdir(parents=True, exist_ok=True)


def core_module():
    return load_original("recompute_sti_free", f"{REV}/recompute_sti_free.py", {
        "ROOT": DATA, "REVISION_ROOT": RESULTS, "OUTPUT_DIR": ANALYSIS, "TABLE_DIR": TABLES,
        "P90_EVENT_PATH": INPUT / "p90_events.csv.gz", "P95_EVENT_PATH": INPUT / "p95_events.csv.gz",
        "CUMULATIVE_PATH": INPUT / "cumulative_events.csv.gz", "BACKGROUND_PATH": INPUT / "pixel_background_regimes.csv",
    })


def run_matching():
    core = core_module()
    core.main()
    p95 = load_original("p95_cumulative", f"{REV}/recompute_p95_cumulative_sti_free.py", {
        "P95_EVENT_LEVEL_PATH": INPUT / "p95_events.csv.gz"})
    p95.main()
    main = pd.read_csv(ANALYSIS / "p90_global_att_sti_free.csv")
    cum = pd.read_csv(ANALYSIS / "cumulative_global_att_sti_free.csv")
    selected = pd.concat([main.loc[main.outcome.isin(["LOSStot", "DECmax", "TIMErec_true"])], cum])
    selected.to_csv(TABLES / "Supplementary_Table_2_global_outcomes.csv", index=False)
    pd.read_csv(ANALYSIS / "p90_balance_sti_free.csv").to_csv(TABLES / "Supplementary_Table_3_balance.csv", index=False)
    pd.read_csv(ANALYSIS / "matching_sensitivity_sti_free.csv").to_csv(TABLES / "Supplementary_Table_4_sensitivity.csv", index=False)
    pd.read_csv(TABLES / "Supplementary_Table_4_sti_free.csv").to_csv(TABLES / "Supplementary_Table_5_periods.csv", index=False)
    pd.read_csv(TABLES / "Supplementary_Table_5_sti_free.csv").to_csv(TABLES / "Supplementary_Table_6_strata.csv", index=False)


def run_fluxnet():
    module = load_original("fluxnet_submission", f"{REV}/build_fluxnet_revision.py", {
        "ROOT": DATA, "REVISION_ROOT": RESULTS, "FIGURE_DIR": FIGURES, "TABLE_DIR": TABLES,
        "ANALYSIS_DIR": ANALYSIS, "PANEL_PATH": INPUT / "fluxnet_monthly_panel.csv"})
    panel = pd.read_csv(INPUT / "fluxnet_monthly_panel.csv")
    site, metadata = module.build_site_contrasts(panel)
    site.to_csv(ANALYSIS / "fluxnet_paired_site_contrasts.csv", index=False)
    metadata.to_csv(ANALYSIS / "fluxnet_eligible_site_metadata.csv", index=False)
    summary = module.contrast_summary(site)
    summary.loc[summary.metric.eq("GPP"), "unit"] = "g C m-2 d-1"
    summary.to_csv(ANALYSIS / "fluxnet_paired_summary.csv", index=False)
    context = module.state_context(panel)
    context.loc[context.metric.eq("GPP"), "unit"] = "g C m-2 d-1"
    context.to_csv(ANALYSIS / "fluxnet_state_context.csv", index=False)
    module.biome_summary(site).to_csv(ANALYSIS / "fluxnet_biome_summary.csv", index=False)
    metrics = [("GPP", "gpp_contrast", "GPP_MEAN_VUT_REF"),
               ("Latent heat flux", "le_contrast", "LE_F_MDS")]
    rows = []
    for label, contrast, raw in metrics:
        base = panel.groupby("site_id")[raw].mean()
        for group in ["All", "Cool", "Warm", "Hot"]:
            f = site if group == "All" else site.loc[site.thermal_background.eq(group)]
            delta = f.set_index("site_id")[contrast].dropna()
            baselines = base.reindex(delta.index)
            denominator = baselines.mean()
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError(f"Invalid group denominator: {label}, {group}")
            rows.append({"metric": label, "group": group, "n_sites": len(delta),
                         "mean_high_minus_low_anomaly": delta.mean(), "mean_site_baseline": denominator,
                         "reduction_percent_of_site_average": -100 * delta.mean() / denominator,
                         "nonpositive_individual_baselines": int(baselines.le(0).sum())})
    normalized = pd.DataFrame(rows)
    normalized.to_csv(ANALYSIS / "fluxnet_baseline_normalized_percentages.csv", index=False)
    table = summary.merge(normalized[["metric", "group", "reduction_percent_of_site_average"]], on=["metric", "group"], how="left")
    table.to_csv(TABLES / "Supplementary_Table_7_FLUXNET.csv", index=False)
    print(normalized.to_string(index=False), flush=True)


def run_pooled():
    panel = INPUT / "pixel_month_panel.csv.gz"
    if not panel.exists():
        raise FileNotFoundError("Full pixel-month input not included. Obtain the full-data archive; no demo is substituted.")
    module = load_original("pooled_submission", f"{REV}/fit_panel_models_sti_free.py", {
        "ROOT": DATA, "REVISION_ROOT": RESULTS, "OUTPUT_DIR": ANALYSIS, "PANEL_PATH": panel})
    module.main()


def run_figures():
    import matplotlib
    matplotlib.use("Agg")
    import cartopy
    cartopy.config["data_dir"] = DATA / "cartopy"
    common = load_original("fig_common", f"{FIG}/fig_common.py", {
        "PACKAGE_ROOT": CODE, "STAGE9_ROOT": ORIGINAL / "codex_stage9_20260516_core_restructure",
        "PROJECT_ROOT": DATA, "OUTPUT_DIR": FIGURES, "SUPP_TABLE_DIR": TABLES,
        "STAGE8_DATA_DIR": DATA / "figure_inputs"})
    plot = load_original("submission_figures", f"{REV}/build_revision_figures.py", {
        "ROOT": DATA, "REVISION_ROOT": RESULTS, "ANALYSIS_DIR": ANALYSIS, "FIGURE_DIR": FIGURES,
        "MAIN_SCRIPT_DIR": ORIGINAL / FIG, "SUPP_DATA_DIR": DATA / "figure_inputs",
        "CUMULATIVE_EVENT_PATH": INPUT / "cumulative_events.csv.gz"})
    # Nested imports use the same path overrides; no output can target the author's workspace.
    def load_nested(name, path):
        overrides = {}
        if path.name == "03_main_fig3_vpd_evidence_nature.py":
            overrides = {"PROJECT_ROOT": DATA, "EVENT_METRICS_PATH": INPUT / "vpd_events.csv.gz"}
        return load_original(name, str(path.relative_to(ORIGINAL)), overrides)
    plot.load_module = load_nested
    plot.apply_style()
    plot.build_figure2()
    if not all((ANALYSIS / name).is_file() for name in POOLED_TABLES):
        import shutil
        for name in POOLED_TABLES:
            shutil.copy2(REFERENCE / name, ANALYSIS / name)
        (ANALYSIS / "pooled_coefficients_provenance.txt").write_text(
            "Archived full-panel estimates displayed in review mode; --mode full recomputes these estimates.\n", encoding="utf-8")
    plot.build_figure3()
    plot.build_figure4()
    for name in ["01_main_fig1_geography_hierarchy_nature.py", "05_supp_event_pathway_maps.py"]:
        mod = load_original("submission_" + name.replace(".py", ""), f"{FIG}/{name}")
        mod.main()


def verify():
    checks = []
    for name in REFERENCE_TABLES:
        actual, expected = pd.read_csv(ANALYSIS / name), pd.read_csv(REFERENCE / name)
        if actual.shape != expected.shape or actual.columns.tolist() != expected.columns.tolist():
            raise AssertionError(f"Reference schema mismatch: {name}")
        maximum = 0.0
        for col in actual:
            if pd.api.types.is_numeric_dtype(expected[col]):
                np.testing.assert_allclose(actual[col], expected[col], equal_nan=True, rtol=1e-9, atol=1e-10,
                                           err_msg=f"Reference numeric mismatch: {name}:{col}")
                finite = np.isfinite(actual[col]) & np.isfinite(expected[col])
                if finite.any():
                    maximum = max(maximum, float(np.max(np.abs(actual.loc[finite, col] - expected.loc[finite, col]))))
            else:
                assert actual[col].fillna("").equals(expected[col].fillna("")), (name, col)
        checks.append({"file": name, "status": "PASS", "maximum_absolute_numeric_error": maximum})
    for name in POOLED_TABLES:
        if (ANALYSIS / name).exists():
            a, b = pd.read_csv(ANALYSIS / name), pd.read_csv(REFERENCE / name)
            np.testing.assert_allclose(a.select_dtypes("number"), b.select_dtypes("number"), rtol=1e-6, atol=1e-9)
            checks.append({"file": name, "status": "PASS", "refitted": not (ANALYSIS / "pooled_coefficients_provenance.txt").exists()})
    flux = pd.read_csv(ANALYSIS / "fluxnet_baseline_normalized_percentages.csv")
    hot = flux.loc[flux.metric.eq("GPP") & flux.group.eq("Hot")].iloc[0]
    assert round(hot.reduction_percent_of_site_average, 2) == 18.20
    assert flux.loc[flux.metric.eq("GPP") & flux.group.eq("All"), "n_sites"].iloc[0] == 88
    report = {"status": "PASS", "checks": checks, "fluxnet_hot_gpp_percent": float(hot.reduction_percent_of_site_average),
              "scope": "Numerical comparison against manuscript reference tables."}
    (RESULTS / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Verified matching outputs and original 88-site tower contrasts.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["review", "full", "pooled", "fluxnet", "figures", "verify"], default="review")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--check-inputs", action="store_true", help="List missing inputs and exit without computing.")
    args = parser.parse_args()
    count = check_inputs(args.mode, args.no_figures)
    if args.check_inputs:
        print(f"Found all {count} required files for mode={args.mode}.")
        return
    prepare()
    if not pd.get_option("compute.use_bottleneck"):
        raise RuntimeError("Bottleneck reductions must be enabled to reproduce archived near-tie matching choices.")
    start = time.time()
    if args.mode in ["review", "full"]:
        run_matching()
        run_fluxnet()
        if args.mode == "full":
            run_pooled()
        if not args.no_figures:
            run_figures()
        verify()
    elif args.mode == "fluxnet":
        run_fluxnet()
    elif args.mode == "pooled":
        run_pooled()
    elif args.mode == "figures":
        run_figures()
    else:
        verify()
    (RESULTS / "run_metadata.json").write_text(json.dumps({
        "mode": args.mode, "seconds": time.time() - start, "python": sys.version,
        "platform": platform.platform(), "data_root": str(DATA), "results_root": str(RESULTS),
        "full_panel_refitted": args.mode == "full", "figure_export_requested": not args.no_figures,
        "raw_raster_preprocessing_rerun": False}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
