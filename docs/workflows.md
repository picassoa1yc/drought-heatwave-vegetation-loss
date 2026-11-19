# Workflow Index

Paths under `code/originals/` group the study's source scripts by analysis and
figure workflow. `code/run_all.py` routes prepared inputs to these scripts and
writes results under `results/` by default.

| Analysis or manuscript item | Source script or output |
| --- | --- |
| Pixel-level event detection and monthly panels | `integrated_vegetation_framework.py` |
| Cumulative anomaly-loss inputs | `analyze_cumulative_anomaly_attribution.py` |
| Figure 1 and supplementary event-pathway maps | `01_main_fig1_geography_hierarchy_nature.py`, `05_supp_event_pathway_maps.py` |
| Figure 2; Supplementary Tables 2-5 | `recompute_sti_free.py`, `recompute_p95_cumulative_sti_free.py`, `build_revision_figures.py` |
| Figure 3: pooled models, VPD strata and trajectories | `fit_panel_models_sti_free.py`, `03_main_fig3_vpd_evidence_nature.py`, `build_revision_figures.py` |
| Figure 4; Supplementary Table 6 | `recompute_sti_free.py`, `03_main_fig3_reordered_biomes_backgrounds_nature.py`, `build_revision_figures.py` |
| Figure 5: R trajectories, local interaction map and moving-window estimates | `04_figure4_dynamics_selectivity.R`, `run_core_temporal_att.py`, figure assembly scripts `05`-`08` |
| Tower contrasts; Supplementary Table 7 | `build_fluxnet_revision.py` and `run_fluxnet()` in `code/run_all.py` |

The tower comparison summarizes within-site high- versus low-VPD drought
contrasts. The GPP reduction is expressed relative to the mean of the site GPP
baselines: `100 * mean_site(low-VPD anomaly - high-VPD anomaly) / mean_site(site raw GPP mean)`.
The main tower comparison uses 88 sites; the Hot group contains 13 sites and its
normalized GPP reduction is 18.20%. Tower GPP is expressed in `g C m-2 d-1`.

## Figure 5 Assembly

Install `requirements-figure5.txt` and the R packages `ggplot2`, `dplyr`,
`tidyr`, `readr`, `forcats`, `stringr`, `patchwork`, `scales`, `viridis`,
`rnaturalearth` and `rnaturalearthdata`. Choose a working directory outside the
source tree, and set `VEGETATION_PROJECT_ROOT` to its absolute path. Paths in
the steps below are relative to that working directory unless otherwise noted.

1. Put `event_trajectory_summary_robust.csv`,
   `functional_trajectory_summary_stage2_robust.csv` and
   `global_local_coefficients_90p_ndvi_min60.csv` in
   `codex_stage8_20260317/figure_r_package/data/`. Set
   `VEGETATION_FIGURE5_INPUTS` to that absolute directory and
   `VEGETATION_FIGURE5_OUTPUTS` to the sibling `outputs/` directory.
2. Run `Rscript` on `code/originals/codex_stage8_20260317/figure_r_package/scripts/04_figure4_dynamics_selectivity.R`
   to export the trajectory base PDF.
3. Place `Supplementary_Table_S7_temporal_diagnostics.csv` in
   `codex_stage9_20260516_core_restructure/supplementary_tables_package/tables/`
   and `global_local_coefficients_90p_ndvi_min60.csv` in
   `output_df/global_v05_90p_main/`. The temporal table contains the 8- and
   10-year window estimates used in panels d-f.
4. Run `05_append_sliding_windows_to_stage8_fig4_pdf.py`,
   `06_recompose_fig4_r_ab_new_c.py`, `07_remove_drought_to_heat_from_fig5.py`,
   and `08_enlarge_fig5_spatial_significance.py` in that order. The scripts are
   under `code/originals/codex_stage8_20260317/figure_python_scripts/`. The last
   script writes the assembled figure to
   `codex_stage9_20260516_core_restructure/main_figures_python/outputs/`.
