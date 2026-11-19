# Prepared Inputs

The analysis reads processed event tables, monthly panels, and figure inputs.
Place these files under `data/` or point `CAPSULE_DATA` to the prepared data
directory. Use the folder layout below when pointing to an extracted analysis
data archive.

| Path relative to the data directory | Contents | Analysis |
| --- | --- | --- |
| `input/p90_events.csv.gz` | Episode labels, duration, matching covariates, and NDVI event outcomes | Main matching |
| `input/p95_events.csv.gz` | Episode labels and outcomes at the stricter heat threshold | Heat-threshold sensitivity |
| `input/cumulative_events.csv.gz` | Episode-level cumulative anomaly losses and matching covariates | Cumulative matching and period comparisons |
| `input/pixel_background_regimes.csv` | Pixel-level climate background classes | Stratified comparison |
| `input/pixel_month_panel.csv.gz` | NDVI, hydroclimate, exposure indicators, and lagged variables at pixel-month resolution | `full`, `pooled` |
| `input/vpd_events.csv.gz` | Episode labels, total event loss, GPP and VPD changes | Figure 3b |
| `input/fluxnet_monthly_panel.csv` | Monthly tower variables, site strata, and gridded drought exposure for the 2001-2014 overlap | Tower analysis |
| `reference/` | Matched outcome and tower reference tables, and pooled-model coefficient tables | Verification and Figure 3a |
| `figure_inputs/` | Episode spatial summaries, trajectories, and ecosystem response summaries | Figure export |
| `cartopy/` | Natural Earth coastline cache for offline map export | Map panels |

`python code/run_all.py --mode review --check-inputs` lists missing inputs by
path. Use `--no-figures` for a table-only run. `reference/` is the reference
directory under `CAPSULE_DATA`; computed tables go to `results/analysis_outputs/`.

The figure inputs for the default run are:

- `spatial_event_summary_by_pixel.csv`
- `event_summary_by_type.csv`
- `vpd_trajectory_summary_robust.csv`
- `functional_trajectory_summary_stage2_robust.csv`
- `event_trajectory_by_zone_vpd_summary_robust.csv`
- `event_multivariate_response_anomaly_by_type_landcover.csv`

For offline maps, place the Natural Earth 110 m coastline shapefile and its
sidecar files in `cartopy/shapefiles/natural_earth/physical/`.

The upstream scripts use `VEGETATION_PROJECT_ROOT` for paths to harmonized
rasters (`used_data/`) and intermediate tables (`output_df/`). The tower data
preparation script also reads `FLUXNET_FULLSET_DIR`, the location of extracted
monthly FLUXNET2015 FULLSET files. See the input constants at the start of each
script for filenames.
