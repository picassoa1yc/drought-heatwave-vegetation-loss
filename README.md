# Concurrent heat amplifies drought-driven vegetation loss

Analysis code for global pixel-level vegetation-loss episodes at 0.5-degree
resolution from 2001 to 2024. The inventory contains 461,375 episodes, including
134,740 compound dry-hot episodes; the principal matched analysis includes
134,406 compound episodes.

## Main result

Concurrent heat was associated with **14.7% additional total event loss**
relative to matched drought-only episodes (ATT = 0.04230). Estimates across
matching specifications ranged from **10.0% to 15.2%**. The relative effect is
`100 * ATT / abs(matched drought-only mean)`.

Compound and drought-only episodes were matched within ecosystem and duration
strata using three nearest neighbours. Covariates include duration and
antecedent NDVI, GPP, transpiration, SPEI, root-zone soil moisture and temperature.
Outcomes include total event loss, peak NDVI decline, recovery time and
cumulative NDVI, SIF, GPP and transpiration anomaly losses.

## Run

Use Python 3.11 or 3.12 and run commands from the repository root in an active
virtual environment. The [input guide](docs/inputs.md) lists the processed files
used by each analysis. Place them in `data/`, or set `CAPSULE_DATA` to their
location. Set `CAPSULE_RESULTS` to change the default output directory, `results/`.

```sh
python -m pip install -r requirements.txt
python code/run_all.py --check-inputs
python code/run_all.py --mode review
```

| Mode | Analysis |
| --- | --- |
| `review` | Main and sensitivity matching, period and subgroup contrasts, tower contrasts, and Figures 1-4 plus the supplementary event-pathway maps. Figure 3a uses reference model coefficients. |
| `full` | The same analyses, with both pooled panel models refitted. Use a fresh output directory for a separate run. |
| `pooled` | Fit both pooled panel models from the monthly pixel panel. |
| `fluxnet` | Calculate paired tower contrasts and baseline-normalized percentages. |
| `figures` | Plot Figures 1-4 and event-pathway maps from the tables in `results/analysis_outputs/`. |
| `verify` | Compare calculated tables with the manuscript reference tables. |

Add `--no-figures` to `review` or `full` to produce tables only. Add
`--check-inputs` to any mode to check file availability without running analyses.
Use [the workflow index](docs/workflows.md) to locate source scripts, inputs and
outputs for event identification and the Figure 5 assembly.

## Repository Layout

- `code/run_all.py`: analysis and plotting entry point.
- `code/originals/`: event processing, statistical models, and Python/R figure scripts.
- `requirements.txt`: pinned Python dependencies for the main workflow.
- `docs/`: input layout and manuscript figure/table index.
