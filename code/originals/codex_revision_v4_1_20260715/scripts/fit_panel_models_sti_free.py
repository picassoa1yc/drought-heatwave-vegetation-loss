from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve()
REVISION_ROOT = ROOT / "codex_revision_v4_1_20260715"
OUTPUT_DIR = REVISION_ROOT / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_PATH = (
    ROOT
    / "output_df"
    / "global_v05_90p_vpd_main"
    / "global_pixel_month_panel_90p_vpd.csv"
)

RESPONSE = "ndvi_anom"
CONTINUOUS = [RESPONSE, "spei", "temp", "vpd", "ndvi_lag1", "sm_lag1"]
BINARY = [
    "drought",
    "heat",
    "drought_lag1",
    "heat_lag1",
    "drought_lag2",
    "heat_lag2",
    "drought_lag3",
    "heat_lag3",
]
MONTH_TERMS = [f"month_{month}" for month in range(2, 13)]
BASE_TERMS = [
    "drought",
    "heat",
    "drought_x_heat",
    "spei",
    "temp",
    "ndvi_lag1",
    "sm_lag1",
    "drought_lag1",
    "heat_lag1",
    "drought_lag2",
    "heat_lag2",
    "drought_lag3",
    "heat_lag3",
    *MONTH_TERMS,
]
VPD_TERMS = [
    "drought",
    "heat",
    "drought_x_heat",
    "spei",
    "temp",
    "vpd",
    "ndvi_lag1",
    "sm_lag1",
    "drought_lag1",
    "heat_lag1",
    "drought_lag2",
    "heat_lag2",
    "drought_lag3",
    "heat_lag3",
    *MONTH_TERMS,
]
RAW_SUM_COLUMNS = [RESPONSE, "spei", "temp", "vpd", "ndvi_lag1", "sm_lag1", *BINARY, "drought_x_heat", *MONTH_TERMS]
READ_COLUMNS = ["pixel_id", "month", "growing_season_flag", *CONTINUOUS, *BINARY]


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk[chunk["growing_season_flag"].eq(1)].copy()
    chunk = chunk.dropna(subset=[*CONTINUOUS, *BINARY, "pixel_id", "month"])
    if chunk.empty:
        return chunk
    chunk["drought_x_heat"] = chunk["drought"] * chunk["heat"]
    month = pd.to_numeric(chunk["month"], errors="coerce").astype(int)
    for value in range(2, 13):
        chunk[f"month_{value}"] = (month == value).astype(float)
    return chunk


def iter_clean_chunks(chunksize: int = 250_000):
    for chunk_number, chunk in enumerate(
        pd.read_csv(PANEL_PATH, usecols=list(dict.fromkeys(READ_COLUMNS)), chunksize=chunksize),
        start=1,
    ):
        clean = clean_chunk(chunk)
        if clean.empty:
            continue
        yield chunk_number, clean


def first_pass() -> tuple[pd.DataFrame, dict[str, float], int]:
    global_count = {column: 0 for column in CONTINUOUS}
    global_sum = {column: 0.0 for column in CONTINUOUS}
    global_sumsq = {column: 0.0 for column in CONTINUOUS}
    grouped_blocks: list[pd.DataFrame] = []
    n_rows = 0
    for chunk_number, chunk in iter_clean_chunks():
        n_rows += len(chunk)
        for column in CONTINUOUS:
            values = chunk[column].to_numpy(dtype=float)
            global_count[column] += int(len(values))
            global_sum[column] += float(values.sum(dtype=np.float64))
            global_sumsq[column] += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
        grouped = chunk.groupby("pixel_id", sort=False)[RAW_SUM_COLUMNS].sum()
        grouped["n_obs"] = chunk.groupby("pixel_id", sort=False).size()
        grouped_blocks.append(grouped)
        if chunk_number % 10 == 0:
            print(f"first_pass_chunks={chunk_number}, rows={n_rows:,}", flush=True)
    if not grouped_blocks:
        raise ValueError("No valid rows were found in the VPD panel.")
    pixel_sums = pd.concat(grouped_blocks).groupby(level=0, sort=False).sum()
    pixel_means = pixel_sums[RAW_SUM_COLUMNS].div(pixel_sums["n_obs"], axis=0)
    pixel_means["n_obs"] = pixel_sums["n_obs"]
    standard_deviations = {}
    for column in CONTINUOUS:
        count = global_count[column]
        mean = global_sum[column] / count
        variance = max(global_sumsq[column] / count - mean * mean, 0.0)
        standard_deviations[column] = float(np.sqrt(variance))
        if standard_deviations[column] == 0:
            raise ValueError(f"Zero global standard deviation for {column}.")
    return pixel_means, standard_deviations, n_rows


def within_arrays(
    chunk: pd.DataFrame,
    pixel_means: pd.DataFrame,
    standard_deviations: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    means = pixel_means.reindex(chunk["pixel_id"].to_numpy())
    if means.isna().any(axis=None):
        raise ValueError("Pixel means could not be aligned to a model chunk.")
    y = (chunk[RESPONSE].to_numpy(dtype=float) - means[RESPONSE].to_numpy(dtype=float)) / standard_deviations[RESPONSE]
    columns: dict[str, np.ndarray] = {}
    for column in ["spei", "temp", "vpd", "ndvi_lag1", "sm_lag1"]:
        columns[column] = (
            chunk[column].to_numpy(dtype=float) - means[column].to_numpy(dtype=float)
        ) / standard_deviations[column]
    for column in [*BINARY, "drought_x_heat", *MONTH_TERMS]:
        columns[column] = chunk[column].to_numpy(dtype=float) - means[column].to_numpy(dtype=float)
    full = np.column_stack([columns[term] for term in VPD_TERMS])
    return y, full


def second_pass(
    pixel_means: pd.DataFrame,
    standard_deviations: dict[str, float],
) -> tuple[dict[str, np.ndarray], int, float]:
    p_full = len(VPD_TERMS)
    p_base = len(BASE_TERMS)
    base_indices = np.asarray([VPD_TERMS.index(term) for term in BASE_TERMS], dtype=int)
    xtx_full = np.zeros((p_full, p_full), dtype=np.float64)
    xty_full = np.zeros(p_full, dtype=np.float64)
    xtx_base = np.zeros((p_base, p_base), dtype=np.float64)
    xty_base = np.zeros(p_base, dtype=np.float64)
    yty = 0.0
    n_rows = 0
    for chunk_number, chunk in iter_clean_chunks():
        y, x_full = within_arrays(chunk, pixel_means, standard_deviations)
        x_base = x_full[:, base_indices]
        xtx_full += x_full.T @ x_full
        xty_full += x_full.T @ y
        xtx_base += x_base.T @ x_base
        xty_base += x_base.T @ y
        yty += float(y @ y)
        n_rows += len(y)
        if chunk_number % 10 == 0:
            print(f"second_pass_chunks={chunk_number}, rows={n_rows:,}", flush=True)
    beta_full = np.linalg.solve(xtx_full, xty_full)
    beta_base = np.linalg.solve(xtx_base, xty_base)
    return {
        "xtx_full": xtx_full,
        "xtx_base": xtx_base,
        "beta_full": beta_full,
        "beta_base": beta_base,
        "base_indices": base_indices,
    }, n_rows, yty


def third_pass(
    pixel_means: pd.DataFrame,
    standard_deviations: dict[str, float],
    matrices: dict[str, np.ndarray],
) -> dict[str, object]:
    xtx_full = matrices["xtx_full"]
    xtx_base = matrices["xtx_base"]
    beta_full = matrices["beta_full"]
    beta_base = matrices["beta_base"]
    base_indices = matrices["base_indices"]
    inv_full = np.linalg.inv(xtx_full)
    inv_base = np.linalg.inv(xtx_base)
    meat_full = np.zeros_like(xtx_full)
    meat_base = np.zeros_like(xtx_base)
    sse_full = 0.0
    sse_base = 0.0
    n_rows = 0
    for chunk_number, chunk in iter_clean_chunks():
        y, x_full = within_arrays(chunk, pixel_means, standard_deviations)
        x_base = x_full[:, base_indices]

        residual_full = y - x_full @ beta_full
        residual_base = y - x_base @ beta_base
        leverage_full = np.einsum("ij,jk,ik->i", x_full, inv_full, x_full, optimize=True)
        leverage_base = np.einsum("ij,jk,ik->i", x_base, inv_base, x_base, optimize=True)
        adjustment_full = np.square(residual_full / np.maximum(1.0 - leverage_full, 1e-8))
        adjustment_base = np.square(residual_base / np.maximum(1.0 - leverage_base, 1e-8))
        meat_full += x_full.T @ (x_full * adjustment_full[:, None])
        meat_base += x_base.T @ (x_base * adjustment_base[:, None])
        sse_full += float(residual_full @ residual_full)
        sse_base += float(residual_base @ residual_base)
        n_rows += len(y)
        if chunk_number % 10 == 0:
            print(f"third_pass_chunks={chunk_number}, rows={n_rows:,}", flush=True)

    covariance_full = inv_full @ meat_full @ inv_full
    covariance_base = inv_base @ meat_base @ inv_base
    return {
        "covariance_full": covariance_full,
        "covariance_base": covariance_base,
        "sse_full": sse_full,
        "sse_base": sse_base,
        "n_rows": n_rows,
    }


def coefficient_table(
    terms: list[str],
    beta: np.ndarray,
    covariance: np.ndarray,
    n_rows: int,
) -> pd.DataFrame:
    standard_error = np.sqrt(np.diag(covariance))
    t_value = beta / standard_error
    degrees_freedom = n_rows - len(terms)
    p_value = 2.0 * stats.t.sf(np.abs(t_value), df=degrees_freedom)
    return pd.DataFrame(
        {
            "term": terms,
            "coef": beta,
            "std_err": standard_error,
            "t_value": t_value,
            "p_value": p_value,
        }
    )


def main() -> None:
    print("Fitting STI-free pooled panel models on a common VPD-complete sample.", flush=True)
    pixel_means, standard_deviations, first_n = first_pass()
    matrices, second_n, yty = second_pass(pixel_means, standard_deviations)
    robust = third_pass(pixel_means, standard_deviations, matrices)
    third_n = int(robust["n_rows"])
    if len({first_n, second_n, third_n}) != 1:
        raise ValueError(f"Pass row counts differ: {first_n}, {second_n}, {third_n}.")

    base = coefficient_table(
        BASE_TERMS,
        matrices["beta_base"],
        robust["covariance_base"],
        third_n,
    )
    vpd = coefficient_table(
        VPD_TERMS,
        matrices["beta_full"],
        robust["covariance_full"],
        third_n,
    )
    base_path = OUTPUT_DIR / "pooled_panel_base_sti_free.csv"
    vpd_path = OUTPUT_DIR / "pooled_panel_vpd_sti_free.csv"
    base.to_csv(base_path, index=False)
    vpd.to_csv(vpd_path, index=False)

    comparison = base.rename(
        columns={"coef": "coef_without_vpd", "p_value": "p_without_vpd", "std_err": "se_without_vpd"}
    )[["term", "coef_without_vpd", "se_without_vpd", "p_without_vpd"]].merge(
        vpd.rename(
            columns={"coef": "coef_with_vpd", "p_value": "p_with_vpd", "std_err": "se_with_vpd"}
        )[["term", "coef_with_vpd", "se_with_vpd", "p_with_vpd"]],
        on="term",
        how="outer",
    )
    comparison["coef_change"] = comparison["coef_with_vpd"] - comparison["coef_without_vpd"]
    comparison.to_csv(OUTPUT_DIR / "pooled_panel_vpd_comparison_sti_free.csv", index=False)

    summary = {
        "panel_path": str(PANEL_PATH),
        "n_common_valid_growing_season_pixel_months": third_n,
        "n_pixels": int(len(pixel_means)),
        "response": RESPONSE,
        "standardization": "Global z-standardization of continuous variables followed by within-pixel demeaning.",
        "sample_rule": "Both models use the same VPD-complete, growing-season pixel-month sample.",
        "sti_included": False,
        "base_terms": BASE_TERMS,
        "vpd_terms": VPD_TERMS,
        "global_standard_deviations": standard_deviations,
        "base_uncentered_r2": float(1.0 - robust["sse_base"] / yty),
        "vpd_uncentered_r2": float(1.0 - robust["sse_full"] / yty),
    }
    (OUTPUT_DIR / "pooled_panel_sti_free_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    key = comparison[comparison["term"].isin(["drought", "heat", "drought_x_heat", "vpd"])]
    print(key.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
