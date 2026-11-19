from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import rasterio


BASE_DIR = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/global_v05_90p_vpd_main")
EVENT_PATH = BASE_DIR / "global_integrated_events_90p_vpd.csv"
OUTPUT_DIR = BASE_DIR / "event_analysis_anomaly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USED_DATA = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data")

VEG_VARS = ["ndvi", "sif", "gpp", "et"]
CLIMATE_VARS = ["spei", "sm", "temp", "sti", "vpd"]
ALL_VARS = VEG_VARS + CLIMATE_VARS

LAND_COVER_MAP = {
    1: "Forest",
    2: "Forest",
    3: "Forest",
    4: "Forest",
    5: "Forest",
    6: "Shrub",
    7: "Shrub",
    8: "Savannas",
    9: "Savannas",
    10: "Grassland",
    12: "Cropland",
    14: "Cropland",
}

CLIM_PATHS = {
    "ndvi": USED_DATA / "ndvi_climatology_2001-2024.tif",
    "sif": USED_DATA / "sif_monthly_climatology_2001-2024.tif",
    "gpp": USED_DATA / "gpp_monthly_climatology_2001-2024.tif",
    "et": USED_DATA / "Et_monthly_climatology_2001-2024.tif",
    "vpd": USED_DATA / "era5_vpd_monthly_climatology_2001-2024.tif",
}
RAW_STACK_PATHS = {
    "temp": USED_DATA / "tmx_2001-2024.tif",
    "sm": USED_DATA / "SMrz_detrend_2001-2024.tif",
}


def parse_series(value: object) -> list[float]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    values: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token or token.lower() == "nan":
            values.append(np.nan)
        else:
            try:
                values.append(float(token))
            except ValueError:
                values.append(np.nan)
    return values


def finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def summarize_stage(values: Iterable[float]) -> Dict[str, float]:
    arr = finite_array(values)
    if arr.size == 0:
        return {"mean": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def land_cover_broad(code: object) -> str:
    if pd.isna(code):
        return "Other"
    return LAND_COVER_MAP.get(int(code), "Other")


def load_monthly_climatology(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            data[data == nodata] = np.nan
    if data.shape[0] != 12:
        raise ValueError(f"Monthly climatology must have 12 bands: {path}")
    return data


def build_monthly_climatology_from_stack(path: Path, cache_path: Optional[Path] = None) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            data[data == nodata] = np.nan
    month_index = np.arange(data.shape[0]) % 12
    climatology = np.full((12, data.shape[1], data.shape[2]), np.nan, dtype=np.float32)
    for month in range(12):
        subset = data[month_index == month]
        with np.errstate(invalid="ignore"):
            climatology[month] = np.nanmean(subset, axis=0)
    if cache_path is not None:
        np.save(cache_path, climatology)
    return climatology


def load_all_climatologies() -> Dict[str, np.ndarray]:
    climatologies: Dict[str, np.ndarray] = {}
    for variable, path in CLIM_PATHS.items():
        climatologies[variable] = load_monthly_climatology(path)
    for variable, path in RAW_STACK_PATHS.items():
        cache_path = CACHE_DIR / f"{variable}_monthly_climatology_cache.npy"
        climatologies[variable] = build_monthly_climatology_from_stack(path, cache_path=cache_path)
    return climatologies


def stage_dates(start_date: str, end_date: str, pre_len: int, post_len: int) -> Dict[str, pd.DatetimeIndex]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    event = pd.date_range(start, end, freq="MS")
    pre = pd.date_range(end=start - pd.DateOffset(months=1), periods=pre_len, freq="MS") if pre_len > 0 else pd.DatetimeIndex([])
    post = pd.date_range(start=end + pd.DateOffset(months=1), periods=post_len, freq="MS") if post_len > 0 else pd.DatetimeIndex([])
    return {"pre": pre, "event": event, "post": post}


def compute_anomaly_series(
    variable: str,
    raw_values: list[float],
    dates: pd.DatetimeIndex,
    row: int,
    col: int,
    climatologies: Dict[str, np.ndarray],
) -> list[float]:
    if variable in {"spei", "sti"}:
        return raw_values
    clim = climatologies[variable]
    n = min(len(raw_values), len(dates))
    out: list[float] = []
    for idx in range(n):
        value = raw_values[idx]
        if not np.isfinite(value):
            out.append(np.nan)
            continue
        month = int(dates[idx].month) - 1
        clim_value = float(clim[month, row, col])
        if not np.isfinite(clim_value):
            out.append(np.nan)
        else:
            out.append(float(value - clim_value))
    return out


def build_event_level_metrics(chunk: pd.DataFrame, climatologies: Dict[str, np.ndarray]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for row in chunk.itertuples(index=False):
        row_idx = int(row.row)
        col_idx = int(row.col)
        pre_dates_len = len(parse_series(getattr(row, "pre_ndvi_path")))
        post_dates_len = len(parse_series(getattr(row, "post_ndvi_path")))
        dates = stage_dates(row.start_date, row.end_date, pre_dates_len, post_dates_len)

        record: Dict[str, object] = {
            "pixel_id": int(row.pixel_id),
            "event_type": row.event_type,
            "land_cover": float(row.land_cover) if pd.notna(row.land_cover) else np.nan,
            "land_cover_broad": land_cover_broad(row.land_cover),
            "duration": float(row.duration),
            "LOSStot": float(row.LOSStot),
            "DECmax": float(row.DECmax),
            "TIMErec_true": float(row.TIMErec_true) if pd.notna(row.TIMErec_true) else np.nan,
            "compound_share": float(row.compound_share),
            "start_date": row.start_date,
            "end_date": row.end_date,
        }

        for variable in ALL_VARS:
            pre_values = parse_series(getattr(row, f"pre_{variable}_path"))
            event_values = parse_series(getattr(row, f"{variable}_path"))
            post_values = parse_series(getattr(row, f"post_{variable}_path"))

            pre_series = compute_anomaly_series(variable, pre_values, dates["pre"], row_idx, col_idx, climatologies)
            event_series = compute_anomaly_series(variable, event_values, dates["event"], row_idx, col_idx, climatologies)
            post_series = compute_anomaly_series(variable, post_values, dates["post"], row_idx, col_idx, climatologies)

            pre_stats = summarize_stage(pre_series)
            event_stats = summarize_stage(event_series)
            post_stats = summarize_stage(post_series)

            record[f"{variable}_anom_pre_mean"] = pre_stats["mean"]
            record[f"{variable}_anom_event_mean"] = event_stats["mean"]
            record[f"{variable}_anom_post_mean"] = post_stats["mean"]
            record[f"{variable}_anom_event_min"] = event_stats["min"]
            record[f"{variable}_anom_event_max"] = event_stats["max"]
            record[f"{variable}_anom_delta_event_minus_pre"] = (
                event_stats["mean"] - pre_stats["mean"]
                if np.isfinite(event_stats["mean"]) and np.isfinite(pre_stats["mean"])
                else np.nan
            )
            record[f"{variable}_anom_delta_post_minus_event"] = (
                post_stats["mean"] - event_stats["mean"]
                if np.isfinite(post_stats["mean"]) and np.isfinite(event_stats["mean"])
                else np.nan
            )

        records.append(record)
    return pd.DataFrame(records)


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"pixel_id", "event_type", "land_cover_broad", "start_date", "end_date"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    agg = {col: ["mean", "median"] for col in numeric_cols}
    summary = df.groupby(group_cols, dropna=False).agg(agg)
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    return summary.reset_index()


def build_variable_highlight_table(summary_by_type: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in summary_by_type.iterrows():
        out: Dict[str, object] = {"event_type": row["event_type"]}
        for variable in ALL_VARS:
            out[f"{variable}_anom_event_mean"] = row.get(f"{variable}_anom_event_mean_mean", np.nan)
            out[f"{variable}_anom_delta_event_minus_pre"] = row.get(f"{variable}_anom_delta_event_minus_pre_mean", np.nan)
            out[f"{variable}_anom_delta_post_minus_event"] = row.get(f"{variable}_anom_delta_post_minus_event_mean", np.nan)
        rows.append(out)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build anomaly-domain multivariate event summaries.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for pilot runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    usecols = [
        "pixel_id",
        "row",
        "col",
        "event_type",
        "land_cover",
        "duration",
        "LOSStot",
        "DECmax",
        "TIMErec_true",
        "compound_share",
        "start_date",
        "end_date",
    ]
    for variable in ALL_VARS:
        usecols.extend([f"pre_{variable}_path", f"{variable}_path", f"post_{variable}_path"])

    climatologies = load_all_climatologies()

    event_frames: List[pd.DataFrame] = []
    rows_loaded = 0
    for chunk_id, chunk in enumerate(pd.read_csv(EVENT_PATH, usecols=usecols, chunksize=10000), start=1):
        if args.max_rows is not None and rows_loaded >= args.max_rows:
            break
        if args.max_rows is not None and rows_loaded + len(chunk) > args.max_rows:
            chunk = chunk.iloc[: args.max_rows - rows_loaded].copy()
        rows_loaded += len(chunk)
        event_frames.append(build_event_level_metrics(chunk, climatologies))
        print(f"processed_chunk={chunk_id} rows_loaded={rows_loaded}")

    event_level = pd.concat(event_frames, ignore_index=True)
    event_level = event_level[event_level["land_cover_broad"] != "Other"].copy()

    summary_by_type = summarize_group(event_level, ["event_type"]).sort_values("LOSStot_mean", ascending=False)
    summary_by_type_landcover = summarize_group(event_level, ["event_type", "land_cover_broad"]).sort_values(
        ["event_type", "LOSStot_mean"], ascending=[True, False]
    )
    summary_by_landcover = summarize_group(event_level, ["land_cover_broad"]).sort_values(
        "LOSStot_mean", ascending=False
    )
    variable_highlights = build_variable_highlight_table(summary_by_type)

    event_level.to_csv(OUTPUT_DIR / "event_multivariate_response_anomaly_event_level.csv", index=False)
    summary_by_type.to_csv(OUTPUT_DIR / "event_multivariate_response_anomaly_by_type.csv", index=False)
    summary_by_type_landcover.to_csv(OUTPUT_DIR / "event_multivariate_response_anomaly_by_type_landcover.csv", index=False)
    summary_by_landcover.to_csv(OUTPUT_DIR / "event_multivariate_response_anomaly_by_landcover.csv", index=False)
    variable_highlights.to_csv(OUTPUT_DIR / "event_multivariate_response_anomaly_highlights.csv", index=False)

    report_lines = [
        "ANOMALY-DOMAIN MULTIVARIATE EVENT RESPONSE SUMMARY",
        f"event_path={EVENT_PATH}",
        "",
        "[top event types by LOSStot]",
        summary_by_type[
            [
                "event_type",
                "LOSStot_mean",
                "DECmax_mean",
                "TIMErec_true_mean",
                "ndvi_anom_delta_event_minus_pre_mean",
                "sif_anom_delta_event_minus_pre_mean",
                "gpp_anom_delta_event_minus_pre_mean",
                "et_anom_delta_event_minus_pre_mean",
                "spei_anom_delta_event_minus_pre_mean",
                "temp_anom_delta_event_minus_pre_mean",
                "sti_anom_delta_event_minus_pre_mean",
                "sm_anom_delta_event_minus_pre_mean",
                "vpd_anom_delta_event_minus_pre_mean",
            ]
        ].to_string(index=False),
        "",
        "[land cover means]",
        summary_by_landcover[
            [
                "land_cover_broad",
                "LOSStot_mean",
                "DECmax_mean",
                "TIMErec_true_mean",
                "ndvi_anom_delta_event_minus_pre_mean",
                "sif_anom_delta_event_minus_pre_mean",
                "gpp_anom_delta_event_minus_pre_mean",
                "et_anom_delta_event_minus_pre_mean",
                "vpd_anom_delta_event_minus_pre_mean",
            ]
        ].to_string(index=False),
    ]
    (OUTPUT_DIR / "event_multivariate_response_anomaly_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved anomaly-domain multivariate event response outputs to {OUTPUT_DIR}; rows={len(event_level)}")


if __name__ == "__main__":
    main()
