from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


BASE_DIR = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/global_v05_90p_main")
EVENT_PATH = BASE_DIR / "global_integrated_events_90p.csv"
OUTPUT_DIR = BASE_DIR / "event_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VEG_VARS = ["ndvi", "sif", "gpp", "et"]
CLIMATE_VARS = ["spei", "sm", "temp", "sti"]
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


def build_event_level_metrics(chunk: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for row in chunk.itertuples(index=False):
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
        }

        for variable in ALL_VARS:
            pre_values = parse_series(getattr(row, f"pre_{variable}_path"))
            event_values = parse_series(getattr(row, f"{variable}_path"))
            post_values = parse_series(getattr(row, f"post_{variable}_path"))

            pre_stats = summarize_stage(pre_values)
            event_stats = summarize_stage(event_values)
            post_stats = summarize_stage(post_values)

            record[f"{variable}_pre_mean"] = pre_stats["mean"]
            record[f"{variable}_event_mean"] = event_stats["mean"]
            record[f"{variable}_post_mean"] = post_stats["mean"]
            record[f"{variable}_event_min"] = event_stats["min"]
            record[f"{variable}_event_max"] = event_stats["max"]
            record[f"{variable}_delta_event_minus_pre"] = (
                event_stats["mean"] - pre_stats["mean"]
                if np.isfinite(event_stats["mean"]) and np.isfinite(pre_stats["mean"])
                else np.nan
            )
            record[f"{variable}_delta_post_minus_event"] = (
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
        if col not in {"pixel_id", "event_type", "land_cover_broad"} and pd.api.types.is_numeric_dtype(df[col])
    ]
    agg = {col: ["mean", "median"] for col in numeric_cols}
    summary = df.groupby(group_cols, dropna=False).agg(agg)
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.reset_index()
    return summary


def build_variable_highlight_table(summary_by_type: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in summary_by_type.iterrows():
        event_type = row["event_type"]
        out: Dict[str, object] = {"event_type": event_type}
        for variable in ALL_VARS:
            out[f"{variable}_event_mean"] = row.get(f"{variable}_event_mean_mean", np.nan)
            out[f"{variable}_delta_event_minus_pre"] = row.get(f"{variable}_delta_event_minus_pre_mean", np.nan)
            out[f"{variable}_delta_post_minus_event"] = row.get(f"{variable}_delta_post_minus_event_mean", np.nan)
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    usecols = [
        "pixel_id",
        "event_type",
        "land_cover",
        "duration",
        "LOSStot",
        "DECmax",
        "TIMErec_true",
        "compound_share",
    ]
    for variable in ALL_VARS:
        usecols.extend([f"pre_{variable}_path", f"{variable}_path", f"post_{variable}_path"])

    event_frames: List[pd.DataFrame] = []
    for chunk in pd.read_csv(EVENT_PATH, usecols=usecols, chunksize=25000):
        event_frames.append(build_event_level_metrics(chunk))

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

    event_level.to_csv(OUTPUT_DIR / "event_multivariate_response_event_level.csv", index=False)
    summary_by_type.to_csv(OUTPUT_DIR / "event_multivariate_response_by_type.csv", index=False)
    summary_by_type_landcover.to_csv(OUTPUT_DIR / "event_multivariate_response_by_type_landcover.csv", index=False)
    summary_by_landcover.to_csv(OUTPUT_DIR / "event_multivariate_response_by_landcover.csv", index=False)
    variable_highlights.to_csv(OUTPUT_DIR / "event_multivariate_response_highlights.csv", index=False)

    report_lines = [
        "MULTIVARIATE EVENT RESPONSE SUMMARY",
        f"event_path={EVENT_PATH}",
        "",
        "[top event types by LOSStot]",
        summary_by_type[
            [
                "event_type",
                "LOSStot_mean",
                "DECmax_mean",
                "TIMErec_true_mean",
                "ndvi_delta_event_minus_pre_mean",
                "sif_delta_event_minus_pre_mean",
                "gpp_delta_event_minus_pre_mean",
                "et_delta_event_minus_pre_mean",
                "spei_delta_event_minus_pre_mean",
                "temp_delta_event_minus_pre_mean",
                "sti_delta_event_minus_pre_mean",
                "sm_delta_event_minus_pre_mean",
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
                "ndvi_delta_event_minus_pre_mean",
                "sif_delta_event_minus_pre_mean",
                "gpp_delta_event_minus_pre_mean",
                "et_delta_event_minus_pre_mean",
            ]
        ].to_string(index=False),
    ]
    (OUTPUT_DIR / "event_multivariate_response_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved multivariate event response outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
