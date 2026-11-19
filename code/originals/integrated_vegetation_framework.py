from __future__ import annotations

import argparse
import csv
import warnings
from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import find_objects, label


@dataclass(frozen=True)
class ProjectPaths:
    test_dir: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "test")
    used_data_dir: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data")
    detrended_ndvi: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/NDVI_detrend_2001-2024.tif")
    anomaly_mask: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/negative_anomaly_mask_2001-2024.tif")
    drought_mask: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/drought_mask_filtered_vegetated_only.tif")
    climatology_ndvi: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/ndvi_climatology_2001-2024.tif")
    landcover: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/landcover_2001-2024_modal.tif")
    spei: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/spei03_2001-2024.tif")
    sti: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/sti_2001-2024.tif")
    temperature: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/tmx_2001-2024.tif")
    ndvi: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/NDVI_detrend_2001-2024.tif")
    soil_moisture: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/SMrz_detrend_2001-2024.tif")
    et: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/Et_detrend_2001-2024.tif")
    sif: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/sif_detrend_2001-2024.tif")
    gpp: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/gpp_detrend_2001-2024_1.tif")
    gpp_climatology: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/gpp_monthly_climatology_2001-2024.tif")
    et_climatology: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/Et_monthly_climatology_2001-2024.tif")
    sif_climatology: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/sif_monthly_climatology_2001-2024.tif")
    radiation: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/ssrd_Wm2_2001-2024.tif")
    vpd: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/era5_vpd_2001-2024.tif")
    vpd_climatology: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "used_data/era5_vpd_monthly_climatology_2001-2024.tif")
    event_output: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/integrated_vna_events.csv")
    panel_output: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/pixel_month_panel.csv")
    model_output: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/pixel_month_model_summary.txt")
    coefficient_output: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/pixel_month_model_coefficients.csv")
    local_coefficient_output: Path = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/pixel_month_local_interaction_coefficients.csv")

    def heatwave_mask_path(self, percentile: int, mask_dir: Optional[Path] = None) -> Path:
        target_dir = mask_dir or self.used_data_dir
        return target_dir / f"heatwave_mask_{percentile}p.tif"


@dataclass(frozen=True)
class EventConfig:
    start_year: int = 2001
    start_month: int = 1
    block_rows: int = 32
    min_event_duration: int = 2
    anomaly_sigma: float = 0.5
    pre_event_months: int = 6
    trigger_lead_months: int = 3
    post_event_months: int = 6
    min_event_temperature_c: float = 5.0
    heat_percentile: int = 90
    heat_mask_dir: Optional[Path] = None
    exclude_landcover: Tuple[int, ...] = (13, 15, 16)
    precision: int = 3
    max_pixels: Optional[int] = None


@dataclass(frozen=True)
class PanelConfig:
    start_year: int = 2001
    start_month: int = 1
    block_rows: int = 16
    lags: Tuple[int, ...] = (1, 2, 3)
    heat_percentile: int = 90
    heat_mask_dir: Optional[Path] = None
    growing_season_mode: str = "temp_and_ndvi"
    growing_temp_c: float = 5.0
    growing_ndvi_threshold: float = 0.1
    exclude_landcover: Tuple[int, ...] = (13, 15, 16)
    pixel_stride: int = 1
    max_pixels: Optional[int] = None
    flush_rows: int = 50000


EVENT_FIELDS: List[str] = [
    "pixel_id",
    "row",
    "col",
    "heat_percentile",
    "land_cover",
    "event_start_index",
    "event_end_index",
    "duration",
    "start_date",
    "end_date",
    "event_type",
    "trigger_start_index",
    "trigger_end_index",
    "stress_lead_months",
    "n_drought_trigger",
    "n_heat_trigger",
    "n_compound_trigger",
    "n_drought_event",
    "n_heat_event",
    "n_compound_event",
    "compound_share",
    "stress_path_trigger",
    "stress_path_event",
    "ndvi_sd",
    "threshold_value",
    "LOSStot",
    "DECmax",
    "TIMEdec",
    "TIMErec_true",
    "max_decrease_date",
    "pre_spei_path",
    "pre_temp_path",
    "pre_vpd_path",
    "pre_sti_path",
    "pre_sm_path",
    "pre_ndvi_path",
    "pre_et_path",
    "pre_sif_path",
    "pre_gpp_path",
    "spei_path",
    "temp_path",
    "vpd_path",
    "sti_path",
    "sm_path",
    "ndvi_path",
    "et_path",
    "sif_path",
    "gpp_path",
    "post_spei_path",
    "post_temp_path",
    "post_vpd_path",
    "post_sti_path",
    "post_sm_path",
    "post_ndvi_path",
    "post_et_path",
    "post_sif_path",
    "post_gpp_path",
    "ndvi_clim_path",
    "gpp_clim_path",
    "et_clim_path",
    "sif_clim_path",
    "vpd_clim_path",
]


PANEL_FIELDS: List[str] = [
    "pixel_id",
    "row",
    "col",
    "heat_percentile",
    "date",
    "year",
    "month",
    "land_cover",
    "growing_season_flag",
    "ndvi_clim_month",
    "vna_flag",
    "drought",
    "heat",
    "compound",
    "ndvi_anom",
    "sif_anom",
    "gpp_anom",
    "et_anom",
    "sm_anom",
    "spei",
    "sti",
    "temp",
    "vpd",
    "radiation",
    "ndvi_lag1",
    "sm_lag1",
    "drought_lag1",
    "heat_lag1",
    "spei_lag1",
    "temp_lag1",
    "vpd_lag1",
    "drought_lag2",
    "heat_lag2",
    "spei_lag2",
    "temp_lag2",
    "vpd_lag2",
    "drought_lag3",
    "heat_lag3",
    "spei_lag3",
    "temp_lag3",
    "vpd_lag3",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_dates(time_steps: int, start_year: int, start_month: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=f"{start_year:04d}-{start_month:02d}-01",
        periods=time_steps,
        freq="MS",
    )


def ensure_heatwave_mask(paths: ProjectPaths, percentile: int, mask_dir: Optional[Path] = None) -> Path:
    mask_path = paths.heatwave_mask_path(percentile, mask_dir)
    if mask_path.exists():
        return mask_path

    ensure_parent(mask_path)
    with rasterio.open(paths.temperature) as src:
        temp_data = src.read().astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            temp_data[temp_data == nodata] = np.nan

    time_steps, height, width = temp_data.shape
    month_indices = np.arange(time_steps) % 12
    monthly_thresholds = np.full((12, height, width), np.nan, dtype=np.float32)

    for month in range(12):
        month_data = temp_data[month_indices == month]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            monthly_thresholds[month] = np.nanpercentile(month_data, percentile, axis=0)

    heatwave_mask = np.zeros((time_steps, height, width), dtype=np.uint8)
    for t in range(time_steps):
        threshold = monthly_thresholds[month_indices[t]]
        valid = np.isfinite(temp_data[t]) & np.isfinite(threshold)
        heatwave_mask[t, valid] = (temp_data[t, valid] >= threshold[valid]).astype(np.uint8)

    profile.update(dtype=rasterio.uint8, count=time_steps, compress="lzw", nodata=255)
    invalid = ~np.isfinite(temp_data)
    heatwave_mask[invalid] = 255

    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(heatwave_mask)
    return mask_path


def serialize(values: np.ndarray, precision: int = 3) -> str:
    if values.size == 0:
        return ""
    out: List[str] = []
    for value in values:
        if np.isnan(value):
            out.append("nan")
        else:
            out.append(f"{value:.{precision}f}")
    return ",".join(out)


def stress_path_string(drought_mask: np.ndarray, heat_mask: np.ndarray) -> str:
    path = np.zeros(drought_mask.shape[0], dtype=np.uint8)
    path[drought_mask & ~heat_mask] = 1
    path[~drought_mask & heat_mask] = 2
    path[drought_mask & heat_mask] = 3
    return "".join(str(int(x)) for x in path)


def find_runs(mask: np.ndarray) -> List[slice]:
    labeled, n_features = label(mask.astype(bool))
    if n_features == 0:
        return []
    return [obj[0] for obj in find_objects(labeled) if obj is not None]


def is_growing_season(
    *,
    temp_value: float,
    ndvi_clim_value: float,
    mode: str,
    temp_threshold: float,
    ndvi_threshold: float,
) -> bool:
    if mode == "none":
        return True

    temp_ok = np.isfinite(temp_value) and temp_value > temp_threshold
    ndvi_ok = np.isfinite(ndvi_clim_value) and ndvi_clim_value >= ndvi_threshold

    if mode == "temp":
        return temp_ok
    if mode == "ndvi":
        return ndvi_ok
    if mode == "temp_and_ndvi":
        return temp_ok and ndvi_ok
    raise ValueError(f"Unsupported growing season mode: {mode}")


def open_readers(
    paths: ProjectPaths,
    *,
    heatwave_mask_path: Optional[Path] = None,
) -> Dict[str, rasterio.io.DatasetReader]:
    items = {
        "detrended_ndvi": paths.detrended_ndvi,
        "anomaly_mask": paths.anomaly_mask,
        "drought_mask": paths.drought_mask,
        "heatwave_mask": heatwave_mask_path or paths.heatwave_mask_path(90),
        "climatology_ndvi": paths.climatology_ndvi,
        "landcover": paths.landcover,
        "spei": paths.spei,
        "sti": paths.sti,
        "temperature": paths.temperature,
        "ndvi": paths.ndvi,
        "soil_moisture": paths.soil_moisture,
        "et": paths.et,
        "sif": paths.sif,
        "gpp": paths.gpp,
        "gpp_climatology": paths.gpp_climatology,
        "et_climatology": paths.et_climatology,
        "sif_climatology": paths.sif_climatology,
        "radiation": paths.radiation,
        "vpd": paths.vpd,
        "vpd_climatology": paths.vpd_climatology,
    }
    stack = ExitStack()
    try:
        readers = {name: stack.enter_context(rasterio.open(path)) for name, path in items.items()}
    except Exception:
        stack.close()
        raise
    readers["_stack"] = stack  # type: ignore[assignment]
    return readers


def close_readers(readers: Dict[str, rasterio.io.DatasetReader]) -> None:
    stack = readers.pop("_stack", None)
    if stack is not None:
        stack.close()


def read_block(
    reader: rasterio.io.DatasetReader,
    row_start: int,
    row_count: int,
    *,
    mask_mode: bool = False,
) -> np.ndarray:
    window = Window(col_off=0, row_off=row_start, width=reader.width, height=row_count)
    data = reader.read(window=window)
    if reader.count == 1:
        data = data[0]
    if mask_mode:
        return data == 1
    data = data.astype(np.float32, copy=False)
    nodata = reader.nodata
    if nodata is not None and np.isfinite(nodata):
        data[data == nodata] = np.nan
    return data


def valid_landcover_mask(landcover_block: np.ndarray, exclude: Sequence[int]) -> np.ndarray:
    mask = np.isfinite(landcover_block)
    if exclude:
        mask &= ~np.isin(landcover_block, list(exclude))
    return mask


def clip_slice(start: int, stop: int, max_len: int) -> slice:
    return slice(max(start, 0), min(stop, max_len))


def first_true_index(mask: np.ndarray) -> Optional[int]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    return int(indices[0])


def compute_true_recovery(
    event_data: np.ndarray,
    tail_data: np.ndarray,
    threshold_value: float,
    peak_index: int,
) -> Optional[int]:
    if tail_data.size == 0:
        return None
    search = tail_data[peak_index:]
    recovery = np.flatnonzero(search >= threshold_value)
    if recovery.size == 0:
        return None
    return int(recovery[0] + 1)


def classify_event(
    drought_trigger: np.ndarray,
    heat_trigger: np.ndarray,
    event_start_offset: int,
) -> Tuple[str, Optional[int]]:
    any_drought = bool(np.any(drought_trigger))
    any_heat = bool(np.any(heat_trigger))
    any_compound = bool(np.any(drought_trigger & heat_trigger))
    first_drought = first_true_index(drought_trigger)
    first_heat = first_true_index(heat_trigger)

    if not any_drought and not any_heat:
        event_type = "Mild_Reference"
    elif any_compound:
        event_type = "Compound_Overlapping"
    elif any_drought and not any_heat:
        event_type = "Drought_Dominant"
    elif any_heat and not any_drought:
        event_type = "Heat_Dominant"
    elif first_drought is not None and first_heat is not None and first_drought < first_heat:
        event_type = "Sequential_Drought_Heat"
    else:
        event_type = "Sequential_Heat_Drought"

    first_stress = min(
        [idx for idx in (first_drought, first_heat) if idx is not None],
        default=None,
    )
    stress_lead = None if first_stress is None else max(0, event_start_offset - first_stress)
    return event_type, stress_lead


def extract_events(
    paths: ProjectPaths,
    config: EventConfig,
    output_path: Optional[Path] = None,
) -> Path:
    output_path = output_path or paths.event_output
    ensure_parent(output_path)
    heatwave_mask_path = ensure_heatwave_mask(paths, config.heat_percentile, config.heat_mask_dir)
    readers = open_readers(paths, heatwave_mask_path=heatwave_mask_path)
    try:
        time_steps = readers["detrended_ndvi"].count
        width = readers["detrended_ndvi"].width
        height = readers["detrended_ndvi"].height
        dates = build_dates(time_steps, config.start_year, config.start_month)

        processed_pixels = 0
        kept_events = 0

        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
            writer.writeheader()

            for row_start in range(0, height, config.block_rows):
                row_count = min(config.block_rows, height - row_start)
                block = {
                    "detrended_ndvi": read_block(readers["detrended_ndvi"], row_start, row_count),
                    "anomaly_mask": read_block(readers["anomaly_mask"], row_start, row_count, mask_mode=True),
                    "drought_mask": read_block(readers["drought_mask"], row_start, row_count, mask_mode=True),
                    "heatwave_mask": read_block(readers["heatwave_mask"], row_start, row_count, mask_mode=True),
                    "climatology_ndvi": read_block(readers["climatology_ndvi"], row_start, row_count),
                    "landcover": read_block(readers["landcover"], row_start, row_count),
                    "spei": read_block(readers["spei"], row_start, row_count),
                    "sti": read_block(readers["sti"], row_start, row_count),
                    "temperature": read_block(readers["temperature"], row_start, row_count),
                    "vpd": read_block(readers["vpd"], row_start, row_count),
                    "ndvi": read_block(readers["ndvi"], row_start, row_count),
                    "soil_moisture": read_block(readers["soil_moisture"], row_start, row_count),
                    "et": read_block(readers["et"], row_start, row_count),
                    "sif": read_block(readers["sif"], row_start, row_count),
                    "gpp": read_block(readers["gpp"], row_start, row_count),
                    "gpp_climatology": read_block(readers["gpp_climatology"], row_start, row_count),
                    "et_climatology": read_block(readers["et_climatology"], row_start, row_count),
                    "sif_climatology": read_block(readers["sif_climatology"], row_start, row_count),
                    "vpd_climatology": read_block(readers["vpd_climatology"], row_start, row_count),
                }

                valid_pixels = valid_landcover_mask(block["landcover"], config.exclude_landcover)

                for local_row in range(row_count):
                    global_row = row_start + local_row
                    valid_cols = np.flatnonzero(valid_pixels[local_row])
                    for col in valid_cols:
                        processed_pixels += 1
                        if config.max_pixels is not None and processed_pixels > config.max_pixels:
                            return output_path

                        pixel_ndvi = block["detrended_ndvi"][:, local_row, col]
                        if np.all(np.isnan(pixel_ndvi)):
                            continue

                        event_slices = [
                            s for s in find_runs(block["anomaly_mask"][:, local_row, col])
                            if (s.stop - s.start) >= config.min_event_duration
                        ]
                        if not event_slices:
                            continue

                        pixel_drought = block["drought_mask"][:, local_row, col]
                        pixel_heat = block["heatwave_mask"][:, local_row, col]
                        pixel_temp = block["temperature"][:, local_row, col]
                        pixel_vpd = block["vpd"][:, local_row, col]
                        pixel_spei = block["spei"][:, local_row, col]
                        pixel_sti = block["sti"][:, local_row, col]
                        pixel_sm = block["soil_moisture"][:, local_row, col]
                        pixel_ndvi_full = block["ndvi"][:, local_row, col]
                        pixel_et = block["et"][:, local_row, col]
                        pixel_sif = block["sif"][:, local_row, col]
                        pixel_gpp = block["gpp"][:, local_row, col]

                        for event_slice in event_slices:
                            event_temp = pixel_temp[event_slice]
                            if event_temp.size == 0 or np.all(np.isnan(event_temp)):
                                continue
                            if np.nanmin(event_temp) <= config.min_event_temperature_c:
                                continue
                            if np.isnan(pixel_sif[event_slice]).all() or np.isnan(pixel_gpp[event_slice]).all():
                                continue

                            pre_slice = clip_slice(event_slice.start - config.pre_event_months, event_slice.start, time_steps)
                            trigger_slice = clip_slice(event_slice.start - config.trigger_lead_months, event_slice.stop, time_steps)
                            post_slice = clip_slice(event_slice.stop, event_slice.stop + config.post_event_months, time_steps)

                            event_data = pixel_ndvi[event_slice]
                            ndvi_sd = float(np.nanstd(pixel_ndvi))
                            threshold_value = -config.anomaly_sigma * ndvi_sd
                            if np.isnan(threshold_value):
                                continue

                            peak_index = int(np.nanargmin(event_data))
                            peak_global_index = event_slice.start + peak_index
                            dec_max_raw = float(event_data[peak_index])
                            ndvi_tail = pixel_ndvi[event_slice.start:]
                            time_rec_true = compute_true_recovery(event_data, ndvi_tail, threshold_value, peak_index)

                            event_indices = np.arange(event_slice.start, event_slice.stop)
                            month_index = dates[event_indices].month.to_numpy() - 1
                            ndvi_clim = block["climatology_ndvi"][month_index, local_row, col]
                            gpp_clim = block["gpp_climatology"][month_index, local_row, col]
                            et_clim = block["et_climatology"][month_index, local_row, col]
                            sif_clim = block["sif_climatology"][month_index, local_row, col]
                            vpd_clim = block["vpd_climatology"][month_index, local_row, col]
                            ndvi_clim_mean = float(np.nanmean(ndvi_clim))
                            if np.isnan(ndvi_clim_mean) or ndvi_clim_mean == 0:
                                continue

                            below_threshold = event_data[event_data < threshold_value]
                            loss_tot = float(np.nansum(np.abs(below_threshold - threshold_value)) / ndvi_clim_mean)
                            dec_max = float(np.abs(dec_max_raw - threshold_value) / ndvi_clim_mean)
                            time_dec = peak_index + 1

                            drought_trigger = pixel_drought[trigger_slice]
                            heat_trigger = pixel_heat[trigger_slice]
                            drought_event = pixel_drought[event_slice]
                            heat_event = pixel_heat[event_slice]
                            event_type, stress_lead = classify_event(
                                drought_trigger,
                                heat_trigger,
                                event_start_offset=event_slice.start - trigger_slice.start,
                            )

                            n_compound_event = int(np.sum(drought_event & heat_event))
                            compound_share = float(n_compound_event / len(drought_event)) if len(drought_event) else np.nan

                            writer.writerow(
                                {
                                    "pixel_id": global_row * width + int(col),
                                    "row": global_row,
                                    "col": int(col),
                                    "heat_percentile": config.heat_percentile,
                                    "land_cover": float(block["landcover"][local_row, col]),
                                    "event_start_index": event_slice.start,
                                    "event_end_index": event_slice.stop - 1,
                                    "duration": len(event_data),
                                    "start_date": dates[event_slice.start].strftime("%Y-%m-%d"),
                                    "end_date": dates[event_slice.stop - 1].strftime("%Y-%m-%d"),
                                    "event_type": event_type,
                                    "trigger_start_index": trigger_slice.start,
                                    "trigger_end_index": trigger_slice.stop - 1,
                                    "stress_lead_months": stress_lead,
                                    "n_drought_trigger": int(np.sum(drought_trigger)),
                                    "n_heat_trigger": int(np.sum(heat_trigger)),
                                    "n_compound_trigger": int(np.sum(drought_trigger & heat_trigger)),
                                    "n_drought_event": int(np.sum(drought_event)),
                                    "n_heat_event": int(np.sum(heat_event)),
                                    "n_compound_event": n_compound_event,
                                    "compound_share": compound_share,
                                    "stress_path_trigger": stress_path_string(drought_trigger, heat_trigger),
                                    "stress_path_event": stress_path_string(drought_event, heat_event),
                                    "ndvi_sd": ndvi_sd,
                                    "threshold_value": threshold_value,
                                    "LOSStot": loss_tot,
                                    "DECmax": dec_max,
                                    "TIMEdec": time_dec,
                                    "TIMErec_true": time_rec_true,
                                    "max_decrease_date": dates[peak_global_index].strftime("%Y-%m-%d"),
                                    "pre_spei_path": serialize(pixel_spei[pre_slice], config.precision),
                                    "pre_temp_path": serialize(pixel_temp[pre_slice], config.precision),
                                    "pre_vpd_path": serialize(pixel_vpd[pre_slice], config.precision),
                                    "pre_sti_path": serialize(pixel_sti[pre_slice], config.precision),
                                    "pre_sm_path": serialize(pixel_sm[pre_slice], config.precision),
                                    "pre_ndvi_path": serialize(pixel_ndvi_full[pre_slice], config.precision),
                                    "pre_et_path": serialize(pixel_et[pre_slice], config.precision),
                                    "pre_sif_path": serialize(pixel_sif[pre_slice], config.precision),
                                    "pre_gpp_path": serialize(pixel_gpp[pre_slice], config.precision),
                                    "spei_path": serialize(pixel_spei[event_slice], config.precision),
                                    "temp_path": serialize(pixel_temp[event_slice], config.precision),
                                    "vpd_path": serialize(pixel_vpd[event_slice], config.precision),
                                    "sti_path": serialize(pixel_sti[event_slice], config.precision),
                                    "sm_path": serialize(pixel_sm[event_slice], config.precision),
                                    "ndvi_path": serialize(pixel_ndvi_full[event_slice], config.precision),
                                    "et_path": serialize(pixel_et[event_slice], config.precision),
                                    "sif_path": serialize(pixel_sif[event_slice], config.precision),
                                    "gpp_path": serialize(pixel_gpp[event_slice], config.precision),
                                    "post_spei_path": serialize(pixel_spei[post_slice], config.precision),
                                    "post_temp_path": serialize(pixel_temp[post_slice], config.precision),
                                    "post_vpd_path": serialize(pixel_vpd[post_slice], config.precision),
                                    "post_sti_path": serialize(pixel_sti[post_slice], config.precision),
                                    "post_sm_path": serialize(pixel_sm[post_slice], config.precision),
                                    "post_ndvi_path": serialize(pixel_ndvi_full[post_slice], config.precision),
                                    "post_et_path": serialize(pixel_et[post_slice], config.precision),
                                    "post_sif_path": serialize(pixel_sif[post_slice], config.precision),
                                    "post_gpp_path": serialize(pixel_gpp[post_slice], config.precision),
                                    "ndvi_clim_path": serialize(ndvi_clim, config.precision),
                                    "gpp_clim_path": serialize(gpp_clim, config.precision),
                                    "et_clim_path": serialize(et_clim, config.precision),
                                    "sif_clim_path": serialize(sif_clim, config.precision),
                                    "vpd_clim_path": serialize(vpd_clim, config.precision),
                                }
                            )
                            kept_events += 1

                print(f"[events] processed rows {row_start + row_count}/{height}, kept events={kept_events}, pixels={processed_pixels}")

        return output_path
    finally:
        close_readers(readers)


def build_panel_dataset(
    paths: ProjectPaths,
    config: PanelConfig,
    output_path: Optional[Path] = None,
) -> Path:
    output_path = output_path or paths.panel_output
    ensure_parent(output_path)
    heatwave_mask_path = ensure_heatwave_mask(paths, config.heat_percentile, config.heat_mask_dir)
    readers = open_readers(paths, heatwave_mask_path=heatwave_mask_path)
    try:
        time_steps = readers["detrended_ndvi"].count
        width = readers["detrended_ndvi"].width
        height = readers["detrended_ndvi"].height
        dates = build_dates(time_steps, config.start_year, config.start_month)
        max_lag = max(config.lags)

        processed_pixels = 0
        buffered_rows: List[Dict[str, object]] = []

        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PANEL_FIELDS)
            writer.writeheader()

            for row_start in range(0, height, config.block_rows):
                row_count = min(config.block_rows, height - row_start)
                block = {
                    "landcover": read_block(readers["landcover"], row_start, row_count),
                    "climatology_ndvi": read_block(readers["climatology_ndvi"], row_start, row_count),
                    "anomaly_mask": read_block(readers["anomaly_mask"], row_start, row_count, mask_mode=True),
                    "drought_mask": read_block(readers["drought_mask"], row_start, row_count, mask_mode=True),
                    "heatwave_mask": read_block(readers["heatwave_mask"], row_start, row_count, mask_mode=True),
                    "detrended_ndvi": read_block(readers["detrended_ndvi"], row_start, row_count),
                    "sif": read_block(readers["sif"], row_start, row_count),
                    "gpp": read_block(readers["gpp"], row_start, row_count),
                    "et": read_block(readers["et"], row_start, row_count),
                    "soil_moisture": read_block(readers["soil_moisture"], row_start, row_count),
                    "spei": read_block(readers["spei"], row_start, row_count),
                    "sti": read_block(readers["sti"], row_start, row_count),
                    "temperature": read_block(readers["temperature"], row_start, row_count),
                    "vpd": read_block(readers["vpd"], row_start, row_count),
                    "radiation": read_block(readers["radiation"], row_start, row_count),
                }

                valid_pixels = valid_landcover_mask(block["landcover"], config.exclude_landcover)

                for local_row in range(row_count):
                    global_row = row_start + local_row
                    valid_cols = np.flatnonzero(valid_pixels[local_row])[:: config.pixel_stride]
                    for col in valid_cols:
                        processed_pixels += 1
                        if config.max_pixels is not None and processed_pixels > config.max_pixels:
                            if buffered_rows:
                                writer.writerows(buffered_rows)
                            return output_path

                        series = {name: values[:, local_row, col] for name, values in block.items() if name != "landcover"}
                        if np.all(np.isnan(series["detrended_ndvi"])):
                            continue

                        pixel_id = global_row * width + int(col)
                        climatology_bands = block["climatology_ndvi"].shape[0]
                        for t in range(max_lag, time_steps):
                            month_index = (dates[t].month - 1) % climatology_bands
                            ndvi_clim_month = block["climatology_ndvi"][month_index, local_row, col]
                            growing_flag = is_growing_season(
                                temp_value=series["temperature"][t],
                                ndvi_clim_value=ndvi_clim_month,
                                mode=config.growing_season_mode,
                                temp_threshold=config.growing_temp_c,
                                ndvi_threshold=config.growing_ndvi_threshold,
                            )
                            if not growing_flag:
                                continue
                            buffered_rows.append(
                                {
                                    "pixel_id": pixel_id,
                                    "row": global_row,
                                    "col": int(col),
                                    "heat_percentile": config.heat_percentile,
                                    "date": dates[t].strftime("%Y-%m-%d"),
                                    "year": int(dates[t].year),
                                    "month": int(dates[t].month),
                                    "land_cover": float(block["landcover"][local_row, col]),
                                    "growing_season_flag": int(growing_flag),
                                    "ndvi_clim_month": ndvi_clim_month,
                                    "vna_flag": int(series["anomaly_mask"][t]),
                                    "drought": int(series["drought_mask"][t]),
                                    "heat": int(series["heatwave_mask"][t]),
                                    "compound": int(series["drought_mask"][t] and series["heatwave_mask"][t]),
                                    "ndvi_anom": series["detrended_ndvi"][t],
                                    "sif_anom": series["sif"][t],
                                    "gpp_anom": series["gpp"][t],
                                    "et_anom": series["et"][t],
                                    "sm_anom": series["soil_moisture"][t],
                                    "spei": series["spei"][t],
                                    "sti": series["sti"][t],
                                    "temp": series["temperature"][t],
                                    "vpd": series["vpd"][t],
                                    "radiation": series["radiation"][t],
                                    "ndvi_lag1": series["detrended_ndvi"][t - 1],
                                    "sm_lag1": series["soil_moisture"][t - 1],
                                    "drought_lag1": int(series["drought_mask"][t - 1]),
                                    "heat_lag1": int(series["heatwave_mask"][t - 1]),
                                    "spei_lag1": series["spei"][t - 1],
                                    "temp_lag1": series["temperature"][t - 1],
                                    "vpd_lag1": series["vpd"][t - 1],
                                    "drought_lag2": int(series["drought_mask"][t - 2]),
                                    "heat_lag2": int(series["heatwave_mask"][t - 2]),
                                    "spei_lag2": series["spei"][t - 2],
                                    "temp_lag2": series["temperature"][t - 2],
                                    "vpd_lag2": series["vpd"][t - 2],
                                    "drought_lag3": int(series["drought_mask"][t - 3]),
                                    "heat_lag3": int(series["heatwave_mask"][t - 3]),
                                    "spei_lag3": series["spei"][t - 3],
                                    "temp_lag3": series["temperature"][t - 3],
                                    "vpd_lag3": series["vpd"][t - 3],
                                }
                            )
                            if len(buffered_rows) >= config.flush_rows:
                                writer.writerows(buffered_rows)
                                buffered_rows.clear()

                print(f"[panel] processed rows {row_start + row_count}/{height}, pixels={processed_pixels}, buffered={len(buffered_rows)}")

            if buffered_rows:
                writer.writerows(buffered_rows)

        return output_path
    finally:
        close_readers(readers)


def fit_interaction_model(
    panel_path: Path,
    summary_output: Path,
    coefficient_output: Path,
    *,
    response: str = "ndvi_anom",
    scale_mode: str = "zscore_continuous",
    max_rows: int = 500000,
    chunk_size: int = 200000,
) -> Tuple[Path, Path]:
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError("statsmodels is required for fit-model.") from exc

    ensure_parent(summary_output)
    ensure_parent(coefficient_output)

    required = [
        "pixel_id",
        "month",
        response,
        "drought",
        "heat",
        "spei",
        "temp",
        "vpd",
        "sti",
        "ndvi_lag1",
        "sm_lag1",
        "drought_lag1",
        "heat_lag1",
        "drought_lag2",
        "heat_lag2",
        "drought_lag3",
        "heat_lag3",
    ]

    chunks: List[pd.DataFrame] = []
    current = 0
    for chunk in pd.read_csv(panel_path, usecols=required, chunksize=chunk_size):
        chunk = chunk.dropna(subset=[response, "spei", "temp", "vpd", "sti", "ndvi_lag1", "sm_lag1"])
        if chunk.empty:
            continue
        remaining = max_rows - current
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining].copy()
        chunks.append(chunk)
        current += len(chunk)
        if current >= max_rows:
            break

    if not chunks:
        raise ValueError("No valid panel rows were available for model fitting.")

    df = pd.concat(chunks, ignore_index=True)
    df["drought_x_heat"] = df["drought"] * df["heat"]
    if scale_mode not in {"none", "zscore_continuous"}:
        raise ValueError(f"Unsupported scale mode: {scale_mode}")

    if scale_mode == "zscore_continuous":
        continuous_cols = [response, "spei", "temp", "vpd", "sti", "ndvi_lag1", "sm_lag1"]
        for col in continuous_cols:
            std = df[col].std(ddof=0)
            if std and np.isfinite(std):
                df[col] = (df[col] - df[col].mean()) / std

    regressors = [
        "drought",
        "heat",
        "drought_x_heat",
        "spei",
        "temp",
        "vpd",
        "sti",
        "ndvi_lag1",
        "sm_lag1",
        "drought_lag1",
        "heat_lag1",
        "drought_lag2",
        "heat_lag2",
        "drought_lag3",
        "heat_lag3",
    ]

    month_dummies = pd.get_dummies(df["month"].astype(int), prefix="month", drop_first=True, dtype=float)
    design = pd.concat([df[["pixel_id", response] + regressors], month_dummies], axis=1)
    group = design.groupby("pixel_id")

    y = design[response] - group[response].transform("mean")
    x_columns = regressors + list(month_dummies.columns)
    x = design[x_columns].copy()
    for col in x_columns:
        x[col] = x[col] - group[col].transform("mean")

    model = sm.OLS(y, x, hasconst=False)
    results = model.fit(cov_type="HC3")

    with summary_output.open("w", encoding="utf-8") as handle:
        handle.write(f"scale_mode: {scale_mode}\n")
        handle.write(f"response: {response}\n")
        handle.write(f"n_rows: {len(df)}\n\n")
        handle.write(results.summary().as_text())

    coef = pd.DataFrame(
        {
            "term": results.params.index,
            "coef": results.params.values,
            "std_err": results.bse.values,
            "t_value": results.tvalues.values,
            "p_value": results.pvalues.values,
        }
    )
    coef.to_csv(coefficient_output, index=False)
    return summary_output, coefficient_output


def fit_spatial_interaction_models(
    panel_path: Path,
    output_path: Path,
    *,
    response: str = "ndvi_anom",
    scale_mode: str = "zscore_by_pixel",
    min_rows: int = 30,
) -> Path:
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError("statsmodels is required for fit-spatial-model.") from exc

    if scale_mode not in {"none", "zscore_by_pixel"}:
        raise ValueError(f"Unsupported scale mode: {scale_mode}")

    ensure_parent(output_path)
    required = [
        "pixel_id",
        "row",
        "col",
        "heat_percentile",
        "land_cover",
        response,
        "drought",
        "heat",
        "ndvi_lag1",
    ]
    df = pd.read_csv(panel_path, usecols=required)
    df = df.dropna(subset=[response, "ndvi_lag1"])
    if df.empty:
        raise ValueError("No valid rows were available for local model fitting.")

    outputs: List[Dict[str, object]] = []
    grouped = df.groupby(["pixel_id", "row", "col", "heat_percentile", "land_cover"], sort=False)

    for (pixel_id, row, col, heat_percentile, land_cover), group in grouped:
        if len(group) < min_rows:
            continue

        local = group.copy()
        local["drought_x_heat"] = local["drought"] * local["heat"]

        if scale_mode == "zscore_by_pixel":
            for col_name in [response, "ndvi_lag1"]:
                std = local[col_name].std(ddof=0)
                if std and np.isfinite(std):
                    local[col_name] = (local[col_name] - local[col_name].mean()) / std

        x = local[["drought", "heat", "drought_x_heat", "ndvi_lag1"]].astype(float)
        x = sm.add_constant(x, has_constant="add")
        y = local[response].astype(float)

        if np.linalg.matrix_rank(x.to_numpy()) < x.shape[1]:
            continue

        try:
            results = sm.OLS(y, x, hasconst=True).fit()
        except Exception:
            continue

        beta_drought = float(results.params.get("drought", np.nan))
        beta_heat = float(results.params.get("heat", np.nan))
        beta_interaction = float(results.params.get("drought_x_heat", np.nan))
        beta_ndvi_lag1 = float(results.params.get("ndvi_lag1", np.nan))
        p_interaction = float(results.pvalues.get("drought_x_heat", np.nan))
        synergy_ratio = np.nan
        if np.isfinite(beta_drought) and beta_drought != 0:
            synergy_ratio = float(beta_interaction / abs(beta_drought))

        outputs.append(
            {
                "pixel_id": int(pixel_id),
                "row": int(row),
                "col": int(col),
                "heat_percentile": int(heat_percentile),
                "land_cover": float(land_cover),
                "n_obs": int(len(local)),
                "n_drought_months": int(local["drought"].sum()),
                "n_heat_months": int(local["heat"].sum()),
                "n_compound_months": int(local["drought_x_heat"].sum()),
                "scale_mode": scale_mode,
                "response": response,
                "beta_const": float(results.params.get("const", np.nan)),
                "beta_drought": beta_drought,
                "beta_heat": beta_heat,
                "beta_interaction": beta_interaction,
                "beta_ndvi_lag1": beta_ndvi_lag1,
                "p_drought": float(results.pvalues.get("drought", np.nan)),
                "p_heat": float(results.pvalues.get("heat", np.nan)),
                "p_interaction": p_interaction,
                "p_ndvi_lag1": float(results.pvalues.get("ndvi_lag1", np.nan)),
                "r_squared": float(results.rsquared),
                "adj_r_squared": float(results.rsquared_adj),
                "synergy_ratio": synergy_ratio,
                "heat_amplifies_drought": bool(np.isfinite(beta_interaction) and beta_interaction < 0),
                "heat_amplifies_drought_sig": bool(np.isfinite(beta_interaction) and beta_interaction < 0 and p_interaction < 0.05),
            }
        )

    result_df = pd.DataFrame(outputs)
    result_df.to_csv(output_path, index=False)
    return output_path


def export_local_coefficients_to_rasters(
    local_coefficient_path: Path,
    output_dir: Path,
    *,
    template_path: Path,
    value_columns: Optional[Sequence[str]] = None,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    default_columns = [
        "beta_drought",
        "beta_heat",
        "beta_interaction",
        "p_drought",
        "p_heat",
        "p_interaction",
        "n_obs",
        "r_squared",
        "adj_r_squared",
        "synergy_ratio",
        "heat_amplifies_drought",
        "heat_amplifies_drought_sig",
    ]
    selected_columns = list(value_columns) if value_columns is not None else default_columns

    df = pd.read_csv(local_coefficient_path)
    required = {"row", "col", *selected_columns}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in local coefficient table: {sorted(missing)}")

    with rasterio.open(template_path) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width

    outputs: List[Path] = []
    bool_columns = {"heat_amplifies_drought", "heat_amplifies_drought_sig"}
    int_columns = {"n_obs"}

    for column in selected_columns:
        if column in bool_columns:
            array = np.full((height, width), 255, dtype=np.uint8)
            valid_values = df[column].fillna(False).astype(np.uint8).to_numpy()
            out_profile = profile.copy()
            out_profile.update(dtype=rasterio.uint8, count=1, nodata=255, compress="lzw")
        elif column in int_columns:
            array = np.full((height, width), -9999.0, dtype=np.float32)
            valid_values = df[column].astype(float).to_numpy()
            out_profile = profile.copy()
            out_profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0, compress="lzw")
        else:
            array = np.full((height, width), -9999.0, dtype=np.float32)
            valid_values = df[column].astype(float).to_numpy()
            out_profile = profile.copy()
            out_profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0, compress="lzw")

        rows = df["row"].astype(int).to_numpy()
        cols = df["col"].astype(int).to_numpy()
        valid_mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        array[rows[valid_mask], cols[valid_mask]] = valid_values[valid_mask]

        output_path = output_dir / f"{local_coefficient_path.stem}_{column}.tif"
        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(array, 1)
        outputs.append(output_path)

    return outputs


def run_sensitivity_suite(
    paths: ProjectPaths,
    *,
    output_dir: Path,
    heat_percentiles: Sequence[int],
    event_max_pixels: Optional[int],
    panel_max_pixels: Optional[int],
    pixel_stride: int,
    scale_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for percentile in heat_percentiles:
        event_path = output_dir / f"test_integrated_events_{percentile}p.csv"
        panel_path = output_dir / f"test_pixel_month_panel_{percentile}p.csv"
        summary_path = output_dir / f"test_model_summary_{percentile}p.txt"
        coefficient_path = output_dir / f"test_model_coefficients_{percentile}p.csv"
        local_path = output_dir / f"test_local_coefficients_{percentile}p.csv"

        event_config = EventConfig(
            max_pixels=event_max_pixels,
            heat_percentile=percentile,
            heat_mask_dir=output_dir,
        )
        panel_config = PanelConfig(
            max_pixels=panel_max_pixels,
            pixel_stride=pixel_stride,
            heat_percentile=percentile,
            heat_mask_dir=output_dir,
        )

        extract_events(paths, event_config, output_path=event_path)
        build_panel_dataset(paths, panel_config, output_path=panel_path)
        fit_interaction_model(
            panel_path=panel_path,
            summary_output=summary_path,
            coefficient_output=coefficient_path,
            scale_mode=scale_mode,
        )
        fit_spatial_interaction_models(
            panel_path=panel_path,
            output_path=local_path,
            scale_mode="zscore_by_pixel",
            min_rows=12,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated vegetation event and panel workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    event_parser = subparsers.add_parser("extract-events", help="Build the integrated VNA event inventory.")
    event_parser.add_argument("--output", type=Path, default=ProjectPaths().event_output)
    event_parser.add_argument("--max-pixels", type=int, default=None)
    event_parser.add_argument("--block-rows", type=int, default=EventConfig().block_rows)
    event_parser.add_argument("--pre-months", type=int, default=EventConfig().pre_event_months)
    event_parser.add_argument("--post-months", type=int, default=EventConfig().post_event_months)
    event_parser.add_argument("--trigger-months", type=int, default=EventConfig().trigger_lead_months)
    event_parser.add_argument("--heat-percentile", type=int, default=90)
    event_parser.add_argument("--heat-mask-dir", type=Path, default=None)

    panel_parser = subparsers.add_parser("build-panel", help="Build the pixel-month panel dataset.")
    panel_parser.add_argument("--output", type=Path, default=ProjectPaths().panel_output)
    panel_parser.add_argument("--max-pixels", type=int, default=None)
    panel_parser.add_argument("--pixel-stride", type=int, default=PanelConfig().pixel_stride)
    panel_parser.add_argument("--block-rows", type=int, default=PanelConfig().block_rows)
    panel_parser.add_argument("--flush-rows", type=int, default=PanelConfig().flush_rows)
    panel_parser.add_argument("--heat-percentile", type=int, default=90)
    panel_parser.add_argument("--heat-mask-dir", type=Path, default=None)
    panel_parser.add_argument("--growing-season-mode", type=str, choices=["none", "temp", "ndvi", "temp_and_ndvi"], default=PanelConfig().growing_season_mode)
    panel_parser.add_argument("--growing-temp", type=float, default=PanelConfig().growing_temp_c)
    panel_parser.add_argument("--growing-ndvi-threshold", type=float, default=PanelConfig().growing_ndvi_threshold)

    model_parser = subparsers.add_parser("fit-model", help="Fit a panel interaction model from the panel CSV.")
    model_parser.add_argument("--panel-path", type=Path, default=ProjectPaths().panel_output)
    model_parser.add_argument("--response", type=str, default="ndvi_anom")
    model_parser.add_argument("--scale-mode", type=str, choices=["none", "zscore_continuous"], default="zscore_continuous")
    model_parser.add_argument("--max-rows", type=int, default=500000)
    model_parser.add_argument("--summary-output", type=Path, default=ProjectPaths().model_output)
    model_parser.add_argument("--coefficient-output", type=Path, default=ProjectPaths().coefficient_output)

    local_parser = subparsers.add_parser("fit-spatial-model", help="Fit per-pixel local interaction models for spatial mapping.")
    local_parser.add_argument("--panel-path", type=Path, default=ProjectPaths().panel_output)
    local_parser.add_argument("--response", type=str, default="ndvi_anom")
    local_parser.add_argument("--scale-mode", type=str, choices=["none", "zscore_by_pixel"], default="zscore_by_pixel")
    local_parser.add_argument("--min-rows", type=int, default=30)
    local_parser.add_argument("--output", type=Path, default=ProjectPaths().local_coefficient_output)

    export_parser = subparsers.add_parser("export-spatial-rasters", help="Export local interaction coefficients to raster maps.")
    export_parser.add_argument("--local-coeff-path", type=Path, required=True)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--template-path", type=Path, default=ProjectPaths().landcover)
    export_parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of local coefficient columns to export.",
    )

    sensitivity_parser = subparsers.add_parser("run-sensitivity", help="Run event, panel, and model outputs for multiple heat thresholds.")
    sensitivity_parser.add_argument("--output-dir", type=Path, default=ProjectPaths().test_dir)
    sensitivity_parser.add_argument("--heat-percentiles", type=int, nargs="+", default=[90, 95])
    sensitivity_parser.add_argument("--event-max-pixels", type=int, default=500)
    sensitivity_parser.add_argument("--panel-max-pixels", type=int, default=50)
    sensitivity_parser.add_argument("--pixel-stride", type=int, default=4)
    sensitivity_parser.add_argument("--scale-mode", type=str, choices=["none", "zscore_continuous"], default="zscore_continuous")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths()

    if args.command == "extract-events":
        config = EventConfig(
            max_pixels=args.max_pixels,
            block_rows=args.block_rows,
            pre_event_months=args.pre_months,
            post_event_months=args.post_months,
            trigger_lead_months=args.trigger_months,
            heat_percentile=args.heat_percentile,
            heat_mask_dir=args.heat_mask_dir,
        )
        output = extract_events(paths, config, output_path=args.output)
        print(f"Event inventory saved to {output}")
        return

    if args.command == "build-panel":
        config = PanelConfig(
            max_pixels=args.max_pixels,
            block_rows=args.block_rows,
            pixel_stride=args.pixel_stride,
            flush_rows=args.flush_rows,
            heat_percentile=args.heat_percentile,
            heat_mask_dir=args.heat_mask_dir,
            growing_season_mode=args.growing_season_mode,
            growing_temp_c=args.growing_temp,
            growing_ndvi_threshold=args.growing_ndvi_threshold,
        )
        output = build_panel_dataset(paths, config, output_path=args.output)
        print(f"Panel dataset saved to {output}")
        return

    if args.command == "fit-model":
        summary_path, coefficient_path = fit_interaction_model(
            panel_path=args.panel_path,
            summary_output=args.summary_output,
            coefficient_output=args.coefficient_output,
            response=args.response,
            scale_mode=args.scale_mode,
            max_rows=args.max_rows,
        )
        print(f"Model summary saved to {summary_path}")
        print(f"Model coefficients saved to {coefficient_path}")
        return

    if args.command == "fit-spatial-model":
        output_path = fit_spatial_interaction_models(
            panel_path=args.panel_path,
            output_path=args.output,
            response=args.response,
            scale_mode=args.scale_mode,
            min_rows=args.min_rows,
        )
        print(f"Local coefficient table saved to {output_path}")
        return

    if args.command == "export-spatial-rasters":
        output_paths = export_local_coefficients_to_rasters(
            local_coefficient_path=args.local_coeff_path,
            output_dir=args.output_dir,
            template_path=args.template_path,
            value_columns=args.columns,
        )
        print(f"Exported {len(output_paths)} raster maps to {args.output_dir}")
        return

    if args.command == "run-sensitivity":
        run_sensitivity_suite(
            paths,
            output_dir=args.output_dir,
            heat_percentiles=args.heat_percentiles,
            event_max_pixels=args.event_max_pixels,
            panel_max_pixels=args.panel_max_pixels,
            pixel_stride=args.pixel_stride,
            scale_mode=args.scale_mode,
        )
        print(f"Sensitivity outputs saved to {args.output_dir}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
