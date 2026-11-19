from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

BASE_DIR = Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve()
MAIN_DIR = BASE_DIR / "output_df" / "global_v05_90p_vpd_main"
EVENT_PATH = MAIN_DIR / "global_integrated_events_90p_vpd.csv"
ANOM_EVENT_LEVEL_PATH = MAIN_DIR / "event_analysis_anomaly" / "event_multivariate_response_anomaly_event_level.csv"
RAW_MATCHES_PATH = BASE_DIR / "output_df" / "global_v05_90p_main" / "matched_attribution" / "compound_vs_drought_matches.csv"
OUTPUT_DIR = MAIN_DIR / "event_analysis_anomaly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
USED_DATA = BASE_DIR / "used_data"

CLIM_PATHS = {
    "ndvi": USED_DATA / "ndvi_climatology_2001-2024.tif",
    "sif": USED_DATA / "sif_monthly_climatology_2001-2024.tif",
    "gpp": USED_DATA / "gpp_monthly_climatology_2001-2024.tif",
    "et": USED_DATA / "Et_monthly_climatology_2001-2024.tif",
}
LAND_COVER_MAP = {
    1: "Forest", 2: "Forest", 3: "Forest", 4: "Forest", 5: "Forest",
    6: "Shrub", 7: "Shrub",
    8: "Savannas", 9: "Savannas",
    10: "Grassland",
    12: "Cropland", 14: "Cropland",
}
LAND_ORDER = ["Forest", "Shrub", "Savannas", "Grassland", "Cropland"]
TREATED_LABEL = "Compound_Overlapping"
CONTROL_LABEL = "Drought_Dominant"
FUNCTIONAL_VARS = ["ndvi", "sif", "gpp", "et"]
CUM_OUTCOMES = [f"{v}_cum_loss" for v in FUNCTIONAL_VARS]
MATCH_FEATURES_ANOM = [
    "duration",
    "ndvi_anom_pre_mean",
    "gpp_anom_pre_mean",
    "et_anom_pre_mean",
    "spei_anom_pre_mean",
    "sm_anom_pre_mean",
    "temp_anom_pre_mean",
    "sti_anom_pre_mean",
]


def parse_series(value: object) -> list[float]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    out: list[float] = []
    for token in text.split(','):
        token = token.strip()
        if not token or token.lower() == 'nan':
            out.append(np.nan)
        else:
            try:
                out.append(float(token))
            except ValueError:
                out.append(np.nan)
    return out


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


def compute_event_anomaly_series(raw_values: list[float], dates: pd.DatetimeIndex, row: int, col: int, clim: np.ndarray) -> list[float]:
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


def cumulative_deficit(pre_mean: float, event_series: Iterable[float]) -> float:
    arr = np.asarray(list(event_series), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    # In anomaly domain, cumulative loss is the accumulated negative departure below the monthly climatology.
    return float(np.maximum(-arr, 0.0).sum())


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 400, seed: int = 42) -> tuple[float, float]:
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = deltas.size
    for i in range(n_boot):
        sample = rng.choice(deltas, size=n, replace=True)
        means[i] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def assign_duration_bin(duration: pd.Series) -> pd.Series:
    return pd.cut(duration, bins=[1, 2, 3, 4, 6, np.inf], labels=["2", "3", "4", "5-6", "7+"])


def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors='coerce')
    b = pd.to_numeric(b, errors='coerce')
    pooled_sd = np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2.0)
    if pooled_sd == 0 or not np.isfinite(pooled_sd):
        return np.nan
    return float((a.mean() - b.mean()) / pooled_sd)


def build_matches_generic(df: pd.DataFrame, k_neighbors: int = 3) -> pd.DataFrame:
    df = df.copy()
    df['duration_bin'] = assign_duration_bin(df['duration'])
    df['event_row_id'] = np.arange(len(df))
    treated = df[df['event_type'] == TREATED_LABEL].copy()
    control = df[df['event_type'] == CONTROL_LABEL].copy()
    matches: List[pd.DataFrame] = []
    for land in LAND_ORDER:
        for duration_bin in ["2", "3", "4", "5-6", "7+"]:
            tg = treated[(treated['land_cover_broad'] == land) & (treated['duration_bin'].astype(str) == duration_bin)].copy()
            cg = control[(control['land_cover_broad'] == land) & (control['duration_bin'].astype(str) == duration_bin)].copy()
            if tg.empty or cg.empty:
                continue
            pooled = pd.concat([tg[MATCH_FEATURES_ANOM], cg[MATCH_FEATURES_ANOM]], axis=0)
            medians = pooled.median(numeric_only=True)
            stds = pooled.std(ddof=0, numeric_only=True).replace(0, 1.0)
            tx = (tg[MATCH_FEATURES_ANOM].fillna(medians) - medians) / stds
            cx = (cg[MATCH_FEATURES_ANOM].fillna(medians) - medians) / stds
            tree = cKDTree(cx.to_numpy(dtype=float))
            k = min(k_neighbors, len(cg))
            dist, idx = tree.query(tx.to_numpy(dtype=float), k=k)
            if k == 1:
                dist = dist[:, None]
                idx = idx[:, None]
            block = tg[['event_row_id', 'land_cover_broad', 'duration_bin']].reset_index(drop=True).copy()
            block['match_distance_mean'] = dist.mean(axis=1)
            for neighbor in range(k):
                block[f'control_row_id_{neighbor+1}'] = cg.iloc[idx[:, neighbor]]['event_row_id'].to_numpy()
            matches.append(block)
    return pd.concat(matches, ignore_index=True)


def compute_att_with_matches(df: pd.DataFrame, matches: pd.DataFrame, outcomes: list[str], group_cols: list[str]) -> pd.DataFrame:
    treated = df.set_index('event_row_id').loc[matches['event_row_id']].reset_index()
    control_cols = [c for c in matches.columns if c.startswith('control_row_id_')]
    control_tables = []
    for col in control_cols:
        control_subset = df.set_index('event_row_id').loc[matches[col]].reset_index(drop=True)
        control_tables.append(control_subset[outcomes])
    control_mean = sum(control_tables) / len(control_tables)
    paired = treated[group_cols + outcomes].copy()
    for outcome in outcomes:
        paired[f'matched_control_{outcome}'] = control_mean[outcome].to_numpy()
        paired[f'delta_{outcome}'] = paired[outcome] - paired[f'matched_control_{outcome}']
    rows: List[Dict[str, object]] = []
    grouped = paired.groupby(group_cols, dropna=False) if group_cols else [(("Global",), paired)]
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_meta = dict(zip(group_cols if group_cols else ['scope'], group_key))
        group_meta['n_treated'] = len(group)
        for outcome in outcomes:
            deltas = group[f'delta_{outcome}'].to_numpy(dtype=float)
            treated_values = group[outcome].to_numpy(dtype=float)
            control_values = group[f'matched_control_{outcome}'].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_ci(deltas)
            control_mean_val = float(np.nanmean(control_values))
            att_mean = float(np.nanmean(deltas))
            rows.append({
                **group_meta,
                'outcome': outcome,
                'treated_mean': float(np.nanmean(treated_values)),
                'matched_control_mean': control_mean_val,
                'att_mean': att_mean,
                'relative_pct': float(att_mean / abs(control_mean_val) * 100.0) if np.isfinite(control_mean_val) and control_mean_val != 0 else np.nan,
                'ci_low': ci_low,
                'ci_high': ci_high,
            })
    return pd.DataFrame(rows)


def bootstrap_mean_difference(a: np.ndarray, b: np.ndarray, n_boot: int = 500, seed: int = 42) -> tuple[float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample_a = rng.choice(a, size=a.size, replace=True)
        sample_b = rng.choice(b, size=b.size, replace=True)
        diffs[i] = sample_b.mean() - sample_a.mean()
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 400, seed: int = 42) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    observed = abs(b.mean() - a.mean())
    combined = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        permuted = rng.permutation(combined)
        diff = abs(permuted[n_a:].mean() - permuted[:n_a].mean())
        if diff >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


def compare_periods(paired_by_period: Dict[str, pd.DataFrame], outcomes: list[str]) -> pd.DataFrame:
    early = paired_by_period['2001-2012']
    late = paired_by_period['2013-2024']
    rows: List[Dict[str, object]] = []
    for idx, outcome in enumerate(outcomes):
        early_delta = early[f'delta_{outcome}'].to_numpy(dtype=float)
        late_delta = late[f'delta_{outcome}'].to_numpy(dtype=float)
        early_mean = float(np.nanmean(early_delta))
        late_mean = float(np.nanmean(late_delta))
        diff = late_mean - early_mean
        ci_low, ci_high = bootstrap_mean_difference(early_delta, late_delta, seed=101 + idx)
        rows.append({
            'outcome': outcome,
            'early_att_mean': early_mean,
            'late_att_mean': late_mean,
            'late_minus_early': diff,
            'relative_change_pct': float(diff / abs(early_mean) * 100.0) if np.isfinite(early_mean) and early_mean != 0 else np.nan,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_permutation': permutation_pvalue(early_delta, late_delta, seed=202 + idx),
        })
    return pd.DataFrame(rows)


def build_inputs() -> pd.DataFrame:
    anom = pd.read_csv(ANOM_EVENT_LEVEL_PATH)
    usecols = ['pixel_id', 'row', 'col', 'event_type', 'land_cover', 'duration', 'start_date', 'end_date', 'LOSStot']
    for v in FUNCTIONAL_VARS:
        usecols.append(f'{v}_path')
    raw = pd.read_csv(EVENT_PATH, usecols=usecols)
    raw['land_cover_broad'] = raw['land_cover'].map(land_cover_broad)
    raw = raw[raw['land_cover_broad'] != 'Other'].reset_index(drop=True)
    anom = anom.reset_index(drop=True)
    if len(raw) != len(anom):
        raise ValueError('Raw and anomaly event tables do not align in length.')
    string_key_cols = ['pixel_id', 'event_type', 'start_date']
    for col in string_key_cols:
        left = raw[col].astype(str).reset_index(drop=True)
        right = anom[col].astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f'Alignment check failed for {col}')
    numeric_key_cols = ['duration', 'LOSStot']
    for col in numeric_key_cols:
        left = pd.to_numeric(raw[col], errors='coerce').to_numpy(dtype=float)
        right = pd.to_numeric(anom[col], errors='coerce').to_numpy(dtype=float)
        if not np.allclose(left, right, equal_nan=True, atol=1e-6):
            raise ValueError(f'Alignment check failed for {col}')
    clim = {v: load_monthly_climatology(path) for v, path in CLIM_PATHS.items()}
    rows: List[Dict[str, object]] = []
    for i, (r, a) in enumerate(zip(raw.itertuples(index=False), anom.itertuples(index=False))):
        rec: Dict[str, object] = {
            'event_row_id': i,
            'event_type': r.event_type,
            'land_cover_broad': r.land_cover_broad,
            'duration': float(r.duration),
            'start_date': r.start_date,
            'start_year': pd.Timestamp(r.start_date).year,
            'period': '2001-2012' if pd.Timestamp(r.start_date).year <= 2012 else '2013-2024',
            'LOSStot': float(r.LOSStot),
            'ndvi_anom_pre_mean': float(getattr(a, 'ndvi_anom_pre_mean')),
            'gpp_anom_pre_mean': float(getattr(a, 'gpp_anom_pre_mean')),
            'et_anom_pre_mean': float(getattr(a, 'et_anom_pre_mean')),
            'spei_anom_pre_mean': float(getattr(a, 'spei_anom_pre_mean')),
            'sm_anom_pre_mean': float(getattr(a, 'sm_anom_pre_mean')),
            'temp_anom_pre_mean': float(getattr(a, 'temp_anom_pre_mean')),
            'sti_anom_pre_mean': float(getattr(a, 'sti_anom_pre_mean')),
        }
        dates = pd.date_range(pd.Timestamp(r.start_date), pd.Timestamp(r.end_date), freq='MS')
        for v in FUNCTIONAL_VARS:
            raw_vals = parse_series(getattr(r, f'{v}_path'))
            event_anom = compute_event_anomaly_series(raw_vals, dates, int(r.row), int(r.col), clim[v])
            rec[f'{v}_cum_loss'] = cumulative_deficit(float(getattr(a, f'{v}_anom_pre_mean')), event_anom)
        rows.append(rec)
        if (i + 1) % 50000 == 0:
            print(f'processed_events={i+1}')
    return pd.DataFrame(rows)


def main() -> None:
    event_level_cache = OUTPUT_DIR / 'event_cumulative_anomaly_event_level.csv'
    if event_level_cache.exists():
        df = pd.read_csv(event_level_cache)
    else:
        df = build_inputs()
        df.to_csv(event_level_cache, index=False)
    raw_matches = pd.read_csv(RAW_MATCHES_PATH)
    if raw_matches['event_row_id'].max() >= len(df):
        raise ValueError('Raw matches event_row_id exceeds anomaly cumulative table length.')
    global_att = compute_att_with_matches(df, raw_matches, CUM_OUTCOMES, [])

    period_paired = {}
    for period in ['2001-2012', '2013-2024']:
        part = df[(df['period'] == period) & (df['event_type'].isin([TREATED_LABEL, CONTROL_LABEL]))].copy().reset_index(drop=True)
        part['event_row_id'] = np.arange(len(part))
        matches = build_matches_generic(part.drop(columns=['event_row_id']))
        control_cols = [c for c in matches.columns if c.startswith('control_row_id_')]
        treated = part.set_index('event_row_id').loc[matches['event_row_id']].reset_index()
        control_tables = []
        for col in control_cols:
            control_subset = part.set_index('event_row_id').loc[matches[col]].reset_index(drop=True)
            control_tables.append(control_subset[CUM_OUTCOMES])
        control_mean = sum(control_tables) / len(control_tables)
        paired = treated[['start_year'] + CUM_OUTCOMES].copy()
        paired['period'] = period
        for outcome in CUM_OUTCOMES:
            paired[f'matched_control_{outcome}'] = control_mean[outcome].to_numpy()
            paired[f'delta_{outcome}'] = paired[outcome] - paired[f'matched_control_{outcome}']
        period_paired[period] = paired
    period_diff = compare_periods(period_paired, CUM_OUTCOMES)

    if not event_level_cache.exists():
        df.to_csv(event_level_cache, index=False)
    global_att.to_csv(OUTPUT_DIR / 'cumulative_global_att_anomaly.csv', index=False)
    period_diff.to_csv(OUTPUT_DIR / 'cumulative_period_difference_anomaly.csv', index=False)

    report_lines = [
        'ANOMALY-DOMAIN CUMULATIVE ATTRIBUTION SUMMARY',
        '',
        '[global matched ATT]',
        global_att.to_string(index=False),
        '',
        '[late-period strengthening]',
        period_diff.to_string(index=False),
    ]
    (OUTPUT_DIR / 'cumulative_anomaly_report.txt').write_text('\n'.join(report_lines), encoding='utf-8')
    print(OUTPUT_DIR / 'cumulative_global_att_anomaly.csv')
    print(OUTPUT_DIR / 'cumulative_period_difference_anomaly.csv')


if __name__ == '__main__':
    main()
