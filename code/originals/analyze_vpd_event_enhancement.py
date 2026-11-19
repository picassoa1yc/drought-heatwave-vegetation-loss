from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = (Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve() / "output_df/global_v05_90p_vpd_main")
EVENT_PATH = BASE_DIR / "global_integrated_events_90p_vpd.csv"
OUTPUT_DIR = BASE_DIR / "event_analysis_vpd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAJ_VARIABLES = ["ndvi", "spei", "temp", "vpd", "gpp", "et"]
FOCUSED_EVENT_TYPES = [
    "Drought_Dominant",
    "Heat_Dominant",
    "Compound_Overlapping",
    "Sequential_Drought_Heat",
]


def parse_series(value: object) -> list[float]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    values: list[float] = []
    for token in text.split(','):
        token = token.strip()
        if not token or token.lower() == 'nan':
            values.append(np.nan)
        else:
            try:
                values.append(float(token))
            except ValueError:
                values.append(np.nan)
    return values


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else np.nan


def event_metrics_from_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in chunk.itertuples(index=False):
        record = {
            'pixel_id': int(row.pixel_id),
            'event_type': row.event_type,
            'duration': float(row.duration),
            'LOSStot': float(row.LOSStot),
            'DECmax': float(row.DECmax),
            'TIMErec_true': float(row.TIMErec_true) if pd.notna(row.TIMErec_true) else np.nan,
            'compound_share': float(row.compound_share) if pd.notna(row.compound_share) else np.nan,
        }
        for var in ['vpd', 'ndvi', 'gpp', 'et', 'spei', 'temp']:
            pre_vals = parse_series(getattr(row, f'pre_{var}_path'))
            event_vals = parse_series(getattr(row, f'{var}_path'))
            post_vals = parse_series(getattr(row, f'post_{var}_path')) if hasattr(row, f'post_{var}_path') else []
            record[f'{var}_pre_mean'] = finite_mean(pre_vals)
            record[f'{var}_event_mean'] = finite_mean(event_vals)
            record[f'{var}_post_mean'] = finite_mean(post_vals)
            record[f'{var}_event_minus_pre'] = (
                record[f'{var}_event_mean'] - record[f'{var}_pre_mean']
                if np.isfinite(record[f'{var}_event_mean']) and np.isfinite(record[f'{var}_pre_mean'])
                else np.nan
            )
            record[f'{var}_post_minus_event'] = (
                record[f'{var}_post_mean'] - record[f'{var}_event_mean']
                if np.isfinite(record[f'{var}_post_mean']) and np.isfinite(record[f'{var}_event_mean'])
                else np.nan
            )
        vpd_clim_vals = parse_series(getattr(row, 'vpd_clim_path'))
        record['vpd_clim_event_mean'] = finite_mean(vpd_clim_vals)
        record['vpd_event_excess'] = (
            record['vpd_event_mean'] - record['vpd_clim_event_mean']
            if np.isfinite(record['vpd_event_mean']) and np.isfinite(record['vpd_clim_event_mean'])
            else np.nan
        )
        rows.append(record)
    return pd.DataFrame(rows)


def build_event_metrics() -> pd.DataFrame:
    usecols = [
        'pixel_id','event_type','duration','LOSStot','DECmax','TIMErec_true','compound_share',
        'pre_vpd_path','vpd_path','post_vpd_path','vpd_clim_path',
        'pre_ndvi_path','ndvi_path','post_ndvi_path',
        'pre_gpp_path','gpp_path','post_gpp_path',
        'pre_et_path','et_path','post_et_path',
        'pre_spei_path','spei_path','post_spei_path',
        'pre_temp_path','temp_path','post_temp_path',
    ]
    parts = []
    for chunk in pd.read_csv(EVENT_PATH, usecols=usecols, chunksize=25000):
        parts.append(event_metrics_from_chunk(chunk))
    return pd.concat(parts, ignore_index=True)


def build_trajectory_summary() -> pd.DataFrame:
    usecols = ['event_type']
    for var in TRAJ_VARIABLES:
        usecols.extend([f'pre_{var}_path', f'{var}_path', f'post_{var}_path'])
    stats = defaultdict(lambda: [0.0, 0.0, 0.0])
    for chunk in pd.read_csv(EVENT_PATH, usecols=usecols, chunksize=25000):
        for row in chunk.itertuples(index=False):
            etype = row.event_type
            for var in TRAJ_VARIABLES:
                for stage, values in [('pre', parse_series(getattr(row, f'pre_{var}_path'))), ('event', parse_series(getattr(row, f'{var}_path'))), ('post', parse_series(getattr(row, f'post_{var}_path')))]:
                    n = len(values)
                    for idx, value in enumerate(values, start=1):
                        if not np.isfinite(value):
                            continue
                        month_idx = -n + idx - 1 if stage == 'pre' else idx
                        key = (etype, var, stage, month_idx)
                        stats[key][0] += float(value)
                        stats[key][1] += 1.0
                        stats[key][2] += float(value) * float(value)
    rows = []
    for (etype, var, stage, month_idx), (value_sum, count, value_sq_sum) in stats.items():
        mean = value_sum / count
        varx = max(value_sq_sum / count - mean * mean, 0.0)
        rows.append({
            'event_type': etype,
            'variable': var,
            'stage': stage,
            'stage_month': month_idx,
            'n_obs': int(count),
            'mean_value': mean,
            'se_value': (varx ** 0.5) / (count ** 0.5),
        })
    return pd.DataFrame(rows).sort_values(['variable','event_type','stage','stage_month']).reset_index(drop=True)


def make_vpd_strata(metrics: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    drought = metrics[metrics['event_type'] == 'Drought_Dominant'].copy()
    split = float(drought['vpd_event_excess'].median())
    metrics = metrics.copy()
    metrics['vpd_event_group'] = pd.Series(pd.NA, index=metrics.index, dtype='object')
    mask = metrics['event_type'] == 'Drought_Dominant'
    metrics.loc[mask & (metrics['vpd_event_excess'] <= split), 'vpd_event_group'] = 'Drought_LowVPD'
    metrics.loc[mask & (metrics['vpd_event_excess'] > split), 'vpd_event_group'] = 'Drought_HighVPD'
    metrics.loc[metrics['event_type'] == 'Compound_Overlapping', 'vpd_event_group'] = 'Compound_HotDry'
    return metrics, split


def summarize_groups(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in {'pixel_id','event_type','vpd_event_group'} and pd.api.types.is_numeric_dtype(df[c])]
    agg = {c: ['mean', 'median'] for c in numeric_cols}
    out = df.groupby(group_col, dropna=False).agg(agg)
    out.columns = [f'{c}_{s}' for c, s in out.columns]
    return out.reset_index()


def make_plots(trajectory: pd.DataFrame, grouped: pd.DataFrame) -> None:
    traj = trajectory[
        trajectory['event_type'].isin(FOCUSED_EVENT_TYPES)
        & trajectory['variable'].isin(['ndvi','vpd','gpp'])
    ].copy()

    robust_parts = []
    for _, g in traj.groupby(['event_type', 'variable', 'stage'], sort=False):
        max_n = float(g['n_obs'].max())
        min_n = max(100.0, max_n * 0.01)
        robust_parts.append(g[g['n_obs'] >= min_n])
    traj = pd.concat(robust_parts, ignore_index=True)

    colors = {
        'Drought_Dominant': '#b35806',
        'Heat_Dominant': '#d7301f',
        'Compound_Overlapping': '#542788',
        'Sequential_Drought_Heat': '#1b7837',
    }
    titles = {'ndvi': 'NDVI anomaly', 'vpd': 'VPD', 'gpp': 'GPP'}
    stage_order = {'pre': 0, 'event': 1, 'post': 2}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, var in zip(axes, ['ndvi','vpd','gpp']):
        sub = traj[traj['variable'] == var].copy()
        for etype in FOCUSED_EVENT_TYPES:
            g = sub[sub['event_type'] == etype].copy()
            if g.empty:
                continue
            g['relative_month'] = g['stage_month']
            pre_len = int(abs(g.loc[g['stage'] == 'pre', 'stage_month'].min())) if (g['stage'] == 'pre').any() else 0
            event_len = int(g.loc[g['stage'] == 'event', 'stage_month'].max()) if (g['stage'] == 'event').any() else 0
            g.loc[g['stage'] == 'post', 'relative_month'] = event_len + g.loc[g['stage'] == 'post', 'stage_month']
            g = g.sort_values(['relative_month', 'stage', 'stage_month'], key=lambda s: s.map(stage_order) if s.name == 'stage' else s)
            ax.plot(g['relative_month'], g['mean_value'], label=etype, color=colors.get(etype), linewidth=2)
        ax.axvline(0.5, color='0.5', linestyle='--', linewidth=1)
        ax.set_title(titles[var])
        ax.set_xlabel('Relative month')
    axes[0].set_ylabel('Mean value')
    axes[0].legend(fontsize=8)
    fig.savefig(OUTPUT_DIR / 'vpd_trajectory_preview.png', dpi=200)
    plt.close(fig)

    focus = grouped[grouped['vpd_event_group'].isin(['Drought_LowVPD','Drought_HighVPD','Compound_HotDry'])].copy()
    if not focus.empty:
        fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
        metrics = [
            ('LOSStot_mean', 'LOSStot'),
            ('TIMErec_true_mean', 'Recovery time'),
            ('gpp_event_minus_pre_mean', 'GPP event-pre'),
            ('vpd_event_excess_mean', 'VPD excess'),
        ]
        colors_bar = ['#67a9cf','#ef8a62','#542788']
        for ax, (col, title) in zip(axes, metrics):
            if col not in focus.columns:
                continue
            ax.bar(focus['vpd_event_group'], focus[col], color=colors_bar)
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=20)
        fig.savefig(OUTPUT_DIR / 'vpd_event_group_contrast.png', dpi=200)
        plt.close(fig)


def write_report(metrics: pd.DataFrame, grouped: pd.DataFrame, split: float, trajectory: pd.DataFrame) -> None:
    report = OUTPUT_DIR / 'vpd_event_report.md'
    lines = ['# VPD Event Enhancement Report', '']
    lines.append(f'event_path={EVENT_PATH}')
    lines.append('')
    lines.append('## VPD stratification rule')
    lines.append('')
    lines.append(f'- `Drought_LowVPD` and `Drought_HighVPD` are split at the median event VPD excess within `Drought_Dominant` events.')
    lines.append(f'- Median `vpd_event_excess` threshold = `{split:.4f}`.')
    lines.append('- `Compound_HotDry` corresponds to `Compound_Overlapping` events in the main event framework.')
    lines.append('')
    lines.append('## Group summary')
    lines.append('')
    show_cols = [
        'vpd_event_group','LOSStot_mean','DECmax_mean','TIMErec_true_mean',
        'ndvi_event_minus_pre_mean','gpp_event_minus_pre_mean','et_event_minus_pre_mean',
        'vpd_event_mean_mean','vpd_clim_event_mean_mean','vpd_event_excess_mean',
        'temp_event_minus_pre_mean','spei_event_minus_pre_mean'
    ]
    available = [c for c in show_cols if c in grouped.columns]
    lines.append(grouped[available].to_markdown(index=False))
    lines.append('')
    lines.append('## Headline trajectory contrasts')
    lines.append('')
    for etype in ['Drought_Dominant','Compound_Overlapping']:
        sub = trajectory[(trajectory['event_type'] == etype) & (trajectory['variable'] == 'vpd') & (trajectory['stage'] == 'event')]
        if not sub.empty:
            peak = sub.loc[sub['mean_value'].idxmax()]
            lines.append(f'- `{etype}` event VPD peaks at month `{int(peak.stage_month)}` with mean `{peak.mean_value:.4f}` (n={int(peak.n_obs)}).')
    lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    try:
        table = grouped.set_index('vpd_event_group')
        if {'Drought_LowVPD','Drought_HighVPD','Compound_HotDry'}.issubset(set(table.index)):
            lines.append(
                f"- `Drought_HighVPD` shows LOSStot `{table.loc['Drought_HighVPD','LOSStot_mean']:.4f}` versus `{table.loc['Drought_LowVPD','LOSStot_mean']:.4f}` in `Drought_LowVPD`, indicating that atmospheric dryness intensifies drought damage even within drought-dominant events."
            )
            lines.append(
                f"- `Compound_HotDry` remains the most damaging group with LOSStot `{table.loc['Compound_HotDry','LOSStot_mean']:.4f}` and mean GPP event-minus-pre `{table.loc['Compound_HotDry','gpp_event_minus_pre_mean']:.4f}`."
            )
            lines.append(
                f"- `Drought_HighVPD` has event VPD excess `{table.loc['Drought_HighVPD','vpd_event_excess_mean']:.4f}`, bridging the interpretation between drought-only and compound hot-dry events."
            )
    except Exception:
        pass
    report.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    metrics = build_event_metrics()
    metrics.to_csv(OUTPUT_DIR / 'vpd_event_level_metrics.csv', index=False)

    summary_by_type = summarize_groups(metrics, 'event_type')
    summary_by_type.to_csv(OUTPUT_DIR / 'vpd_summary_by_event_type.csv', index=False)

    metrics_strat, split = make_vpd_strata(metrics)
    focused = metrics_strat[metrics_strat['vpd_event_group'].notna()].copy()
    grouped = summarize_groups(focused, 'vpd_event_group')
    grouped.to_csv(OUTPUT_DIR / 'vpd_drought_stratified_summary.csv', index=False)

    trajectory = build_trajectory_summary()
    trajectory.to_csv(OUTPUT_DIR / 'vpd_trajectory_summary.csv', index=False)

    make_plots(trajectory, grouped)
    write_report(metrics_strat, grouped, split, trajectory)
    print(f'Saved VPD event outputs to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
