import os
from pathlib import Path
import pandas as pd
import numpy as np
import rasterio
import statsmodels.formula.api as smf

base = Path(os.environ.get("VEGETATION_PROJECT_ROOT", Path.cwd())).resolve()
flux_unzipped = Path(os.environ.get("FLUXNET_FULLSET_DIR", "fluxnet_fullset"))
stage8 = base / 'codex_stage8_20260317'
reports = stage8 / 'reports'
reports.mkdir(parents=True, exist_ok=True)
shortlist_path = reports / 'fluxnet_fullset_v2_shortlist_stage8.csv'
metadata_path = next(p for p in stage8.glob('*.csv') if 'availability' not in p.name.lower())

# metadata
last_err = None
for enc in ['utf-8-sig', 'gbk', 'gb18030', 'cp936']:
    try:
        meta = pd.read_csv(metadata_path, encoding=enc)
        break
    except Exception as e:
        last_err = e
else:
    raise last_err
meta = meta.rename(columns=lambda c: c.strip()).rename(columns={
    'SITE_ID':'site_id','SITE_NAME':'site_name','LOCATION_LAT':'lat','LOCATION_LONG':'lon',
    'IGBP':'igbp','MAT':'mat_c','MAP':'map_mm','FLUXNET2015':'has_fluxnet2015'
})
meta = meta[['site_id','site_name','lat','lon','igbp','mat_c','map_mm','has_fluxnet2015']].copy()
short = pd.read_csv(shortlist_path)
eligible_ids = set(short['site_id'].astype(str))

biome_map = {
    'CRO': 'Cropland', 'GRA': 'Grassland', 'SAV': 'Savanna',
    'WSA': 'Shrub', 'OSH': 'Shrub', 'CSH': 'Shrub',
    'ENF': 'Forest', 'EBF': 'Forest', 'DBF': 'Forest', 'MF': 'Forest', 'DNF': 'Forest'
}

def thermal_label(mat):
    if pd.isna(mat):
        return np.nan
    if mat > 15:
        return 'Hot'
    if mat > 5:
        return 'Warm'
    return 'Cool'

records = []
for folder in sorted(flux_unzipped.glob('FLX_*_FLUXNET2015_FULLSET_*')):
    mm_files = list(folder.glob('*_FULLSET_MM_*.csv'))
    if not mm_files:
        continue
    mm_path = mm_files[0]
    site_id = mm_path.stem.split('_')[1]
    if site_id not in eligible_ids:
        continue
    df = pd.read_csv(mm_path, usecols=lambda c: c in ['TIMESTAMP','TA_F','VPD_F','P_F','LE_F_MDS','GPP_NT_VUT_REF','GPP_DT_VUT_REF'])
    df['site_id'] = site_id
    df['date'] = pd.to_datetime(df['TIMESTAMP'].astype(str), format='%Y%m', errors='coerce')
    df = df.dropna(subset=['date']).copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df = df[(df['year'] >= 2001) & (df['year'] <= 2014)].copy()
    for col in ['TA_F','VPD_F','P_F','LE_F_MDS','GPP_NT_VUT_REF','GPP_DT_VUT_REF']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] <= -9990, col] = np.nan
    df['GPP_MEAN_VUT_REF'] = df[['GPP_NT_VUT_REF','GPP_DT_VUT_REF']].mean(axis=1)
    records.append(df)

tower = pd.concat(records, ignore_index=True)
tower = tower.merge(meta, on='site_id', how='left')
tower['land_cover_broad'] = tower['igbp'].map(biome_map).fillna(tower['igbp'])
tower['thermal_background'] = tower['mat_c'].map(thermal_label)
for var in ['TA_F','VPD_F','P_F','LE_F_MDS','GPP_MEAN_VUT_REF']:
    clim = tower.groupby(['site_id','month'])[var].transform('mean')
    tower[f'{var}_clim'] = clim
    tower[f'{var}_anom'] = tower[var] - clim

# extract grid vars
paths = {
    'grid_drought': base / 'used_data' / 'drought_month_mask_2001-2024.tif',
    'grid_heat': base / 'used_data' / 'heatwave_mask_90p.tif',
    'grid_spei': base / 'used_data' / 'spei03_2001-2024.tif',
    'grid_tmp': base / 'used_data' / 'tmp_2001-2024.tif',
    'grid_vpd': base / 'used_data' / 'era5_vpd_2001-2024.tif',
}
site_unique = tower[['site_id','lat','lon']].drop_duplicates().reset_index(drop=True)
with rasterio.open(paths['grid_drought']) as ref:
    site_unique[['row','col']] = site_unique.apply(lambda r: pd.Series(ref.index(r['lon'], r['lat'])), axis=1)
    height, width = ref.height, ref.width
site_unique = site_unique[(site_unique['row'] >= 0) & (site_unique['row'] < height) & (site_unique['col'] >= 0) & (site_unique['col'] < width)].copy()
month_index = pd.date_range('2001-01-01','2024-12-01',freq='MS')
parts = []
for key, path in paths.items():
    with rasterio.open(path) as src:
        arr = src.read()[:, site_unique['row'].to_numpy(), site_unique['col'].to_numpy()]
    df = pd.DataFrame(arr, columns=site_unique['site_id'].tolist())
    df['date'] = month_index
    long = df.melt(id_vars='date', var_name='site_id', value_name=key)
    parts.append(long)
grid = parts[0]
for p in parts[1:]:
    grid = grid.merge(p, on=['date','site_id'], how='left')
for col in ['grid_spei','grid_tmp','grid_vpd']:
    grid[col] = pd.to_numeric(grid[col], errors='coerce')
    grid.loc[grid[col] <= -9990, col] = np.nan
grid['grid_drought'] = pd.to_numeric(grid['grid_drought'], errors='coerce').fillna(0).astype(int)
grid['grid_heat'] = pd.to_numeric(grid['grid_heat'], errors='coerce').fillna(0).astype(int)

a = tower.merge(grid, on=['site_id','date'], how='left')
# keep overlapping period only where gridded data exist
# site-level VPD threshold
q90 = a.groupby('site_id')['VPD_F_anom'].quantile(0.9).rename('vpd_anom_q90').reset_index()
a = a.merge(q90, on='site_id', how='left')
a['site_high_vpd'] = ((a['VPD_F_anom'] >= a['vpd_anom_q90']) & a['VPD_F_anom'].notna()).astype(int)
a['validation_state_v2'] = np.select(
    [
        (a['grid_drought'] == 1) & (a['site_high_vpd'] == 1),
        (a['grid_drought'] == 1) & (a['site_high_vpd'] == 0),
        (a['grid_drought'] == 0) & (a['site_high_vpd'] == 1),
    ],
    ['DroughtPlusHighVPD', 'DroughtOnly_LowVPD', 'HighVPD_Only'],
    default='Reference'
)

# outputs
panel_cols = [
    'site_id','site_name','date','year','month','lat','lon','igbp','land_cover_broad','thermal_background',
    'TA_F','TA_F_anom','VPD_F','VPD_F_anom','P_F','P_F_anom','LE_F_MDS','LE_F_MDS_anom','GPP_MEAN_VUT_REF','GPP_MEAN_VUT_REF_anom',
    'grid_drought','grid_heat','grid_spei','grid_tmp','grid_vpd','site_high_vpd','vpd_anom_q90','validation_state_v2'
]
a[panel_cols].sort_values(['site_id','date']).to_csv(reports / 'fluxnet_validation_panel_v2_stage8.csv', index=False)

site_summary = a.groupby('site_id').agg(
    site_name=('site_name','first'), land_cover_broad=('land_cover_broad','first'), thermal_background=('thermal_background','first'),
    n_months=('date','count'), start_date=('date','min'), end_date=('date','max'),
    mean_ta=('TA_F','mean'), mean_vpd=('VPD_F','mean')
).reset_index()
site_summary['start_date'] = site_summary['start_date'].dt.strftime('%Y-%m')
site_summary['end_date'] = site_summary['end_date'].dt.strftime('%Y-%m')
site_summary.to_csv(reports / 'fluxnet_validation_site_summary_v2_stage8.csv', index=False)

state_summary = a.groupby('validation_state_v2').agg(
    n_months=('date','count'), n_sites=('site_id','nunique'),
    mean_gpp_anom=('GPP_MEAN_VUT_REF_anom','mean'), median_gpp_anom=('GPP_MEAN_VUT_REF_anom','median'),
    mean_le_anom=('LE_F_MDS_anom','mean'), median_le_anom=('LE_F_MDS_anom','median'),
    mean_vpd_anom=('VPD_F_anom','mean'), mean_ta_anom=('TA_F_anom','mean'),
    mean_grid_spei=('grid_spei','mean'), mean_grid_vpd=('grid_vpd','mean')
).reset_index()
state_summary.to_csv(reports / 'fluxnet_validation_state_summary_v2_stage8.csv', index=False)

biome_summary = a.groupby(['land_cover_broad','validation_state_v2']).agg(
    n_months=('date','count'), n_sites=('site_id','nunique'),
    mean_gpp_anom=('GPP_MEAN_VUT_REF_anom','mean'), mean_le_anom=('LE_F_MDS_anom','mean'), mean_vpd_anom=('VPD_F_anom','mean')
).reset_index()
biome_summary.to_csv(reports / 'fluxnet_validation_biome_summary_v2_stage8.csv', index=False)

thermal_summary = a.groupby(['thermal_background','validation_state_v2']).agg(
    n_months=('date','count'), n_sites=('site_id','nunique'),
    mean_gpp_anom=('GPP_MEAN_VUT_REF_anom','mean'), mean_le_anom=('LE_F_MDS_anom','mean'), mean_vpd_anom=('VPD_F_anom','mean')
).reset_index()
thermal_summary.to_csv(reports / 'fluxnet_validation_thermal_summary_v2_stage8.csv', index=False)

# regressions
rows = []
texts = []
for response in ['GPP_MEAN_VUT_REF_anom','LE_F_MDS_anom']:
    dfm = a[['site_id','month','grid_drought','site_high_vpd',response,'land_cover_broad','thermal_background']].dropna().copy()
    fit = smf.ols(f'{response} ~ grid_drought + site_high_vpd + grid_drought:site_high_vpd + C(site_id) + C(month)', data=dfm).fit(cov_type='HC3')
    for term in ['grid_drought','site_high_vpd','grid_drought:site_high_vpd']:
        rows.append({'model':'all_sites','response':response,'term':term,'coef':fit.params.get(term,np.nan),'p_value':fit.pvalues.get(term,np.nan),'n_obs':int(fit.nobs),'r_squared':fit.rsquared})
    texts.append(f'## all_sites :: {response}')
    texts.append(fit.summary().as_text())
    texts.append('')

    hot = dfm[dfm['thermal_background'] == 'Hot'].copy()
    if len(hot) >= 100:
        fit_hot = smf.ols(f'{response} ~ grid_drought + site_high_vpd + grid_drought:site_high_vpd + C(site_id) + C(month)', data=hot).fit(cov_type='HC3')
        for term in ['grid_drought','site_high_vpd','grid_drought:site_high_vpd']:
            rows.append({'model':'hot_sites','response':response,'term':term,'coef':fit_hot.params.get(term,np.nan),'p_value':fit_hot.pvalues.get(term,np.nan),'n_obs':int(fit_hot.nobs),'r_squared':fit_hot.rsquared})
        texts.append(f'## hot_sites :: {response}')
        texts.append(fit_hot.summary().as_text())
        texts.append('')

pd.DataFrame(rows).to_csv(reports / 'fluxnet_validation_models_v2_stage8.csv', index=False)
(reports / 'fluxnet_validation_models_v2_stage8.txt').write_text('\n'.join(texts), encoding='utf-8')

# concise report
lookup = {r['validation_state_v2']: r for _, r in state_summary.iterrows()}
lines = ['# FLUXNET Validation V2 Stage 8','']
lines.append(f'eligible_sites={site_summary.shape[0]}')
lines.append('')
lines.append('## Overall state summary')
lines.append('')
lines.append(state_summary.to_markdown(index=False))
lines.append('')
lines.append('## Headline interpretation')
lines.append('')
if 'DroughtPlusHighVPD' in lookup and 'DroughtOnly_LowVPD' in lookup:
    c = lookup['DroughtPlusHighVPD']
    d = lookup['DroughtOnly_LowVPD']
    lines.append(f"- Across all eligible sites, drought months with high site-level VPD show mean GPP anomaly `{c['mean_gpp_anom']:.4f}` versus `{d['mean_gpp_anom']:.4f}` in drought months without high VPD.")
    lines.append(f"- The corresponding LE anomaly is `{c['mean_le_anom']:.4f}` versus `{d['mean_le_anom']:.4f}`.")
    lines.append(f"- Mean site VPD anomaly rises from `{d['mean_vpd_anom']:.4f}` to `{c['mean_vpd_anom']:.4f}`.")
lines.append('- This second-pass validation is designed as independent support for the global event and VPD results, not as a replacement for the main attribution framework.')
(reports / 'fluxnet_validation_report_v2_stage8.md').write_text('\n'.join(lines), encoding='utf-8')

print('sites=', site_summary.shape[0])
print(state_summary.to_string(index=False))
print(pd.DataFrame(rows).to_string(index=False))
