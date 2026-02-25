#!/usr/bin/env python3
"""
update_model.py
Downloads HRRR and NAM-3km precipitation type fields from NOAA NOMADS,
renders transparent PNG overlays, and writes model_bounds.js / model_frames.js
for the weather radar viewer.

Precipitation type colour scheme
  0 → transparent (no precip)
  1 → rain       #1E90FF (blue)
  2 → snow       #DDEEFF (light blue/white)
  3 → frz rain   #FF69B4 (pink)
  4 → ice pell.  #C0C0C0 (silver)
"""

import datetime
import json
import os
import shutil
import tempfile
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import requests

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
OUTPUT_DIR    = 'docs/assets'
MANIFEST_FILE = os.path.join(OUTPUT_DIR, 'model_manifest.json')
MAX_HISTORY   = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Precipitation type colourmap  (0=no-precip transparent, 1=rain, 2=snow, 3=frzr, 4=ice)
_COLORS  = [(0, 0, 0, 0),          # 0 transparent
            (0.118, 0.565, 1.0, 0.85),   # 1 rain  #1E90FF
            (0.867, 0.933, 1.0, 0.85),   # 2 snow  #DDEEFF
            (1.0,   0.412, 0.706, 0.85), # 3 frzr  #FF69B4
            (0.753, 0.753, 0.753, 0.85)] # 4 ice   #C0C0C0
PRECIP_CMAP  = mcolors.ListedColormap(_COLORS)
PRECIP_NORM  = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], PRECIP_CMAP.N)

# NOMADS GRIB filter base URLs
NOMADS_HRRR_URL = (
    'https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl'
    '?file=hrrr.t{HH}z.wrfsfcf01.grib2'
    '&lev_surface=on'
    '&var_CRAIN=on&var_CSNOW=on&var_CFRZR=on&var_CICE=on'
    '&dir=%2Fhrrr.{YYYYMMDD}%2Fconus'
)
NOMADS_NAM_URL = (
    'https://nomads.ncep.noaa.gov/cgi-bin/filter_nam.pl'
    '?file=nam.t{HH}z.conusnest.hiresf01.tm00.grib2'
    '&lev_surface=on'
    '&var_CRAIN=on&var_CSNOW=on&var_CFRZR=on&var_CICE=on'
    '&dir=%2Fnam.{YYYYMMDD}'
)

# ── Manifest helpers ───────────────────────────────────────────────────────────

def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)


# ── NOMADS download helpers ────────────────────────────────────────────────────

def _download_nomads(url_template, date_obj, hour_str):
    """Try to download a filtered GRIB2 from NOMADS. Returns bytes or None."""
    url = url_template.format(HH=hour_str, YYYYMMDD=date_obj.strftime('%Y%m%d'))
    try:
        resp = requests.get(url, stream=True, timeout=90)
        if resp.status_code == 200 and len(resp.content) > 1000:
            return resp.content, url
    except Exception as e:
        print(f'  NOMADS request failed: {e}')
    return None, None


def fetch_hrrr():
    """Return (grib_bytes, run_datetime, cache_key) for latest available HRRR run."""
    now = datetime.datetime.now(datetime.timezone.utc)
    for offset in range(6):
        dt = now - datetime.timedelta(hours=offset)
        data, url = _download_nomads(NOMADS_HRRR_URL, dt, dt.strftime('%H'))
        if data:
            key = f"hrrr.{dt.strftime('%Y%m%d')}.t{dt.strftime('%H')}z.f01"
            return data, dt, key
    return None, None, None


def fetch_nam3k():
    """Return (grib_bytes, run_datetime, cache_key) for latest available NAM 3km run."""
    now = datetime.datetime.now(datetime.timezone.utc)
    # NAM runs at 00, 06, 12, 18 UTC
    for offset in range(10):
        dt = now - datetime.timedelta(hours=offset)
        run_hour = (dt.hour // 6) * 6
        run_dt   = dt.replace(hour=run_hour, minute=0, second=0, microsecond=0)
        data, url = _download_nomads(NOMADS_NAM_URL, run_dt, f'{run_hour:02d}')
        if data:
            key = f"nam3k.{run_dt.strftime('%Y%m%d')}.t{run_hour:02d}z.f01"
            return data, run_dt, key
        # Only check each unique run hour once
        if offset > 0 and run_hour == (now - datetime.timedelta(hours=offset - 1)).replace().hour // 6 * 6:
            continue
    return None, None, None


# ── GRIB2 parsing ──────────────────────────────────────────────────────────────

def extract_precip_type(grib_path):
    """
    Read CRAIN/CSNOW/CFRZR/CICE from a GRIB2 file and combine into a
    single precip-type array.  Returns (precip_type_2d, lats_2d, lons_2d)
    or raises on failure.
    """
    try:
        import cfgrib
        import xarray as xr
    except ImportError:
        raise ImportError('cfgrib and xarray are required: conda install -c conda-forge cfgrib xarray')

    field_map = {'CRAIN': 1, 'CSNOW': 2, 'CFRZR': 3, 'CICE': 4}
    datasets  = {}

    for short_name, code in field_map.items():
        try:
            ds = cfgrib.open_dataset(
                grib_path,
                filter_by_keys={'shortName': short_name, 'typeOfLevel': 'surface'},
                indexpath=None,
            )
            var  = list(ds.data_vars)[0]
            data = ds[var].values.astype(float)
            lats = ds['latitude'].values
            lons = ds['longitude'].values
            # Normalise longitudes to [-180, 180]
            lons = np.where(lons > 180, lons - 360, lons)
            datasets[code] = (data, lats, lons)
        except Exception as e:
            print(f'  Warning: {short_name} not read — {e}')

    if not datasets:
        raise ValueError('No precipitation type fields found in GRIB2 file')

    # Combine fields — higher-priority types overwrite lower ones
    ref_data, lats, lons = next(iter(datasets.values()))
    precip_type = np.zeros_like(ref_data, dtype=float)

    # Priority order: rain > snow > freezing rain > ice (applied last = highest)
    for code in [4, 3, 1, 2]:
        if code in datasets:
            d, _, _ = datasets[code]
            precip_type[d >= 0.5] = float(code)

    return precip_type, lats, lons


# ── PNG generation ─────────────────────────────────────────────────────────────

def generate_model_png(precip_type, lats, lons, output_path):
    """
    Render precip_type array to a transparent PNG overlay and return
    (lat_min, lon_min, lat_max, lon_max).
    """
    masked = np.ma.masked_where(precip_type < 0.5, precip_type)

    lon_min = max(float(np.nanmin(lons)), -135)
    lon_max = min(float(np.nanmax(lons)),  -60)
    lat_min = max(float(np.nanmin(lats)),   20)
    lat_max = min(float(np.nanmax(lats)),   55)

    fig = plt.figure(figsize=(16, 10), dpi=200)
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)

    if lons.ndim == 2:
        ax.pcolormesh(lons, lats, masked,
                      cmap=PRECIP_CMAP, norm=PRECIP_NORM,
                      transform=projection, rasterized=True)
    else:
        ax.pcolormesh(lons, lats, masked,
                      cmap=PRECIP_CMAP, norm=PRECIP_NORM,
                      transform=projection, rasterized=True)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    actual_lon_min, actual_lon_max = ax.get_xlim()
    actual_lat_min, actual_lat_max = ax.get_ylim()

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0,
                transparent=True, dpi=200)
    plt.close()

    return (actual_lat_min, actual_lon_min, actual_lat_max, actual_lon_max)


# ── Frame rotation ─────────────────────────────────────────────────────────────

def rotate_frames(prod_key):
    base    = f'{OUTPUT_DIR}/{prod_key}'
    oldest  = f'{base}_{MAX_HISTORY - 1}.png'
    if os.path.exists(oldest):
        os.remove(oldest)
    for i in range(MAX_HISTORY - 2, -1, -1):
        src = f'{base}_{i}.png' if i > 0 else f'{base}.png'
        dst = f'{base}_{i + 1}.png'
        if os.path.exists(src):
            shutil.copy2(src, dst)


# ── Main ───────────────────────────────────────────────────────────────────────

MODELS = [
    ('HRRR',  fetch_hrrr),
    ('NAM3k', fetch_nam3k),
]

manifest    = load_manifest()
bounds_data = {}
frames_data = {}
any_changes = False

for model_name, fetch_fn in MODELS:
    print(f'\n=== {model_name} ===')
    try:
        grib_bytes, run_dt, cache_key = fetch_fn()
        if grib_bytes is None:
            print(f'  No data available, skipping.')
            prod_key = f'{model_name}_precip_type'
            if prod_key in manifest:
                bounds_data[prod_key] = manifest[prod_key].get('bounds')
                frames_data[prod_key] = manifest[prod_key].get('frames')
            continue

        mkey = f'{model_name}_last_key'
        if manifest.get(mkey) == cache_key:
            print(f'  Same run as last time ({cache_key}), skipping.')
            prod_key = f'{model_name}_precip_type'
            if prod_key in manifest:
                bounds_data[prod_key] = manifest[prod_key].get('bounds')
                frames_data[prod_key] = manifest[prod_key].get('frames')
            continue

        print(f'  New run: {cache_key}')
        any_changes = True
        scan_time   = run_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Write GRIB2 to temp file
        tmp = tempfile.NamedTemporaryFile(suffix='.grib2', delete=False)
        tmp.write(grib_bytes)
        tmp.close()
        grib_path = tmp.name

        try:
            precip_type, lats, lons = extract_precip_type(grib_path)
        finally:
            os.unlink(grib_path)

        prod_key = f'{model_name}_precip_type'
        rotate_frames(prod_key)

        png_path = f'{OUTPUT_DIR}/{prod_key}.png'
        bounds   = generate_model_png(precip_type, lats, lons, png_path)
        print(f'  Saved {png_path}')

        bounds_data[prod_key] = list(bounds)

        frame_list = [{'file': f'{prod_key}.png', 'time': scan_time}]
        old_frames = manifest.get(prod_key, {}).get('frames', [])
        for i, old_frame in enumerate(old_frames):
            hist_idx  = i + 1
            if hist_idx >= MAX_HISTORY:
                break
            hist_file = f'{prod_key}_{hist_idx}.png'
            if os.path.exists(os.path.join(OUTPUT_DIR, hist_file)):
                frame_list.append({'file': hist_file, 'time': old_frame.get('time', '')})

        frames_data[prod_key] = frame_list
        manifest[prod_key]    = {'bounds': list(bounds), 'frames': frame_list}
        manifest[mkey]        = cache_key

    except Exception as e:
        import traceback
        print(f'  {model_name}: Error — {e}')
        traceback.print_exc()
        prod_key = f'{model_name}_precip_type'
        if prod_key in manifest:
            bounds_data[prod_key] = manifest[prod_key].get('bounds')
            frames_data[prod_key] = manifest[prod_key].get('frames')

# ── Write output files ─────────────────────────────────────────────────────────

if any_changes or not os.path.exists(os.path.join(OUTPUT_DIR, 'model_bounds.js')):
    bounds_js = 'window.MODEL_BOUNDS = ' + json.dumps(bounds_data, indent=2) + ';\n'
    with open(os.path.join(OUTPUT_DIR, 'model_bounds.js'), 'w') as f:
        f.write(bounds_js)

    frames_js = 'window.MODEL_FRAMES = ' + json.dumps(frames_data, indent=2) + ';\n'
    with open(os.path.join(OUTPUT_DIR, 'model_frames.js'), 'w') as f:
        f.write(frames_js)

    save_manifest(manifest)
    print('\nModel data updated.')
else:
    print('\nNo changes — skipping output.')
