# update_radar.py
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import datetime
import gzip
import json
import os
import re
import shutil
import urllib.request
import pyart
import pygrib
import matplotlib
matplotlib.use('Agg')   # headless backend — must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Configurable radars
RADARS = ['KCCX', 'KDIX', 'KDOX', 'KBGM', 'KOKX', 'KPBZ']

# NEXRAD Products: output_name -> PyART field name
PRODUCTS = {
    'reflectivity':            'reflectivity',
    'velocity':                'velocity',
    'correlation_coefficient': 'cross_correlation_ratio',
}

# MRMS Products: output_name -> { mrms_product_dir, cmap, vmin, vmax, label }
MRMS_PRODUCTS = {
    'rotation_instant': {
        'mrms_dir': 'AzShear_0-2kmAGL',
        'cmap': 'rotation_cmap',
        'vmin': -0.01,
        'vmax': 0.01,
        'label': 'Instant Rotation AGL',
        'units': 's⁻¹',
    },
    'rotation_1hr': {
        'mrms_dir': 'RotationTrack60min_0-2kmAGL',
        'cmap': 'rotation_track_cmap',
        'vmin': 0,
        'vmax': 0.01,
        'label': '1hr Rotation AGL',
        'units': 's⁻¹',
    },
    'rotation_24hr': {
        'mrms_dir': 'RotationTrack1440min_0-2kmAGL',
        'cmap': 'rotation_track_cmap',
        'vmin': 0,
        'vmax': 0.01,
        'label': '24hr Rotation AGL',
        'units': 's⁻¹',
    },
    'mesh': {
        'mrms_dir': 'MESH',
        'cmap': 'mesh_cmap',
        'vmin': 0,
        'vmax': 100,
        'label': 'Max Est. Hail Size',
        'units': 'mm',
    },
    'posh': {
        'mrms_dir': 'POSH',
        'cmap': 'posh_cmap',
        'vmin': 0,
        'vmax': 100,
        'label': 'Prob. Severe Hail',
        'units': '%',
    },
    'qpe_24hr': {
        'mrms_dir': 'MultiSensor_QPE_24H_Pass2',
        'cmap': 'qpe_cmap',
        'vmin': 0,
        'vmax': 150,
        'label': '24hr QPE',
        'units': 'mm',
    },
}

# MRMS base URL
MRMS_BASE_URL = 'https://mrms.ncep.noaa.gov/data/2D'

# Regional crop for MRMS (covers northeastern US where our radars are)
MRMS_LAT_MIN, MRMS_LAT_MAX = 35.0, 45.0
MRMS_LON_MIN, MRMS_LON_MAX = -83.0, -69.0

# Number of historical frames to keep (plus the current one)
MAX_HISTORY = 10

# Output directory
OUTPUT_DIR = 'docs/assets'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Manifest file tracks last-processed S3 keys and frame history
MANIFEST_FILE = os.path.join(OUTPUT_DIR, 'manifest.json')

# S3 client (anonymous access)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# ── Custom colormaps for MRMS products ────────────────────────────────────────

def _register_mrms_cmaps():
    """Register custom colormaps for MRMS products."""
    # Rotation (diverging): blue (cyclonic-) → transparent → red (cyclonic+)
    rotation_colors = [
        (0.0, (0.0, 0.2, 0.8, 0.9)),
        (0.3, (0.3, 0.5, 1.0, 0.6)),
        (0.45, (0.7, 0.7, 0.7, 0.05)),
        (0.55, (0.7, 0.7, 0.7, 0.05)),
        (0.7, (1.0, 0.5, 0.3, 0.6)),
        (1.0, (0.8, 0.0, 0.0, 0.9)),
    ]
    rotation_cmap = mcolors.LinearSegmentedColormap.from_list('rotation_cmap',
        [(pos, rgba[:3]) for pos, rgba in rotation_colors], N=256)
    plt.colormaps.register(cmap=rotation_cmap, name='rotation_cmap')

    # Rotation track (sequential): transparent → yellow → orange → red → magenta
    rt_colors = ['#00000000', '#FFFF00', '#FF8C00', '#FF0000', '#FF00FF']
    rt_cmap = mcolors.LinearSegmentedColormap.from_list('rotation_track_cmap',
        ['#333333', '#FFFF00', '#FF8C00', '#FF0000', '#FF00FF'], N=256)
    plt.colormaps.register(cmap=rt_cmap, name='rotation_track_cmap')

    # MESH (sequential): green → yellow → orange → red → purple
    mesh_cmap = mcolors.LinearSegmentedColormap.from_list('mesh_cmap',
        ['#333333', '#00CC00', '#FFFF00', '#FF8C00', '#FF0000', '#CC00CC'], N=256)
    plt.colormaps.register(cmap=mesh_cmap, name='mesh_cmap')

    # POSH (sequential): blue → cyan → green → yellow → red
    posh_cmap = mcolors.LinearSegmentedColormap.from_list('posh_cmap',
        ['#333333', '#0044FF', '#00CCCC', '#00CC00', '#FFFF00', '#FF0000'], N=256)
    plt.colormaps.register(cmap=posh_cmap, name='posh_cmap')

    # QPE (sequential): light blue → blue → green → yellow → orange → red → magenta
    qpe_cmap = mcolors.LinearSegmentedColormap.from_list('qpe_cmap',
        ['#333333', '#88BBFF', '#0044FF', '#00CC00', '#FFFF00', '#FF8C00', '#FF0000', '#CC00CC'], N=256)
    plt.colormaps.register(cmap=qpe_cmap, name='qpe_cmap')

_register_mrms_cmaps()

def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_manifest(manifest):
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)

def get_latest_file(radar):
    now = datetime.datetime.now(datetime.timezone.utc)
    prefix = f"{now.year}/{now.strftime('%m')}/{now.strftime('%d')}/{radar}/"

    # Paginate through all S3 objects for the day
    all_files = []
    continuation_token = None
    while True:
        kwargs = {'Bucket': 'unidata-nexrad-level2', 'Prefix': prefix}
        if continuation_token:
            kwargs['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**kwargs)
        if 'Contents' not in response:
            break
        all_files.extend(response['Contents'])
        if response.get('IsTruncated'):
            continuation_token = response['NextContinuationToken']
        else:
            break

    if not all_files:
        print(f"No objects found for prefix {prefix}")
        return None

    files = [obj for obj in all_files if obj['Key'].endswith(('_V06', '_V07', '_V08'))]
    if not files:
        print(f"No valid Level-2 files in {prefix}")
        return None

    latest = max(files, key=lambda x: x['LastModified'])
    return latest['Key']

def download_file(key):
    local_path = '/tmp/' + os.path.basename(key)
    s3.download_file('unidata-nexrad-level2', key, local_path)
    return local_path

def find_best_sweep(radar_data, field):
    """Return the lowest-elevation sweep with 0.25 km gate spacing (super-res),
    falling back to any sweep that has non-masked data."""
    gate_spacing = radar_data.range.get('meters_between_gates', 1000)
    is_super_res = abs(gate_spacing - 250) < 50  # 250 m = 0.25 km

    best_standard = None
    for i in range(radar_data.nsweeps):
        start = int(radar_data.sweep_start_ray_index['data'][i])
        end   = int(radar_data.sweep_end_ray_index['data'][i]) + 1
        data  = radar_data.fields[field]['data'][start:end]
        mask  = np.ma.getmaskarray(data)
        if not np.all(mask):
            if is_super_res:
                return i  # super-res data — first good sweep is best
            if best_standard is None:
                best_standard = i
    return best_standard if best_standard is not None else 0

def generate_png(radar_data, pyart_field, output_name, radar_code, output_path):
    display = pyart.graph.RadarMapDisplay(radar_data)

    fig = plt.figure(figsize=(16, 16), dpi=150)
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)

    # Colormap / range keyed on output_name
    if output_name == 'reflectivity':
        cmap = 'NWSRef'
        vmin, vmax = -20, 75
    elif output_name == 'velocity':
        cmap = 'NWSVel'
        vmin, vmax = -50, 50
    else:  # correlation_coefficient
        cmap = 'RefDiff'
        vmin, vmax = 0.5, 1.05

    sweep_idx = find_best_sweep(radar_data, pyart_field)

    # Apply gate filter to remove noise and produce cleaner imagery
    gatefilter = pyart.filters.GateFilter(radar_data)
    gatefilter.exclude_transition()
    gatefilter.exclude_masked(pyart_field)

    display.plot_ppi_map(
        pyart_field, sweep_idx,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        ax=ax,
        title='',
        colorbar_flag=False,
        lat_lines=[],
        lon_lines=[],
        gatefilter=gatefilter,
        raster=True,
        embellish=False,
    )

    # Remove all axes decorations so the PNG is pure radar data on transparency
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Capture geographic extent before saving
    lon_min, lon_max = ax.get_xlim()
    lat_min, lat_max = ax.get_ylim()

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0,
                transparent=True, dpi=150)
    plt.close()

    return (lat_min, lon_min, lat_max, lon_max)

def rotate_frames(radar_code, output_name):
    """Shift existing frames: delete frame 4, move 3→4, 2→3, 1→2, 0→1."""
    base = f"{OUTPUT_DIR}/{radar_code}_{output_name}"

    # Delete the oldest frame if it exists
    oldest = f"{base}_{MAX_HISTORY - 1}.png"
    if os.path.exists(oldest):
        os.remove(oldest)

    # Shift frames down: N-2 → N-1, ... , 1 → 2, 0 → 1
    for i in range(MAX_HISTORY - 2, -1, -1):
        src = f"{base}_{i}.png" if i > 0 else f"{base}.png"
        dst = f"{base}_{i + 1}.png"
        if os.path.exists(src):
            shutil.copy2(src, dst)

def extract_scan_time(s3_key):
    """Extract the scan timestamp from the S3 key filename.
    Filename format: KCCX20260224_123456_V06
    """
    basename = os.path.basename(s3_key)
    # Remove the radar prefix and version suffix to get the timestamp part
    # e.g. KCCX20260224_123456_V06 → 20260224_123456
    try:
        parts = basename.split('_')
        if len(parts) >= 2:
            # First part: KCCX20260224 → date is chars 4:
            date_str = parts[0][4:]  # '20260224'
            time_str = parts[1]       # '123456'
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}Z"
    except Exception:
        pass
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def cleanup_old_frames():
    """Delete any frame PNGs with index >= MAX_HISTORY (older than the 10th newest)."""
    for fname in os.listdir(OUTPUT_DIR):
        if not fname.endswith('.png'):
            continue
        # Match patterns like KCCX_reflectivity_12.png or MRMS_mesh_15.png
        m = re.match(r'^(.+)_(\d+)\.png$', fname)
        if m:
            idx = int(m.group(2))
            if idx >= MAX_HISTORY:
                path = os.path.join(OUTPUT_DIR, fname)
                print(f"  Cleanup: removing old frame {fname} (index {idx} >= {MAX_HISTORY})")
                os.remove(path)

# ── MRMS functions ────────────────────────────────────────────────────────────

def get_latest_mrms_file(mrms_dir):
    """Fetch the latest GRIB2 file URL from an MRMS product directory."""
    url = f"{MRMS_BASE_URL}/{mrms_dir}/"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode('utf-8')
    except Exception as e:
        print(f"  MRMS: Could not list {mrms_dir}: {e}")
        return None

    # Parse Apache directory listing for .grib2.gz files
    pattern = r'href="(MRMS_[^"]+\.grib2\.gz)"'
    files = re.findall(pattern, html)
    if not files:
        print(f"  MRMS: No grib2.gz files found in {mrms_dir}")
        return None

    # The latest file is typically the last one (sorted by name which includes timestamp)
    latest = sorted(files)[-1]
    return f"{MRMS_BASE_URL}/{mrms_dir}/{latest}"

def extract_mrms_time(filename):
    """Extract timestamp from MRMS filename.
    Format: MRMS_ProductName_00.00_YYYYMMDD-HHMMSS.grib2.gz
    """
    try:
        m = re.search(r'(\d{8})-(\d{6})', filename)
        if m:
            d, t = m.group(1), m.group(2)
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}Z"
    except Exception:
        pass
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def download_mrms_grib2(file_url):
    """Download and decompress a MRMS GRIB2 file."""
    local_gz = '/tmp/mrms_temp.grib2.gz'
    local_grib = '/tmp/mrms_temp.grib2'
    try:
        req = urllib.request.Request(file_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(local_gz, 'wb') as f:
                f.write(resp.read())
        with gzip.open(local_gz, 'rb') as gz:
            with open(local_grib, 'wb') as f:
                f.write(gz.read())
        return local_grib
    except Exception as e:
        print(f"  MRMS: Download failed for {file_url}: {e}")
        # Clean up partial files
        for p in [local_gz, local_grib]:
            if os.path.exists(p):
                os.remove(p)
        return None
    finally:
        if os.path.exists(local_gz):
            os.remove(local_gz)

def generate_mrms_png(grib_path, output_name, mrms_cfg, output_path):
    """Read MRMS GRIB2 data and render a regional PNG."""
    grbs = pygrib.open(grib_path)
    grb = grbs[1]  # First (and usually only) message

    data, lats, lons = grb.data(
        lat1=MRMS_LAT_MIN, lat2=MRMS_LAT_MAX,
        lon1=MRMS_LON_MIN, lon2=MRMS_LON_MAX,
    )
    grbs.close()

    # Mask invalid/missing values (MRMS uses large negative values for missing)
    data = np.ma.masked_where((data < -900) | (data > 1e10), data)

    # For rotation products, also mask very small values for cleaner display
    if 'rotation' in output_name:
        data = np.ma.masked_where(np.abs(data) < 0.0005, data)
    else:
        # For MESH/POSH/QPE, mask zero or near-zero values
        data = np.ma.masked_where(data <= 0.01, data)

    fig = plt.figure(figsize=(12, 10), dpi=150)
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    ax.set_extent([MRMS_LON_MIN, MRMS_LON_MAX, MRMS_LAT_MIN, MRMS_LAT_MAX],
                  crs=projection)

    mesh = ax.pcolormesh(
        lons, lats, data,
        cmap=mrms_cfg['cmap'],
        vmin=mrms_cfg['vmin'],
        vmax=mrms_cfg['vmax'],
        transform=ccrs.PlateCarree(),
        shading='nearest',
        rasterized=True,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0,
                transparent=True, dpi=150)
    plt.close()

    return (MRMS_LAT_MIN, MRMS_LON_MIN, MRMS_LAT_MAX, MRMS_LON_MAX)

# ── Main ───────────────────────────────────────────────────────────────────────
manifest = load_manifest()
bounds_data = {}
frames_data = {}
any_changes = False

for radar in RADARS:
    try:
        key = get_latest_file(radar)
        if not key:
            print(f"No data for {radar}, skipping.")
            # Preserve existing manifest data for this radar
            for output_name in PRODUCTS:
                prod_key = f"{radar}_{output_name}"
                if prod_key in manifest:
                    entry = manifest[prod_key]
                    if 'bounds' in entry:
                        bounds_data[prod_key] = entry['bounds']
                    if 'frames' in entry:
                        frames_data[prod_key] = entry['frames']
            continue

        # Check if this is a new scan vs. what we last processed
        radar_manifest_key = f"{radar}_last_key"
        last_key = manifest.get(radar_manifest_key)

        if last_key == key:
            print(f"  {radar}: No new scan (same as last run), skipping.")
            # Preserve existing data
            for output_name in PRODUCTS:
                prod_key = f"{radar}_{output_name}"
                if prod_key in manifest:
                    entry = manifest[prod_key]
                    if 'bounds' in entry:
                        bounds_data[prod_key] = entry['bounds']
                    if 'frames' in entry:
                        frames_data[prod_key] = entry['frames']
            continue

        print(f"  {radar}: New scan detected ({os.path.basename(key)})")
        scan_time = extract_scan_time(key)
        any_changes = True

        local_file = download_file(key)
        radar_data = pyart.io.read_nexrad_archive(local_file)

        for output_name, pyart_field in PRODUCTS.items():
            prod_key = f"{radar}_{output_name}"

            if pyart_field not in radar_data.fields:
                print(f"    {radar}: field '{pyart_field}' not available, skipping.")
                # Preserve existing data
                if prod_key in manifest:
                    entry = manifest[prod_key]
                    if 'bounds' in entry:
                        bounds_data[prod_key] = entry['bounds']
                    if 'frames' in entry:
                        frames_data[prod_key] = entry['frames']
                continue

            try:
                # Rotate existing frames before generating the new one
                rotate_frames(radar, output_name)

                # Generate the new (latest) frame
                png_path = f"{OUTPUT_DIR}/{radar}_{output_name}.png"
                bounds = generate_png(radar_data, pyart_field, output_name, radar, png_path)
                bounds_data[prod_key] = list(bounds)
                print(f"    Saved {png_path}")

                # Build the frame list for this product
                frame_list = [{'file': f"{radar}_{output_name}.png", 'time': scan_time}]
                # Add existing history frames
                old_frames = manifest.get(prod_key, {}).get('frames', [])
                for i, old_frame in enumerate(old_frames):
                    hist_idx = i + 1
                    if hist_idx >= MAX_HISTORY:
                        break
                    hist_file = f"{radar}_{output_name}_{hist_idx}.png"
                    if os.path.exists(os.path.join(OUTPUT_DIR, hist_file)):
                        frame_list.append({'file': hist_file, 'time': old_frame.get('time', '')})

                frames_data[prod_key] = frame_list

                # Update manifest entry for this product
                manifest[prod_key] = {
                    'bounds': list(bounds),
                    'frames': frame_list,
                }

            except Exception as e:
                print(f"    {radar}/{output_name}: render failed — {e}")
                # Preserve existing data on failure
                if prod_key in manifest:
                    entry = manifest[prod_key]
                    if 'bounds' in entry:
                        bounds_data[prod_key] = entry['bounds']
                    if 'frames' in entry:
                        frames_data[prod_key] = entry['frames']

        # Update the last processed key for this radar
        manifest[radar_manifest_key] = key

        os.remove(local_file)

    except Exception as e:
        print(f"Error processing {radar}: {e}")
        # Preserve existing data on error
        for output_name in PRODUCTS:
            prod_key = f"{radar}_{output_name}"
            if prod_key in manifest:
                entry = manifest[prod_key]
                if 'bounds' in entry:
                    bounds_data[prod_key] = entry['bounds']
                if 'frames' in entry:
                    frames_data[prod_key] = entry['frames']

# ── Process MRMS products ─────────────────────────────────────────────────────
print("\n── MRMS Products ──")
for output_name, mrms_cfg in MRMS_PRODUCTS.items():
    prod_key = f"MRMS_{output_name}"
    try:
        file_url = get_latest_mrms_file(mrms_cfg['mrms_dir'])
        if not file_url:
            # Preserve existing data
            if prod_key in manifest:
                entry = manifest[prod_key]
                if 'bounds' in entry:
                    bounds_data[prod_key] = entry['bounds']
                if 'frames' in entry:
                    frames_data[prod_key] = entry['frames']
            continue

        # Check if this is a new file vs. what we last processed
        mrms_manifest_key = f"MRMS_{output_name}_last_url"
        last_url = manifest.get(mrms_manifest_key)

        if last_url == file_url:
            print(f"  MRMS {output_name}: No new data (same as last run), skipping.")
            if prod_key in manifest:
                entry = manifest[prod_key]
                if 'bounds' in entry:
                    bounds_data[prod_key] = entry['bounds']
                if 'frames' in entry:
                    frames_data[prod_key] = entry['frames']
            continue

        print(f"  MRMS {output_name}: New data detected ({os.path.basename(file_url)})")
        scan_time = extract_mrms_time(os.path.basename(file_url))
        any_changes = True

        local_grib = download_mrms_grib2(file_url)
        if not local_grib:
            # Preserve existing data
            if prod_key in manifest:
                entry = manifest[prod_key]
                if 'bounds' in entry:
                    bounds_data[prod_key] = entry['bounds']
                if 'frames' in entry:
                    frames_data[prod_key] = entry['frames']
            continue

        try:
            # Rotate existing frames
            rotate_frames('MRMS', output_name)

            # Generate the new (latest) frame
            png_path = f"{OUTPUT_DIR}/MRMS_{output_name}.png"
            bounds = generate_mrms_png(local_grib, output_name, mrms_cfg, png_path)
            bounds_data[prod_key] = list(bounds)
            print(f"    Saved {png_path}")

            # Build the frame list
            frame_list = [{'file': f"MRMS_{output_name}.png", 'time': scan_time}]
            old_frames = manifest.get(prod_key, {}).get('frames', [])
            for i, old_frame in enumerate(old_frames):
                hist_idx = i + 1
                if hist_idx >= MAX_HISTORY:
                    break
                hist_file = f"MRMS_{output_name}_{hist_idx}.png"
                if os.path.exists(os.path.join(OUTPUT_DIR, hist_file)):
                    frame_list.append({'file': hist_file, 'time': old_frame.get('time', '')})

            frames_data[prod_key] = frame_list
            manifest[prod_key] = {
                'bounds': list(bounds),
                'frames': frame_list,
            }
            manifest[mrms_manifest_key] = file_url

        except Exception as e:
            print(f"    MRMS/{output_name}: render failed — {e}")
            if prod_key in manifest:
                entry = manifest[prod_key]
                if 'bounds' in entry:
                    bounds_data[prod_key] = entry['bounds']
                if 'frames' in entry:
                    frames_data[prod_key] = entry['frames']
        finally:
            if os.path.exists(local_grib):
                os.remove(local_grib)

    except Exception as e:
        print(f"Error processing MRMS {output_name}: {e}")
        if prod_key in manifest:
            entry = manifest[prod_key]
            if 'bounds' in entry:
                bounds_data[prod_key] = entry['bounds']
            if 'frames' in entry:
                frames_data[prod_key] = entry['frames']

# ── Cleanup old frames (older than 10th newest) ──────────────────────────────
print("\n── Frame Cleanup ──")
cleanup_old_frames()

# Write bounds to JS file.
# Use window.RADAR_BOUNDS (not const) so the script can be reloaded at runtime
# without a "Cannot redeclare block-scoped variable" SyntaxError on the second call.
with open(f"{OUTPUT_DIR}/bounds.js", 'w') as f:
    f.write("window.RADAR_BOUNDS = " + json.dumps(bounds_data) + ";")

# Write frames manifest to JS file for playback support
with open(f"{OUTPUT_DIR}/frames.js", 'w') as f:
    f.write("window.RADAR_FRAMES = " + json.dumps(frames_data) + ";")

# Save the manifest for next run
save_manifest(manifest)

if any_changes:
    print("Done. New radar data processed.")
else:
    print("Done. No new scans detected — all radars up to date.")
print(f"Bounds written to {OUTPUT_DIR}/bounds.js")
print(f"Frames written to {OUTPUT_DIR}/frames.js")
