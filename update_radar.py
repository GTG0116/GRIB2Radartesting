# update_radar.py
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import datetime
import json
import os
import shutil
import pyart
import matplotlib
matplotlib.use('Agg')   # headless backend — must come before pyplot import
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Configurable radars
RADARS = ['KCCX', 'KDIX', 'KDOX', 'KBGM', 'KOKX', 'KPBZ']

# Products: output_name -> PyART field name
# Note: correlation coefficient is 'cross_correlation_ratio' in PyART's NEXRAD reader
PRODUCTS = {
    'reflectivity':            'reflectivity',
    'velocity':                'velocity',
    'correlation_coefficient': 'cross_correlation_ratio',
}

# Number of historical frames to keep (plus the current one)
MAX_HISTORY = 5

# Output directory
OUTPUT_DIR = 'docs/assets'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Manifest file tracks last-processed S3 keys and frame history
MANIFEST_FILE = os.path.join(OUTPUT_DIR, 'manifest.json')

# S3 client (anonymous access)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

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
    """Return the first sweep index that contains non-masked data for field."""
    for i in range(radar_data.nsweeps):
        start = int(radar_data.sweep_start_ray_index['data'][i])
        end   = int(radar_data.sweep_end_ray_index['data'][i]) + 1
        data  = radar_data.fields[field]['data'][start:end]
        mask  = np.ma.getmaskarray(data)
        if not np.all(mask):
            return i
    return 0  # fallback

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

    # Plot with rasterized output for smooth pixel rendering
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

    # Ensure rasterized elements render with smooth interpolation
    for artist in ax.get_children():
        if hasattr(artist, 'set_rasterized'):
            artist.set_rasterized(True)
        if hasattr(artist, 'set_interpolation'):
            artist.set_interpolation('bilinear')

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
                transparent=True, dpi=800)
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
