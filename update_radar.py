# update_radar.py
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import datetime
import json
import os
import pyart
import matplotlib
matplotlib.use('Agg')   # headless backend — must come before pyplot import
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import warnings
warnings.filterwarnings("ignore")
 
# Configurable radars
RADARS = ['KCCX', 'KDOX', 'KIND', 'KDIX', 'KBGM']
 
# Products: output_name -> PyART field name
# Note: correlation coefficient is 'cross_correlation_ratio' in PyART's NEXRAD reader
PRODUCTS = {
    'reflectivity':            'reflectivity',
    'velocity':                'velocity',
    'correlation_coefficient': 'cross_correlation_ratio',
}
 
# Output directory
OUTPUT_DIR = 'docs/assets'
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# S3 client (anonymous access)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
 
def get_latest_file(radar):
    now = datetime.datetime.utcnow()
    prefix = f"{now.year}/{now.strftime('%m')}/{now.strftime('%d')}/{radar}/"
 
    response = s3.list_objects_v2(Bucket='unidata-nexrad-level2', Prefix=prefix)
    if 'Contents' not in response:
        print(f"No objects found for prefix {prefix}")
        return None
 
    files = [obj for obj in response['Contents'] if obj['Key'].endswith(('_V06', '_V07'))]
    if not files:
        print(f"No valid Level-2 files in {prefix}")
        return None
 
    return max(files, key=lambda x: x['LastModified'])['Key']
 
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
 
def generate_png(radar_data, pyart_field, output_name, radar_code):
    display = pyart.graph.RadarMapDisplay(radar_data)
 
    fig = plt.figure(figsize=(12, 12))
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
 
    # Colormap / range keyed on output_name (not the internal PyART field name)
    if output_name == 'reflectivity':
        cmap = 'NWSRef'
        vmin, vmax = -30, 70
    elif output_name == 'velocity':
        cmap = 'BuDRd18'
        vmin, vmax = -30, 30
    else:  # correlation_coefficient
        cmap = 'RefDiff'
        vmin, vmax = 0, 1
 
    sweep_idx = find_best_sweep(radar_data, pyart_field)
 
    # Plot — empty lat/lon lists suppress coordinate gridlines on the image
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
 
    png_path = f"{OUTPUT_DIR}/{radar_code}_{output_name}.png"
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0,
                transparent=True, dpi=150)
    plt.close()
 
    return png_path, (lat_min, lon_min, lat_max, lon_max)
 
# ── Main ───────────────────────────────────────────────────────────────────────
bounds_data = {}
 
for radar in RADARS:
    try:
        key = get_latest_file(radar)
        if not key:
            print(f"No data for {radar}, skipping.")
            continue
 
        local_file = download_file(key)
        radar_data = pyart.io.read_nexrad_archive(local_file)
 
        for output_name, pyart_field in PRODUCTS.items():
            if pyart_field not in radar_data.fields:
                print(f"  {radar}: field '{pyart_field}' not available, skipping.")
                continue
            try:
                png_path, bounds = generate_png(radar_data, pyart_field, output_name, radar)
                bounds_data[f"{radar}_{output_name}"] = list(bounds)
                print(f"  Saved {png_path}")
            except Exception as e:
                print(f"  {radar}/{output_name}: render failed — {e}")
 
        os.remove(local_file)
 
    except Exception as e:
        print(f"Error processing {radar}: {e}")
 
# Write bounds to JS file.
# Use window.RADAR_BOUNDS (not const) so the script can be reloaded at runtime
# without a "Cannot redeclare block-scoped variable" SyntaxError on the second call.
with open(f"{OUTPUT_DIR}/bounds.js", 'w') as f:
    f.write("window.RADAR_BOUNDS = " + json.dumps(bounds_data) + ";")
 
print("Done. Bounds written to", f"{OUTPUT_DIR}/bounds.js")

