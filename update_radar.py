# update_radar.py
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import datetime
import os
import pyart
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Configurable radars
RADARS = ['KCCX', 'KDOX']  # Change this list to switch radars

# Products to display
PRODUCTS = {
    'reflectivity': 'reflectivity',
    'velocity': 'velocity',
    'correlation_coefficient': 'correlation_coefficient'
}

# Output directory
OUTPUT_DIR = 'docs/assets'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# S3 client (anonymous access)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def get_latest_file(radar):
    now = datetime.datetime.utcnow()
    prefix = f"{now.year}/{now.strftime('%m')}/{now.strftime('%d')}/{radar}/"
    
    # List objects to find the latest file
    response = s3.list_objects_v2(Bucket='unidata-nexrad-level2', Prefix=prefix)
    if 'Contents' not in response:
        print(f"No objects found for prefix {prefix}")
        return None
    
    # Filter to valid Level-2 files (usually end with _V06 or similar)
    files = [obj for obj in response['Contents'] if obj['Key'].endswith(('_V06', '_V07'))]
    if not files:
        print(f"No valid Level-2 files in {prefix}")
        return None
    
    latest_key = max(files, key=lambda x: x['LastModified'])['Key']
    return latest_key

def download_file(key):
    local_path = '/tmp/' + os.path.basename(key)
    s3.download_file('unidata-nexrad-level2', key, local_path)
    return local_path

def generate_png(radar_data, field, radar_code):
    display = pyart.graph.RadarMapDisplay(radar_data)
    
    fig = plt.figure(figsize=(10, 10))
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    
    # Plot the radar data
    mesh = display.plot_ppi_map(field, 0, vmin=-30 if field == 'reflectivity' else -20 if field == 'velocity' else 0,
                                vmax=70 if field == 'reflectivity' else 20 if field == 'velocity' else 1.05,
                                cmap='pyart_NWSRef' if field == 'reflectivity' else 'pyart_BuDRd18' if field == 'velocity' else 'pyart_RefDiff',
                                ax=ax, title='', colorbar_flag=False,
                                lat_lines=np.arange(30, 50, 1), lon_lines=np.arange(-90, -70, 1))
    
    # Set transparent background
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Remove axes
    ax.axis('off')
    
    # Get bounds
    lon_min, lon_max = ax.get_xlim()
    lat_min, lat_max = ax.get_ylim()
    
    # Save PNG with alpha
    png_path = f"{OUTPUT_DIR}/{radar_code}_{field}.png"
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    
    return png_path, (lat_min, lon_min, lat_max, lon_max)  # bounds: south, west, north, east

# Bounds dict to save in JS file
bounds_data = {}

for radar in RADARS:
    key = get_latest_file(radar)
    if not key:
        print(f"No data for {radar}")
        continue
    
    local_file = download_file(key)
    radar_data = pyart.io.read_nexrad_archive(local_file)
    
    for display_name, field in PRODUCTS.items():
        png_path, bounds = generate_png(radar_data, field, radar)
        bounds_data[f"{radar}_{display_name}"] = list(bounds)
    
    os.remove(local_file)

# Write bounds to JS file
with open(f"{OUTPUT_DIR}/bounds.js", 'w') as f:
    f.write("const RADAR_BOUNDS = " + str(bounds_data) + ";")
