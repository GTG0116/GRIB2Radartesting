import os
import sys
import shutil
import base64
import nexradaws
import pyart
import numpy as np
import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta

# --- Configuration ---
RADAR_SITES = ['KCCX', 'KDIX']
# Region: PA/NJ
GRID_LAT_MIN, GRID_LAT_MAX = 39.0, 42.0
GRID_LON_MIN, GRID_LON_MAX = -79.0, -73.0
GRID_SHAPE = (1, 600, 800) 

def get_latest_scans():
    conn = nexradaws.NexradAwsInterface()
    scans = []
    
    # We check "Today" and "Yesterday" to handle the UTC midnight edge case
    check_dates = [datetime.utcnow(), datetime.utcnow() - timedelta(days=1)]
    
    for site in RADAR_SITES:
        site_scans = []
        print(f"Checking {site}...")
        
        for date in check_dates:
            try:
                found = conn.get_avail_scans(date.year, date.month, date.day, site)
                if found:
                    site_scans.extend(found)
            except Exception as e:
                print(f"  - Error checking {date.strftime('%Y-%m-%d')}: {e}")

        if site_scans:
            # Sort by time and pick the absolute latest
            site_scans.sort(key=lambda x: x.scan_time)
            latest = site_scans[-1]
            print(f"  -> Found newest scan: {latest.filename} ({latest.scan_time})")
            scans.append(latest)
        else:
            print(f"  -> Warning: No scans found for {site} in the last 48 hours.")
            
    return scans

def download_and_read_scans(scans, download_dir='radar_data'):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    conn = nexradaws.NexradAwsInterface()
    results = conn.download(scans, download_dir)
    
    radars = []
    for download in results.success:
        try:
            # pyart.io.read handles compression automatically
            radar = pyart.io.read(download.filepath)
            radars.append(radar)
            print(f"Loaded data for {download.filename}")
        except Exception as e:
            print(f"Failed to read {download.filename}: {e}")
            
    return radars

def generate_image_overlay(grid, field, filename, vmin, vmax, cmap_name):
    if field not in grid.fields:
        print(f"Skipping {field}: Data not in grid.")
        return False

    data = grid.fields[field]['data'][0]
    
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        cmap = plt.get_cmap('viridis') # Fallback

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    image_data = mappable.to_rgba(data)
    
    if np.ma.is_masked(data):
        image_data[data.mask, 3] = 0.0 # Make 'no data' pixels transparent
    
    plt.imsave(filename, image_data, origin='lower')
    print(f"Generated {filename}")
    return True

def create_map(radars):
    if not radars:
        print("Error: No radar objects available to grid.")
        sys.exit(1) # Force GitHub Action to fail

    print("Gridding data... (This merges the radars)")
    try:
        grid = pyart.map.grid_from_radars(
            radars,
            grid_shape=GRID_SHAPE,
            grid_limits=((0, 2000), 
                         (GRID_LAT_MIN, GRID_LAT_MAX), 
                         (GRID_LON_MIN, GRID_LON_MAX)),
            grid_origin_lat=(GRID_LAT_MIN + GRID_LAT_MAX) / 2,
            grid_origin_lon=(GRID_LON_MIN + GRID_LON_MAX) / 2,
            fields=['reflectivity', 'velocity', 'cross_correlation_ratio']
        )
    except Exception as e:
        print(f"Gridding Failed: {e}")
        sys.exit(1)

    # Base Map
    center_lat = (GRID_LAT_MIN + GRID_LAT_MAX) / 2
    center_lon = (GRID_LON_MIN + GRID_LON_MAX) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles='cartodbpositron')

    # Fix 404 Favicon Error
    m.get_root().header.add_child(folium.Element('<link rel="shortcut icon" href="favicon.ico" type="image/x-icon">'))

    image_bounds = [[GRID_LAT_MIN, GRID_LON_MIN], [GRID_LAT_MAX, GRID_LON_MAX]]

    # Generate Layers
    has_layer = False
    
    if generate_image_overlay(grid, 'reflectivity', 'overlay_ref.png', -10, 70, 'HomeyerRainbow'):
        folium.raster_layers.ImageOverlay(
            image='overlay_ref.png', bounds=image_bounds, opacity=0.7, name='Reflectivity', show=True
        ).add_to(m)
        has_layer = True

    if generate_image_overlay(grid, 'velocity', 'overlay_vel.png', -30, 30, 'BuDRd18'):
        folium.raster_layers.ImageOverlay(
            image='overlay_vel.png', bounds=image_bounds, opacity=0.7, name='Velocity', show=False
        ).add_to(m)

    if generate_image_overlay(grid, 'cross_correlation_ratio', 'overlay_cc.png', 0.8, 1.0, 'RefDiff'):
        folium.raster_layers.ImageOverlay(
            image='overlay_cc.png', bounds=image_bounds, opacity=0.7, name='Correlation Coeff', show=False
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    
    # Save Map
    m.save('index.html')
    print("Map saved to index.html")
    
    if not has_layer:
        print("Warning: Map created but no layers were generated.")

def main():
    # 1. Create Dummy Favicon (prevents browser 404s)
    with open("favicon.ico", "wb") as f:
        f.write(base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"))

    # 2. Get Data
    scans = get_latest_scans()
    if not scans:
        print("CRITICAL: No scans found for any radar site. Exiting.")
        sys.exit(1) # Fail the action so we know!

    # 3. Download & Process
    radars = download_and_read_scans(scans)
    if not radars:
        print("CRITICAL: Scans found but failed to read. Exiting.")
        sys.exit(1)

    # 4. Generate Map
    create_map(radars)
    
    # 5. Cleanup
    if os.path.exists('radar_data'):
        shutil.rmtree('radar_data')

if __name__ == "__main__":
    main()
