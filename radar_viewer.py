import os
import shutil
import nexradaws
import pyart
import numpy as np
import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

# --- Configuration ---
RADAR_SITES = ['KCCX', 'KDIX']
# Define the area for the map (Central/Eastern PA)
GRID_LAT_MIN, GRID_LAT_MAX = 39.0, 42.0
GRID_LON_MIN, GRID_LON_MAX = -79.0, -73.0
GRID_SHAPE = (1, 600, 800) # (z, y, x) - Higher x/y = higher resolution but slower

def get_latest_scans():
    conn = nexradaws.NexradAwsInterface()
    scans = []
    
    # Get scans for the current day
    today = datetime.utcnow()
    print(f"Searching for scans on {today.strftime('%Y-%m-%d')}...")
    
    for site in RADAR_SITES:
        # Get all scans for today
        avail_scans = conn.get_avail_scans(today.year, today.month, today.day, site)
        if avail_scans:
            # Sort and pick the very latest one
            avail_scans.sort(key=lambda x: x.scan_time)
            latest = avail_scans[-1]
            print(f"Found {site}: {latest.filename}")
            scans.append(latest)
        else:
            print(f"No scans found for {site} yet today.")
            
    return scans

def download_and_read_scans(scans, download_dir='radar_data'):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    conn = nexradaws.NexradAwsInterface()
    results = conn.download(scans, download_dir)
    
    radars = []
    for download in results.success:
        try:
            radar = pyart.io.read_nexrad_archive(download.filepath)
            radars.append(radar)
        except Exception as e:
            print(f"Failed to read {download.filename}: {e}")
            
    return radars

def generate_image_overlay(grid, field, filename, vmin, vmax, cmap_name):
    """
    Converts a Py-ART grid field into a transparent PNG for Folium.
    """
    data = grid.fields[field]['data'][0] # Get the first vertical level
    
    # Normalize data for colormap
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    
    # Map data to RGBA
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    image_data = mappable.to_rgba(data)
    
    # Make masked (NaN) values transparent
    # Py-ART uses masked arrays. Where mask is True, set Alpha to 0
    if np.ma.is_masked(data):
        image_data[data.mask, 3] = 0.0
    
    # Save to PNG
    plt.imsave(filename, image_data, origin='lower')
    return filename

def create_map(radars):
    # 1. Grid the data (Mosaic KCCX and KDIX together)
    print("Gridding data (this may take a moment)...")
    grid = pyart.map.grid_from_radars(
        radars,
        grid_shape=GRID_SHAPE,
        grid_limits=((0, 2000), # 0-2km altitude (lowest sweep focus)
                     (GRID_LAT_MIN, GRID_LAT_MAX), 
                     (GRID_LON_MIN, GRID_LON_MAX)),
        grid_origin_lat=(GRID_LAT_MIN + GRID_LAT_MAX) / 2,
        grid_origin_lon=(GRID_LON_MIN + GRID_LON_MAX) / 2,
        fields=['reflectivity', 'velocity', 'cross_correlation_ratio']
    )

    # 2. Setup Folium Map
    center_lat = (GRID_LAT_MIN + GRID_LAT_MAX) / 2
    center_lon = (GRID_LON_MIN + GRID_LON_MAX) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles='cartodbpositron')

    # Define Image Bounds for Folium
    # Note: Py-ART grids are defined by center points, but images fill the box.
    # We use the grid limits we defined earlier.
    image_bounds = [[GRID_LAT_MIN, GRID_LON_MIN], [GRID_LAT_MAX, GRID_LON_MAX]]

    # 3. Generate Overlays
    
    # -- Reflectivity --
    gen_file_ref = 'overlay_ref.png'
    generate_image_overlay(grid, 'reflectivity', gen_file_ref, vmin=-10, vmax=70, cmap_name='pyart_HomeyerRainbow')
    folium.raster_layers.ImageOverlay(
        image=gen_file_ref,
        bounds=image_bounds,
        opacity=0.8,
        name='Reflectivity (dBZ)',
        show=True
    ).add_to(m)

    # -- Velocity --
    gen_file_vel = 'overlay_vel.png'
    generate_image_overlay(grid, 'velocity', gen_file_vel, vmin=-30, vmax=30, cmap_name='pyart_BuDRd18')
    folium.raster_layers.ImageOverlay(
        image=gen_file_vel,
        bounds=image_bounds,
        opacity=0.8,
        name='Velocity (m/s)',
        show=False
    ).add_to(m)

    # -- Correlation Coefficient --
    gen_file_cc = 'overlay_cc.png'
    generate_image_overlay(grid, 'cross_correlation_ratio', gen_file_cc, vmin=0.8, vmax=1.05, cmap_name='pyart_RefDiff')
    folium.raster_layers.ImageOverlay(
        image=gen_file_cc,
        bounds=image_bounds,
        opacity=0.8,
        name='Correlation Coefficient',
        show=False
    ).add_to(m)

    # 4. Add Controls
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add a timestamp annotation
    scan_time = radars[0].time['units'].replace('seconds since ', '')
    timestamp_html = f'<div style="position: fixed; bottom: 50px; left: 50px; z-index:9999; font-size:14px; background-color: white; padding: 10px; border: 2px solid grey;"><b>Data Time (UTC):</b> {scan_time}</div>'
    m.get_root().html.add_child(folium.Element(timestamp_html))

    print("Saving map to index.html...")
    m.save('index.html')

def main():
    scans = get_latest_scans()
    if not scans:
        print("No scans found.")
        return

    radars = download_and_read_scans(scans)
    if not radars:
        print("Could not read any radar files.")
        return

    create_map(radars)
    
    # Cleanup downloaded files
    if os.path.exists('radar_data'):
        shutil.rmtree('radar_data')

if __name__ == "__main__":
    main()
