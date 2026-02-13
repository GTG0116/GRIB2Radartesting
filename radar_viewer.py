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
# Define the area for the map (Pennsylvania / New Jersey region)
GRID_LAT_MIN, GRID_LAT_MAX = 39.0, 42.0
GRID_LON_MIN, GRID_LON_MAX = -79.0, -73.0
GRID_SHAPE = (1, 600, 800) 

def get_latest_scans():
    conn = nexradaws.NexradAwsInterface()
    scans = []
    today = datetime.utcnow()
    print(f"Searching for scans on {today.strftime('%Y-%m-%d')}...")
    
    for site in RADAR_SITES:
        try:
            avail_scans = conn.get_avail_scans(today.year, today.month, today.day, site)
            if avail_scans:
                avail_scans.sort(key=lambda x: x.scan_time)
                latest = avail_scans[-1]
                print(f"Found {site}: {latest.filename}")
                scans.append(latest)
        except Exception as e:
            print(f"Could not fetch availability for {site}: {e}")
            
    return scans

def download_and_read_scans(scans, download_dir='radar_data'):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        
    conn = nexradaws.NexradAwsInterface()
    results = conn.download(scans, download_dir)
    
    radars = []
    for download in results.success:
        try:
            # Using pyart.io.read is more robust than read_nexrad_archive
            radar = pyart.io.read(download.filepath)
            radars.append(radar)
            print(f"Successfully loaded {download.filename}")
        except Exception as e:
            print(f"Failed to read {download.filename}: {e}")
            
    return radars

def generate_image_overlay(grid, field, filename, vmin, vmax, cmap_name):
    """Converts a Py-ART grid field into a transparent PNG."""
    # Check if field exists in the grid
    if field not in grid.fields:
        print(f"Warning: Field {field} not found in grid. Skipping overlay.")
        return None

    data = grid.fields[field]['data'][0] 
    
    # Use standard Matplotlib colormap names
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        print(f"Colormap {cmap_name} not found, falling back to viridis.")
        cmap = plt.get_cmap('viridis')

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    image_data = mappable.to_rgba(data)
    
    # Apply transparency to masked areas
    if np.ma.is_masked(data):
        image_data[data.mask, 3] = 0.0
    
    plt.imsave(filename, image_data, origin='lower')
    return filename

def create_map(radars):
    if not radars:
        print("No valid radar data objects found. Exiting.")
        return

    print("Gridding data... This might take a minute.")
    # Mosaic multiple radars into one grid
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

    # Initialize Folium Map
    center_lat = (GRID_LAT_MIN + GRID_LAT_MAX) / 2
    center_lon = (GRID_LON_MIN + GRID_LON_MAX) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles='cartodbpositron')

    image_bounds = [[GRID_LAT_MIN, GRID_LON_MIN], [GRID_LAT_MAX, GRID_LON_MAX]]

    # --- Reflectivity Layer ---
    if generate_image_overlay(grid, 'reflectivity', 'overlay_ref.png', -10, 70, 'HomeyerRainbow'):
        folium.raster_layers.ImageOverlay(
            image='overlay_ref.png',
            bounds=image_bounds,
            opacity=0.7,
            name='Reflectivity (dBZ)',
            show=True
        ).add_to(m)

    # --- Velocity Layer ---
    if generate_image_overlay(grid, 'velocity', 'overlay_vel.png', -30, 30, 'BuDRd18'):
        folium.raster_layers.ImageOverlay(
            image='overlay_vel.png',
            bounds=image_bounds,
            opacity=0.7,
            name='Velocity (m/s)',
            show=False
        ).add_to(m)

    # --- CC Layer ---
    if generate_image_overlay(grid, 'cross_correlation_ratio', 'overlay_cc.png', 0.8, 1.0, 'RefDiff'):
        folium.raster_layers.ImageOverlay(
            image='overlay_cc.png',
            bounds=image_bounds,
            opacity=0.7,
            name='Correlation Coefficient',
            show=False
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add timestamp
    data_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    title_html = f'''
             <div style="position: fixed; bottom: 50px; left: 50px; width: 250px; height: 70px; 
             background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
             padding: 10px;">
             <b>KCCX & KDIX Mosaic</b><br>
             Last Updated: {data_time}
             </div>
             '''
    m.get_root().html.add_child(folium.Element(title_html))

    print("Saving map to index.html...")
    m.save('index.html')

def main():
    scans = get_latest_scans()
    if not scans:
        print("No scans found.")
        return

    radars = download_and_read_scans(scans)
    create_map(radars)
    
    if os.path.exists('radar_data'):
        shutil.rmtree('radar_data')

if __name__ == "__main__":
    main()
