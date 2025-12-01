"""
Visualize visibility analysis results.
"""

import pandas as pd
import geopandas as gpd
from shapely import from_wkt
import matplotlib.pyplot as plt
import json
import re

def load_buildings(csv_path):
    """Load and parse building data from CSV."""
    df = pd.read_csv(csv_path)

    def fix_wkt(wkt_str):
        coords = re.findall(r'(-?\d+\.\d+)\s+(-?\d+\.\d+)', wkt_str)
        if coords:
            polygon_coords = [[float(lon), float(lat)] for lon, lat in coords]
            coord_str = ", ".join([f"{lon} {lat}" for lon, lat in polygon_coords])
            return f"POLYGON (({coord_str}))"
        return wkt_str

    df['geometry'] = df['shape'].apply(lambda x: from_wkt(fix_wkt(x)))
    return gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

# Load data
gdf = load_buildings('data/test_buildings_grid.csv')

with open('viewpoints.json') as f:
    viewpoints = json.load(f)

with open('visibility_results.json') as f:
    results = json.load(f)

# Create a 2x3 grid of subplots (one for each viewpoint)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, vp in enumerate(viewpoints):
    ax = axes[idx]
    vp_name = vp['name']
    visible_building_ids = set(results[vp_name])

    # Plot buildings
    for building_idx, (_, row) in enumerate(gdf.iterrows()):
        geom = row['geometry']
        x, y = geom.exterior.xy

        building_id = building_idx + 1
        is_visible = building_id in visible_building_ids

        if is_visible:
            color = 'lightgreen'
            edgecolor = 'darkgreen'
            alpha = 0.7
        else:
            color = 'lightcoral'
            edgecolor = 'darkred'
            alpha = 0.3

        ax.fill(x, y, alpha=alpha, edgecolor=edgecolor, facecolor=color, linewidth=2)

        # Label buildings
        centroid = geom.centroid
        height = row['hgt_median_m']
        label_text = f"B{building_id}\n{height:.0f}m"
        ax.text(centroid.x, centroid.y, label_text,
                ha='center', va='center', fontsize=8, fontweight='bold')

    # Plot viewpoint
    ax.plot(vp['lon'], vp['lat'], 'r*', markersize=25, markeredgecolor='darkred', markeredgewidth=2)

    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title(f"Viewpoint: {vp_name}\n{len(visible_building_ids)} visible buildings",
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

plt.suptitle('Visibility Analysis Results (Green=Visible, Red=Blocked)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('visibility_results.png', dpi=150, bbox_inches='tight')
print("Saved visibility results to visibility_results.png")

# Print summary
print("\n=== Visibility Summary ===\n")
for vp in viewpoints:
    vp_name = vp['name']
    visible_ids = results[vp_name]
    print(f"{vp_name:6s}: {len(visible_ids):2d} visible - Buildings {visible_ids}")
