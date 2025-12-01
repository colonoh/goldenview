"""
Visualize visibility analysis results.
Shows which buildings can see which viewpoints.
Buildings that can't see any viewpoints are greyed out.
"""

import pandas as pd
import geopandas as gpd
from shapely import from_wkt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import re
import numpy as np

# Load the results
print("Loading results...")
with open('visibility_results_russian_hill.json') as f:
    results = json.load(f)

# Load buildings
print("Loading buildings...")
df = pd.read_csv('data/Building_Footprints_RussianHill_10k.csv')

def fix_wkt(wkt_str):
    coords = re.findall(r'(-?\d+\.\d+)\s+(-?\d+\.\d+)', wkt_str)
    if coords:
        polygon_coords = [[float(lon), float(lat)] for lon, lat in coords]
        if polygon_coords[0] != polygon_coords[-1]:
            polygon_coords.append(polygon_coords[0])
        coord_str = ", ".join([f"{lon} {lat}" for lon, lat in polygon_coords])
        return f"POLYGON (({coord_str}))"
    return wkt_str

def parse_geometry(wkt_str):
    try:
        geom = from_wkt(fix_wkt(wkt_str))
        if not geom.is_valid:
            geom = geom.buffer(0)
        return geom
    except Exception:
        return None

df['geometry'] = df['shape'].apply(parse_geometry)
df = df[df['geometry'].notna()].reset_index(drop=True)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

# Create a mapping of which buildings see which viewpoints
print("Processing visibility data...")
viewpoint_names = list(results.keys())
building_visibility = {}  # building_id -> set of viewpoints that can see it

for vp_name, visible_building_ids in results.items():
    for building_id in visible_building_ids:
        if building_id not in building_visibility:
            building_visibility[building_id] = set()
        building_visibility[building_id].add(vp_name)

# Assign colors to buildings based on visibility
def get_building_color(building_idx):
    """Get color for a building based on how many viewpoints can see it."""
    building_id = building_idx + 1  # IDs are 1-indexed

    if building_id not in building_visibility:
        return 'lightgrey'  # Can't see any viewpoints

    num_visible = len(building_visibility[building_id])

    # Color based on number of viewpoints
    if num_visible == 4:
        return 'lime'  # Can see all 4
    elif num_visible == 3:
        return 'darkgreen'
    elif num_visible == 2:
        return 'yellow'
    elif num_visible == 1:
        return 'orange'
    else:
        return 'lightgrey'

def plot_geometry(ax, geom, color, alpha, linewidth):
    """Plot a geometry (handles both Polygon and MultiPolygon)."""
    if geom.geom_type == 'Polygon':
        x, y = geom.exterior.xy
        ax.fill(x, y, alpha=alpha, edgecolor='black', facecolor=color, linewidth=linewidth)
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=alpha, edgecolor='black', facecolor=color, linewidth=linewidth)
    else:
        # Handle other geometry types if needed
        pass

# Create visualization with 5 subplots (1 overview + 4 per-viewpoint)
fig = plt.figure(figsize=(20, 12))

# Main overview plot
ax_main = plt.subplot(2, 3, 1)

# Plot overview - all buildings colored by visibility
for building_idx in range(len(gdf)):
    geom = gdf.iloc[building_idx]['geometry']
    color = get_building_color(building_idx)
    plot_geometry(ax_main, geom, color, alpha=0.6, linewidth=0.5)

ax_main.set_xlabel('Longitude', fontsize=10)
ax_main.set_ylabel('Latitude', fontsize=10)
ax_main.set_title('Overview: Buildings by Visibility Count\n(Bright Green = All 4 Viewpoints)', fontsize=12, fontweight='bold')
ax_main.grid(True, alpha=0.3, linestyle='--')
ax_main.set_aspect('equal')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='lime', edgecolor='black', label='Visible from all 4 viewpoints'),
    mpatches.Patch(facecolor='darkgreen', edgecolor='black', label='Visible from 3 viewpoints'),
    mpatches.Patch(facecolor='yellow', edgecolor='black', label='Visible from 2 viewpoints'),
    mpatches.Patch(facecolor='orange', edgecolor='black', label='Visible from 1 viewpoint'),
    mpatches.Patch(facecolor='lightgrey', edgecolor='black', label='Not visible from any'),
]
ax_main.legend(handles=legend_elements, loc='upper left', fontsize=8)

# Per-viewpoint plots
for vp_idx, vp_name in enumerate(viewpoint_names):
    ax = plt.subplot(2, 3, vp_idx + 2)
    visible_ids = set(results[vp_name])

    # Plot buildings
    for building_idx in range(len(gdf)):
        geom = gdf.iloc[building_idx]['geometry']
        building_id = building_idx + 1

        if building_id in visible_ids:
            color = 'lightgreen'
            alpha = 0.7
            linewidth = 1
        else:
            color = 'lightgrey'
            alpha = 0.3
            linewidth = 0.5

        plot_geometry(ax, geom, color, alpha=alpha, linewidth=linewidth)

    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title(f'{vp_name}\n({len(visible_ids)} visible)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

plt.suptitle('Russian Hill Visibility Analysis Results', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('visibility_analysis_results.png', dpi=150, bbox_inches='tight')
print("Saved overview visualization to visibility_analysis_results.png")

# Create detailed statistics visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart: Buildings by visibility count
visibility_counts = {}
for building_id in building_visibility:
    num_vp = len(building_visibility[building_id])
    if num_vp not in visibility_counts:
        visibility_counts[num_vp] = 0
    visibility_counts[num_vp] += 1

# Add buildings with 0 visibility
num_no_visibility = len(gdf) - len(building_visibility)
visibility_counts[0] = num_no_visibility

counts = [visibility_counts.get(i, 0) for i in range(5)]
labels = ['0 viewpoints\n(greyed out)', '1 viewpoint', '2 viewpoints', '3 viewpoints', '4 viewpoints']
colors_chart = ['lightgrey', 'orange', 'yellow', 'darkgreen', 'lime']

ax1.bar(range(5), counts, color=colors_chart, edgecolor='black', linewidth=2)
ax1.set_ylabel('Number of Buildings', fontsize=12)
ax1.set_xlabel('Visible From', fontsize=12)
ax1.set_title('Buildings by Viewpoint Visibility Count', fontsize=13, fontweight='bold')
ax1.set_xticks(range(5))
ax1.set_xticklabels(labels, fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (count, label) in enumerate(zip(counts, labels)):
    ax1.text(i, count + 20, str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)

# Bar chart: Per-viewpoint visibility
vp_names_short = ['GG North', 'GG South', 'Alcatraz', 'Bay Bridge']
vp_counts = [len(results[vp_name]) for vp_name in viewpoint_names]
colors_vp = ['red', 'darkred', 'orange', 'blue']

ax2.bar(range(len(viewpoint_names)), vp_counts, color=colors_vp, edgecolor='black', linewidth=2)
ax2.set_ylabel('Number of Visible Buildings', fontsize=12)
ax2.set_xlabel('Viewpoint', fontsize=12)
ax2.set_title('Buildings Visible from Each Viewpoint', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(viewpoint_names)))
ax2.set_xticklabels(vp_names_short, fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (count, name) in enumerate(zip(vp_counts, vp_names_short)):
    ax2.text(i, count + 20, str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('visibility_analysis_statistics.png', dpi=150, bbox_inches='tight')
print("Saved statistics visualization to visibility_analysis_statistics.png")

# Print summary
print("\n" + "="*70)
print("VISIBILITY SUMMARY")
print("="*70)
print()
print(f"Total buildings analyzed: {len(gdf)}")
print()
print("Buildings by visibility count:")
for num_vp in sorted(visibility_counts.keys()):
    count = visibility_counts[num_vp]
    pct = (count / len(gdf)) * 100
    print(f"  {num_vp} viewpoints: {count:6d} buildings ({pct:5.1f}%)")

print()
print("Buildings visible from each viewpoint:")
for vp_name in viewpoint_names:
    count = len(results[vp_name])
    pct = (count / len(gdf)) * 100
    print(f"  {vp_name:35s}: {count:6d} buildings ({pct:5.1f}%)")

print()
print("="*70)
