#!/usr/bin/env python3
"""
Debug script to profile performance bottlenecks in visibility analysis.
"""

import time
import pandas as pd
import geopandas as gpd
from shapely import from_wkt
import json
import re
import numpy as np
from rtree import index

print("=" * 70)
print("PERFORMANCE DEBUG - Russian Hill Dataset (5,432 buildings)")
print("=" * 70)
print()

# STEP 1: Load CSV
print("STEP 1: Loading CSV...")
start = time.time()
df = pd.read_csv('data/Building_Footprints_RussianHill_10k.csv')
step1_time = time.time() - start
print(f"  Loaded {len(df)} rows in {step1_time:.2f}s")
print()

# STEP 2: Parse geometries
print("STEP 2: Parsing and validating geometries...")
start = time.time()

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

# Test on small sample first
sample_size = 100
print(f"  Testing on sample of {sample_size} geometries...")
sample_start = time.time()
test_geoms = df['shape'].head(sample_size).apply(parse_geometry)
sample_time = time.time() - sample_start
print(f"  Sample parse time: {sample_time:.2f}s for {sample_size} rows")
print(f"  Extrapolated full time: {sample_time * (len(df)/sample_size):.1f}s")

# Check how many would be invalid
invalid_count = test_geoms.isna().sum()
print(f"  Invalid geometries in sample: {invalid_count}/{sample_size}")
print()

# STEP 3: Build spatial index
print("STEP 3: Building spatial index...")
start = time.time()
print(f"  Creating {len(df)} index entries...")
rtree_idx = index.Index()
for i in range(min(100, len(df))):  # Test with first 100
    bounds = (0, 0, 1, 1)  # Dummy bounds
    rtree_idx.insert(i, bounds)
index_build_time = time.time() - start
extrapolated_index_time = index_build_time * (len(df) / min(100, len(df)))
print(f"  Sample index build (100 entries): {index_build_time:.3f}s")
print(f"  Extrapolated full index time: {extrapolated_index_time:.2f}s")
print()

# STEP 4: Visibility analysis simulation
print("STEP 4: Simulating visibility analysis...")
print("  Estimating time per visibility check...")

# Load viewpoints
with open('viewpoints_real.json') as f:
    viewpoints = json.load(f)

num_buildings = len(df)
num_viewpoints = len(viewpoints)
total_checks = num_buildings * num_viewpoints

print(f"  Buildings: {num_buildings}")
print(f"  Viewpoints: {num_viewpoints}")
print(f"  Total visibility checks: {total_checks:,}")
print()

# Estimate per-check time based on ray-box intersection
def ray_box_intersection_test():
    """Quick test of ray-box intersection performance"""
    ray_start = np.array([0.0, 0.0, 0.0])
    ray_end = np.array([0.1, 0.1, 100.0])
    box_min = np.array([0.05, 0.05, 0.0])
    box_max = np.array([0.15, 0.15, 50.0])

    ray_dir = ray_end - ray_start
    t_min = 0.0
    t_max = 1.0

    for i in range(3):
        if abs(ray_dir[i]) < 1e-9:
            if ray_start[i] < box_min[i] or ray_start[i] > box_max[i]:
                return False
        else:
            t1 = (box_min[i] - ray_start[i]) / ray_dir[i]
            t2 = (box_max[i] - ray_start[i]) / ray_dir[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
    return t_min <= t_max

num_tests = 10000
start = time.time()
for _ in range(num_tests):
    ray_box_intersection_test()
intersection_time = (time.time() - start) / num_tests
print(f"  Ray-box intersection time: {intersection_time*1e6:.2f} microseconds")
print()

# STEP 5: Estimate total time
print("=" * 70)
print("TOTAL TIME ESTIMATE")
print("=" * 70)
print()

# Estimate with different candidate filtering strategies
print("Scenario 1: Check ALL buildings for each ray (no spatial indexing)")
estimated_checks_per_ray = num_buildings  # Check all
time_per_check = intersection_time  # Just intersection
estimated_total_1 = estimated_checks_per_ray * num_viewpoints * time_per_check
print(f"  {estimated_checks_per_ray} checks per viewpoint")
print(f"  {num_viewpoints} viewpoints")
print(f"  Time: {estimated_total_1:.1f}s ({estimated_total_1/60:.1f} minutes)")
print()

print("Scenario 2: Spatial indexing filters to 1% of buildings")
estimated_checks_per_ray = int(num_buildings * 0.01)  # Only 1% are candidates
time_per_check = intersection_time
estimated_total_2 = estimated_checks_per_ray * num_viewpoints * time_per_check
print(f"  {estimated_checks_per_ray} checks per viewpoint (1%)")
print(f"  {num_viewpoints} viewpoints")
print(f"  Time: {estimated_total_2:.1f}s ({estimated_total_2/60:.1f} minutes)")
print()

print("Scenario 3: Including I/O and geometry processing overhead")
overhead_per_check = 0.001  # millisecond overhead per check
estimated_total_3 = step1_time + (sample_time * (len(df)/sample_size)) + extrapolated_index_time
estimated_total_3 += (estimated_checks_per_ray * num_viewpoints * (time_per_check + overhead_per_check))
print(f"  CSV load: {step1_time:.1f}s")
print(f"  Geometry parsing: {sample_time * (len(df)/sample_size):.1f}s")
print(f"  Index building: {extrapolated_index_time:.1f}s")
print(f"  Visibility checks: {estimated_checks_per_ray * num_viewpoints * (time_per_check + overhead_per_check):.1f}s")
print(f"  Total: {estimated_total_3:.1f}s ({estimated_total_3/60:.1f} minutes)")
print()

print("=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print()
print("If Scenario 3 is > 30 seconds, the bottleneck is likely:")
print("  1. Geometry parsing (if > 10s) - use parallel processing or pre-parse")
print("  2. Spatial indexing (if > 5s) - might be unavoidable, but one-time cost")
print("  3. Visibility checks (if > 15s) - consider ray-box implementation or reduce checks")
print()
