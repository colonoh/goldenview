"""
Optimized visibility analysis with spatial indexing.
Uses R-tree index to filter candidate buildings before ray-box intersection tests.
"""

import pandas as pd
import geopandas as gpd
from shapely import from_wkt, box
import numpy as np
import json
import re
from typing import List, Dict, Tuple
from rtree import index
import time

class VisibilityCheckerOptimized:
    def __init__(self, buildings_csv: str, viewpoints_json: str):
        """
        Initialize the optimized visibility checker with spatial indexing.

        Args:
            buildings_csv: Path to CSV file with building footprints and heights
            viewpoints_json: Path to JSON file with viewpoint locations
        """
        self.gdf = self._load_buildings(buildings_csv)
        self.viewpoints = self._load_viewpoints(viewpoints_json)
        self._build_spatial_index()

    def _load_buildings(self, csv_path: str) -> gpd.GeoDataFrame:
        """Load and parse building data from CSV."""
        df = pd.read_csv(csv_path)

        def fix_wkt(wkt_str):
            coords = re.findall(r'(-?\d+\.\d+)\s+(-?\d+\.\d+)', wkt_str)
            if coords:
                polygon_coords = [[float(lon), float(lat)] for lon, lat in coords]
                # Ensure polygon is closed (first point = last point)
                if polygon_coords[0] != polygon_coords[-1]:
                    polygon_coords.append(polygon_coords[0])
                coord_str = ", ".join([f"{lon} {lat}" for lon, lat in polygon_coords])
                return f"POLYGON (({coord_str}))"
            return wkt_str

        def parse_geometry(wkt_str):
            try:
                geom = from_wkt(fix_wkt(wkt_str))
                # Make valid if needed
                if not geom.is_valid:
                    geom = geom.buffer(0)  # Common fix for invalid geometries
                return geom
            except Exception:
                return None

        df['geometry'] = df['shape'].apply(parse_geometry)

        # Remove rows with invalid geometries
        invalid_count = df['geometry'].isna().sum()
        if invalid_count > 0:
            print(f"Warning: {invalid_count} buildings had invalid geometries and were skipped")
            df = df[df['geometry'].notna()].reset_index(drop=True)

        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

        print(f"Loaded {len(gdf)} buildings (valid geometries)")
        return gdf

    def _load_viewpoints(self, json_path: str) -> List[Dict]:
        """Load viewpoint data from JSON."""
        with open(json_path) as f:
            viewpoints = json.load(f)
        print(f"Loaded {len(viewpoints)} viewpoints")
        return viewpoints

    def _build_spatial_index(self):
        """Build an R-tree spatial index for 2D building footprints."""
        print("Building spatial index...")
        self.rtree_idx = index.Index()

        # Pre-extract bounds for all geometries (faster than row-by-row access)
        bounds_array = np.array([geom.bounds for geom in self.gdf['geometry']])

        for building_idx, bounds in enumerate(bounds_array):
            self.rtree_idx.insert(building_idx, bounds)

        print("Spatial index built")

    def get_building_box_3d(self, building_idx: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get 3D bounding box for a building.

        Returns:
            ((min_lon, min_lat, min_z), (max_lon, max_lat, max_z))
        """
        row = self.gdf.iloc[building_idx]
        geom = row['geometry']
        bounds = geom.bounds  # (minx, miny, maxx, maxy)

        ground_elev = row.get('gnd_min_m', 0)
        height = row['hgt_median_m']

        min_z = ground_elev
        max_z = ground_elev + height

        return (
            (bounds[0], bounds[1], min_z),
            (bounds[2], bounds[3], max_z)
        )

    def ray_box_intersection(self, ray_start: Tuple[float, float, float],
                            ray_end: Tuple[float, float, float],
                            box_min: Tuple[float, float, float],
                            box_max: Tuple[float, float, float]) -> bool:
        """
        Check if a 3D ray intersects an axis-aligned box using the slab method.
        """
        ray_start = np.array(ray_start, dtype=float)
        ray_end = np.array(ray_end, dtype=float)
        box_min = np.array(box_min, dtype=float)
        box_max = np.array(box_max, dtype=float)

        ray_dir = ray_end - ray_start
        t_min = 0.0
        t_max = 1.0

        for i in range(3):  # Check x, y, z dimensions
            if abs(ray_dir[i]) < 1e-9:  # Ray parallel to slab
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

    def get_candidate_buildings(self, ray_start: Tuple[float, float, float],
                                ray_end: Tuple[float, float, float],
                                buffer: float = 0.0001) -> List[int]:
        """
        Use spatial index to find buildings that might intersect the ray.
        Returns building indices to check for occlusion.

        Args:
            ray_start: (lon, lat, elevation) ray origin
            ray_end: (lon, lat, elevation) ray destination
            buffer: Expand the ray bbox by this amount to catch edge cases

        Returns:
            List of candidate building indices
        """
        # Create bounding box for the ray in 2D (ignoring height for initial filter)
        ray_2d_box = box(
            min(ray_start[0], ray_end[0]) - buffer,
            min(ray_start[1], ray_end[1]) - buffer,
            max(ray_start[0], ray_end[0]) + buffer,
            max(ray_start[1], ray_end[1]) + buffer
        )

        # Query spatial index for buildings in this box
        candidates = list(self.rtree_idx.intersection(ray_2d_box.bounds))
        return candidates

    def is_building_visible(self, viewpoint: Dict, target_building_idx: int) -> bool:
        """
        Check if a building is visible from a viewpoint using spatial indexing.

        Args:
            viewpoint: Dict with 'lon', 'lat', 'elevation_m'
            target_building_idx: Index of building to check visibility

        Returns:
            True if building is visible from viewpoint
        """
        target_min, target_max = self.get_building_box_3d(target_building_idx)
        ray_start = (viewpoint['lon'], viewpoint['lat'], viewpoint['elevation_m'])

        target_geom = self.gdf.iloc[target_building_idx]['geometry']
        target_centroid = target_geom.centroid
        target_height = self.gdf.iloc[target_building_idx]['hgt_median_m']
        target_ground = self.gdf.iloc[target_building_idx].get('gnd_min_m', 0)

        ray_end = (target_centroid.x, target_centroid.y, target_ground + target_height)

        # Get candidate buildings using spatial index
        candidate_indices = self.get_candidate_buildings(ray_start, ray_end)

        # Check for occlusion by candidate buildings only
        for other_idx in candidate_indices:
            if other_idx == target_building_idx:
                continue

            other_min, other_max = self.get_building_box_3d(other_idx)

            if self.ray_box_intersection(ray_start, ray_end, other_min, other_max):
                return False

        return True

    def analyze_visibility(self) -> Dict:
        """
        Analyze visibility for all viewpoints and buildings.

        Returns:
            Dict mapping viewpoint names to lists of visible building indices
        """
        import time
        import sys

        results = {}
        total_buildings = len(self.gdf)
        total_viewpoints = len(self.viewpoints)
        overall_start = time.time()

        for vp_idx, vp in enumerate(self.viewpoints):
            vp_name = vp['name']
            visible_buildings = []

            vp_start = time.time()
            print(f"\n[{vp_idx + 1}/{total_viewpoints}] Processing: {vp_name}")
            print(f"{'='*70}")

            for building_idx in range(total_buildings):
                # Print progress every 500 buildings
                if building_idx % 500 == 0 and building_idx > 0:
                    elapsed = time.time() - vp_start
                    rate = building_idx / elapsed
                    remaining = (total_buildings - building_idx) / rate
                    percent = (building_idx / total_buildings) * 100

                    # Calculate overall progress
                    total_checks_done = vp_idx * total_buildings + building_idx
                    total_checks = total_buildings * total_viewpoints
                    overall_percent = (total_checks_done / total_checks) * 100
                    overall_elapsed = time.time() - overall_start
                    overall_rate = total_checks_done / overall_elapsed
                    overall_remaining = (total_checks - total_checks_done) / overall_rate

                    print(f"  [{building_idx}/{total_buildings}] {percent:5.1f}% | "
                          f"{rate:.0f} checks/sec | "
                          f"~{remaining:.0f}s remaining for this viewpoint")
                    print(f"  Overall: {overall_percent:5.1f}% | ~{overall_remaining:.0f}s remaining")
                    sys.stdout.flush()

                if self.is_building_visible(vp, building_idx):
                    visible_buildings.append(building_idx + 1)

            vp_time = time.time() - vp_start
            results[vp_name] = visible_buildings
            print(f"\nCompleted {vp_name}: {len(visible_buildings)} visible buildings in {vp_time:.1f}s")

        overall_time = time.time() - overall_start
        print(f"\n{'='*70}")
        print(f"ALL VIEWPOINTS COMPLETE in {overall_time:.1f}s ({overall_time/60:.1f} minutes)")
        print(f"{'='*70}\n")

        return results


if __name__ == "__main__":
    print("=== Optimized Visibility Checker with Spatial Indexing ===\n")

    start_time = time.time()

    checker = VisibilityCheckerOptimized(
        'data/test_buildings_grid.csv',
        'viewpoints.json'
    )

    print("\n=== Visibility Analysis Results ===\n")
    results = checker.analyze_visibility()

    # Save results to JSON
    with open('visibility_results_optimized.json', 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.3f} seconds")
    print("Results saved to visibility_results_optimized.json")
