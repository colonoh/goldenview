"""
Instrumented visibility checker with detailed timing for each step.
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

class VisibilityCheckerInstrumented:
    def __init__(self, buildings_csv: str, viewpoints_json: str):
        """Initialize with detailed timing."""
        print(f"[INIT] Starting initialization...")
        start = time.time()

        self.gdf = self._load_buildings(buildings_csv)
        load_time = time.time() - start
        print(f"[LOAD] Buildings loaded in {load_time:.3f}s")

        start = time.time()
        self.viewpoints = self._load_viewpoints(viewpoints_json)
        vp_time = time.time() - start
        print(f"[LOAD] Viewpoints loaded in {vp_time:.3f}s")

        start = time.time()
        self._build_spatial_index()
        index_time = time.time() - start
        print(f"[INDEX] Spatial index built in {index_time:.3f}s")

        print(f"[INIT] Total init time: {load_time + vp_time + index_time:.3f}s\n")

    def _load_buildings(self, csv_path: str) -> gpd.GeoDataFrame:
        """Load and parse building data from CSV."""
        df = pd.read_csv(csv_path)

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
        invalid_count = df['geometry'].isna().sum()
        if invalid_count > 0:
            print(f"[WARN] {invalid_count} buildings had invalid geometries")
            df = df[df['geometry'].notna()].reset_index(drop=True)

        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        return gdf

    def _load_viewpoints(self, json_path: str) -> List[Dict]:
        """Load viewpoint data from JSON."""
        with open(json_path) as f:
            viewpoints = json.load(f)
        return viewpoints

    def _build_spatial_index(self):
        """Build an R-tree spatial index for 2D building footprints."""
        self.rtree_idx = index.Index()
        for building_idx, (_, row) in enumerate(self.gdf.iterrows()):
            bounds = row['geometry'].bounds
            self.rtree_idx.insert(building_idx, bounds)

    def get_building_box_3d(self, building_idx: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get 3D bounding box for a building."""
        row = self.gdf.iloc[building_idx]
        geom = row['geometry']
        bounds = geom.bounds
        ground_elev = row.get('gnd_min_m', 0)
        height = row['hgt_median_m']
        return ((bounds[0], bounds[1], ground_elev), (bounds[2], bounds[3], ground_elev + height))

    def ray_box_intersection(self, ray_start: Tuple[float, float, float],
                            ray_end: Tuple[float, float, float],
                            box_min: Tuple[float, float, float],
                            box_max: Tuple[float, float, float]) -> bool:
        """Check if a 3D ray intersects an axis-aligned box."""
        ray_start = np.array(ray_start, dtype=float)
        ray_end = np.array(ray_end, dtype=float)
        box_min = np.array(box_min, dtype=float)
        box_max = np.array(box_max, dtype=float)

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

    def get_candidate_buildings(self, ray_start: Tuple[float, float, float],
                                ray_end: Tuple[float, float, float],
                                buffer: float = 0.0001) -> List[int]:
        """Use spatial index to find candidate buildings."""
        ray_2d_box = box(
            min(ray_start[0], ray_end[0]) - buffer,
            min(ray_start[1], ray_end[1]) - buffer,
            max(ray_start[0], ray_end[0]) + buffer,
            max(ray_start[1], ray_end[1]) + buffer
        )
        candidates = list(self.rtree_idx.intersection(ray_2d_box.bounds))
        return candidates

    def is_building_visible(self, viewpoint: Dict, target_building_idx: int) -> bool:
        """Check if a building is visible from a viewpoint."""
        target_min, target_max = self.get_building_box_3d(target_building_idx)
        ray_start = (viewpoint['lon'], viewpoint['lat'], viewpoint['elevation_m'])

        target_geom = self.gdf.iloc[target_building_idx]['geometry']
        target_centroid = target_geom.centroid
        target_height = self.gdf.iloc[target_building_idx]['hgt_median_m']
        target_ground = self.gdf.iloc[target_building_idx].get('gnd_min_m', 0)

        ray_end = (target_centroid.x, target_centroid.y, target_ground + target_height)

        candidate_indices = self.get_candidate_buildings(ray_start, ray_end)

        for other_idx in candidate_indices:
            if other_idx == target_building_idx:
                continue
            other_min, other_max = self.get_building_box_3d(other_idx)
            if self.ray_box_intersection(ray_start, ray_end, other_min, other_max):
                return False

        return True

    def analyze_visibility(self) -> Dict:
        """Analyze visibility with detailed timing."""
        results = {}

        for vp_idx, vp in enumerate(self.viewpoints):
            vp_name = vp['name']
            print(f"[VIS] Viewpoint {vp_idx+1}/{len(self.viewpoints)}: {vp_name}")

            start_vp = time.time()
            visible_buildings = []

            num_buildings = len(self.gdf)
            print_interval = max(1000, num_buildings // 10)  # Print every 10%

            for building_idx in range(num_buildings):
                if building_idx % print_interval == 0 and building_idx > 0:
                    elapsed = time.time() - start_vp
                    rate = building_idx / elapsed
                    remaining = (num_buildings - building_idx) / rate
                    print(f"  [{building_idx}/{num_buildings}] {rate:.0f} checks/sec, ~{remaining:.1f}s remaining")

                if self.is_building_visible(vp, building_idx):
                    visible_buildings.append(building_idx + 1)

            vp_time = time.time() - start_vp
            print(f"  Completed in {vp_time:.2f}s, {len(visible_buildings)} visible")
            print()

            results[vp_name] = visible_buildings

        return results


if __name__ == "__main__":
    print("=" * 70)
    print("INSTRUMENTED VISIBILITY CHECKER - Russian Hill")
    print("=" * 70)
    print()

    checker = VisibilityCheckerInstrumented(
        'data/Building_Footprints_RussianHill_10k.csv',
        'viewpoints_real.json'
    )

    start_total = time.time()
    results = checker.analyze_visibility()
    total_time = time.time() - start_total

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    for vp_name in results:
        print(f"{vp_name:40s}: {len(results[vp_name]):6d} visible")

    print()
    print(f"Total analysis time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print()

    with open('visibility_results_russian_hill_debug.json', 'w') as f:
        json.dump(results, f, indent=2)
