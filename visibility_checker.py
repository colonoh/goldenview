"""
Visibility analysis for building footprints.
Determines which buildings are visible from specified viewpoints.
"""

import pandas as pd
import geopandas as gpd
from shapely import from_wkt, box
from shapely.geometry import Point, LineString
import numpy as np
import json
import re
from typing import List, Dict, Tuple

class VisibilityChecker:
    def __init__(self, buildings_csv: str, viewpoints_json: str):
        """
        Initialize the visibility checker with building and viewpoint data.

        Args:
            buildings_csv: Path to CSV file with building footprints and heights
            viewpoints_json: Path to JSON file with viewpoint locations
        """
        self.gdf = self._load_buildings(buildings_csv)
        self.viewpoints = self._load_viewpoints(viewpoints_json)

    def _load_buildings(self, csv_path: str) -> gpd.GeoDataFrame:
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
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

        print(f"Loaded {len(gdf)} buildings")
        return gdf

    def _load_viewpoints(self, json_path: str) -> List[Dict]:
        """Load viewpoint data from JSON."""
        with open(json_path) as f:
            viewpoints = json.load(f)
        print(f"Loaded {len(viewpoints)} viewpoints")
        return viewpoints

    def get_building_box_3d(self, building_idx: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get 3D bounding box for a building.

        Returns:
            ((min_lon, min_lat, min_z), (max_lon, max_lat, max_z))
        """
        row = self.gdf.iloc[building_idx]
        geom = row['geometry']
        bounds = geom.bounds  # (minx, miny, maxx, maxy)

        # Height: from ground (gnd_min_m) to ground + building height
        ground_elev = row.get('gnd_min_m', 0)
        height = row['hgt_median_m']

        min_z = ground_elev
        max_z = ground_elev + height

        return (
            (bounds[0], bounds[1], min_z),  # min corner
            (bounds[2], bounds[3], max_z)   # max corner
        )

    def ray_box_intersection(self, ray_start: Tuple[float, float, float],
                            ray_end: Tuple[float, float, float],
                            box_min: Tuple[float, float, float],
                            box_max: Tuple[float, float, float]) -> bool:
        """
        Check if a 3D ray intersects an axis-aligned box using the slab method.

        Args:
            ray_start: (lon, lat, elevation) of ray origin
            ray_end: (lon, lat, elevation) of ray destination
            box_min: (min_lon, min_lat, min_z) of box
            box_max: (max_lon, max_lat, max_z) of box

        Returns:
            True if ray intersects the box
        """
        ray_start = np.array(ray_start, dtype=float)
        ray_end = np.array(ray_end, dtype=float)
        box_min = np.array(box_min, dtype=float)
        box_max = np.array(box_max, dtype=float)

        ray_dir = ray_end - ray_start

        # Slab intersection method
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

    def is_building_visible(self, viewpoint: Dict, target_building_idx: int) -> bool:
        """
        Check if a building is visible from a viewpoint.

        Args:
            viewpoint: Dict with 'lon', 'lat', 'elevation_m'
            target_building_idx: Index of building to check visibility

        Returns:
            True if building is visible from viewpoint
        """
        # Get target building bounding box
        target_min, target_max = self.get_building_box_3d(target_building_idx)

        # Ray origin (viewpoint)
        ray_start = (viewpoint['lon'], viewpoint['lat'], viewpoint['elevation_m'])

        # Ray destination (center of target building top)
        target_geom = self.gdf.iloc[target_building_idx]['geometry']
        target_centroid = target_geom.centroid
        target_height = self.gdf.iloc[target_building_idx]['hgt_median_m']
        target_ground = self.gdf.iloc[target_building_idx].get('gnd_min_m', 0)

        ray_end = (target_centroid.x, target_centroid.y, target_ground + target_height)

        # Check for occlusion by other buildings
        for other_idx in range(len(self.gdf)):
            if other_idx == target_building_idx:
                continue

            other_min, other_max = self.get_building_box_3d(other_idx)

            # Check if the ray intersects this building
            if self.ray_box_intersection(ray_start, ray_end, other_min, other_max):
                return False  # Building is occluded

        return True  # No occlusion found

    def analyze_visibility(self) -> Dict:
        """
        Analyze visibility for all viewpoints and buildings.

        Returns:
            Dict mapping viewpoint names to lists of visible building indices
        """
        results = {}

        for vp in self.viewpoints:
            vp_name = vp['name']
            visible_buildings = []

            for building_idx in range(len(self.gdf)):
                if self.is_building_visible(vp, building_idx):
                    visible_buildings.append(building_idx + 1)  # 1-indexed for display

            results[vp_name] = visible_buildings
            print(f"{vp_name:10s}: {len(visible_buildings)} visible buildings: {visible_buildings}")

        return results


if __name__ == "__main__":
    # Test the visibility checker
    checker = VisibilityChecker(
        'data/test_buildings_grid.csv',
        'viewpoints.json'
    )

    print("\n=== Visibility Analysis Results ===\n")
    results = checker.analyze_visibility()

    # Save results to JSON
    with open('visibility_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to visibility_results.json")
