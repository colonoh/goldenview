#!/usr/bin/env python3
"""
Full dataset visibility analysis.
Analyzes which buildings are visible from 4 iconic San Francisco viewpoints.
"""

import time
import sys
import json
from visibility_checker_optimized import VisibilityCheckerOptimized

def main():
    print("=" * 70)
    print("FULL DATASET VISIBILITY ANALYSIS")
    print("=" * 70)
    print()

    start_total = time.time()

    try:
        print("Loading full dataset and viewpoints...")
        start = time.time()

        checker = VisibilityCheckerOptimized(
            'data/Building_Footprints_20251130.csv',
            'viewpoints_real.json'
        )

        load_time = time.time() - start
        print(f"Load time: {load_time:.2f}s\n")

        print("Running visibility analysis on all buildings and viewpoints...")
        print("This may take several minutes...\n")
        start = time.time()

        results = checker.analyze_visibility()

        analysis_time = time.time() - start

        print()
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print()

        total_visible = 0
        for viewpoint_name in sorted(results.keys()):
            visible_buildings = results[viewpoint_name]
            num_visible = len(visible_buildings)
            total_visible += num_visible
            print(f"{viewpoint_name:40s}: {num_visible:8d} visible buildings")

        print()
        print(f"{'Total visible (sum)':40s}: {total_visible:8d}")
        print()

        # Timing summary
        total_time = load_time + analysis_time
        print("=" * 70)
        print("TIMING")
        print("=" * 70)
        print(f"Load time:     {load_time:8.2f}s")
        print(f"Analysis time: {analysis_time:8.2f}s ({analysis_time/60:.1f} minutes)")
        print(f"Total time:    {total_time:8.2f}s ({total_time/60:.1f} minutes)")
        print()

        # Save results
        with open('visibility_results_full.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("Results saved to visibility_results_full.json")
        print()
        print("=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
