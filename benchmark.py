"""
Benchmark comparison between original and optimized visibility checkers.
Also estimates performance on full 180k building dataset.
"""

import time
from visibility_checker import VisibilityChecker
from visibility_checker_optimized import VisibilityCheckerOptimized

def benchmark_original():
    """Benchmark the original visibility checker."""
    print("Benchmarking ORIGINAL checker (no spatial index)...")
    start = time.time()
    checker = VisibilityChecker('data/test_buildings_grid.csv', 'viewpoints.json')
    load_time = time.time() - start

    start = time.time()
    results = checker.analyze_visibility()
    analysis_time = time.time() - start

    total_time = load_time + analysis_time
    print(f"  Load time: {load_time:.4f}s")
    print(f"  Analysis time: {analysis_time:.4f}s")
    print(f"  Total time: {total_time:.4f}s\n")
    return total_time

def benchmark_optimized():
    """Benchmark the optimized visibility checker."""
    print("Benchmarking OPTIMIZED checker (with R-tree spatial index)...")
    start = time.time()
    checker = VisibilityCheckerOptimized('data/test_buildings_grid.csv', 'viewpoints.json')
    load_time = time.time() - start

    start = time.time()
    results = checker.analyze_visibility()
    analysis_time = time.time() - start

    total_time = load_time + analysis_time
    print(f"  Load time: {load_time:.4f}s")
    print(f"  Analysis time: {analysis_time:.4f}s")
    print(f"  Total time: {total_time:.4f}s\n")
    return total_time

if __name__ == "__main__":
    print("=" * 60)
    print("VISIBILITY CHECKER BENCHMARK")
    print("=" * 60)
    print()

    # Run benchmarks
    original_time = benchmark_original()
    optimized_time = benchmark_optimized()

    # Calculate speedup
    speedup = original_time / optimized_time
    print(f"Speedup: {speedup:.2f}x")
    print()

    # Estimate for full dataset
    print("=" * 60)
    print("ESTIMATED PERFORMANCE ON FULL DATASET")
    print("=" * 60)
    print()

    num_buildings_test = 9
    num_buildings_full = 180000
    num_viewpoints = 10

    scale_factor = num_buildings_full / num_buildings_test

    print(f"Test dataset: {num_buildings_test} buildings, {num_viewpoints} viewpoints")
    print(f"Full dataset: {num_buildings_full} buildings, {num_viewpoints} viewpoints")
    print(f"Scale factor: {scale_factor:.0f}x")
    print()

    # Naive scaling (original approach)
    estimated_original = original_time * scale_factor * num_viewpoints / num_viewpoints
    print(f"Original approach (no spatial index):")
    print(f"  Estimated time: {estimated_original:.1f} seconds ({estimated_original/60:.1f} minutes)")
    print()

    # Optimized approach with spatial indexing
    # Spatial index overhead scales linearly with dataset
    # But ray filtering becomes much more effective with larger datasets
    # Conservative estimate: sqrt(scale_factor) effectiveness improvement
    estimated_optimized_base = optimized_time * scale_factor * num_viewpoints / num_viewpoints
    # With spatial indexing, only nearby buildings are checked
    # Assuming ~1% of buildings are actually candidates per ray in large dataset
    candidate_ratio = 0.01
    estimated_optimized = estimated_optimized_base * candidate_ratio

    print(f"Optimized approach (with R-tree spatial index):")
    print(f"  Base time (no spatial filtering): {estimated_optimized_base:.1f} seconds")
    print(f"  With spatial filtering (~{candidate_ratio*100:.0f}% candidates): {estimated_optimized:.1f} seconds ({estimated_optimized/60:.1f} minutes)")
    print()

    estimated_speedup = estimated_original / estimated_optimized
    print(f"Estimated speedup on full dataset: {estimated_speedup:.0f}x")
    print()
    print(f"Full analysis should complete in approximately {estimated_optimized/60:.1f} - {estimated_optimized/30:.1f} minutes")
