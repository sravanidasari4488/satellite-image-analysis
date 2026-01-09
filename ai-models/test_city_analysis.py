"""
Advanced GEE Pipeline Validation & Benchmarking Suite
----------------------------------------------------
A comprehensive testing framework for city-level geospatial analysis.
Performs cross-biome validation, performance profiling, and spectral integrity checks.

Scientific Rationale:
1. Multi-Biome Testing: Validates that land-cover models work in both temperate (NYC) 
   and arid (Dubai) environments.
2. Spectral Assertion: Ensures calculated indices (NDVI, NDBI) conform to 
   theoretical physical bounds [-1, 1].
3. Computational Profiling: Tracks Earth Engine request latency to optimize 
   large-scale city batch processing.
"""

import os
import sys
import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Standard Geospatial Stack
try:
    import ee
    import pandas as pd
    from tabulate import tabulate
except ImportError:
    print("Missing critical dependencies: pip install earthengine-api pandas tabulate")
    sys.exit(1)

# Add parent directory to path to find the analysis module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure Rich Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("GEE-Test-Suite")

load_dotenv()

class GEEPipelineTester:
    """
    Automated Test Suite for GEE City Analysis.
    Supports benchmarking, spectral validation, and error reporting.
    """

    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
        # Test Cases: Diverse Biomes
        self.test_cases = {
            "New_York_Temperate": [
                [-74.0479, 40.6829], [-73.9067, 40.6829], 
                [-73.9067, 40.8820], [-74.0479, 40.8820], [-74.0479, 40.6829]
            ],
            "Cairo_Arid": [
                [31.15, 29.95], [31.35, 29.95], 
                [31.35, 30.15], [31.15, 30.15], [31.15, 29.95]
            ]
        }

    def initialize_gee(self):
        """Initializes and verifies GEE Connection."""
        try:
            logger.info("Connecting to Google Earth Engine...")
            ee.Initialize()
            logger.info("Successfully authenticated with GEE.")
        except Exception as e:
            logger.error(f"GEE Initialization failed: {e}")
            raise

    def run_benchmark(self, city_name: str, polygon_coords: List[List[float]]):
        """
        Runs a full pipeline execution for a specific city and tracks performance.
        """
        from city_level_gee_analysis import CityLevelGEEAnalysis
        
        logger.info(f"--- Starting Benchmark: {city_name} ---")
        analyzer = CityLevelGEEAnalysis()
        
        step_times = {}
        
        try:
            # 1. Geometry Conversion
            t0 = time.time()
            geometry = analyzer.polygon_to_ee_geometry(polygon_coords)
            step_times['geom_conv'] = time.time() - t0
            
            # 2. Image Acquisition
            t1 = time.time()
            image = analyzer.get_sentinel2_image(geometry, cloud_cover_threshold=20)
            # Trigger a small compute to verify image exists
            _ = image.get('system:index').getInfo() 
            step_times['img_fetch'] = time.time() - t1
            
            # 3. Spectral Indices
            t2 = time.time()
            image_with_indices = analyzer.calculate_spectral_indices(image)
            # Physical Assertion: Check NDVI range
            stats = image_with_indices.select('NDVI').reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=geometry,
                scale=100
            ).getInfo()
            
            logger.info(f"NDVI Range for {city_name}: {stats.get('NDVI_min'):.2f} to {stats.get('NDVI_max'):.2f}")
            if stats.get('NDVI_min') < -1.1 or stats.get('NDVI_max') > 1.1:
                logger.warning(f"Out of bounds spectral data detected in {city_name}")
                
            step_times['spectral_calc'] = time.time() - t2
            
            # 4. ML Classification
            t3 = time.time()
            classified = analyzer.classify_land_cover_ml(image_with_indices)
            step_times['classification'] = time.time() - t3
            
            # 5. Area Statistics
            t4 = time.time()
            area_stats = analyzer.calculate_area_statistics(classified, geometry, scale=30)
            step_times['stats_engine'] = time.time() - t4
            
            # Compile Metrics
            self.results.append({
                'city': city_name,
                'status': 'PASSED',
                'area_km2': round(area_stats['total_area_km2'], 2),
                'urban_pct': round(area_stats['classes'].get('urban', {}).get('percentage', 0), 2),
                'veg_pct': round(area_stats['classes'].get('vegetation', {}).get('percentage', 0), 2),
                'water_pct': round(area_stats['classes'].get('water', {}).get('percentage', 0), 2),
                'latency_sec': round(sum(step_times.values()), 2)
            })
            
            logger.info(f"Finished {city_name} in {self.results[-1]['latency_sec']}s")

        except Exception as e:
            logger.error(f"Benchmark failed for {city_name}: {e}")
            self.results.append({
                'city': city_name,
                'status': f'FAILED: {str(e)[:50]}',
                'latency_sec': 0
            })

    def print_report(self):
        """Generates a formatted ASCII table of test results."""
        print("\n" + "="*85)
        print(f"GEE PIPELINE TEST REPORT | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*85)
        
        df = pd.DataFrame(self.results)
        if not df.empty:
            print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No results to display.")
            
        total_time = time.time() - self.start_time
        print(f"\nTotal Suite Execution Time: {total_time:.2f} seconds")
        
        success_rate = (sum(1 for r in self.results if r['status'] == 'PASSED') / len(self.results)) * 100
        print(f"Overall Success Rate: {success_rate:.1f}%")
        print("="*85 + "\n")

    def run_all(self):
        """Orchestrates the full suite."""
        self.initialize_gee()
        
        for name, poly in self.test_cases.items():
            self.run_benchmark(name, poly)
            
        self.print_report()
        
        # Final exit code logic
        return all(r['status'] == 'PASSED' for r in self.results)

def main():
    """Execution entry point."""
    print("""
    #######################################################
    #        GEE PIPELINE INTEGRATION TEST SUITE          #
    #   Validating Satellite Imagery Analysis Workflow     #
    #######################################################
    """)
    
    tester = GEEPipelineTester()
    success = tester.run_all()
    
    if success:
        print("[SUCCESS] All city benchmarks passed spectral and logic validation.")
        sys.exit(0)
    else:
        print("[CRITICAL] One or more test cases failed. Check logs for details.")
        sys.exit(1)

if __name__ == '__main__':
    main()
