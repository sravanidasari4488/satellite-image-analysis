"""
Test script for city-level GEE analysis
Tests the complete pipeline with a sample city
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

def test_city_analysis():
    """Test city-level analysis with a sample city"""
    print("=" * 60)
    print("City-Level GEE Analysis Test")
    print("=" * 60)
    
    try:
        from city_level_gee_analysis import CityLevelGEEAnalysis
        
        # Initialize analyzer
        print("\n[1/5] Initializing GEE analyzer...")
        analyzer = CityLevelGEEAnalysis()
        print("[OK] GEE analyzer initialized")
        
        # Sample city polygon (New York City - simplified)
        # In production, this would come from OpenCage API
        sample_polygon = [
            [-74.0479, 40.6829],  # Southwest
            [-73.9067, 40.6829],  # Southeast
            [-73.9067, 40.8820],   # Northeast
            [-74.0479, 40.8820],  # Northwest
            [-74.0479, 40.6829]   # Close polygon
        ]
        
        print("\n[2/5] Testing polygon conversion...")
        geometry = analyzer.polygon_to_ee_geometry(sample_polygon)
        print(f"[OK] Polygon converted to GEE geometry")
        
        print("\n[3/5] Testing Sentinel-2 image loading...")
        image = analyzer.get_sentinel2_image(geometry, cloud_cover_threshold=20)
        print("[OK] Sentinel-2 image loaded and masked")
        
        print("\n[4/5] Testing spectral index calculation...")
        image_with_indices = analyzer.calculate_spectral_indices(image)
        print("[OK] Spectral indices calculated (NDVI, NDWI, NDBI, EVI)")
        
        print("\n[5/5] Testing land cover classification...")
        classified = analyzer.classify_land_cover_ml(image_with_indices)
        print("[OK] Land cover classified")
        
        print("\n[6/6] Testing area statistics calculation...")
        area_stats = analyzer.calculate_area_statistics(classified, geometry, scale=10)
        print("[OK] Area statistics calculated")
        
        # Display results
        print("\n" + "=" * 60)
        print("Test Results")
        print("=" * 60)
        print(f"Total Area: {area_stats['total_area_km2']} km²")
        print(f"Total Pixels: {area_stats['total_pixels']:,}")
        print(f"\nLand Cover Distribution:")
        
        total_percentage = 0
        for class_name, class_data in area_stats['classes'].items():
            print(f"  {class_name.capitalize():15} {class_data['percentage']:6.2f}%  "
                  f"({class_data['area_km2']:8.2f} km²)")
            total_percentage += class_data['percentage']
        
        print(f"\nTotal Percentage: {total_percentage:.2f}%")
        
        if abs(total_percentage - 100.0) < 1.0:
            print("[OK] Percentages sum to ~100% (within tolerance)")
        else:
            print(f"[WARNING] Percentages sum to {total_percentage:.2f}% (expected ~100%)")
        
        print("\n" + "=" * 60)
        print("[OK] All tests passed!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_city_analysis()
    sys.exit(0 if success else 1)

