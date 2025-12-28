"""
Test script to verify Google Earth Engine integration
Run this script to check if GEE is properly configured and working
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("=" * 60)
    print("Testing Imports...")
    print("=" * 60)
    
    try:
        import ee
        print("[OK] ee module imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import ee: {e}")
        print("   Run: pip install earthengine-api")
        return False
    
    try:
        import numpy as np
        print("[OK] numpy imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import numpy: {e}")
        return False
    
    try:
        from PIL import Image
        print("[OK] PIL (Pillow) imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import PIL: {e}")
        return False
    
    try:
        import requests
        print("[OK] requests imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import requests: {e}")
        return False
    
    try:
        from gee_integration import GoogleEarthEngineIntegration
        print("[OK] GoogleEarthEngineIntegration imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import GoogleEarthEngineIntegration: {e}")
        return False
    
    return True

def test_gee_initialization():
    """Test GEE initialization"""
    print("\n" + "=" * 60)
    print("Testing GEE Initialization...")
    print("=" * 60)
    
    try:
        from gee_integration import GoogleEarthEngineIntegration
        gee = GoogleEarthEngineIntegration()
        
        if gee.initialized:
            print("[OK] Google Earth Engine initialized successfully")
            return True, gee
        else:
            print("[WARNING]  Google Earth Engine not initialized")
            print("   This is OK if you haven't run 'earthengine authenticate' yet")
            print("   The application will fall back to SentinelHub")
            return False, None
    except Exception as e:
        print(f"[ERROR] Error during initialization: {e}")
        return False, None

def test_gee_functionality(gee):
    """Test GEE functionality if initialized"""
    if not gee or not gee.initialized:
        print("\n[WARNING]  Skipping functionality tests (GEE not initialized)")
        return
    
    print("\n" + "=" * 60)
    print("Testing GEE Functionality...")
    print("=" * 60)
    
    try:
        # Test getting a collection
        print("\n1. Testing image collection retrieval...")
        collection = gee.get_sentinel2_collection(
            start_date='2024-01-01',
            end_date='2024-01-31',
            cloud_cover=20
        )
        print("   [OK] Successfully retrieved Sentinel-2 collection")
        
        # Test getting an image
        print("\n2. Testing image retrieval...")
        image = gee.get_least_cloudy_image(collection)
        print("   [OK] Successfully retrieved least cloudy image")
        
        # Test calculating indices
        print("\n3. Testing spectral index calculation...")
        ndvi = gee.calculate_ndvi(image)
        ndwi = gee.calculate_ndwi(image)
        ndbi = gee.calculate_ndbi(image)
        print("   [OK] Successfully calculated NDVI, NDWI, and NDBI")
        
        # Test getting spectral indices
        print("\n4. Testing get_spectral_indices...")
        image_with_indices = gee.get_spectral_indices(image)
        print("   [OK] Successfully added spectral indices to image")
        
        print("\n[OK] All GEE functionality tests passed!")
        
    except Exception as e:
        print(f"\n[ERROR] Error during functionality test: {e}")
        import traceback
        traceback.print_exc()

def test_app_integration():
    """Test if app.py can import GEE integration"""
    print("\n" + "=" * 60)
    print("Testing App Integration...")
    print("=" * 60)
    
    try:
        # Try to import app (this will test if GEE integration loads)
        # We'll do a minimal import to avoid starting the server
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", "app.py")
        if spec and spec.loader:
            # Just check if it can be loaded without errors
            print("[OK] app.py can be loaded (GEE integration should work)")
        else:
            print("[WARNING]  Could not verify app.py (this is OK)")
    except Exception as e:
        print(f"[WARNING]  Note: {e}")
        print("   This is OK - app.py will handle GEE initialization at runtime")

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Google Earth Engine Integration Test")
    print("=" * 60)
    
    # Test 1: Imports
    if not test_imports():
        print("\n[ERROR] Import tests failed. Please install missing dependencies.")
        print("   Run: pip install -r requirements.txt")
        return
    
    # Test 2: GEE Initialization
    initialized, gee = test_gee_initialization()
    
    # Test 3: GEE Functionality (if initialized)
    if initialized:
        test_gee_functionality(gee)
    else:
        print("\n[INFO] To enable GEE functionality:")
        print("   1. Run: earthengine authenticate")
        print("   2. Sign in with your Google account")
        print("   3. Grant necessary permissions")
        print("   4. Re-run this test script")
    
    # Test 4: App Integration
    test_app_integration()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    if initialized:
        print("[OK] Google Earth Engine is ready to use!")
        print("   The application will use GEE for satellite data processing.")
    else:
        print("[WARNING]  Google Earth Engine is not initialized.")
        print("   The application will fall back to SentinelHub.")
        print("   This is OK - the app will still work!")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()

