"""
Test script to run after completing Earth Engine registration
"""

import os
from dotenv import load_dotenv
import ee

def test_gee():
    """Test Google Earth Engine after registration"""
    print("=" * 60)
    print("Testing Google Earth Engine After Registration")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv('ai-models/.env')
    project_id = os.getenv('GEE_PROJECT_ID')
    
    if not project_id:
        print("[ERROR] GEE_PROJECT_ID not found in .env file")
        return False
    
    print(f"\n[INFO] Project ID: {project_id}")
    print("[INFO] Attempting to initialize...")
    
    try:
        # Initialize Earth Engine
        ee.Initialize(project=project_id)
        print("[OK] Google Earth Engine initialized successfully!")
        
        # Test accessing a Sentinel-2 collection
        print("\n[INFO] Testing data access...")
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1)
        count = collection.size().getInfo()
        print(f"[OK] Successfully accessed Sentinel-2 collection! (Found {count} images)")
        
        # Test getting an image
        print("\n[INFO] Testing image retrieval...")
        image = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20220101T000241_20220101T000238_T60HVE')
        image_id = image.get('system:id').getInfo()
        print(f"[OK] Successfully retrieved image: {image_id}")
        
        print("\n" + "=" * 60)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 60)
        print("\nGoogle Earth Engine is fully operational!")
        print("Your application will now use GEE for satellite data processing.")
        return True
        
    except ee.ee_exception.EEException as e:
        error_msg = str(e)
        if "not registered" in error_msg.lower():
            print("\n[WARNING] Project is still not registered")
            print("[INFO] Please complete the registration process:")
            print("   1. Choose commercial or non-commercial use")
            print("   2. Complete the registration form")
            print("   3. Wait a few minutes for registration to propagate")
            print("   4. Run this test again")
        else:
            print(f"\n[ERROR] Earth Engine error: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return False

if __name__ == '__main__':
    test_gee()

