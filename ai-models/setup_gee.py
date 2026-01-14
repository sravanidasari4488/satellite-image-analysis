"""
Setup script for Google Earth Engine
This script helps configure GEE with proper project settings
"""

import os
import ee

def setup_gee():
    """Setup Google Earth Engine"""
    print("=" * 60)
    print("Google Earth Engine Setup")
    print("=" * 60)
    
    # Check if already authenticated
    try:
        ee.Initialize()
        print("[OK] Google Earth Engine is already initialized!")
        return True
    except Exception as e:
        print(f"[INFO] Not initialized yet: {e}")
    
    # Try to authenticate
    print("\n[INFO] Starting authentication...")
    try:
        ee.Authenticate()
        print("[OK] Authentication successful!")
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        print("\nPlease visit: https://code.earthengine.google.com/")
        print("Sign in and accept the terms to enable Earth Engine API")
        return False
    
    # Try to initialize with a default project
    # For user accounts, you can use 'earthengine-api' as a default
    print("\n[INFO] Attempting to initialize...")
    
    # Option 1: Try without project (may work if project is set in credentials)
    try:
        ee.Initialize()
        print("[OK] Initialized successfully without explicit project!")
        return True
    except Exception as e:
        print(f"[INFO] Initialization without project failed: {e}")
    
    # Option 2: Try with a common project name
    # Users can set their own project ID
    project_id = os.getenv('GEE_PROJECT_ID')
    if not project_id:
        print("\n[INFO] No GEE_PROJECT_ID found in environment.")
        print("[INFO] You can:")
        print("  1. Set GEE_PROJECT_ID environment variable")
        print("  2. Or use your Google Cloud project ID")
        print("  3. Or leave it empty - the app will work with fallback")
        
        # Ask user for project ID
        user_input = input("\nEnter your Google Cloud Project ID (or press Enter to skip): ").strip()
        if user_input:
            project_id = user_input
            print(f"[INFO] Using project ID: {project_id}")
            # Save to .env file
            env_file = os.path.join(os.path.dirname(__file__), '.env')
            with open(env_file, 'a') as f:
                f.write(f'\nGEE_PROJECT_ID={project_id}\n')
            print(f"[OK] Saved to {env_file}")
    
    if project_id:
        try:
            ee.Initialize(project=project_id)
            print(f"[OK] Initialized successfully with project: {project_id}")
            return True
        except Exception as e:
            print(f"[WARNING] Initialization with project failed: {e}")
            print("[INFO] This is OK - the app will use SentinelHub as fallback")
            return False
    
    print("\n[WARNING] Could not initialize GEE with a project.")
    print("[INFO] The application will still work using SentinelHub as fallback.")
    print("[INFO] To use GEE later, set GEE_PROJECT_ID in your .env file")
    return False

if __name__ == '__main__':
    setup_gee()

