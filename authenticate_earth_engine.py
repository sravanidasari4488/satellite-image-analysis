"""
Google Earth Engine Authentication Helper Script
Run this script to authenticate with Google Earth Engine
"""

import ee
import sys
import os

print("=" * 60)
print("Google Earth Engine Authentication")
print("=" * 60)
print()

try:
    print("Step 1: Attempting to authenticate...")
    print("This will open your web browser for authentication.")
    print()
    
    # Authenticate
    ee.Authenticate()
    
    print("✓ Authentication successful!")
    print()
    
    print("Step 2: Initializing Earth Engine...")
    
    # Try to get project from environment variable
    project = os.getenv('EARTHENGINE_PROJECT')
    
    if project:
        print(f"Using project: {project}")
        ee.Initialize(project=project)
    else:
        try:
            ee.Initialize()
        except Exception as init_error:
            if "no project found" in str(init_error).lower():
                print("⚠ No Google Cloud project found.")
                print()
                print("You need to set up a Google Cloud project. Here's how:")
                print()
                print("OPTION 1: Set project via environment variable")
                print("  Set EARTHENGINE_PROJECT environment variable:")
                print("  PowerShell: $env:EARTHENGINE_PROJECT='your-project-id'")
                print("  Or add to .env file: EARTHENGINE_PROJECT=your-project-id")
                print()
                print("OPTION 2: Set up project in Google Cloud Console")
                print("  1. Visit: https://console.cloud.google.com/")
                print("  2. Create a new project or select existing one")
                print("  3. Enable Earth Engine API for that project")
                print("  4. Visit: https://code.earthengine.google.com/")
                print("  5. Link your project to Earth Engine")
                print()
                print("OPTION 3: Use default project (if available)")
                print("  Visit: https://code.earthengine.google.com/")
                print("  Your default project should be listed there")
                print()
                raise init_error
            else:
                raise init_error
    
    print("✓ Earth Engine initialized successfully!")
    print()
    print("You can now use the geospatial intelligence system.")
    print()
    
except Exception as e:
    print("✗ Setup incomplete!")
    print()
    print("Error:", str(e))
    print()
    print("Please make sure:")
    print("1. You have signed up for Google Earth Engine at https://earthengine.google.com/")
    print("2. Your account has been approved (usually takes 1-2 business days)")
    print("3. You have a Google Cloud project set up and linked")
    print()
    print("For more help, see EARTH_ENGINE_SETUP.md")
    sys.exit(1)

