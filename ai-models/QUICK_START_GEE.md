# Quick Start: Google Earth Engine Setup

## Fast Setup (5 minutes)

### Step 1: Get Your Google Cloud Project ID

1. Go to https://console.cloud.google.com/
2. Create or select a project
3. Copy your **Project ID** (looks like `my-project-123456`)
4. Enable Earth Engine API:
   - Go to "APIs & Services" > "Library"
   - Search "Earth Engine API" > Click "Enable"

### Step 2: Authenticate

Run this command:
```bash
python -c "import ee; ee.Authenticate()"
```

Follow the browser prompts to sign in.

### Step 3: Set Project ID

Add to `ai-models/.env`:
```env
GEE_PROJECT_ID=your-project-id-here
```

Replace `your-project-id-here` with your actual project ID from Step 1.

### Step 4: Test

Run:
```bash
python -c "import ee; ee.Initialize(project='your-project-id'); print('Success!')"
```

## That's It! 🎉

Your application will now use Google Earth Engine automatically.

**Note**: If GEE is not configured, the app automatically falls back to SentinelHub - it will still work!

## Troubleshooting

**Error: "no project found"**
- Make sure you set `GEE_PROJECT_ID` in your `.env` file
- Verify the project ID is correct in Google Cloud Console

**Error: "Authentication failed"**
- Make sure you've enabled Earth Engine API in Google Cloud Console
- Try running authentication again: `python -c "import ee; ee.Authenticate()"`

**Still having issues?**
- The app will work with SentinelHub as fallback
- Check `ai-models/GEE_SETUP.md` for detailed instructions

