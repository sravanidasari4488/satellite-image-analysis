# Google Earth Engine Authentication Guide

## Step 1: Sign up for Google Earth Engine

1. Visit https://earthengine.google.com/
2. Click "Sign up" and register with your Google account
3. Wait for approval (usually takes 1-2 business days)

## Step 2: Authenticate

Once your account is approved, authenticate using one of these methods:

### Method 1: Using Python (Recommended)

Open PowerShell or Command Prompt in the project directory and run:

```powershell
python -c "import ee; ee.Authenticate()"
```

This will:
1. Open your default web browser
2. Ask you to sign in with your Google account
3. Generate an authentication token
4. Save the credentials automatically

### Method 2: Using Command Line Tool

If you have the `earthengine` command installed:

```bash
earthengine authenticate
```

### Method 3: Manual Authentication

1. Visit: https://code.earthengine.google.com/
2. Sign in with your Google account
3. Go to "Settings" → "Authentication"
4. Follow the instructions to generate credentials

## Step 3: Set Up Google Cloud Project

Earth Engine requires a Google Cloud project. You have two options:

### Option A: Use Existing Project (Recommended)

1. Visit https://code.earthengine.google.com/
2. Sign in with your Google account
3. You'll see your default project or can create one
4. Copy the project ID (it looks like: `ee-your-project-name`)

Then set it as an environment variable:

**PowerShell:**
```powershell
$env:EARTHENGINE_PROJECT="your-project-id"
```

**Or add to `.env` file:**
```
EARTHENGINE_PROJECT=your-project-id
```

### Option B: Create New Project

1. Visit https://console.cloud.google.com/
2. Create a new project or select an existing one
3. Enable the Earth Engine API for that project
4. Visit https://code.earthengine.google.com/ and link your project
5. Set the project ID as shown in Option A

## Step 4: Verify Setup

Test that everything works by running:

```powershell
python authenticate_earth_engine.py
```

Or manually:
```python
python -c "import ee; ee.Initialize(); print('Earth Engine initialized successfully!')"
```

## Troubleshooting

### Error: "no project found"

If you see this error, you need to:
1. Visit https://code.earthengine.google.com/
2. Create a new project or select an existing one
3. Make sure your Google Cloud project is linked to Earth Engine

### Error: "Authentication failed"

1. Make sure you've signed up and been approved for Earth Engine
2. Try running the authentication command again
3. Clear cached credentials and re-authenticate:
   ```python
   python -c "import ee; ee.Authenticate(auth_mode='notebook')"
   ```

### Credentials Location

Credentials are typically stored in:
- Windows: `C:\Users\<username>\.config\earthengine\credentials`
- Linux/Mac: `~/.config/earthengine/credentials`

## Need Help?

- Earth Engine Documentation: https://developers.google.com/earth-engine
- Earth Engine Forum: https://groups.google.com/g/google-earth-engine-developers

