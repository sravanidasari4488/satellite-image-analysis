# 🚀 How to Start the Application

This guide will help you start all services for the Satellite Image Analysis application.

## Prerequisites

Before starting, ensure you have:
- ✅ Node.js installed (v18 or higher)
- ✅ Python 3.8+ installed
- ✅ MongoDB Atlas account (or local MongoDB)
- ✅ All environment variables configured

## Quick Start (All Services)

### Option 1: Start Everything at Once (Recommended)

Open **3 separate terminal windows** and run:

#### Terminal 1: AI Service (Flask)
```powershell
cd C:\Users\Sravani\Desktop\Satellite_Image_Analysis\ai-models
.\venv\Scripts\Activate.ps1
python app.py
```

**Expected output:**
```
 * Running on http://127.0.0.1:5001
 * Model loaded successfully
```

#### Terminal 2: Backend (Node.js)
```powershell
cd C:\Users\Sravani\Desktop\Satellite_Image_Analysis\backend
npm run dev
```

**Expected output:**
```
Server running on port 5000
MongoDB connected
```

#### Terminal 3: Frontend (React)
```powershell
cd C:\Users\Sravani\Desktop\Satellite_Image_Analysis\frontend
npm start
```

**Expected output:**
```
Compiled successfully!
Local: http://localhost:3000
```

### Option 2: Using Root Scripts

You can also use the root package.json scripts:

```powershell
# Terminal 1: AI Service
cd C:\Users\Sravani\Desktop\Satellite_Image_Analysis
npm run ai-service

# Terminal 2: Backend + Frontend (together)
cd C:\Users\Sravani\Desktop\Satellite_Image_Analysis
npm run dev
```

## Service URLs

Once all services are running:

| Service | URL | Status |
|---------|-----|--------|
| **Frontend** | http://localhost:3000 | Main application |
| **Backend API** | http://localhost:5000 | REST API |
| **AI Service** | http://localhost:5001 | ML/AI endpoints |

## Step-by-Step Startup

### Step 1: Start AI Service (Python/Flask)

```powershell
# Navigate to AI models directory
cd C:\Users\Sravani\Desktop\Satellite_Image_Analysis\ai-models

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start Flask service
python app.py
```

**What to check:**
- ✅ Port 5001 is available
- ✅ Model loads: `INFO:multispectral_analysis:Loaded multispectral model from models/multispectral_landcover_model.h5`
- ✅ No errors in console

### Step 2: Start Backend (Node.js/Express)

```powershell
# Navigate to backend directory
cd C:\Users\Sravani\Desktop\Satellite_Image_Analysis\backend

# Install dependencies (if not already done)
npm install

# Start development server
npm run dev
```

**What to check:**
- ✅ Port 5000 is available
- ✅ MongoDB connection successful
- ✅ No errors in console

### Step 3: Start Frontend (React)

```powershell
# Navigate to frontend directory
cd C:\Users\Sravani\Desktop\Satellite_Image_Analysis\frontend

# Install dependencies (if not already done)
npm install

# Start React development server
npm start
```

**What to check:**
- ✅ Port 3000 is available
- ✅ Browser opens automatically
- ✅ No compilation errors

## Environment Variables

Make sure these are configured:

### Backend (.env in `backend/` folder)
```env
PORT=5000
MONGODB_URI=your_mongodb_connection_string
JWT_SECRET=your_jwt_secret
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
OPENCAGE_API_KEY=your_opencage_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
AI_SERVICE_URL=http://localhost:5001
```

### AI Service (.env in `ai-models/` folder)
```env
PORT=5001
DEBUG=False
```

### Frontend (.env in `frontend/` folder)
```env
REACT_APP_API_URL=http://localhost:5000/api
```

## Troubleshooting

### Port Already in Use

**Error:** `Port 3000 is already in use` or similar

**Solution:**
```powershell
# Find process using the port
netstat -ano | findstr :3000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

Or change the port in the respective configuration files.

### AI Service Not Starting

**Check:**
1. Virtual environment is activated
2. All Python dependencies installed: `pip install -r requirements.txt`
3. Model file exists: `ai-models/models/multispectral_landcover_model.h5`

### Backend Connection Errors

**Check:**
1. MongoDB URI is correct in `backend/.env`
2. MongoDB Atlas IP whitelist includes your IP
3. Network connection is active

### Frontend Can't Connect to Backend

**Check:**
1. Backend is running on port 5000
2. `REACT_APP_API_URL` in `frontend/.env` is correct
3. CORS is enabled in backend (should be automatic)

## Verification

Once all services are running, verify they're working:

### 1. Check AI Service
```powershell
curl http://localhost:5001/health
```
Or visit: http://localhost:5001/health

### 2. Check Backend
```powershell
curl http://localhost:5000/api/health
```
Or visit: http://localhost:5000/api/health

### 3. Check Frontend
Open browser: http://localhost:3000

You should see the Satellite Image Analysis application.

## Development Workflow

### Hot Reload
- **Frontend**: Automatically reloads on file changes
- **Backend**: Automatically restarts with nodemon
- **AI Service**: Requires manual restart (or use Flask debug mode)

### Stopping Services

Press `Ctrl+C` in each terminal to stop the respective service.

## Production Build

For production deployment:

```powershell
# Build frontend
cd frontend
npm run build

# Start backend (production mode)
cd backend
npm start

# Start AI service (use gunicorn for production)
cd ai-models
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `npm run dev` | Start backend + frontend |
| `npm run ai-service` | Start AI service |
| `npm run install-all` | Install all dependencies |
| `npm run build` | Build frontend for production |

## Need Help?

- Check logs in each terminal for error messages
- Verify environment variables are set correctly
- Ensure all dependencies are installed
- Check that ports are not in use by other applications

---

**Status**: ✅ Ready to start all services!






