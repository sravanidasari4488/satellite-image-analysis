# Environment Setup Instructions

## Backend Environment Setup

Create a file named `.env` in the `backend/` directory with the following content:

```env
# Server Configuration
PORT=5000
NODE_ENV=development

# Database
MONGODB_URI=mongodb://localhost:27017/satellite-analysis
MONGODB_ATLAS_URI=mongodb+srv://username:password@cluster.mongodb.net/satellite-analysis

# JWT
JWT_SECRET=your-super-secret-jwt-key-here-change-this-in-production
JWT_EXPIRE=7d

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# API Keys
OPENWEATHER_API_KEY=your-openweathermap-api-key
SENTINELHUB_CLIENT_ID=your-sentinelhub-client-id
SENTINELHUB_CLIENT_SECRET=your-sentinelhub-client-secret
SENTINELHUB_TOKEN=your-sentinelhub-token
OPENCAGE_API_KEY=your-opencage-api-key
GOOGLE_EARTH_ENGINE_KEY=your-google-earth-engine-key

# Cloud Storage
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_BUCKET_NAME=your-s3-bucket-name
AWS_REGION=us-east-1

# Frontend URL
CLIENT_URL=http://localhost:3000
```

## Frontend Environment Setup

Create a file named `.env` in the `frontend/` directory with the following content:

```env
# React App Configuration
REACT_APP_API_URL=http://localhost:5000/api
REACT_APP_GOOGLE_CLIENT_ID=your-google-client-id
REACT_APP_SENTINELHUB_CLIENT_ID=your-sentinelhub-client-id
```

## API Keys Required

To make the application fully functional, you'll need to obtain the following API keys:

1. **OpenWeatherMap API**: https://openweathermap.org/api
   - Free tier available
   - Used for weather data

2. **OpenCage Geocoding API**: https://opencagedata.com/api
   - Free tier available
   - Used for location geocoding

3. **SentinelHub API**: https://www.sentinel-hub.com/
   - Free tier available
   - Used for satellite imagery

4. **Google OAuth**: https://console.developers.google.com/
   - Free
   - Used for user authentication

5. **MongoDB Atlas**: https://www.mongodb.com/atlas
   - Free tier available
   - Used for database

## Quick Setup Commands

```bash
# Copy environment files
cp backend/env.example backend/.env
cp frontend/.env.example frontend/.env

# Install dependencies
npm run install-all

# Start the application
npm run dev
```


