# Satellite Image Analysis Application - Setup Guide

## Prerequisites

- Node.js (v16 or higher)
- MongoDB (local or MongoDB Atlas)
- Python 3.8+ (for AI service)
- Git

## Quick Setup

### 1. Install Dependencies

```bash
# Install all dependencies
npm run install-all

# Or install individually:
cd backend && npm install
cd ../frontend && npm install
cd ../ai-models && pip install -r requirements.txt
```

### 2. Environment Configuration

#### Backend Environment (.env)
Create `backend/.env` file with the following content:

```env
# Server Configuration
PORT=5000
NODE_ENV=development

# Database
MONGODB_URI=mongodb://localhost:27017/satellite-analysis
# OR for MongoDB Atlas:
# MONGODB_ATLAS_URI=mongodb+srv://username:password@cluster.mongodb.net/satellite-analysis

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

# Frontend URL
CLIENT_URL=http://localhost:3000

# AI Service
AI_SERVICE_URL=http://localhost:5001
```

#### Frontend Environment (.env)
Create `frontend/.env` file with the following content:

```env
# API Configuration
REACT_APP_API_URL=http://localhost:5000/api

# Google OAuth (if using client-side OAuth)
REACT_APP_GOOGLE_CLIENT_ID=your-google-client-id
```

### 3. API Keys Setup

#### OpenWeatherMap API
1. Go to [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for a free account
3. Get your API key
4. Add it to `OPENWEATHER_API_KEY` in backend/.env

#### OpenCage Geocoding API
1. Go to [OpenCage](https://opencagedata.com/api)
2. Sign up for a free account
3. Get your API key
4. Add it to `OPENCAGE_API_KEY` in backend/.env

#### SentinelHub API (Optional)
1. Go to [SentinelHub](https://www.sentinel-hub.com/)
2. Sign up for an account
3. Get your client ID and secret
4. Add them to backend/.env

#### Google OAuth (Optional)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Add client ID and secret to backend/.env

### 4. Database Setup

#### Local MongoDB
```bash
# Install MongoDB locally
# Start MongoDB service
mongod
```

#### MongoDB Atlas (Cloud)
1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a free cluster
3. Get connection string
4. Add it to `MONGODB_ATLAS_URI` in backend/.env

### 5. Start the Application

#### Development Mode
```bash
# Start all services
npm run dev

# Or start individually:
# Backend
cd backend && npm run dev

# Frontend
cd frontend && npm start

# AI Service
cd ai-models && python app.py
```

#### Production Mode with Docker
```bash
# Build and start all services
docker-compose up --build

# Or start in background
docker-compose up -d --build
```

### 6. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **AI Service**: http://localhost:5001
- **API Documentation**: http://localhost:5000/api/health

## Features

### ✅ Implemented Features
- User authentication (JWT + Google OAuth)
- Satellite image analysis
- Weather data integration
- Report generation (PDF/CSV)
- Dashboard with statistics
- Responsive UI with Tailwind CSS
- Docker containerization
- AI-powered land classification
- Risk assessment (flood, drought, deforestation)

### 🔧 Configuration Options
- Custom color themes
- Unit preferences (metric/imperial)
- Notification settings
- Public/private reports
- Export formats

## Troubleshooting

### Common Issues

1. **MongoDB Connection Error**
   - Check if MongoDB is running
   - Verify connection string in .env
   - Check network connectivity

2. **API Key Errors**
   - Verify API keys are correct
   - Check API key permissions
   - Ensure sufficient API quota

3. **Port Conflicts**
   - Change ports in .env files
   - Check if ports are already in use
   - Update docker-compose.yml if using Docker

4. **Dependencies Issues**
   - Run `npm install` in each directory
   - Clear node_modules and reinstall
   - Check Node.js version compatibility

### Logs and Debugging

```bash
# Backend logs
cd backend && npm run dev

# Frontend logs
cd frontend && npm start

# AI service logs
cd ai-models && python app.py

# Docker logs
docker-compose logs -f
```

## Production Deployment

### Environment Variables
- Set `NODE_ENV=production`
- Use strong JWT secrets
- Configure proper CORS origins
- Set up SSL certificates
- Use production database

### Security Considerations
- Change default JWT secret
- Use HTTPS in production
- Implement rate limiting
- Validate all inputs
- Use environment variables for secrets

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs
3. Check API documentation
4. Verify environment configuration

## License

MIT License - see LICENSE file for details.
