# Satellite Image Analysis Application

A full-stack application that analyzes satellite images based on user location and generates comprehensive weather and land reports.

## Features

- **User Authentication**: Login/sign-up with email or Google OAuth
- **Location Input**: Support for city names, PIN codes, or lat-long coordinates
- **Satellite Image Analysis**: 
  - Land type detection (forest, water, urban, agricultural)
  - Vegetation health analysis using NDVI
  - Deforestation and land-use change detection
  - Flood/drought risk assessment
- **Weather Integration**: Live weather data from OpenWeatherMap
- **Reports Dashboard**: Interactive visualization with downloadable reports

## Tech Stack

- **Frontend**: React, TailwindCSS
- **Backend**: Node.js, Express.js
- **Database**: MongoDB
- **AI/ML**: Python with TensorFlow/PyTorch for image analysis
- **APIs**: SentinelHub, Google Earth Engine, OpenWeatherMap
- **Storage**: Cloud storage for images and reports

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   npm run install-all
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env` in both `backend/` and `frontend/` directories
   - Fill in your API keys and configuration

4. Start the development servers:
   ```bash
   npm run dev
   ```

## Project Structure

```
satellite-image-analysis/
├── frontend/          # React application
├── backend/           # Node.js API server
├── ai-models/         # Python AI/ML models
├── docs/             # Documentation
└── README.md
```

## API Keys Required

- Google Earth Engine API
- SentinelHub API
- OpenWeatherMap API
- MongoDB Atlas connection string
- Google OAuth credentials

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request