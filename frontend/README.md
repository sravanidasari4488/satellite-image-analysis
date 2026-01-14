# Geospatial Intelligence System - Frontend

React.js frontend for the Satellite-based Geospatial Intelligence System.

## Installation

```bash
cd frontend
npm install
```

## Running the Application

1. Make sure the Flask backend is running on `http://localhost:5000`
2. Start the React development server:

```bash
npm start
```

The app will open at `http://localhost:3000`

## Building for Production

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## Environment Variables

Create a `.env` file in the `frontend` directory to customize the API URL:

```
REACT_APP_API_URL=http://localhost:5000
```

If not set, it defaults to `http://localhost:5000`.


