const express = require('express');
const axios = require('axios');
const { body, validationResult } = require('express-validator');
const { authenticateToken, optionalAuth } = require('../middleware/auth');

const router = express.Router();

// Fetch weather data from OpenWeatherMap
const fetchWeatherData = async (coordinates, units = 'metric') => {
  try {
    const { latitude, longitude } = coordinates;
    
    const response = await axios.get('https://api.openweathermap.org/data/2.5/weather', {
      params: {
        lat: latitude,
        lon: longitude,
        appid: process.env.OPENWEATHER_API_KEY,
        units: units
      }
    });

    const data = response.data;
    
    console.log('✅ Using REAL weather data from OpenWeather API');
    
    return {
      temperature: {
        current: Math.round(data.main.temp),
        min: Math.round(data.main.temp_min),
        max: Math.round(data.main.temp_max),
        unit: units === 'metric' ? 'celsius' : 'fahrenheit'
      },
      humidity: data.main.humidity,
      pressure: data.main.pressure,
      windSpeed: data.wind.speed,
      windDirection: data.wind.deg || 0,
      visibility: data.visibility ? Math.round(data.visibility / 1000) : null, // Convert to km
      uvIndex: null, // Will be fetched separately
      precipitation: {
        current: data.rain ? data.rain['1h'] || 0 : 0,
        probability: data.clouds.all
      },
      description: data.weather[0].description,
      icon: data.weather[0].icon,
      timestamp: new Date(),
      _source: 'openweather-api', // Flag to indicate real data
      _realData: true
    };
  } catch (error) {
    console.error('Weather API error:', error);
    
    // Check if it's an API key error
    if (error.response?.status === 401) {
      const errorMsg = error.response?.data?.message || 'Invalid API key';
      console.error('OpenWeather API Key Error:', errorMsg);
      throw new Error(`OpenWeather API Error: ${errorMsg}. Please check your API key in .env file. New keys may take up to 2 hours to activate.`);
    }
    
    // Return mock data as fallback if API fails
    console.warn('⚠️  Using fallback weather data due to API error');
    return {
      temperature: {
        current: 25,
        min: 20,
        max: 30,
        unit: units === 'metric' ? 'celsius' : 'fahrenheit'
      },
      humidity: 65,
      pressure: 1013,
      windSpeed: 10,
      windDirection: 180,
      visibility: 10,
      uvIndex: null,
      precipitation: {
        current: 0,
        probability: 20
      },
      description: 'Weather data unavailable',
      icon: '02d',
      timestamp: new Date(),
      _fallback: true
    };
  }
};

// Fetch UV index data
const fetchUVIndex = async (coordinates) => {
  try {
    const { latitude, longitude } = coordinates;
    
    const response = await axios.get('https://api.openweathermap.org/data/2.5/uvi', {
      params: {
        lat: latitude,
        lon: longitude,
        appid: process.env.OPENWEATHER_API_KEY
      }
    });

    return response.data.value;
  } catch (error) {
    console.error('UV Index API error:', error);
    if (error.response?.status === 401) {
      console.warn('⚠️  OpenWeather API key invalid for UV Index');
    }
    return null; // UV index is optional
  }
};

// Fetch 5-day weather forecast
const fetchWeatherForecast = async (coordinates, units = 'metric') => {
  try {
    const { latitude, longitude } = coordinates;
    
    const response = await axios.get('https://api.openweathermap.org/data/2.5/forecast', {
      params: {
        lat: latitude,
        lon: longitude,
        appid: process.env.OPENWEATHER_API_KEY,
        units: units
      }
    });

    const forecasts = response.data.list.map(item => ({
      timestamp: new Date(item.dt * 1000),
      temperature: {
        current: Math.round(item.main.temp),
        min: Math.round(item.main.temp_min),
        max: Math.round(item.main.temp_max),
        unit: units === 'metric' ? 'celsius' : 'fahrenheit'
      },
      humidity: item.main.humidity,
      pressure: item.main.pressure,
      windSpeed: item.wind.speed,
      windDirection: item.wind.deg || 0,
      precipitation: {
        current: item.rain ? item.rain['3h'] || 0 : 0,
        probability: item.clouds.all
      },
      description: item.weather[0].description,
      icon: item.weather[0].icon
    }));

    return forecasts;
  } catch (error) {
    console.error('Weather forecast API error:', error);
    if (error.response?.status === 401) {
      throw new Error('OpenWeather API Key Error: Invalid API key. Please check your API key in .env file.');
    }
    throw new Error('Failed to fetch weather forecast');
  }
};

// Get current weather for a location
router.post('/current', optionalAuth, [
  body('latitude').isFloat({ min: -90, max: 90 }).withMessage('Valid latitude required'),
  body('longitude').isFloat({ min: -180, max: 180 }).withMessage('Valid longitude required'),
  body('units').optional().isIn(['metric', 'imperial']).withMessage('Units must be metric or imperial')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { latitude, longitude, units = 'metric' } = req.body;
    const coordinates = { latitude, longitude };

    // Determine units based on user preference or request
    const userUnits = req.user?.preferences?.units || units;

    const [weatherData, uvIndex] = await Promise.all([
      fetchWeatherData(coordinates, userUnits),
      fetchUVIndex(coordinates)
    ]);

    // Add UV index to weather data
    weatherData.uvIndex = uvIndex;

    res.json({
      success: true,
      weather: weatherData,
      location: {
        coordinates,
        timestamp: new Date()
      },
      dataSource: weatherData._realData ? 'OpenWeather API (Real Data)' : 'Mock Data (Fallback)'
    });
  } catch (error) {
    console.error('Current weather error:', error);
    res.status(500).json({ 
      message: error.message || 'Failed to fetch current weather' 
    });
  }
});

// Get weather forecast for a location
router.post('/forecast', optionalAuth, [
  body('latitude').isFloat({ min: -90, max: 90 }).withMessage('Valid latitude required'),
  body('longitude').isFloat({ min: -180, max: 180 }).withMessage('Valid longitude required'),
  body('units').optional().isIn(['metric', 'imperial']).withMessage('Units must be metric or imperial')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { latitude, longitude, units = 'metric' } = req.body;
    const coordinates = { latitude, longitude };

    // Determine units based on user preference or request
    const userUnits = req.user?.preferences?.units || units;

    const forecast = await fetchWeatherForecast(coordinates, userUnits);

    res.json({
      success: true,
      forecast,
      location: {
        coordinates,
        timestamp: new Date()
      }
    });
  } catch (error) {
    console.error('Weather forecast error:', error);
    res.status(500).json({ 
      message: error.message || 'Failed to fetch weather forecast' 
    });
  }
});

// Get weather alerts for a location
router.post('/alerts', optionalAuth, [
  body('latitude').isFloat({ min: -90, max: 90 }).withMessage('Valid latitude required'),
  body('longitude').isFloat({ min: -180, max: 180 }).withMessage('Valid longitude required')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { latitude, longitude } = req.body;
    const coordinates = { latitude, longitude };

    // Fetch weather alerts (OpenWeatherMap One Call API)
    const response = await axios.get('https://api.openweathermap.org/data/2.5/onecall', {
      params: {
        lat: latitude,
        lon: longitude,
        appid: process.env.OPENWEATHER_API_KEY,
        exclude: 'minutely,hourly,daily'
      }
    });

    const alerts = response.data.alerts || [];

    res.json({
      success: true,
      alerts: alerts.map(alert => ({
        title: alert.event,
        description: alert.description,
        start: new Date(alert.start * 1000),
        end: new Date(alert.end * 1000),
        severity: alert.tags ? alert.tags[0] : 'moderate',
        source: alert.sender_name
      })),
      location: {
        coordinates,
        timestamp: new Date()
      }
    });
  } catch (error) {
    console.error('Weather alerts error:', error);
    res.status(500).json({ 
      message: error.message || 'Failed to fetch weather alerts' 
    });
  }
});

// Get historical weather data (requires premium API)
router.post('/historical', authenticateToken, [
  body('latitude').isFloat({ min: -90, max: 90 }).withMessage('Valid latitude required'),
  body('longitude').isFloat({ min: -180, max: 180 }).withMessage('Valid longitude required'),
  body('date').isISO8601().withMessage('Valid date required'),
  body('units').optional().isIn(['metric', 'imperial']).withMessage('Units must be metric or imperial')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { latitude, longitude, date, units = 'metric' } = req.body;
    const coordinates = { latitude, longitude };

    // Convert date to timestamp
    const timestamp = Math.floor(new Date(date).getTime() / 1000);

    // Fetch historical weather data
    const response = await axios.get('https://api.openweathermap.org/data/2.5/onecall/timemachine', {
      params: {
        lat: latitude,
        lon: longitude,
        dt: timestamp,
        appid: process.env.OPENWEATHER_API_KEY,
        units: units
      }
    });

    const data = response.data.current;
    
    const historicalWeather = {
      temperature: {
        current: Math.round(data.temp),
        unit: units === 'metric' ? 'celsius' : 'fahrenheit'
      },
      humidity: data.humidity,
      pressure: data.pressure,
      windSpeed: data.wind_speed,
      windDirection: data.wind_deg || 0,
      uvIndex: data.uvi,
      precipitation: {
        current: data.rain ? data.rain['1h'] || 0 : 0
      },
      description: data.weather[0].description,
      icon: data.weather[0].icon,
      timestamp: new Date(data.dt * 1000)
    };

    res.json({
      success: true,
      weather: historicalWeather,
      location: {
        coordinates,
        date: new Date(date)
      }
    });
  } catch (error) {
    console.error('Historical weather error:', error);
    res.status(500).json({ 
      message: error.message || 'Failed to fetch historical weather' 
    });
  }
});

// Export fetchWeatherData for use in other routes
module.exports = router;
module.exports.fetchWeatherData = fetchWeatherData;

