const express = require('express');
const axios = require('axios');
const { body, validationResult } = require('express-validator');
const { authenticateToken, optionalAuth } = require('../middleware/auth');

const router = express.Router();

// Geocoding service to convert location to coordinates
const geocodeLocation = async (location) => {
  try {
    // Using OpenCage Geocoding API (free tier available)
    const response = await axios.get('https://api.opencagedata.com/geocode/v1/json', {
      params: {
        q: location,
        key: process.env.OPENCAGE_API_KEY,
        limit: 1,
        no_annotations: 1
      }
    });

    if (response.data.results && response.data.results.length > 0) {
      const result = response.data.results[0];
      return {
        coordinates: {
          latitude: result.geometry.lat,
          longitude: result.geometry.lng
        },
        address: result.formatted,
        components: result.components
      };
    }
    throw new Error('Location not found');
  } catch (error) {
    console.error('Geocoding error:', error);
    throw new Error('Failed to geocode location');
  }
};

// Reverse geocoding - convert coordinates to address
const reverseGeocode = async (latitude, longitude) => {
  try {
    const response = await axios.get('https://api.opencagedata.com/geocode/v1/json', {
      params: {
        q: `${latitude},${longitude}`,
        key: process.env.OPENCAGE_API_KEY,
        limit: 1,
        no_annotations: 1
      }
    });

    if (response.data.results && response.data.results.length > 0) {
      const result = response.data.results[0];
      return {
        address: result.formatted,
        components: result.components
      };
    }
    throw new Error('Address not found');
  } catch (error) {
    console.error('Reverse geocoding error:', error);
    throw new Error('Failed to reverse geocode coordinates');
  }
};

// Geocode a location (convert text to coordinates)
router.post('/geocode', optionalAuth, [
  body('location').notEmpty().withMessage('Location is required')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { location } = req.body;
    const geocodedLocation = await geocodeLocation(location);

    res.json({
      success: true,
      location: geocodedLocation
    });
  } catch (error) {
    console.error('Geocoding error:', error);
    res.status(500).json({ 
      message: error.message || 'Failed to geocode location' 
    });
  }
});

// Reverse geocode coordinates (convert coordinates to address)
router.post('/reverse-geocode', optionalAuth, [
  body('latitude').isFloat({ min: -90, max: 90 }).withMessage('Valid latitude required'),
  body('longitude').isFloat({ min: -180, max: 180 }).withMessage('Valid longitude required')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { latitude, longitude } = req.body;
    const address = await reverseGeocode(latitude, longitude);

    res.json({
      success: true,
      coordinates: { latitude, longitude },
      address
    });
  } catch (error) {
    console.error('Reverse geocoding error:', error);
    res.status(500).json({ 
      message: error.message || 'Failed to reverse geocode coordinates' 
    });
  }
});

// Get location suggestions (autocomplete)
router.post('/suggestions', optionalAuth, [
  body('query').notEmpty().withMessage('Query is required')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { query } = req.body;
    
    // Using OpenCage Geocoding API for suggestions
    const response = await axios.get('https://api.opencagedata.com/geocode/v1/json', {
      params: {
        q: query,
        key: process.env.OPENCAGE_API_KEY,
        limit: 5,
        no_annotations: 1
      }
    });

    const suggestions = response.data.results.map(result => ({
      formatted: result.formatted,
      coordinates: {
        latitude: result.geometry.lat,
        longitude: result.geometry.lng
      },
      components: result.components
    }));

    res.json({
      success: true,
      suggestions
    });
  } catch (error) {
    console.error('Location suggestions error:', error);
    res.status(500).json({ 
      message: error.message || 'Failed to get location suggestions' 
    });
  }
});

// Validate coordinates
router.post('/validate-coordinates', optionalAuth, [
  body('latitude').isFloat({ min: -90, max: 90 }).withMessage('Valid latitude required'),
  body('longitude').isFloat({ min: -180, max: 180 }).withMessage('Valid longitude required')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { latitude, longitude } = req.body;

    // Basic validation - coordinates are within valid ranges
    const isValid = latitude >= -90 && latitude <= 90 && longitude >= -180 && longitude <= 180;

    res.json({
      success: true,
      valid: isValid,
      coordinates: { latitude, longitude }
    });
  } catch (error) {
    console.error('Coordinate validation error:', error);
    res.status(500).json({ 
      message: error.message || 'Failed to validate coordinates' 
    });
  }
});

// Get location information (detailed)
router.post('/info', optionalAuth, [
  body('location').optional().notEmpty().withMessage('Location cannot be empty'),
  body('latitude').optional().isFloat({ min: -90, max: 90 }).withMessage('Valid latitude required'),
  body('longitude').optional().isFloat({ min: -180, max: 180 }).withMessage('Valid longitude required')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { location, latitude, longitude } = req.body;

    let locationData;

    if (location) {
      // Geocode the location
      locationData = await geocodeLocation(location);
    } else if (latitude && longitude) {
      // Reverse geocode the coordinates
      const address = await reverseGeocode(latitude, longitude);
      locationData = {
        coordinates: { latitude, longitude },
        address: address.address,
        components: address.components
      };
    } else {
      return res.status(400).json({ 
        message: 'Either location or coordinates (latitude, longitude) must be provided' 
      });
    }

    // Get additional information about the location
    const additionalInfo = {
      country: locationData.components?.country || 'Unknown',
      state: locationData.components?.state || 'Unknown',
      city: locationData.components?.city || locationData.components?.town || 'Unknown',
      postalCode: locationData.components?.postcode || 'Unknown',
      continent: locationData.components?.continent || 'Unknown'
    };

    res.json({
      success: true,
      location: {
        ...locationData,
        additionalInfo
      }
    });
  } catch (error) {
    console.error('Location info error:', error);
    res.status(500).json({ 
      message: error.message || 'Failed to get location information' 
    });
  }
});

module.exports = router;