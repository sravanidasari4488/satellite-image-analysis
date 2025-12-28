/**
 * City-Level GEE Analysis Route
 * Performs region-based land cover classification using exact city boundaries
 */

const express = require('express');
const axios = require('axios');
const { body, validationResult } = require('express-validator');
const { authenticateToken } = require('../middleware/auth');
const Report = require('../models/Report');

const router = express.Router();

// Get AI service URL
const getAIServiceURL = () => {
  return process.env.AI_SERVICE_URL || 'http://localhost:5001';
};

// Helper to call AI service
const callAIService = async (endpoint, data) => {
  const aiServiceUrl = getAIServiceURL();
  const fullUrl = `${aiServiceUrl}${endpoint}`;
  
  try {
    const response = await axios.post(fullUrl, data, {
      headers: { 'Content-Type': 'application/json' },
      timeout: 120000, // 2 minutes for GEE processing
      validateStatus: (status) => status < 500
    });
    
    if (response.status >= 400) {
      const errorMsg = response.data?.error || response.data?.message || `HTTP ${response.status}`;
      throw new Error(errorMsg);
    }
    
    return response.data;
  } catch (error) {
    if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT') {
      throw new Error(`AI Service unavailable at ${fullUrl}`);
    }
    throw error;
  }
};

// Geocode location and get polygon
const geocodeLocation = async (location) => {
  try {
    const params = {
      q: location,
      key: process.env.OPENCAGE_API_KEY,
      limit: 1,
      no_annotations: 0
    };
    
    const response = await axios.get('https://api.opencagedata.com/geocode/v1/json', { params });

    if (response.data.results && response.data.results.length > 0) {
      const result = response.data.results[0];
      const geometry = result.geometry;
      
      // Extract polygon coordinates
      let polygon = null;
      let bounds = null;
      
      if (geometry.type === 'Polygon' && geometry.coordinates) {
        polygon = geometry.coordinates[0]; // First ring
      } else if (result.bounds) {
        bounds = {
          northeast: { lat: result.bounds.northeast.lat, lng: result.bounds.northeast.lng },
          southwest: { lat: result.bounds.southwest.lat, lng: result.bounds.southwest.lng }
        };
      }
      
      // Calculate bounds from polygon
      if (polygon && polygon.length > 0) {
        const lats = polygon.map(coord => coord[1]);
        const lngs = polygon.map(coord => coord[0]);
        bounds = {
          northeast: { lat: Math.max(...lats), lng: Math.max(...lngs) },
          southwest: { lat: Math.min(...lats), lng: Math.min(...lngs) }
        };
      }
      
      return {
        coordinates: { latitude: geometry.lat, longitude: geometry.lng },
        address: result.formatted,
        polygon: polygon, // [[lng, lat], ...] format
        bounds: bounds,
        geometryType: geometry.type || 'Point'
      };
    }
    
    throw new Error('Location not found');
  } catch (error) {
    console.error('Geocoding error:', error);
    throw new Error('Failed to geocode location');
  }
};

/**
 * City-level land cover analysis using Google Earth Engine
 * 
 * POST /api/city-gee/analyze
 * 
 * Body: {
 *   "location": "City name",
 *   "start_date": "2024-01-01" (optional),
 *   "end_date": "2024-01-31" (optional),
 *   "cloud_cover_threshold": 20 (optional)
 * }
 */
router.post('/analyze', authenticateToken, [
  body('location').notEmpty().withMessage('Location is required')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { location, start_date, end_date, cloud_cover_threshold, title } = req.body;

    console.log(`🏙️  Starting city-level GEE analysis for: ${location}`);

    // Step 1: Geocode location and get polygon
    const geocodedLocation = await geocodeLocation(location);
    
    if (!geocodedLocation.polygon) {
      return res.status(400).json({
        message: 'City polygon boundary not available. Please provide a valid city name with administrative boundaries.'
      });
    }

    console.log(`📍 Retrieved polygon with ${geocodedLocation.polygon.length} coordinates`);

    // Step 2: Call GEE analysis endpoint
    const analysisResult = await callAIService('/gee/analyze-city', {
      city_name: location,
      polygon_coords: geocodedLocation.polygon, // Already in [lng, lat] format
      start_date: start_date,
      end_date: end_date,
      cloud_cover_threshold: cloud_cover_threshold || 20
    });

    if (!analysisResult.success) {
      return res.status(500).json({
        message: analysisResult.error || 'Analysis failed'
      });
    }

    // Step 3: Format response
    const landCover = analysisResult.land_cover;
    
    // Map to standard format
    const landClassification = {
      water: {
        percentage: Math.round(landCover.water.percentage * 100) / 100,
        areaKm2: landCover.water.area_km2,
        pixels: landCover.water.pixels
      },
      forest: {
        percentage: Math.round(landCover.forest.percentage * 100) / 100,
        areaKm2: landCover.forest.area_km2,
        pixels: landCover.forest.pixels
      },
      urban: {
        percentage: Math.round(landCover.urban.percentage * 100) / 100,
        areaKm2: landCover.urban.area_km2,
        pixels: landCover.urban.pixels
      },
      agricultural: {
        percentage: Math.round(landCover.agricultural.percentage * 100) / 100,
        areaKm2: landCover.agricultural.area_km2,
        pixels: landCover.agricultural.pixels
      },
      barren: {
        percentage: Math.round(landCover.barren.percentage * 100) / 100,
        areaKm2: landCover.barren.area_km2,
        pixels: landCover.barren.pixels
      },
      _method: 'gee_city_level',
      _totalAreaKm2: analysisResult.summary.total_area_km2,
      _percentageSum: analysisResult.summary.percentage_sum
    };

    // Step 4: Save report (optional)
    let report = null;
    try {
      report = new Report({
        user: req.user._id,
        title: title || `City-Level Analysis: ${location}`,
        location: {
          name: geocodedLocation.address,
          coordinates: geocodedLocation.coordinates,
          address: geocodedLocation.address,
          bounds: geocodedLocation.bounds,
          polygon: geocodedLocation.polygon,
          geometryType: geocodedLocation.geometryType
        },
        landClassification: landClassification,
        analysisMetadata: {
          processingTime: Date.now(),
          modelVersion: '2.0.0-gee',
          method: 'city_level_gee',
          confidence: 95,
          dataQuality: 'excellent',
          methodology: analysisResult.methodology
        },
        status: 'completed'
      });

      await report.save();
    } catch (dbError) {
      console.warn('Failed to save report:', dbError.message);
    }

    // Step 5: Return response
    res.json({
      success: true,
      city_name: location,
      location: geocodedLocation,
      land_classification: landClassification,
      summary: analysisResult.summary,
      methodology: analysisResult.methodology,
      imagery_date_range: analysisResult.imagery_date_range,
      report_id: report?._id,
      message: 'City-level analysis completed using Google Earth Engine'
    });

  } catch (error) {
    console.error('City-level GEE analysis error:', error);
    res.status(500).json({
      message: error.message || 'Failed to perform city-level analysis',
      error: error.message
    });
  }
});

module.exports = router;

