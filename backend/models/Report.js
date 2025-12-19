const mongoose = require('mongoose');

const locationSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true
  },
  coordinates: {
    latitude: {
      type: Number,
      required: true,
      min: -90,
      max: 90
    },
    longitude: {
      type: Number,
      required: true,
      min: -180,
      max: 180
    }
  },
  address: String,
  city: String,
  country: String,
  postalCode: String
});

const weatherDataSchema = new mongoose.Schema({
  temperature: {
    current: Number,
    min: Number,
    max: Number,
    unit: {
      type: String,
      enum: ['celsius', 'fahrenheit'],
      default: 'celsius'
    }
  },
  humidity: Number,
  pressure: Number,
  windSpeed: Number,
  windDirection: Number,
  visibility: Number,
  uvIndex: Number,
  precipitation: {
    current: Number,
    probability: Number
  },
  description: String,
  icon: String,
  timestamp: {
    type: Date,
    default: Date.now
  }
});

const landClassificationSchema = new mongoose.Schema({
  forest: {
    percentage: Number,
    health: {
      type: String,
      enum: ['excellent', 'good', 'fair', 'poor', 'critical']
    },
    ndvi: Number
  },
  water: {
    percentage: Number,
    quality: {
      type: String,
      enum: ['excellent', 'good', 'fair', 'poor']
    }
  },
  urban: {
    percentage: Number,
    density: {
      type: String,
      enum: ['low', 'medium', 'high', 'very_high']
    }
  },
  agricultural: {
    percentage: Number,
    cropHealth: {
      type: String,
      enum: ['excellent', 'good', 'fair', 'poor']
    },
    ndvi: Number
  },
  barren: {
    percentage: Number
  }
});

const riskAssessmentSchema = new mongoose.Schema({
  floodRisk: {
    level: {
      type: String,
      enum: ['low', 'medium', 'high', 'very_high']
    },
    probability: Number,
    factors: [String]
  },
  droughtRisk: {
    level: {
      type: String,
      enum: ['low', 'medium', 'high', 'very_high']
    },
    probability: Number,
    factors: [String]
  },
  deforestation: {
    detected: Boolean,
    severity: {
      type: String,
      enum: ['low', 'medium', 'high']
    },
    area: Number,
    timeframe: String
  }
});

const reportSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  title: {
    type: String,
    required: true,
    trim: true
  },
  location: {
    type: locationSchema,
    required: true
  },
  satelliteImage: {
    url: String,
    timestamp: Date,
    resolution: String,
    source: {
      type: String,
      enum: ['sentinel', 'sentinel-2', 'sentinel-1', 'landsat', 'landsat-8', 'google_earth', 'user_provided']
    },
    bands: [String]
  },
  weatherData: {
    type: weatherDataSchema,
    required: true
  },
  landClassification: {
    type: landClassificationSchema,
    required: true
  },
  riskAssessment: {
    type: riskAssessmentSchema,
    required: true
  },
  analysisMetadata: {
    processingTime: Number,
    modelVersion: String,
    confidence: Number,
    dataQuality: {
      type: String,
      enum: ['excellent', 'good', 'fair', 'poor']
    }
  },
  visualizations: [{
    type: {
      type: String,
      enum: ['ndvi_map', 'land_classification', 'risk_heatmap', 'comparison']
    },
    url: String,
    description: String
  }],
  status: {
    type: String,
    enum: ['processing', 'completed', 'failed'],
    default: 'processing'
  },
  isPublic: {
    type: Boolean,
    default: false
  },
  tags: [String],
  notes: String
}, {
  timestamps: true
});

// Indexes for better query performance
reportSchema.index({ user: 1, createdAt: -1 });
reportSchema.index({ 'location.coordinates.latitude': 1, 'location.coordinates.longitude': 1 });
reportSchema.index({ status: 1 });
reportSchema.index({ isPublic: 1 });

// Virtual for report age
reportSchema.virtual('age').get(function() {
  return Math.floor((Date.now() - this.createdAt) / (1000 * 60 * 60 * 24));
});

// Method to calculate overall health score
reportSchema.methods.getOverallHealthScore = function() {
  const classification = this.landClassification;
  let score = 0;
  let factors = 0;

  if (classification.forest.health) {
    const healthScores = { excellent: 100, good: 80, fair: 60, poor: 40, critical: 20 };
    score += healthScores[classification.forest.health] * (classification.forest.percentage / 100);
    factors++;
  }

  if (classification.agricultural.cropHealth) {
    const healthScores = { excellent: 100, good: 80, fair: 60, poor: 40 };
    score += healthScores[classification.agricultural.cropHealth] * (classification.agricultural.percentage / 100);
    factors++;
  }

  return factors > 0 ? Math.round(score / factors) : 0;
};

module.exports = mongoose.model('Report', reportSchema);

