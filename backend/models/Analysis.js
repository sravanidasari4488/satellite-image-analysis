const mongoose = require('mongoose');

const analysisSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  location: {
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
    country: String,
    state: String,
    city: String,
    postalCode: String
  },
  satelliteImage: {
    url: String,
    cloudStoragePath: String,
    acquisitionDate: Date,
    resolution: String,
    bands: [String], // e.g., ['B02', 'B03', 'B04', 'B08'] for Sentinel-2
    metadata: {
      satellite: String,
      sensor: String,
      cloudCoverage: Number,
      sunElevation: Number,
      sunAzimuth: Number
    }
  },
  analysisResults: {
    landClassification: {
      forest: {
        percentage: Number,
        confidence: Number,
        areas: [{
          coordinates: [[Number, Number]],
          area: Number
        }]
      },
      water: {
        percentage: Number,
        confidence: Number,
        areas: [{
          coordinates: [[Number, Number]],
          area: Number
        }]
      },
      urban: {
        percentage: Number,
        confidence: Number,
        areas: [{
          coordinates: [[Number, Number]],
          area: Number
        }]
      },
      agricultural: {
        percentage: Number,
        confidence: Number,
        areas: [{
          coordinates: [[Number, Number]],
          area: Number
        }]
      },
      barren: {
        percentage: Number,
        confidence: Number,
        areas: [{
          coordinates: [[Number, Number]],
          area: Number
        }]
      }
    },
    vegetationHealth: {
      ndvi: {
        average: Number,
        min: Number,
        max: Number,
        standardDeviation: Number,
        distribution: {
          healthy: Number, // NDVI > 0.6
          moderate: Number, // 0.3 < NDVI <= 0.6
          poor: Number // NDVI <= 0.3
        }
      },
      evi: {
        average: Number,
        min: Number,
        max: Number
      }
    },
    riskAssessment: {
      floodRisk: {
        level: {
          type: String,
          enum: ['low', 'medium', 'high', 'very-high'],
          default: 'low'
        },
        score: Number,
        factors: [String]
      },
      droughtRisk: {
        level: {
          type: String,
          enum: ['low', 'medium', 'high', 'very-high'],
          default: 'low'
        },
        score: Number,
        factors: [String]
      },
      deforestationRisk: {
        level: {
          type: String,
          enum: ['low', 'medium', 'high', 'very-high'],
          default: 'low'
        },
        score: Number,
        factors: [String]
      }
    }
  },
  weatherData: {
    current: {
      temperature: Number,
      humidity: Number,
      pressure: Number,
      windSpeed: Number,
      windDirection: Number,
      visibility: Number,
      uvIndex: Number,
      description: String,
      icon: String,
      timestamp: Date
    },
    forecast: [{
      date: Date,
      temperature: {
        min: Number,
        max: Number
      },
      humidity: Number,
      precipitation: {
        probability: Number,
        amount: Number
      },
      windSpeed: Number,
      description: String,
      icon: String
    }]
  },
  processingStatus: {
    type: String,
    enum: ['pending', 'processing', 'completed', 'failed'],
    default: 'pending'
  },
  processingLog: [{
    timestamp: Date,
    status: String,
    message: String,
    details: mongoose.Schema.Types.Mixed
  }],
  reportGenerated: {
    type: Boolean,
    default: false
  },
  reportUrl: String,
  tags: [String],
  isPublic: {
    type: Boolean,
    default: false
  }
}, {
  timestamps: true
});

// Indexes for better query performance
analysisSchema.index({ user: 1, createdAt: -1 });
analysisSchema.index({ 'location.coordinates.latitude': 1, 'location.coordinates.longitude': 1 });
analysisSchema.index({ processingStatus: 1 });
analysisSchema.index({ createdAt: -1 });

// Virtual for total analysis time
analysisSchema.virtual('processingTime').get(function() {
  if (this.processingStatus === 'completed' && this.processingLog.length > 0) {
    const startTime = this.processingLog[0].timestamp;
    const endTime = this.processingLog[this.processingLog.length - 1].timestamp;
    return endTime - startTime;
  }
  return null;
});

// Method to add processing log entry
analysisSchema.methods.addProcessingLog = function(status, message, details = null) {
  this.processingLog.push({
    timestamp: new Date(),
    status,
    message,
    details
  });
  return this.save();
};

// Method to update processing status
analysisSchema.methods.updateStatus = function(status, message = null, details = null) {
  this.processingStatus = status;
  if (message) {
    this.addProcessingLog(status, message, details);
  }
  return this.save();
};

// Static method to find analyses by location
analysisSchema.statics.findByLocation = function(latitude, longitude, radius = 1000) {
  // This would need to be implemented with proper geospatial queries
  // For now, returning a simple coordinate-based search
  return this.find({
    'location.coordinates.latitude': {
      $gte: latitude - 0.01,
      $lte: latitude + 0.01
    },
    'location.coordinates.longitude': {
      $gte: longitude - 0.01,
      $lte: longitude + 0.01
    }
  });
};

module.exports = mongoose.model('Analysis', analysisSchema);
