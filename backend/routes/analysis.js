const express = require('express');
const axios = require('axios');
const { body, validationResult } = require('express-validator');
const { authenticateToken, optionalAuth } = require('../middleware/auth');
const Analysis = require('../models/Analysis');
const Report = require('../models/Report');

const router = express.Router();

// Get analysis history for user
router.get('/history', authenticateToken, async (req, res) => {
  try {
    const { page = 1, limit = 10, status } = req.query;
    const skip = (page - 1) * limit;

    // Build query
    const query = { user: req.user._id };
    if (status) {
      query.processingStatus = status;
    }

    const analyses = await Analysis.find(query)
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(parseInt(limit))
      .select('-satelliteImage.url') // Exclude large image data
      .populate('user', 'name email');

    const total = await Analysis.countDocuments(query);

    res.json({
      success: true,
      analyses,
      pagination: {
        current: parseInt(page),
        total: Math.ceil(total / limit),
        count: analyses.length,
        totalCount: total
      }
    });
  } catch (error) {
    console.error('Get analysis history error:', error);
    res.status(500).json({ message: 'Failed to fetch analysis history' });
  }
});

// Get a specific analysis
router.get('/:id', authenticateToken, async (req, res) => {
  try {
    const analysis = await Analysis.findOne({
      _id: req.params.id,
      user: req.user._id
    }).populate('user', 'name email');

    if (!analysis) {
      return res.status(404).json({ message: 'Analysis not found' });
    }

    res.json({
      success: true,
      analysis
    });
  } catch (error) {
    console.error('Get analysis error:', error);
    res.status(500).json({ message: 'Failed to fetch analysis' });
  }
});

// Create new analysis
router.post('/create', authenticateToken, [
  body('location').notEmpty().withMessage('Location is required'),
  body('title').optional().trim().isLength({ min: 1, max: 100 }).withMessage('Title must be 1-100 characters'),
  body('tags').optional().isArray().withMessage('Tags must be array'),
  body('isPublic').optional().isBoolean().withMessage('isPublic must be boolean')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { location, title, tags, isPublic } = req.body;

    // Create new analysis record
    const analysis = new Analysis({
      user: req.user._id,
      location: {
        name: location.name || location,
        coordinates: location.coordinates,
        address: location.address,
        country: location.country,
        state: location.state,
        city: location.city,
        postalCode: location.postalCode
      },
      title: title || `Analysis for ${location.name || location}`,
      tags: tags || [],
      isPublic: isPublic || false,
      processingStatus: 'pending'
    });

    // Add initial processing log
    await analysis.addProcessingLog('pending', 'Analysis request created');

    await analysis.save();

    res.status(201).json({
      success: true,
      message: 'Analysis created successfully',
      analysis: {
        id: analysis._id,
        title: analysis.title,
        location: analysis.location,
        processingStatus: analysis.processingStatus,
        createdAt: analysis.createdAt
      }
    });
  } catch (error) {
    console.error('Create analysis error:', error);
    res.status(500).json({ message: 'Failed to create analysis' });
  }
});

// Update analysis
router.put('/:id', authenticateToken, [
  body('title').optional().trim().isLength({ min: 1, max: 100 }).withMessage('Title must be 1-100 characters'),
  body('tags').optional().isArray().withMessage('Tags must be array'),
  body('isPublic').optional().isBoolean().withMessage('isPublic must be boolean'),
  body('notes').optional().trim().isLength({ max: 1000 }).withMessage('Notes must be less than 1000 characters')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const analysis = await Analysis.findOne({
      _id: req.params.id,
      user: req.user._id
    });

    if (!analysis) {
      return res.status(404).json({ message: 'Analysis not found' });
    }

    const updates = req.body;
    const updatedAnalysis = await Analysis.findByIdAndUpdate(
      req.params.id,
      { $set: updates },
      { new: true, runValidators: true }
    );

    res.json({
      success: true,
      message: 'Analysis updated successfully',
      analysis: updatedAnalysis
    });
  } catch (error) {
    console.error('Update analysis error:', error);
    res.status(500).json({ message: 'Failed to update analysis' });
  }
});

// Delete analysis
router.delete('/:id', authenticateToken, async (req, res) => {
  try {
    const analysis = await Analysis.findOne({
      _id: req.params.id,
      user: req.user._id
    });

    if (!analysis) {
      return res.status(404).json({ message: 'Analysis not found' });
    }

    await Analysis.findByIdAndDelete(req.params.id);

    res.json({
      success: true,
      message: 'Analysis deleted successfully'
    });
  } catch (error) {
    console.error('Delete analysis error:', error);
    res.status(500).json({ message: 'Failed to delete analysis' });
  }
});

// Start analysis processing
router.post('/:id/start', authenticateToken, async (req, res) => {
  try {
    const analysis = await Analysis.findOne({
      _id: req.params.id,
      user: req.user._id
    });

    if (!analysis) {
      return res.status(404).json({ message: 'Analysis not found' });
    }

    if (analysis.processingStatus !== 'pending') {
      return res.status(400).json({ message: 'Analysis is not in pending status' });
    }

    // Update status to processing
    await analysis.updateStatus('processing', 'Analysis processing started');

    // Here you would typically trigger the actual analysis process
    // For now, we'll simulate the process
    setTimeout(async () => {
      try {
        // Simulate analysis completion
        await analysis.updateStatus('completed', 'Analysis completed successfully');
        
        // Create a report from the analysis
        const report = new Report({
          user: req.user._id,
          title: analysis.title,
          location: analysis.location,
          satelliteImage: analysis.satelliteImage,
          weatherData: analysis.weatherData,
          landClassification: analysis.analysisResults?.landClassification,
          riskAssessment: analysis.analysisResults?.riskAssessment,
          analysisMetadata: {
            processingTime: Date.now() - analysis.createdAt.getTime(),
            modelVersion: '1.0.0',
            confidence: 85,
            dataQuality: 'good'
          },
          status: 'completed'
        });

        await report.save();
        await analysis.updateStatus('completed', 'Report generated successfully');
      } catch (error) {
        console.error('Analysis processing error:', error);
        await analysis.updateStatus('failed', 'Analysis processing failed');
      }
    }, 5000); // Simulate 5 second processing time

    res.json({
      success: true,
      message: 'Analysis processing started',
      analysis: {
        id: analysis._id,
        processingStatus: 'processing'
      }
    });
  } catch (error) {
    console.error('Start analysis error:', error);
    res.status(500).json({ message: 'Failed to start analysis' });
  }
});

// Get analysis status
router.get('/:id/status', authenticateToken, async (req, res) => {
  try {
    const analysis = await Analysis.findOne({
      _id: req.params.id,
      user: req.user._id
    }).select('processingStatus processingLog createdAt updatedAt');

    if (!analysis) {
      return res.status(404).json({ message: 'Analysis not found' });
    }

    res.json({
      success: true,
      status: {
        processingStatus: analysis.processingStatus,
        processingLog: analysis.processingLog,
        createdAt: analysis.createdAt,
        updatedAt: analysis.updatedAt
      }
    });
  } catch (error) {
    console.error('Get analysis status error:', error);
    res.status(500).json({ message: 'Failed to get analysis status' });
  }
});

// Get analysis statistics
router.get('/stats/overview', authenticateToken, async (req, res) => {
  try {
    const userId = req.user._id;

    const [
      totalAnalyses,
      pendingAnalyses,
      processingAnalyses,
      completedAnalyses,
      failedAnalyses,
      recentAnalyses
    ] = await Promise.all([
      Analysis.countDocuments({ user: userId }),
      Analysis.countDocuments({ user: userId, processingStatus: 'pending' }),
      Analysis.countDocuments({ user: userId, processingStatus: 'processing' }),
      Analysis.countDocuments({ user: userId, processingStatus: 'completed' }),
      Analysis.countDocuments({ user: userId, processingStatus: 'failed' }),
      Analysis.countDocuments({ 
        user: userId, 
        createdAt: { $gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) } // Last 30 days
      })
    ]);

    // Get most analyzed locations
    const topLocations = await Analysis.aggregate([
      { $match: { user: userId } },
      { $group: { _id: '$location.name', count: { $sum: 1 } } },
      { $sort: { count: -1 } },
      { $limit: 5 }
    ]);

    res.json({
      success: true,
      stats: {
        totalAnalyses,
        pendingAnalyses,
        processingAnalyses,
        completedAnalyses,
        failedAnalyses,
        recentAnalyses,
        topLocations
      }
    });
  } catch (error) {
    console.error('Get analysis stats error:', error);
    res.status(500).json({ message: 'Failed to fetch analysis statistics' });
  }
});

// Get public analyses
router.get('/public/list', optionalAuth, async (req, res) => {
  try {
    const { page = 1, limit = 10, location, tags } = req.query;
    const skip = (page - 1) * limit;

    // Build query for public analyses
    const query = { isPublic: true, processingStatus: 'completed' };
    
    if (location) {
      query['location.name'] = { $regex: location, $options: 'i' };
    }
    
    if (tags) {
      const tagArray = tags.split(',');
      query.tags = { $in: tagArray };
    }

    const analyses = await Analysis.find(query)
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(parseInt(limit))
      .select('-satelliteImage.url') // Exclude large image data
      .populate('user', 'name');

    const total = await Analysis.countDocuments(query);

    res.json({
      success: true,
      analyses,
      pagination: {
        current: parseInt(page),
        total: Math.ceil(total / limit),
        count: analyses.length,
        totalCount: total
      }
    });
  } catch (error) {
    console.error('Get public analyses error:', error);
    res.status(500).json({ message: 'Failed to fetch public analyses' });
  }
});

module.exports = router;