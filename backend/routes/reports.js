const express = require('express');
const PDFDocument = require('jspdf');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;
const fs = require('fs').promises;
const path = require('path');
const { body, validationResult } = require('express-validator');
const { authenticateToken, optionalAuth } = require('../middleware/auth');
const Report = require('../models/Report');

const router = express.Router();

// Generate PDF report
const generatePDFReport = async (report) => {
  const doc = new PDFDocument();
  const buffers = [];
  
  doc.on('data', buffers.push.bind(buffers));
  
  return new Promise((resolve, reject) => {
    doc.on('end', () => {
      const pdfData = Buffer.concat(buffers);
      resolve(pdfData);
    });
    
    doc.on('error', reject);

    // Add content to PDF
    doc.fontSize(20).text('Satellite Image Analysis Report', 50, 50);
    
    doc.fontSize(14).text(`Location: ${report.location.name}`, 50, 100);
    doc.text(`Generated: ${report.createdAt.toLocaleDateString()}`, 50, 120);
    doc.text(`Analysis ID: ${report._id}`, 50, 140);
    
    // Weather Summary
    doc.fontSize(16).text('Weather Summary', 50, 180);
    doc.fontSize(12);
    doc.text(`Temperature: ${report.weatherData.temperature.current}°${report.weatherData.temperature.unit === 'celsius' ? 'C' : 'F'}`, 50, 210);
    doc.text(`Humidity: ${report.weatherData.humidity}%`, 50, 230);
    doc.text(`Wind Speed: ${report.weatherData.windSpeed} m/s`, 50, 250);
    doc.text(`Precipitation: ${report.weatherData.precipitation.current} mm`, 50, 270);
    doc.text(`Description: ${report.weatherData.description}`, 50, 290);
    
    // Land Classification
    doc.fontSize(16).text('Land Classification', 50, 330);
    doc.fontSize(12);
    doc.text(`Forest: ${report.landClassification.forest.percentage}% (${report.landClassification.forest.health})`, 50, 360);
    doc.text(`Water: ${report.landClassification.water.percentage}% (${report.landClassification.water.quality})`, 50, 380);
    doc.text(`Urban: ${report.landClassification.urban.percentage}% (${report.landClassification.urban.density})`, 50, 400);
    doc.text(`Agricultural: ${report.landClassification.agricultural.percentage}% (${report.landClassification.agricultural.cropHealth})`, 50, 420);
    doc.text(`Barren: ${report.landClassification.barren.percentage}%`, 50, 440);
    
    // Risk Assessment
    doc.fontSize(16).text('Risk Assessment', 50, 480);
    doc.fontSize(12);
    doc.text(`Flood Risk: ${report.riskAssessment.floodRisk.level} (${report.riskAssessment.floodRisk.probability.toFixed(1)}%)`, 50, 510);
    doc.text(`Drought Risk: ${report.riskAssessment.droughtRisk.level} (${report.riskAssessment.droughtRisk.probability.toFixed(1)}%)`, 50, 530);
    
    if (report.riskAssessment.deforestation.detected) {
      doc.text(`Deforestation: Detected (${report.riskAssessment.deforestation.severity})`, 50, 550);
    } else {
      doc.text('Deforestation: Not detected', 50, 550);
    }
    
    // Analysis Metadata
    doc.fontSize(16).text('Analysis Details', 50, 590);
    doc.fontSize(12);
    doc.text(`Confidence: ${report.analysisMetadata.confidence.toFixed(1)}%`, 50, 620);
    doc.text(`Data Quality: ${report.analysisMetadata.dataQuality}`, 50, 640);
    doc.text(`Model Version: ${report.analysisMetadata.modelVersion}`, 50, 660);
    
    doc.end();
  });
};

// Generate CSV report
const generateCSVReport = async (report) => {
  const csvData = [
    {
      'Report ID': report._id,
      'Title': report.title,
      'Location': report.location.name,
      'Latitude': report.location.coordinates.latitude,
      'Longitude': report.location.coordinates.longitude,
      'Generated Date': report.createdAt.toISOString(),
      'Temperature': report.weatherData.temperature.current,
      'Temperature Unit': report.weatherData.temperature.unit,
      'Humidity': report.weatherData.humidity,
      'Wind Speed': report.weatherData.windSpeed,
      'Precipitation': report.weatherData.precipitation.current,
      'Weather Description': report.weatherData.description,
      'Forest Percentage': report.landClassification.forest.percentage,
      'Forest Health': report.landClassification.forest.health,
      'Water Percentage': report.landClassification.water.percentage,
      'Water Quality': report.landClassification.water.quality,
      'Urban Percentage': report.landClassification.urban.percentage,
      'Urban Density': report.landClassification.urban.density,
      'Agricultural Percentage': report.landClassification.agricultural.percentage,
      'Crop Health': report.landClassification.agricultural.cropHealth,
      'Barren Percentage': report.landClassification.barren.percentage,
      'Flood Risk Level': report.riskAssessment.floodRisk.level,
      'Flood Risk Probability': report.riskAssessment.floodRisk.probability,
      'Drought Risk Level': report.riskAssessment.droughtRisk.level,
      'Drought Risk Probability': report.riskAssessment.droughtRisk.probability,
      'Deforestation Detected': report.riskAssessment.deforestation.detected,
      'Deforestation Severity': report.riskAssessment.deforestation.severity,
      'Analysis Confidence': report.analysisMetadata.confidence,
      'Data Quality': report.analysisMetadata.dataQuality
    }
  ];

  const csvWriter = createCsvWriter({
    path: 'temp_report.csv',
    header: Object.keys(csvData[0]).map(key => ({ id: key, title: key }))
  });

  await csvWriter.writeRecords(csvData);
  const csvContent = await fs.readFile('temp_report.csv', 'utf8');
  await fs.unlink('temp_report.csv'); // Clean up temp file

  return csvContent;
};

// Get all reports for a user
router.get('/', authenticateToken, async (req, res) => {
  try {
    const { page = 1, limit = 10, status, sortBy = 'createdAt', sortOrder = 'desc' } = req.query;
    const skip = (page - 1) * limit;

    // Build query
    const query = { user: req.user._id };
    if (status) {
      query.status = status;
    }

    // Build sort object
    const sort = {};
    sort[sortBy] = sortOrder === 'desc' ? -1 : 1;

    const reports = await Report.find(query)
      .sort(sort)
      .skip(skip)
      .limit(parseInt(limit))
      .select('-satelliteImage.url') // Exclude large image data
      .populate('user', 'name email');

    const total = await Report.countDocuments(query);

    res.json({
      success: true,
      reports,
      pagination: {
        current: parseInt(page),
        total: Math.ceil(total / limit),
        count: reports.length,
        totalCount: total
      }
    });
  } catch (error) {
    console.error('Get reports error:', error);
    res.status(500).json({ message: 'Failed to fetch reports' });
  }
});

// Get a specific report
router.get('/:id', authenticateToken, async (req, res) => {
  try {
    const report = await Report.findOne({
      _id: req.params.id,
      user: req.user._id
    }).populate('user', 'name email');

    if (!report) {
      return res.status(404).json({ message: 'Report not found' });
    }

    res.json({
      success: true,
      report
    });
  } catch (error) {
    console.error('Get report error:', error);
    res.status(500).json({ message: 'Failed to fetch report' });
  }
});

// Update report
router.put('/:id', authenticateToken, [
  body('title').optional().trim().isLength({ min: 1, max: 100 }).withMessage('Title must be 1-100 characters'),
  body('isPublic').optional().isBoolean().withMessage('isPublic must be boolean'),
  body('tags').optional().isArray().withMessage('Tags must be array'),
  body('notes').optional().trim().isLength({ max: 1000 }).withMessage('Notes must be less than 1000 characters')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const report = await Report.findOne({
      _id: req.params.id,
      user: req.user._id
    });

    if (!report) {
      return res.status(404).json({ message: 'Report not found' });
    }

    const updates = req.body;
    const updatedReport = await Report.findByIdAndUpdate(
      req.params.id,
      { $set: updates },
      { new: true, runValidators: true }
    );

    res.json({
      success: true,
      message: 'Report updated successfully',
      report: updatedReport
    });
  } catch (error) {
    console.error('Update report error:', error);
    res.status(500).json({ message: 'Failed to update report' });
  }
});

// Delete report
router.delete('/:id', authenticateToken, async (req, res) => {
  try {
    const report = await Report.findOne({
      _id: req.params.id,
      user: req.user._id
    });

    if (!report) {
      return res.status(404).json({ message: 'Report not found' });
    }

    await Report.findByIdAndDelete(req.params.id);

    res.json({
      success: true,
      message: 'Report deleted successfully'
    });
  } catch (error) {
    console.error('Delete report error:', error);
    res.status(500).json({ message: 'Failed to delete report' });
  }
});

// Download report as PDF
router.get('/:id/download/pdf', authenticateToken, async (req, res) => {
  try {
    const report = await Report.findOne({
      _id: req.params.id,
      user: req.user._id
    });

    if (!report) {
      return res.status(404).json({ message: 'Report not found' });
    }

    const pdfBuffer = await generatePDFReport(report);

    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', `attachment; filename="report_${report._id}.pdf"`);
    res.send(pdfBuffer);
  } catch (error) {
    console.error('PDF generation error:', error);
    res.status(500).json({ message: 'Failed to generate PDF report' });
  }
});

// Download report as CSV
router.get('/:id/download/csv', authenticateToken, async (req, res) => {
  try {
    const report = await Report.findOne({
      _id: req.params.id,
      user: req.user._id
    });

    if (!report) {
      return res.status(404).json({ message: 'Report not found' });
    }

    const csvContent = await generateCSVReport(report);

    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename="report_${report._id}.csv"`);
    res.send(csvContent);
  } catch (error) {
    console.error('CSV generation error:', error);
    res.status(500).json({ message: 'Failed to generate CSV report' });
  }
});

// Get public reports
router.get('/public/list', optionalAuth, async (req, res) => {
  try {
    const { page = 1, limit = 10, location, tags } = req.query;
    const skip = (page - 1) * limit;

    // Build query for public reports
    const query = { isPublic: true, status: 'completed' };
    
    if (location) {
      query['location.name'] = { $regex: location, $options: 'i' };
    }
    
    if (tags) {
      const tagArray = tags.split(',');
      query.tags = { $in: tagArray };
    }

    const reports = await Report.find(query)
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(parseInt(limit))
      .select('-satelliteImage.url') // Exclude large image data
      .populate('user', 'name');

    const total = await Report.countDocuments(query);

    res.json({
      success: true,
      reports,
      pagination: {
        current: parseInt(page),
        total: Math.ceil(total / limit),
        count: reports.length,
        totalCount: total
      }
    });
  } catch (error) {
    console.error('Get public reports error:', error);
    res.status(500).json({ message: 'Failed to fetch public reports' });
  }
});

// Get report statistics
router.get('/stats/overview', authenticateToken, async (req, res) => {
  try {
    const userId = req.user._id;

    const [
      totalReports,
      completedReports,
      processingReports,
      publicReports,
      recentReports
    ] = await Promise.all([
      Report.countDocuments({ user: userId }),
      Report.countDocuments({ user: userId, status: 'completed' }),
      Report.countDocuments({ user: userId, status: 'processing' }),
      Report.countDocuments({ user: userId, isPublic: true }),
      Report.countDocuments({ 
        user: userId, 
        createdAt: { $gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) } // Last 30 days
      })
    ]);

    // Get most analyzed locations
    const topLocations = await Report.aggregate([
      { $match: { user: userId } },
      { $group: { _id: '$location.name', count: { $sum: 1 } } },
      { $sort: { count: -1 } },
      { $limit: 5 }
    ]);

    res.json({
      success: true,
      stats: {
        totalReports,
        completedReports,
        processingReports,
        publicReports,
        recentReports,
        topLocations
      }
    });
  } catch (error) {
    console.error('Get stats error:', error);
    res.status(500).json({ message: 'Failed to fetch statistics' });
  }
});

module.exports = router;

