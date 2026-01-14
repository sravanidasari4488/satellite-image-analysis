const mongoose = require('mongoose');

/**
 * Middleware to check if MongoDB is connected before processing requests
 * Waits for connection if it's in progress
 */
const checkDatabaseConnection = async (req, res, next) => {
  const readyState = mongoose.connection.readyState;
  
  // 0 = disconnected, 1 = connected, 2 = connecting, 3 = disconnecting
  if (readyState === 1) {
    // Connected - proceed
    return next();
  } else if (readyState === 2) {
    // Connecting - wait a bit and check again
    return new Promise((resolve) => {
      const checkInterval = setInterval(() => {
        if (mongoose.connection.readyState === 1) {
          clearInterval(checkInterval);
          next();
          resolve();
        } else if (mongoose.connection.readyState === 0 || mongoose.connection.readyState === 3) {
          clearInterval(checkInterval);
          res.status(503).json({
            message: 'Database connection failed. Please try again later.',
            error: 'Database connection unavailable'
          });
          resolve();
        }
      }, 100); // Check every 100ms
      
      // Timeout after 5 seconds
      setTimeout(() => {
        clearInterval(checkInterval);
        if (mongoose.connection.readyState !== 1) {
          res.status(503).json({
            message: 'Database connection timeout. Please try again later.',
            error: 'Database connection timeout'
          });
          resolve();
        }
      }, 5000);
    });
  } else {
    // Disconnected (0) or Disconnecting (3)
    return res.status(503).json({
      message: 'Database is not connected. Please check the connection and try again.',
      error: 'Database connection unavailable',
      state: readyState === 0 ? 'disconnected' : 'disconnecting'
    });
  }
};

module.exports = { checkDatabaseConnection };

