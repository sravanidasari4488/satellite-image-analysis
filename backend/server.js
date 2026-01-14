const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const passport = require('passport');
const session = require('express-session');
require('dotenv').config();

// Import passport configuration
require('./config/passport');

const app = express();

// Security middleware
app.use(helmet());
app.use(compression());

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});
app.use('/api/', limiter);

// CORS configuration
app.use(cors({
  origin: true, // Allow all origins for testing
  credentials: true
}));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Session configuration
app.use(session({
  secret: process.env.JWT_SECRET || 'your-secret-key',
  resave: false,
  saveUninitialized: false,
  cookie: { secure: false } // Set to true in production with HTTPS
}));

// Passport middleware
app.use(passport.initialize());
app.use(passport.session());

// Database connection with improved timeout settings
mongoose.set('strictQuery', true);
// Enable buffering with longer timeout for OAuth callbacks
mongoose.set('bufferCommands', true);

const mongoOptions = {
  serverSelectionTimeoutMS: 30000, // 30 seconds
  socketTimeoutMS: 45000, // 45 seconds
  connectTimeoutMS: 30000, // 30 seconds
  maxPoolSize: 10,
  retryWrites: true,
  w: 'majority'
};

const mongoUri = process.env.MONGODB_URI || process.env.MONGODB_ATLAS_URI;

// Connection event handlers
mongoose.connection.on('connected', () => {
  console.log('✅ MongoDB connected successfully');
  const dbName = mongoUri && mongoUri.includes('@') 
    ? mongoUri.split('@')[1]?.split('/')[1]?.split('?')[0] 
    : 'local';
  console.log(`📊 Database: ${dbName}`);
});

mongoose.connection.on('error', (err) => {
  console.error('❌ MongoDB connection error:', err.message);
});

mongoose.connection.on('disconnected', () => {
  console.warn('⚠️  MongoDB disconnected');
});

mongoose.connection.on('reconnected', () => {
  console.log('🔄 MongoDB reconnected');
});

// Function to connect to MongoDB
async function connectToDatabase() {
  if (!mongoUri) {
    console.error('❌ MongoDB URI not found in environment variables!');
    console.error('Please set MONGODB_URI or MONGODB_ATLAS_URI in your .env file');
    return false;
  }

  try {
    await mongoose.connect(mongoUri, mongoOptions);
    console.log('✅ MongoDB connection established');
    return true;
  } catch (err) {
    console.error('❌ MongoDB connection failed:', err.message);
    console.error('💡 Troubleshooting:');
    console.error('   1. Check if MongoDB Atlas cluster is running (not paused)');
    console.error('   2. Verify your connection string in .env file');
    console.error('   3. Check if your IP is whitelisted in MongoDB Atlas (use 0.0.0.0/0 for all IPs)');
    console.error('   4. Verify network connectivity');
    console.error('   5. Check if credentials are correct');
    return false;
  }
}

// Health check endpoint (before routes)
app.get('/api/health', (req, res) => {
  res.status(200).json({
    status: 'OK',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Routes
app.use('/api/auth', require('./routes/auth'));
app.use('/api/location', require('./routes/location'));
app.use('/api/satellite', require('./routes/satellite'));
app.use('/api/city-gee', require('./routes/city-gee'));
app.use('/api/weather', require('./routes/weather'));
app.use('/api/analysis', require('./routes/analysis'));
app.use('/api/reports', require('./routes/reports'));
app.use('/api/users', require('./routes/users'));

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    message: 'Something went wrong!',
    error: process.env.NODE_ENV === 'development' ? err.message : {}
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Endpoint not found', path: req.originalUrl });
});

const PORT = process.env.PORT || 5000;

// Start server only after MongoDB connection is established
async function startServer() {
  console.log('🔄 Connecting to MongoDB...');
  const dbConnected = await connectToDatabase();
  
  if (!dbConnected) {
    console.warn('⚠️  Starting server without database connection. Some features may not work.');
    console.warn('💡 The server will continue to retry connecting in the background.');
  }

  app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
    console.log(`📝 Environment: ${process.env.NODE_ENV || 'development'}`);
    if (dbConnected) {
      console.log('✅ All systems ready!');
    } else {
      console.log('⚠️  Waiting for database connection...');
    }
  });
}

// Start the server
startServer().catch(err => {
  console.error('❌ Failed to start server:', err);
  process.exit(1);
});

module.exports = app;