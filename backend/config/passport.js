const passport = require('passport');
const mongoose = require('mongoose');
const GoogleStrategy = require('passport-google-oauth20').Strategy;
const JwtStrategy = require('passport-jwt').Strategy;
const ExtractJwt = require('passport-jwt').ExtractJwt;
const User = require('../models/User');

// Helper function to wait for MongoDB connection
async function waitForConnection(maxWait = 30000) {
  const startTime = Date.now();
  while (mongoose.connection.readyState !== 1 && (Date.now() - startTime) < maxWait) {
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  return mongoose.connection.readyState === 1;
}

// JWT Strategy
passport.use(new JwtStrategy({
  jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
  secretOrKey: process.env.JWT_SECRET
}, async (payload, done) => {
  try {
    // Wait for MongoDB connection if not ready
    if (mongoose.connection.readyState !== 1) {
      const connected = await waitForConnection(10000); // Wait up to 10 seconds
      if (!connected) {
        return done(new Error('Database connection unavailable'), false);
      }
    }

    const user = await User.findById(payload.userId).select('-password');
    if (user) {
      return done(null, user);
    }
    return done(null, false);
  } catch (error) {
    return done(error, false);
  }
}));

// Google OAuth Strategy (only initialize if credentials are provided)
if (process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET && 
    process.env.GOOGLE_CLIENT_ID !== 'your-google-client-id' && 
    process.env.GOOGLE_CLIENT_SECRET !== 'your-google-client-secret') {
  // Callback URL should be the backend URL
  const backendURL = process.env.BACKEND_URL || 'http://localhost:5000';
  const callbackURL = `${backendURL}/api/auth/google/callback`;
  
  passport.use(new GoogleStrategy({
    clientID: process.env.GOOGLE_CLIENT_ID,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    callbackURL: callbackURL
  }, async (accessToken, refreshToken, profile, done) => {
  try {
    // Wait for MongoDB connection if not ready
    if (mongoose.connection.readyState !== 1) {
      const connected = await waitForConnection(30000); // Wait up to 30 seconds
      if (!connected) {
        return done(new Error('Database connection timeout. Please try again.'), null);
      }
    }

    // Check if user already exists
    let user = await User.findOne({ googleId: profile.id });
    
    if (user) {
      // Update last login
      user.lastLogin = new Date();
      await user.save();
      return done(null, user);
    }

    // Check if user exists with same email
    user = await User.findOne({ email: profile.emails[0].value });
    
    if (user) {
      // Link Google account to existing user
      user.googleId = profile.id;
      user.avatar = profile.photos[0]?.value;
      user.lastLogin = new Date();
      await user.save();
      return done(null, user);
    }

    // Create new user
    user = new User({
      name: profile.displayName,
      email: profile.emails[0].value,
      googleId: profile.id,
      avatar: profile.photos[0]?.value,
      isEmailVerified: true, // Google emails are verified
      lastLogin: new Date()
    });

    await user.save();
    return done(null, user);
  } catch (error) {
    console.error('Google OAuth error:', error);
    return done(error, null);
  }
  }));
} else {
  console.warn('⚠️  Google OAuth credentials not configured. Google sign-in will not be available.');
}

// Serialize user for session
passport.serializeUser((user, done) => {
  done(null, user._id);
});

// Deserialize user from session
passport.deserializeUser(async (id, done) => {
  try {
    const user = await User.findById(id).select('-password');
    done(null, user);
  } catch (error) {
    done(error, null);
  }
});

module.exports = passport;

