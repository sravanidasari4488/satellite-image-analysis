#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const question = (query) => new Promise((resolve) => rl.question(query, resolve));

async function configureKeys() {
  console.log('🔑 API Keys Configuration Tool\n');
  console.log('This tool will help you configure your API keys.\n');
  console.log('Press Enter to skip any key you don\'t have yet.\n');

  const keys = {};

  // OpenWeatherMap API
  console.log('🌤️  OpenWeatherMap API (Weather Data)');
  console.log('Get your key at: https://openweathermap.org/api');
  keys.OPENWEATHER_API_KEY = await question('Enter OpenWeatherMap API Key: ');

  // OpenCage API
  console.log('\n📍 OpenCage Geocoding API (Location Services)');
  console.log('Get your key at: https://opencagedata.com/api');
  keys.OPENCAGE_API_KEY = await question('Enter OpenCage API Key: ');

  // Google OAuth
  console.log('\n🔐 Google OAuth (User Authentication)');
  console.log('Get your credentials at: https://console.cloud.google.com/');
  keys.GOOGLE_CLIENT_ID = await question('Enter Google Client ID: ');
  keys.GOOGLE_CLIENT_SECRET = await question('Enter Google Client Secret: ');

  // SentinelHub API
  console.log('\n🛰️  SentinelHub API (Satellite Images)');
  console.log('Get your credentials at: https://www.sentinel-hub.com/');
  keys.SENTINELHUB_CLIENT_ID = await question('Enter SentinelHub Client ID: ');
  keys.SENTINELHUB_CLIENT_SECRET = await question('Enter SentinelHub Client Secret: ');
  keys.SENTINELHUB_TOKEN = await question('Enter SentinelHub Token: ');

  // MongoDB Atlas
  console.log('\n🗄️  MongoDB Atlas (Database)');
  console.log('Get your connection string at: https://www.mongodb.com/atlas');
  keys.MONGODB_ATLAS_URI = await question('Enter MongoDB Atlas URI (or press Enter for local MongoDB): ');

  rl.close();

  // Update backend .env
  const backendEnvPath = path.join(__dirname, 'backend', '.env');
  if (fs.existsSync(backendEnvPath)) {
    let backendEnv = fs.readFileSync(backendEnvPath, 'utf8');
    
    Object.entries(keys).forEach(([key, value]) => {
      if (value.trim()) {
        const regex = new RegExp(`^${key}=.*$`, 'm');
        if (regex.test(backendEnv)) {
          backendEnv = backendEnv.replace(regex, `${key}=${value}`);
        } else {
          backendEnv += `\n${key}=${value}`;
        }
      }
    });

    fs.writeFileSync(backendEnvPath, backendEnv);
    console.log('\n✅ Backend .env file updated!');
  }

  // Update frontend .env
  const frontendEnvPath = path.join(__dirname, 'frontend', '.env');
  if (fs.existsSync(frontendEnvPath)) {
    let frontendEnv = fs.readFileSync(frontendEnvPath, 'utf8');
    
    if (keys.GOOGLE_CLIENT_ID.trim()) {
      const regex = /^REACT_APP_GOOGLE_CLIENT_ID=.*$/m;
      if (regex.test(frontendEnv)) {
        frontendEnv = frontendEnv.replace(regex, `REACT_APP_GOOGLE_CLIENT_ID=${keys.GOOGLE_CLIENT_ID}`);
      } else {
        frontendEnv += `\nREACT_APP_GOOGLE_CLIENT_ID=${keys.GOOGLE_CLIENT_ID}`;
      }
    }

    fs.writeFileSync(frontendEnvPath, frontendEnv);
    console.log('✅ Frontend .env file updated!');
  }

  console.log('\n🎉 Configuration complete!');
  console.log('\n📋 Next steps:');
  console.log('1. Start MongoDB (local or use the Atlas connection string)');
  console.log('2. Run: npm run dev');
  console.log('3. Access your app at: http://localhost:3000');
  
  console.log('\n📖 For detailed setup instructions, see SETUP.md');
}

configureKeys().catch(console.error);




