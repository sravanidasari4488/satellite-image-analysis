#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('🚀 Setting up Satellite Image Analysis Application...\n');

// Check if Node.js version is compatible
const nodeVersion = process.version;
const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0]);

if (majorVersion < 16) {
  console.error('❌ Node.js version 16 or higher is required. Current version:', nodeVersion);
  process.exit(1);
}

console.log('✅ Node.js version check passed:', nodeVersion);

// Create .env files if they don't exist
const createEnvFile = (filePath, content) => {
  if (!fs.existsSync(filePath)) {
    fs.writeFileSync(filePath, content);
    console.log(`✅ Created ${filePath}`);
  } else {
    console.log(`⚠️  ${filePath} already exists, skipping...`);
  }
};

// Backend .env
const backendEnvPath = path.join(__dirname, 'backend', '.env');
const backendEnvContent = `# Server Configuration
PORT=5000
NODE_ENV=development

# Database
MONGODB_URI=mongodb://localhost:27017/satellite-analysis

# JWT
JWT_SECRET=your-super-secret-jwt-key-here-change-this-in-production
JWT_EXPIRE=7d

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# API Keys
OPENWEATHER_API_KEY=your-openweathermap-api-key
SENTINELHUB_CLIENT_ID=your-sentinelhub-client-id
SENTINELHUB_CLIENT_SECRET=your-sentinelhub-client-secret
SENTINELHUB_TOKEN=your-sentinelhub-token
OPENCAGE_API_KEY=your-opencage-api-key

# Frontend URL
CLIENT_URL=http://localhost:3000

# AI Service
AI_SERVICE_URL=http://localhost:5001
`;

// Frontend .env
const frontendEnvPath = path.join(__dirname, 'frontend', '.env');
const frontendEnvContent = `# API Configuration
REACT_APP_API_URL=http://localhost:5000/api

# Google OAuth (if using client-side OAuth)
REACT_APP_GOOGLE_CLIENT_ID=your-google-client-id
`;

// AI Models .env
const aiEnvPath = path.join(__dirname, 'ai-models', '.env');
const aiEnvContent = `# AI Service Configuration
FLASK_ENV=development
PORT=5001
DEBUG=True

# Model Configuration
MODEL_PATH=./models
LOG_LEVEL=INFO
`;

console.log('\n📝 Creating environment files...');
createEnvFile(backendEnvPath, backendEnvContent);
createEnvFile(frontendEnvPath, frontendEnvContent);
createEnvFile(aiEnvPath, aiEnvContent);

// Install dependencies
console.log('\n📦 Installing dependencies...');

try {
  console.log('Installing root dependencies...');
  execSync('npm install', { stdio: 'inherit' });
  
  console.log('Installing backend dependencies...');
  execSync('cd backend && npm install', { stdio: 'inherit' });
  
  console.log('Installing frontend dependencies...');
  execSync('cd frontend && npm install', { stdio: 'inherit' });
  
  console.log('Installing AI service dependencies...');
  execSync('cd ai-models && pip install -r requirements.txt', { stdio: 'inherit' });
  
  console.log('✅ All dependencies installed successfully!');
} catch (error) {
  console.error('❌ Error installing dependencies:', error.message);
  console.log('\nPlease install dependencies manually:');
  console.log('1. npm install');
  console.log('2. cd backend && npm install');
  console.log('3. cd frontend && npm install');
  console.log('4. cd ai-models && pip install -r requirements.txt');
}

// Create directories
console.log('\n📁 Creating necessary directories...');
const directories = [
  'backend/uploads',
  'backend/logs',
  'ai-models/models',
  'ai-models/temp'
];

directories.forEach(dir => {
  const dirPath = path.join(__dirname, dir);
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
    console.log(`✅ Created directory: ${dir}`);
  }
});

console.log('\n🎉 Setup completed successfully!');
console.log('\n📋 Next steps:');
console.log('1. Configure your API keys in the .env files');
console.log('2. Start MongoDB (local or use MongoDB Atlas)');
console.log('3. Run the application:');
console.log('   - Development: npm run dev');
console.log('   - Docker: docker-compose up --build');
console.log('\n📖 For detailed setup instructions, see SETUP.md');
console.log('\n🌐 Access the application at:');
console.log('   - Frontend: http://localhost:3000');
console.log('   - Backend API: http://localhost:5000');
console.log('   - AI Service: http://localhost:5001');

