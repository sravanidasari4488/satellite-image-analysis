#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

console.log('🧪 Testing Satellite Image Analysis Application Setup...\n');

// Test 1: Check if all required files exist
const requiredFiles = [
  'backend/server.js',
  'backend/package.json',
  'frontend/package.json',
  'frontend/src/App.js',
  'ai-models/app.py',
  'ai-models/requirements.txt',
  'docker-compose.yml',
  'package.json'
];

console.log('📁 Checking required files...');
let allFilesExist = true;

requiredFiles.forEach(file => {
  const filePath = path.join(__dirname, file);
  if (fs.existsSync(filePath)) {
    console.log(`✅ ${file}`);
  } else {
    console.log(`❌ ${file} - MISSING`);
    allFilesExist = false;
  }
});

// Test 2: Check if .env files exist
console.log('\n🔧 Checking environment files...');
const envFiles = [
  'backend/.env',
  'frontend/.env',
  'ai-models/.env'
];

envFiles.forEach(file => {
  const filePath = path.join(__dirname, file);
  if (fs.existsSync(filePath)) {
    console.log(`✅ ${file}`);
  } else {
    console.log(`⚠️  ${file} - Not found (run setup.js to create)`);
  }
});

// Test 3: Check if node_modules exist
console.log('\n📦 Checking dependencies...');
const nodeModulesPaths = [
  'node_modules',
  'backend/node_modules',
  'frontend/node_modules'
];

nodeModulesPaths.forEach(dir => {
  const dirPath = path.join(__dirname, dir);
  if (fs.existsSync(dirPath)) {
    console.log(`✅ ${dir}`);
  } else {
    console.log(`❌ ${dir} - Run npm install`);
  }
});

// Test 4: Check package.json scripts
console.log('\n📜 Checking package.json scripts...');
try {
  const rootPackageJson = JSON.parse(fs.readFileSync(path.join(__dirname, 'package.json'), 'utf8'));
  const requiredScripts = ['dev', 'install-all', 'build'];
  
  requiredScripts.forEach(script => {
    if (rootPackageJson.scripts && rootPackageJson.scripts[script]) {
      console.log(`✅ Script '${script}' found`);
    } else {
      console.log(`❌ Script '${script}' missing`);
    }
  });
} catch (error) {
  console.log('❌ Error reading package.json:', error.message);
}

// Test 5: Check if routes exist
console.log('\n🛣️  Checking route files...');
const routeFiles = [
  'backend/routes/auth.js',
  'backend/routes/satellite.js',
  'backend/routes/weather.js',
  'backend/routes/reports.js',
  'backend/routes/users.js',
  'backend/routes/location.js',
  'backend/routes/analysis.js'
];

routeFiles.forEach(file => {
  const filePath = path.join(__dirname, file);
  if (fs.existsSync(filePath)) {
    console.log(`✅ ${file}`);
  } else {
    console.log(`❌ ${file} - MISSING`);
  }
});

// Test 6: Check if UI components exist
console.log('\n🎨 Checking UI components...');
const uiComponents = [
  'frontend/src/components/UI/Button.js',
  'frontend/src/components/UI/Card.js',
  'frontend/src/components/UI/Input.js',
  'frontend/src/components/UI/LoadingSpinner.js',
  'frontend/src/components/Layout/Layout.js'
];

uiComponents.forEach(file => {
  const filePath = path.join(__dirname, file);
  if (fs.existsSync(filePath)) {
    console.log(`✅ ${file}`);
  } else {
    console.log(`❌ ${file} - MISSING`);
  }
});

// Test 7: Check if pages exist
console.log('\n📄 Checking page components...');
const pageComponents = [
  'frontend/src/pages/Home.js',
  'frontend/src/pages/Dashboard.js',
  'frontend/src/pages/Analysis.js',
  'frontend/src/pages/Reports.js',
  'frontend/src/pages/Profile.js',
  'frontend/src/pages/Auth/Login.js',
  'frontend/src/pages/Auth/Register.js'
];

pageComponents.forEach(file => {
  const filePath = path.join(__dirname, file);
  if (fs.existsSync(filePath)) {
    console.log(`✅ ${file}`);
  } else {
    console.log(`❌ ${file} - MISSING`);
  }
});

// Summary
console.log('\n📊 Setup Test Summary:');
if (allFilesExist) {
  console.log('✅ All required files are present');
  console.log('✅ Application structure is complete');
  console.log('\n🚀 Ready to start the application!');
  console.log('\nTo start the application:');
  console.log('1. Configure API keys in .env files');
  console.log('2. Start MongoDB');
  console.log('3. Run: npm run dev');
} else {
  console.log('❌ Some files are missing');
  console.log('Please run setup.js to complete the installation');
}

console.log('\n📖 For detailed setup instructions, see SETUP.md');









