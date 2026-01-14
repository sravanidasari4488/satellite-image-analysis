#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const question = (query) => new Promise((resolve) => rl.question(query, resolve));

async function setupGoogleOAuth() {
  console.log('🔐 Google OAuth Setup for Satellite Image Analysis\n');
  
  console.log('📋 Before we start, you need to:');
  console.log('1. Go to https://console.cloud.google.com/');
  console.log('2. Create a new project or select existing one');
  console.log('3. Enable Google+ API');
  console.log('4. Create OAuth 2.0 Client ID');
  console.log('5. Add redirect URIs:');
  console.log('   - http://localhost:5000/api/auth/google/callback');
  console.log('   - http://localhost:3000/api/auth/google/callback\n');

  const continueSetup = await question('Have you completed the Google Console setup? (y/n): ');
  
  if (continueSetup.toLowerCase() !== 'y') {
    console.log('\n📖 Please follow the steps in GOOGLE_OAUTH_SETUP.md first.');
    console.log('Then run this script again.');
    rl.close();
    return;
  }

  console.log('\n🔑 Now let\'s configure your credentials:\n');

  const clientId = await question('Enter your Google Client ID: ');
  const clientSecret = await question('Enter your Google Client Secret: ');

  if (!clientId || !clientSecret) {
    console.log('\n❌ Both Client ID and Client Secret are required!');
    rl.close();
    return;
  }

  // Read existing .env file or create new one
  const envPath = path.join(__dirname, 'backend', '.env');
  let envContent = '';

  if (fs.existsSync(envPath)) {
    envContent = fs.readFileSync(envPath, 'utf8');
  } else {
    // Create default .env content
    envContent = `# Server Configuration
PORT=5000
NODE_ENV=development

# Database - MongoDB Atlas
MONGODB_URI=mongodb+srv://satellite-analysis:satellite123@cluster0.aj8ttad.mongodb.net/satellite-analysis?retryWrites=true&w=majority&appName=Cluster0

# JWT
JWT_SECRET=your-super-secret-jwt-key-here-change-this-in-production-12345
JWT_EXPIRE=7d

# Weather API
OPENWEATHER_API_KEY=your-openweather-api-key

# Geocoding API
OPENCAGE_API_KEY=your-opencage-api-key

# Sentinel Hub API
SENTINEL_HUB_CLIENT_ID=937e58bf-b2c6-43bc-96d1-ba03f872acb6
SENTINEL_HUB_CLIENT_SECRET=aXTTmp06qyd1dTK8m4XIaMsVhNiGp6cp
SENTINEL_HUB_TOKEN=eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ3dE9hV1o2aFJJeUowbGlsYXctcWd4NzlUdm1hX3ZKZlNuMW1WNm5HX0tVIn0.eyJleHAiOjE3NTg0NjAxMzgsImlhdCI6MTc1ODQ1NjUzOCwianRpIjoiNjAzMDViMmMtNGMwYy00ZmEwLTljZDItZTg1MjIxOWUyNDQyIiwiaXNzIjoiaHR0cHM6Ly9zZXJ2aWNlcy5zZW50aW5lbC1odWIuY29tL2F1dGgvcmVhbG1zL21haW4iLCJhdWQiOiJodHRwczovL2FwaS5wbGFuZXQuY29tLyIsInN1YiI6ImI5MTIyNTg1LTcyMGMtNGU0Yi1hZDJmLWU3MjEwYWEyOWM2NiIsInR5cCI6IkJlYXJlciIsImF6cCI6IjkzN2U1OGJmLWIyYzYtNDNiYy05NmQxLWJhMDNmODcyYWNiNiIsInNjb3BlIjoiZW1haWwgcHJvZmlsZSIsImNsaWVudEhvc3QiOiI0OS40Ny4yNTUuMjUzIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJwbF9wcm9qZWN0IjoiNmVlNWJkMDUtODFjMy00OTY1LWJmMTAtNTc1ZmJiZjMzYWI1IiwicHJlZmVycmVkX3VzZXJuYW1lIjoic2VydmljZS1hY2NvdW50LTkzN2U1OGJmLWIyYzYtNDNiYy05NmQxLWJhMDNmODcyYWNiNiIsImNsaWVudEFkZHJlc3MiOiI0OS40Ny4yNTUuMjUzIiwiY2xpZW50X2lkIjoiOTM3ZTU4YmYtYjJjNi00M2JjLTk2ZDEtYmEwM2Y4NzJhY2I2IiwiYWNjb3VudCI6IjZlZTViZDA1LTgxYzMtNDk2NS1iZjEwLTU3NWZiYmYzM2FiNSIsInBsX3dvcmtzcGFjZSI6Ijk1NTM0MGYwLTg5MzUtNDRmZC1hOWY4LTllN2YxZGQ5ZTJmZCJ9.lMtcZNA5Hh_CtaXZVDhYqpHJo1evWrXaP6XMJ-01fCl_QRzGGk4yuvc_jLkcLMSH5_kLCaLeU1PYY8XlDVG_qEgqDVKCAkokus_pETY1-CF38PAbS4MsqhyPTHOvUdaCIQ8NvJc2LJTFBvE-KpGZ5BEXXgYPXTN7YnWUBhusKAjeeDoYJca_ytQYBFylG2oOv1behBlKjolBeNpmd600XU1FDJ58YcceIlOo3EMcJIf7Ptcr89yQMGAS12hOmGPAMzekifLeQCs9iS3QYeO4VbPJmB29ZDEk_WKX-qDsLgOqEc8Dp4oQMSuDmO30BTbVZpuREBM5O1nke3eq5r8Ofw

# AI Service
AI_SERVICE_URL=http://localhost:8000

# File Upload
MAX_FILE_SIZE=10485760
UPLOAD_PATH=./uploads
`;
  }

  // Update or add Google OAuth credentials
  envContent = envContent.replace(/GOOGLE_CLIENT_ID=.*/g, `GOOGLE_CLIENT_ID=${clientId}`);
  envContent = envContent.replace(/GOOGLE_CLIENT_SECRET=.*/g, `GOOGLE_CLIENT_SECRET=${clientSecret}`);
  
  // Add if not present
  if (!envContent.includes('GOOGLE_CLIENT_ID')) {
    envContent += `\n# Google OAuth\nGOOGLE_CLIENT_ID=${clientId}\nGOOGLE_CLIENT_SECRET=${clientSecret}`;
  }

  // Write the updated .env file
  fs.writeFileSync(envPath, envContent);
  
  console.log('\n✅ Google OAuth credentials configured successfully!');
  console.log('📁 Updated backend/.env file');
  
  console.log('\n🚀 Next steps:');
  console.log('1. Restart your server: npm run dev');
  console.log('2. Open http://localhost:3000');
  console.log('3. Click "Sign in with Google"');
  console.log('4. Should work now! 🎉');

  rl.close();
}

setupGoogleOAuth();









