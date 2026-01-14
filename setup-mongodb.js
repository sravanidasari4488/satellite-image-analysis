#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const question = (query) => new Promise((resolve) => rl.question(query, resolve));

async function setupMongoDB() {
  console.log('🗄️  MongoDB Setup for Satellite Image Analysis\n');
  console.log('Since Docker is not available, we\'ll use MongoDB Atlas (cloud database).\n');

  console.log('📋 Steps to get MongoDB Atlas connection string:\n');
  console.log('1. Go to https://www.mongodb.com/atlas');
  console.log('2. Sign up for a free account');
  console.log('3. Create a new cluster (free tier)');
  console.log('4. Create a database user');
  console.log('5. Get your connection string\n');

  const useAtlas = await question('Do you want to use MongoDB Atlas? (y/n): ');
  
  if (useAtlas.toLowerCase() === 'y') {
    const connectionString = await question('Enter your MongoDB Atlas connection string: ');
    
    if (connectionString) {
      // Create .env file content
      const envContent = `# Server Configuration
PORT=5000
NODE_ENV=development

# Database - MongoDB Atlas
MONGODB_URI=${connectionString}

# JWT
JWT_SECRET=your-super-secret-jwt-key-here-change-this-in-production-12345
JWT_EXPIRE=7d

# Google OAuth (Optional)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

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
UPLOAD_PATH=./uploads`;

      // Write to backend/.env
      const envPath = path.join(__dirname, 'backend', '.env');
      fs.writeFileSync(envPath, envContent);
      
      console.log('✅ Created backend/.env file with MongoDB Atlas configuration!');
      console.log('\n🚀 You can now start your application with: npm run dev');
    }
  } else {
    console.log('\n💡 Alternative options:');
    console.log('1. Install MongoDB locally: https://www.mongodb.com/try/download/community');
    console.log('2. Install Docker Desktop: https://www.docker.com/products/docker-desktop');
    console.log('3. Use MongoDB Atlas (recommended for development)');
  }

  rl.close();
}

setupMongoDB();









