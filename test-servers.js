#!/usr/bin/env node

const http = require('http');

console.log('🧪 Testing Satellite Image Analysis Servers...\n');

// Test backend server
function testBackend() {
  return new Promise((resolve) => {
    const req = http.get('http://localhost:5000/api/health', (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        console.log('✅ Backend Server (Port 5000): RUNNING');
        console.log('   Response:', data);
        resolve(true);
      });
    });
    
    req.on('error', (err) => {
      console.log('❌ Backend Server (Port 5000): NOT RUNNING');
      console.log('   Error:', err.message);
      resolve(false);
    });
    
    req.setTimeout(5000, () => {
      console.log('❌ Backend Server (Port 5000): TIMEOUT');
      req.destroy();
      resolve(false);
    });
  });
}

// Test frontend server
function testFrontend() {
  return new Promise((resolve) => {
    const req = http.get('http://localhost:3000', (res) => {
      console.log('✅ Frontend Server (Port 3000): RUNNING');
      console.log('   Status:', res.statusCode);
      resolve(true);
    });
    
    req.on('error', (err) => {
      console.log('❌ Frontend Server (Port 3000): NOT RUNNING');
      console.log('   Error:', err.message);
      resolve(false);
    });
    
    req.setTimeout(5000, () => {
      console.log('❌ Frontend Server (Port 3000): TIMEOUT');
      req.destroy();
      resolve(false);
    });
  });
}

async function testServers() {
  console.log('Testing servers...\n');
  
  const backendRunning = await testBackend();
  const frontendRunning = await testFrontend();
  
  console.log('\n📊 Test Results:');
  console.log('Backend:', backendRunning ? '✅ Running' : '❌ Not Running');
  console.log('Frontend:', frontendRunning ? '✅ Running' : '❌ Not Running');
  
  if (backendRunning && frontendRunning) {
    console.log('\n🎉 Both servers are running!');
    console.log('🌐 Open your browser and go to: http://localhost:3000');
  } else {
    console.log('\n⚠️  Some servers are not running. Check the terminal for errors.');
  }
}

testServers();









