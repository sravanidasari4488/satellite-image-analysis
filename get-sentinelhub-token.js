#!/usr/bin/env node

const axios = require('axios');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const question = (query) => new Promise((resolve) => rl.question(query, resolve));

async function getSentinelHubToken() {
  console.log('🔑 Sentinel Hub Token Generator\n');
  console.log('This script will help you get an OAuth token for Sentinel Hub.\n');

  try {
    // Get credentials from user
    const clientId = await question('Enter your Sentinel Hub Client ID: ');
    const clientSecret = await question('Enter your Sentinel Hub Client Secret: ');

    if (!clientId || !clientSecret) {
      console.log('❌ Both Client ID and Client Secret are required!');
      rl.close();
      return;
    }

    console.log('\n🔄 Generating token...');

    // Generate OAuth token
    const response = await axios.post('https://services.sentinel-hub.com/oauth/token', {
      grant_type: 'client_credentials',
      client_id: clientId,
      client_secret: clientSecret
    }, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      }
    });

    const { access_token, expires_in } = response.data;

    console.log('✅ Token generated successfully!\n');
    console.log('📋 Your credentials:');
    console.log(`Client ID: ${clientId}`);
    console.log(`Client Secret: ${clientSecret}`);
    console.log(`Access Token: ${access_token}`);
    console.log(`Expires in: ${expires_in} seconds\n`);

    // Update .env file
    const fs = require('fs');
    const path = require('path');
    
    const envPath = path.join(__dirname, 'backend', '.env');
    
    if (fs.existsSync(envPath)) {
      let envContent = fs.readFileSync(envPath, 'utf8');
      
      // Update or add Sentinel Hub credentials
      envContent = envContent.replace(/SENTINEL_HUB_CLIENT_ID=.*/g, `SENTINEL_HUB_CLIENT_ID=${clientId}`);
      envContent = envContent.replace(/SENTINEL_HUB_CLIENT_SECRET=.*/g, `SENTINEL_HUB_CLIENT_SECRET=${clientSecret}`);
      envContent = envContent.replace(/SENTINEL_HUB_TOKEN=.*/g, `SENTINEL_HUB_TOKEN=${access_token}`);
      
      // Add if not present
      if (!envContent.includes('SENTINEL_HUB_CLIENT_ID')) {
        envContent += `\nSENTINEL_HUB_CLIENT_ID=${clientId}`;
      }
      if (!envContent.includes('SENTINEL_HUB_CLIENT_SECRET')) {
        envContent += `\nSENTINEL_HUB_CLIENT_SECRET=${clientSecret}`;
      }
      if (!envContent.includes('SENTINEL_HUB_TOKEN')) {
        envContent += `\nSENTINEL_HUB_TOKEN=${access_token}`;
      }
      
      fs.writeFileSync(envPath, envContent);
      console.log('✅ Updated backend/.env file with your credentials!');
    } else {
      console.log('⚠️  backend/.env file not found. Please create it manually.');
    }

    console.log('\n🎉 Setup complete! You can now use Sentinel Hub in your application.');
    console.log('\n📝 Note: Tokens expire after a certain time. Your app will automatically refresh them.');

  } catch (error) {
    console.error('❌ Error generating token:', error.response?.data || error.message);
    console.log('\n💡 Make sure your Client ID and Client Secret are correct.');
    console.log('   You can find them in Sentinel Hub Dashboard > User settings > OAuth clients');
  }

  rl.close();
}

getSentinelHubToken();




