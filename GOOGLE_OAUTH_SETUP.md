# Google OAuth Setup Guide

## 🚨 **Current Issue**
The Google OAuth is showing "invalid_client" error because the Google OAuth credentials are not properly configured.

## 🔧 **How to Fix This**

### **Step 1: Create Google OAuth Credentials**

1. **Go to Google Cloud Console**: https://console.cloud.google.com/
2. **Create a new project** or select existing one
3. **Enable Google+ API**:
   - Go to "APIs & Services" > "Library"
   - Search for "Google+ API"
   - Click "Enable"

4. **Create OAuth 2.0 Credentials**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth 2.0 Client IDs"
   - Choose "Web application"
   - Add these **Authorized redirect URIs**:
     ```
     http://localhost:5000/api/auth/google/callback
     http://localhost:3000/api/auth/google/callback
     ```
   - Click "Create"

5. **Copy your credentials**:
   - **Client ID**: Copy this value
   - **Client Secret**: Copy this value

### **Step 2: Update Your Environment File**

Create or update `backend/.env` file with:

```env
# Google OAuth Credentials
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here

# Other existing variables...
PORT=5000
NODE_ENV=development
MONGODB_URI=mongodb+srv://satellite-analysis:satellite123@cluster0.aj8ttad.mongodb.net/satellite-analysis?retryWrites=true&w=majority&appName=Cluster0
JWT_SECRET=your-super-secret-jwt-key-here-change-this-in-production-12345
JWT_EXPIRE=7d
OPENWEATHER_API_KEY=your-openweather-api-key
OPENCAGE_API_KEY=your-opencage-api-key
SENTINEL_HUB_CLIENT_ID=937e58bf-b2c6-43bc-96d1-ba03f872acb6
SENTINEL_HUB_CLIENT_SECRET=aXTTmp06qyd1dTK8m4XIaMsVhNiGp6cp
SENTINEL_HUB_TOKEN=eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJ3dE9hV1o2aFJJeUowbGlsYXctcWd4NzlUdm1hX3ZKZlNuMW1WNm5HX0tVIn0.eyJleHAiOjE3NTg0NjAxMzgsImlhdCI6MTc1ODQ1NjUzOCwianRpIjoiNjAzMDViMmMtNGMwYy00ZmEwLTljZDItZTg1MjIxOWUyNDQyIiwiaXNzIjoiaHR0cHM6Ly9zZXJ2aWNlcy5zZW50aW5lbC1odWIuY29tL2F1dGgvcmVhbG1zL21haW4iLCJhdWQiOiJodHRwczovL2FwaS5wbGFuZXQuY29tLyIsInN1YiI6ImI5MTIyNTg1LTcyMGMtNGU0Yi1hZDJmLWU3MjEwYWEyOWM2NiIsInR5cCI6IkJlYXJlciIsImF6cCI6IjkzN2U1OGJmLWIyYzYtNDNiYy05NmQxLWJhMDNmODcyYWNiNiIsInNjb3BlIjoiZW1haWwgcHJvZmlsZSIsImNsaWVudEhvc3QiOiI0OS40Ny4yNTUuMjUzIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJwbF9wcm9qZWN0IjoiNmVlNWJkMDUtODFjMy00OTY1LWJmMTAtNTc1ZmJiZjMzYWI1IiwicHJlZmVycmVkX3VzZXJuYW1lIjoic2VydmljZS1hY2NvdW50LTkzN2U1OGJmLWIyYzYtNDNiYy05NmQxLWJhMDNmODcyYWNiNiIsImNsaWVudEFkZHJlc3MiOiI0OS40Ny4yNTUuMjUzIiwiY2xpZW50X2lkIjoiOTM3ZTU4YmYtYjJjNi00M2JjLTk2ZDEtYmEwM2Y4NzJhY2I2IiwiYWNjb3VudCI6IjZlZTViZDA1LTgxYzMtNDk2NS1iZjEwLTU3NWZiYmYzM2FiNSIsInBsX3dvcmtzcGFjZSI6Ijk1NTM0MGYwLTg5MzUtNDRmZC1hOWY4LTllN2YxZGQ5ZTJmZCJ9.lMtcZNA5Hh_CtaXZVDhYqpHJo1evWrXaP6XMJ-01fCl_QRzGGk4yuvc_jLkcLMSH5_kLCaLeU1PYY8XlDVG_qEgqDVKCAkokus_pETY1-CF38PAbS4MsqhyPTHOvUdaCIQ8NvJc2LJTFBvE-KpGZ5BEXXgYPXTN7YnWUBhusKAjeeDoYJca_ytQYBFylG2oOv1behBlKjolBeNpmd600XU1FDJ58YcceIlOo3EMcJIf7Ptcr89yQMGAS12hOmGPAMzekifLeQCs9iS3QYeO4VbPJmB29ZDEk_WKX-qDsLgOqEc8Dp4oQMSuDmO30BTbVZpuREBM5O1nke3eq5r8Ofw
AI_SERVICE_URL=http://localhost:8000
MAX_FILE_SIZE=10485760
UPLOAD_PATH=./uploads
```

### **Step 3: Restart the Server**

After updating the .env file:

1. **Stop the current server** (Ctrl+C in terminal)
2. **Restart the application**:
   ```bash
   npm run dev
   ```

### **Step 4: Test Google OAuth**

1. **Open**: http://localhost:3000
2. **Click**: "Sign in with Google"
3. **Should redirect** to Google OAuth page successfully

## 🔍 **Troubleshooting**

### **If you still get errors:**

1. **Check redirect URIs** in Google Console:
   - Must include: `http://localhost:5000/api/auth/google/callback`
   - Must include: `http://localhost:3000/api/auth/google/callback`

2. **Verify environment variables**:
   - Make sure `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` are set
   - No extra spaces or quotes around the values

3. **Check Google Console**:
   - Make sure the OAuth consent screen is configured
   - Add your email as a test user if in testing mode

## 📝 **Quick Setup Commands**

If you want to set up Google OAuth quickly:

1. **Get your Google OAuth credentials** (follow Step 1 above)
2. **Update the .env file** with your credentials
3. **Restart the server**

## ✅ **Expected Result**

After proper setup:
- Google OAuth should redirect to Google's login page
- After login, should redirect back to your application
- User should be logged in successfully

---

**Need help?** Make sure to:
1. ✅ Enable Google+ API in Google Cloud Console
2. ✅ Add correct redirect URIs
3. ✅ Copy exact Client ID and Client Secret
4. ✅ Restart the server after updating .env









