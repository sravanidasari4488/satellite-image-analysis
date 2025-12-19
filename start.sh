#!/bin/bash

echo "Starting Satellite Image Analysis Application..."
echo

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed or not in PATH"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    echo "Please install Python from https://python.org/"
    exit 1
fi

echo "Installing dependencies..."
npm install
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install root dependencies"
    exit 1
fi

npm run install-all
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install all dependencies"
    exit 1
fi

echo
echo "Setting up environment files..."

# Create .env files from examples if they don't exist
if [ ! -f "backend/.env" ] && [ -f "backend/env.example" ]; then
    cp backend/env.example backend/.env
    echo "Created backend/.env from example"
fi

if [ ! -f "frontend/.env" ] && [ -f "frontend/.env.example" ]; then
    cp frontend/.env.example frontend/.env
    echo "Created frontend/.env from example"
fi

if [ ! -f "ai-models/.env" ] && [ -f "ai-models/.env.example" ]; then
    cp ai-models/.env.example ai-models/.env
    echo "Created ai-models/.env from example"
fi

echo
echo "Installing Python dependencies for AI service..."
cd ai-models
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Python dependencies"
    exit 1
fi
cd ..

echo
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo
echo "IMPORTANT: Before running the application:"
echo "1. Edit backend/.env and add your API keys"
echo "2. Edit frontend/.env and add your API keys"
echo "3. Make sure MongoDB is running"
echo
echo "To start the application, run:"
echo "  npm run dev"
echo
echo "Or start individual services:"
echo "  npm run server  (backend only)"
echo "  npm run client  (frontend only)"
echo
echo "For more information, see setup.md"
echo


