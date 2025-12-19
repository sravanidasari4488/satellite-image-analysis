@echo off
echo Starting Satellite Image Analysis Application...
echo.

echo Checking if Node.js is installed...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo Checking if Python is installed...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org/
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install root dependencies
    pause
    exit /b 1
)

call npm run install-all
if %errorlevel% neq 0 (
    echo ERROR: Failed to install all dependencies
    pause
    exit /b 1
)

echo.
echo Setting up environment files...
if not exist "backend\.env" (
    if exist "backend\env.example" (
        copy "backend\env.example" "backend\.env"
        echo Created backend\.env from example
    ) else (
        echo WARNING: backend\env.example not found
    )
)

if not exist "frontend\.env" (
    if exist "frontend\.env.example" (
        copy "frontend\.env.example" "frontend\.env"
        echo Created frontend\.env from example
    ) else (
        echo WARNING: frontend\.env.example not found
    )
)

if not exist "ai-models\.env" (
    if exist "ai-models\.env.example" (
        copy "ai-models\.env.example" "ai-models\.env"
        echo Created ai-models\.env from example
    ) else (
        echo WARNING: ai-models\.env.example not found
    )
)

echo.
echo Installing Python dependencies for AI service...
cd ai-models
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)
cd ..

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo IMPORTANT: Before running the application:
echo 1. Edit backend\.env and add your API keys
echo 2. Edit frontend\.env and add your API keys
echo 3. Make sure MongoDB is running
echo.
echo To start the application, run:
echo   npm run dev
echo.
echo Or start individual services:
echo   npm run server  (backend only)
echo   npm run client  (frontend only)
echo.
echo For more information, see setup.md
echo.
pause


