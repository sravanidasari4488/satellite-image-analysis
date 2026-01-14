@echo off
echo Starting Satellite Image Analysis Application...
echo.

echo Installing dependencies...
call npm run install-all

echo.
echo Starting development servers...
echo Backend will run on http://localhost:5000
echo Frontend will run on http://localhost:3000
echo.

call npm run dev


