<<<<<<< HEAD
import React, { useState } from 'react';
import './App.css';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [city, setCity] = useState('');
  const [locality, setLocality] = useState('');
  const [localities, setLocalities] = useState([]);
  const [loadingLocalities, setLoadingLocalities] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleCitySubmit = async (e) => {
    e.preventDefault();
    
    if (!city.trim()) {
      setError('Please enter a city name');
      return;
    }

    setLoadingLocalities(true);
    setError(null);
    setLocalities([]);
    setLocality('');
    setResult(null);

    try {
      const response = await axios.post(`${API_URL}/localities`, {
        city: city.trim(),
        radius_km: 18
      });
      
      setLocalities(response.data.localities || []);
      if (response.data.localities.length === 0) {
        setError('No localities found for this city. Try a different city name.');
      }
    } catch (err) {
      setError(
        err.response?.data?.error || 
        err.message || 
        'Failed to fetch localities. Please try again.'
      );
    } finally {
      setLoadingLocalities(false);
    }
  };

  const handleLocalityAnalyze = async (e) => {
    e.preventDefault();
    
    if (!city.trim() || !locality.trim()) {
      setError('Please select a locality');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_URL}/analyze`, {
        city: city.trim(),
        locality: locality.trim()
      });
      
      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.error || 
        err.message || 
        'Failed to analyze locality. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1 className="title">
            <span className="icon">🛰️</span>
            Geospatial Intelligence System
          </h1>
          <p className="subtitle">
            Real-time satellite-based land cover analysis and climate risk assessment
          </p>
        </header>

        <form onSubmit={handleCitySubmit} className="search-form">
          <div className="input-group">
            <input
              type="text"
              value={city}
              onChange={(e) => setCity(e.target.value)}
              placeholder="Enter city name (e.g., Hyderabad, Delhi)"
              className="location-input"
              disabled={loadingLocalities || loading}
            />
            <button 
              type="submit" 
              className="submit-button"
              disabled={loadingLocalities || loading}
            >
              {loadingLocalities ? (
                <>
                  <span className="spinner"></span>
                  Loading...
                </>
              ) : (
                <>
                  <span>🔍</span>
                  Find Localities
                </>
              )}
            </button>
          </div>
        </form>

        {localities.length > 0 && (
          <form onSubmit={handleLocalityAnalyze} className="search-form">
            <div className="input-group">
              <select
                value={locality}
                onChange={(e) => setLocality(e.target.value)}
                className="location-input"
                disabled={loading}
                style={{ padding: '12px', fontSize: '16px', borderRadius: '8px', border: '1px solid #ddd' }}
              >
                <option value="">Select a locality...</option>
                {localities.map((loc, index) => (
                  <option key={index} value={loc.name}>
                    {loc.name}
                  </option>
                ))}
              </select>
              <button 
                type="submit" 
                className="submit-button"
                disabled={loading || !locality}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <span>🔍</span>
                    Analyze
                  </>
                )}
              </button>
            </div>
          </form>
        )}

        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}

        {result && (
          <div className="results">
            <div className="result-section">
              <h2 className="section-title">📍 Location Information</h2>
              <div className="info-grid">
                <div className="info-item">
                  <span className="info-label">City:</span>
                  <span className="info-value">{result.city}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Locality:</span>
                  <span className="info-value">{result.locality}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Satellite Source:</span>
                  <span className="info-value">{result.satellite_source}</span>
                </div>
              </div>
            </div>

            <div className="result-section">
              <h2 className="section-title">🌍 Land Cover Classification</h2>
              <div className="land-cover-grid">
                <div className="land-cover-item">
                  <div className="land-cover-header">
                    <span className="land-cover-icon">🏙️</span>
                    <span className="land-cover-label">Urban</span>
                  </div>
                  <div className="land-cover-value">{result.landcover_percentages.urban.toFixed(2)}%</div>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill urban" 
                      style={{ width: `${result.landcover_percentages.urban}%` }}
                    ></div>
                  </div>
                </div>

                <div className="land-cover-item">
                  <div className="land-cover-header">
                    <span className="land-cover-icon">🌲</span>
                    <span className="land-cover-label">Forest</span>
                  </div>
                  <div className="land-cover-value">{result.landcover_percentages.forest.toFixed(2)}%</div>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill forest" 
                      style={{ width: `${result.landcover_percentages.forest}%` }}
                    ></div>
                  </div>
                </div>

                <div className="land-cover-item">
                  <div className="land-cover-header">
                    <span className="land-cover-icon">🌿</span>
                    <span className="land-cover-label">Vegetation</span>
                  </div>
                  <div className="land-cover-value">{result.landcover_percentages.vegetation.toFixed(2)}%</div>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill vegetation" 
                      style={{ width: `${result.landcover_percentages.vegetation}%` }}
                    ></div>
                  </div>
                </div>

                <div className="land-cover-item">
                  <div className="land-cover-header">
                    <span className="land-cover-icon">💧</span>
                    <span className="land-cover-label">Water</span>
                  </div>
                  <div className="land-cover-value">{result.landcover_percentages.water.toFixed(2)}%</div>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill water" 
                      style={{ width: `${result.landcover_percentages.water}%` }}
                    ></div>
                  </div>
                </div>

                <div className="land-cover-item">
                  <div className="land-cover-header">
                    <span className="land-cover-icon">🏜️</span>
                    <span className="land-cover-label">Bare Land</span>
                  </div>
                  <div className="land-cover-value">{result.landcover_percentages.bare_land.toFixed(2)}%</div>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill bare-land" 
                      style={{ width: `${result.landcover_percentages.bare_land}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>

            {result.weather && (
              <div className="result-section">
                <h2 className="section-title">🌤️ Weather Data</h2>
                <div className="weather-grid">
                  <div className="weather-item">
                    <span className="weather-icon">🌡️</span>
                    <div className="weather-details">
                      <span className="weather-label">Temperature</span>
                      <span className="weather-value">{result.weather.temperature}°C</span>
                    </div>
                  </div>
                  <div className="weather-item">
                    <span className="weather-icon">🌧️</span>
                    <div className="weather-details">
                      <span className="weather-label">Rainfall</span>
                      <span className="weather-value">{result.weather.rainfall} mm</span>
                    </div>
                  </div>
                  <div className="weather-item">
                    <span className="weather-icon">💨</span>
                    <div className="weather-details">
                      <span className="weather-label">Humidity</span>
                      <span className="weather-value">{result.weather.humidity}%</span>
                    </div>
                  </div>
                  <div className="weather-item">
                    <span className="weather-icon">💨</span>
                    <div className="weather-details">
                      <span className="weather-label">Wind Speed</span>
                      <span className="weather-value">{result.weather.wind_speed} m/s</span>
                    </div>
                  </div>
                  <div className="weather-item">
                    <span className="weather-icon">📊</span>
                    <div className="weather-details">
                      <span className="weather-label">Pressure</span>
                      <span className="weather-value">{result.weather.pressure} hPa</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="result-section">
              <h2 className="section-title">⚠️ Climate Risk Assessment</h2>
              <div className="risk-grid">
                <div className="risk-item">
                  <div className="risk-header">
                    <span className="risk-icon">🌊</span>
                    <span className="risk-label">Flood Risk</span>
                  </div>
                  <div 
                    className="risk-value"
                    style={{ 
                      color: result.flood_risk === 'High' ? '#ef4444' : 
                             result.flood_risk === 'Medium' ? '#f59e0b' : '#10b981' 
                    }}
                  >
                    {result.flood_risk}
                  </div>
                </div>

                <div className="risk-item">
                  <div className="risk-header">
                    <span className="risk-icon">🔥</span>
                    <span className="risk-label">Heat Risk</span>
                  </div>
                  <div 
                    className="risk-value"
                    style={{ 
                      color: result.heat_risk === 'High' ? '#ef4444' : 
                             result.heat_risk === 'Medium' ? '#f59e0b' : '#10b981' 
                    }}
                  >
                    {result.heat_risk}
                  </div>
                </div>

                <div className="risk-item">
                  <div className="risk-header">
                    <span className="risk-icon">🌵</span>
                    <span className="risk-label">Drought Risk</span>
                  </div>
                  <div 
                    className="risk-value"
                    style={{ 
                      color: result.drought_risk === 'High' ? '#ef4444' : 
                             result.drought_risk === 'Medium' ? '#f59e0b' : '#10b981' 
                    }}
                  >
                    {result.drought_risk}
                  </div>
                </div>
              </div>
            </div>

            {result.disasters !== undefined && (
              <div className="result-section">
                <h2 className="section-title">🚨 Natural Disaster Alerts</h2>
                {result.disasters && result.disasters.length > 0 ? (
                  <div className="disaster-grid">
                    {result.disasters.map((disaster, index) => (
                      <div key={index} className="disaster-card" style={{
                        borderLeft: `4px solid ${
                          disaster.severity === 'High' ? '#ef4444' : 
                          disaster.severity === 'Medium' ? '#f59e0b' : '#10b981'
                        }`
                      }}>
                        <div className="disaster-header">
                          <span className="disaster-icon">
                            {disaster.type === 'earthquake' ? '🌍' : 
                             disaster.type === 'cyclone' ? '🌀' : '⚠️'}
                          </span>
                          <span className="disaster-title">{disaster.title}</span>
                          <span className="disaster-severity" style={{
                            color: disaster.severity === 'High' ? '#ef4444' : 
                                   disaster.severity === 'Medium' ? '#f59e0b' : '#10b981'
                          }}>
                            {disaster.severity}
                          </span>
                        </div>
                        <div className="disaster-details">
                          {disaster.distance_km > 0 && (
                            <span className="disaster-distance">{disaster.distance_km} km away</span>
                          )}
                          <span className="disaster-time">{disaster.time}</span>
                        </div>
                        {disaster.description && (
                          <div className="disaster-description">{disaster.description}</div>
                        )}
                        <div className="disaster-source">Source: {disaster.source}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="no-disasters">
                    <span className="no-disasters-icon">✅</span>
                    <span className="no-disasters-text">No active natural disasters near this locality</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
=======
import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './contexts/AuthContext';
import Layout from './components/Layout/Layout';
import Home from './pages/Home';
import Login from './pages/Auth/Login';
import Register from './pages/Auth/Register';
import AuthCallback from './pages/Auth/Callback';
import Dashboard from './pages/Dashboard';
import Analysis from './pages/Analysis';
import Reports from './pages/Reports';
import Profile from './pages/Profile';
import LoadingSpinner from './components/UI/LoadingSpinner';

// Protected Route Component
const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();
  
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }
  
  return user ? children : <Navigate to="/login" replace />;
};

// Public Route Component (redirect to dashboard if logged in)
const PublicRoute = ({ children }) => {
  const { user, loading } = useAuth();
  
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }
  
  return user ? <Navigate to="/dashboard" replace /> : children;
};

function App() {
  return (
    <div className="App">
      <Routes>
        {/* Public Routes */}
        <Route path="/" element={<Layout><Home /></Layout>} />
        <Route 
          path="/login" 
          element={
            <PublicRoute>
              <Layout><Login /></Layout>
            </PublicRoute>
          } 
        />
        <Route 
          path="/register" 
          element={
            <PublicRoute>
              <Layout><Register /></Layout>
            </PublicRoute>
          } 
        />
        <Route path="/auth/callback" element={<AuthCallback />} />
        
        {/* Protected Routes */}
        <Route 
          path="/dashboard" 
          element={
            <ProtectedRoute>
              <Layout><Dashboard /></Layout>
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/analysis" 
          element={
            <ProtectedRoute>
              <Layout><Analysis /></Layout>
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/reports" 
          element={
            <ProtectedRoute>
              <Layout><Reports /></Layout>
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/profile" 
          element={
            <ProtectedRoute>
              <Layout><Profile /></Layout>
            </ProtectedRoute>
          } 
        />
        
        {/* Catch all route */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
>>>>>>> a141e052c246e59fa901f6cb767b91a1e45ab2d2
    </div>
  );
}

export default App;
<<<<<<< HEAD
=======

>>>>>>> a141e052c246e59fa901f6cb767b91a1e45ab2d2
