import React, { useState } from 'react';
import { useMutation } from 'react-query';
import { satelliteEndpoints, weatherEndpoints, reportEndpoints, api } from '../services/api';
import { useAuth } from '../contexts/AuthContext';
import Card from '../components/UI/Card';
import Button from '../components/UI/Button';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import toast from 'react-hot-toast';
import {
  Map,
  BarChart3,
  Download,
  AlertTriangle,
  CheckCircle,
  Globe,
} from 'lucide-react';

const Analysis = () => {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const { user } = useAuth();

  // Weather data mutation
  const weatherMutation = useMutation(weatherEndpoints.getCurrent, {
    onSuccess: (data) => {
      console.log('Weather data:', data.data);
    },
    onError: (error) => {
      console.error('Weather fetch error:', error);
    },
  });

  // Satellite analysis mutation
  const analysisMutation = useMutation(satelliteEndpoints.analyze, {
    onSuccess: (data) => {
      console.log('Analysis result received:', data.data.report);
      console.log('Satellite image available:', !!data.data.report?.satelliteImage);
      setAnalysisResult(data.data.report);
      setIsAnalyzing(false);
      toast.success('Analysis completed successfully!');
    },
    onError: (error) => {
      setIsAnalyzing(false);
      toast.error(error.response?.data?.message || 'Analysis failed');
    },
  });

  const onSubmit = async (data) => {
    console.log('Form submitted with data:', data);
    
    // Check if user is authenticated
    if (!user) {
      toast.error('Please log in to perform analysis');
      return;
    }
    
    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      // Step 1: Geocode the location to get coordinates
      const geocodeResponse = await api.post('/location/geocode', {
        location: data.location
      });

      const { coordinates } = geocodeResponse.data.location;

      // Step 2: Get weather data using coordinates
      const weatherResponse = await weatherMutation.mutateAsync({
        latitude: coordinates.latitude,
        longitude: coordinates.longitude,
        units: 'metric',
      });

      // Step 3: Perform satellite analysis
      await analysisMutation.mutateAsync({
        location: data.location,
        title: data.title,
        weatherData: weatherResponse.data.weather,
      });
    } catch (error) {
      setIsAnalyzing(false);
      console.error('Analysis error:', error);
      toast.error(error.response?.data?.message || 'Failed to complete analysis');
    }
  };

  const handleDownloadReport = async (format) => {
    if (!analysisResult?.report?.id) return;

    try {
      const endpoint = format === 'pdf' 
        ? reportEndpoints.downloadPDF 
        : reportEndpoints.downloadCSV;
      
      const response = await endpoint(analysisResult.report.id);
      const blob = new Blob([response.data]);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `report_${analysisResult.report.id}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      toast.success(`${format.toUpperCase()} report downloaded successfully!`);
    } catch (error) {
      toast.error(`Failed to download ${format.toUpperCase()} report`);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="md:flex md:items-center md:justify-between">
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
            Satellite Image Analysis
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Analyze satellite images for land classification, weather data, and environmental insights.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Analysis Form */}
        <div className="lg:col-span-1">
          <Card>
            <Card.Header>
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Start New Analysis
              </h3>
            </Card.Header>
            <Card.Body>
              <form onSubmit={(e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const data = {
                  title: formData.get('title'),
                  location: formData.get('location')
                };
                console.log('Form submitted with data:', data);
                
                // Simple validation
                if (!data.title || data.title.length < 3) {
                  alert('Title must be at least 3 characters');
                  return;
                }
                if (!data.location || data.location.length < 2) {
                  alert('Location must be at least 2 characters');
                  return;
                }
                
                onSubmit(data);
              }} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Analysis Title
                  </label>
                  <input
                    name="title"
                    type="text"
                    placeholder="Enter analysis title"
                    className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                    required
                    minLength={3}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Location
                  </label>
                  <input
                    name="location"
                    type="text"
                    placeholder="Enter city name, coordinates, or postal code"
                    className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                    required
                    minLength={2}
                  />
                  <p className="mt-1 text-xs text-gray-500">
                    Examples: "New York", "40.7128, -74.0060", "10001"
                  </p>
                </div>

                <Button
                  type="submit"
                  className="w-full"
                  loading={isAnalyzing}
                  disabled={isAnalyzing || !user}
                >
                  {isAnalyzing ? (
                    <>
                      <LoadingSpinner size="sm" className="mr-2" />
                      Analyzing...
                    </>
                  ) : !user ? (
                    <>
                      <Map className="w-4 h-4 mr-2" />
                      Please Log In to Start Analysis
                    </>
                  ) : (
                    <>
                      <Map className="w-4 h-4 mr-2" />
                      Start Analysis
                    </>
                  )}
                </Button>
              </form>
            </Card.Body>
          </Card>

          {/* Quick Tips */}
          <Card className="mt-6">
            <Card.Header>
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Analysis Tips
              </h3>
            </Card.Header>
            <Card.Body>
              <div className="space-y-3">
                <div className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-secondary-500 mt-0.5 mr-2 flex-shrink-0" />
                  <p className="text-sm text-gray-600">
                    Enter any location: city name, coordinates, or postal code
                  </p>
                </div>
                <div className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-secondary-500 mt-0.5 mr-2 flex-shrink-0" />
                  <p className="text-sm text-gray-600">
                    Coordinates provide the most precise analysis
                  </p>
                </div>
                <div className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-secondary-500 mt-0.5 mr-2 flex-shrink-0" />
                  <p className="text-sm text-gray-600">
                    Analysis typically takes 1-3 minutes
                  </p>
                </div>
                <div className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-secondary-500 mt-0.5 mr-2 flex-shrink-0" />
                  <p className="text-sm text-gray-600">
                    Results include weather and land classification data
                  </p>
                </div>
              </div>
            </Card.Body>
          </Card>
        </div>

        {/* Analysis Results */}
        <div className="lg:col-span-2">
          {isAnalyzing && (
            <Card>
              <Card.Body>
                <div className="text-center py-12">
                  <LoadingSpinner size="lg" className="mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Analyzing Satellite Images
                  </h3>
                  <p className="text-sm text-gray-500">
                    Processing satellite data and generating insights...
                  </p>
                  <div className="mt-4 flex justify-center">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-primary-600 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              </Card.Body>
            </Card>
          )}

          {analysisResult && (
            <div className="space-y-6">
              {/* Satellite Image - Full City Area */}
              {analysisResult.satelliteImage && analysisResult.satelliteImage.url ? (
                <Card>
                  <Card.Header>
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-lg leading-6 font-medium text-gray-900">
                          Satellite Image - Analyzed Area
                        </h3>
                        <p className="text-sm text-gray-500 mt-1">
                          Full city polygon area used for calculations
                        </p>
                      </div>
                    </div>
                  </Card.Header>
                  <Card.Body>
                    <div className="relative rounded-lg overflow-hidden bg-gray-100 border-2 border-primary-200">
                      <img
                        src={analysisResult.satelliteImage.url}
                        alt="Satellite image of analyzed city area"
                        className="w-full h-auto object-contain"
                        style={{ maxHeight: '700px', minHeight: '300px' }}
                        onError={(e) => {
                          console.error('Image load error:', e);
                          e.target.style.display = 'none';
                          e.target.nextSibling.style.display = 'block';
                        }}
                      />
                      <div className="hidden text-center py-8 text-gray-500">
                        <p>Failed to load satellite image</p>
                      </div>
                      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black via-black/80 to-transparent text-white p-3 text-xs">
                        <div className="flex flex-wrap justify-between gap-2">
                          <span className="flex items-center gap-1">
                            <Globe className="w-3 h-3" />
                            Source: {analysisResult.satelliteImage.source || 'Sentinel-2'}
                          </span>
                          {analysisResult.satelliteImage.timestamp && (
                            <span>
                              📅 {new Date(analysisResult.satelliteImage.timestamp).toLocaleDateString()}
                            </span>
                          )}
                          {analysisResult.satelliteImage.resolution && (
                            <span>
                              📏 Resolution: {analysisResult.satelliteImage.resolution}
                            </span>
                          )}
                          {analysisResult.location?.bounds && (
                            <span className="text-xs opacity-75">
                              🏙️ Full city polygon analyzed
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    {analysisResult.location?.bounds && (
                      <div className="mt-3 p-3 bg-primary-50 rounded-lg">
                        <p className="text-xs text-primary-700">
                          <strong>Analysis Coverage:</strong> This image covers the full administrative boundary of {analysisResult.location.name || 'the selected area'}.
                          All calculations (NDVI, land classification, risk assessment) are based on this complete area.
                        </p>
                      </div>
                    )}
                  </Card.Body>
                </Card>
              ) : (
                <Card>
                  <Card.Body>
                    <div className="text-center py-8 text-gray-500">
                      <Map className="mx-auto h-12 w-12 text-gray-400 mb-2" />
                      <p>Satellite image not available</p>
                    </div>
                  </Card.Body>
                </Card>
              )}

              {/* Analysis Summary */}
              <Card>
                <Card.Header>
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">
                      Analysis Results
                    </h3>
                    <div className="flex space-x-2">
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => handleDownloadReport('pdf')}
                      >
                        <Download className="w-4 h-4 mr-1" />
                        PDF
                      </Button>
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => handleDownloadReport('csv')}
                      >
                        <Download className="w-4 h-4 mr-1" />
                        CSV
                      </Button>
                    </div>
                  </div>
                </Card.Header>
                <Card.Body>
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                    <div className="bg-secondary-50 p-4 rounded-lg">
                      <div className="flex items-center">
                        <BarChart3 className="h-8 w-8 text-secondary-600" />
                        <div className="ml-3">
                          <p className="text-sm font-medium text-secondary-600">
                            Forest Coverage
                          </p>
                          <p className="text-2xl font-bold text-secondary-900">
                            {analysisResult.landClassification?.forest?.percentage || 0}%
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-primary-50 p-4 rounded-lg">
                      <div className="flex items-center">
                        <Globe className="h-8 w-8 text-primary-600" />
                        <div className="ml-3">
                          <p className="text-sm font-medium text-primary-600">
                            Water Bodies
                          </p>
                          <p className="text-2xl font-bold text-primary-900">
                            {analysisResult.landClassification?.water?.percentage || 0}%
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-warning-50 p-4 rounded-lg">
                      <div className="flex items-center">
                        <Map className="h-8 w-8 text-warning-600" />
                        <div className="ml-3">
                          <p className="text-sm font-medium text-warning-600">
                            Urban Area
                          </p>
                          <p className="text-2xl font-bold text-warning-900">
                            {analysisResult.landClassification?.urban?.percentage || 0}%
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="bg-danger-50 p-4 rounded-lg">
                      <div className="flex items-center">
                        <AlertTriangle className="h-8 w-8 text-danger-600" />
                        <div className="ml-3">
                          <p className="text-sm font-medium text-danger-600">
                            Risk Level
                          </p>
                          <p className="text-lg font-bold text-danger-900 capitalize">
                            {analysisResult.riskAssessment?.floodRisk?.level || 'Low'}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </Card.Body>
              </Card>

              {/* Detailed Analysis */}
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                {/* Land Classification */}
                <Card>
                  <Card.Header>
                    <h3 className="text-lg leading-6 font-medium text-gray-900">
                      Land Classification
                    </h3>
                  </Card.Header>
                  <Card.Body>
                    <div className="space-y-4">
                      {analysisResult.landClassification && Object.entries(analysisResult.landClassification).map(([type, data]) => (
                        <div key={type} className="flex items-center justify-between">
                          <div className="flex items-center">
                            <div className="w-3 h-3 rounded-full bg-primary-500 mr-3"></div>
                            <span className="text-sm font-medium text-gray-900 capitalize">
                              {type}
                            </span>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-bold text-gray-900">
                              {data.percentage}%
                            </p>
                            {data.health && (
                              <p className="text-xs text-gray-500 capitalize">
                                {data.health}
                              </p>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card.Body>
                </Card>

                {/* Risk Assessment */}
                <Card>
                  <Card.Header>
                    <h3 className="text-lg leading-6 font-medium text-gray-900">
                      Risk Assessment
                    </h3>
                  </Card.Header>
                  <Card.Body>
                    <div className="space-y-4">
                      {analysisResult.riskAssessment && Object.entries(analysisResult.riskAssessment).map(([risk, data]) => (
                        <div key={risk} className="flex items-center justify-between">
                          <div>
                            <p className="text-sm font-medium text-gray-900 capitalize">
                              {risk.replace(/([A-Z])/g, ' $1').trim()}
                            </p>
                            {data.probability && (
                              <p className="text-xs text-gray-500">
                                {data.probability.toFixed(1)}% probability
                              </p>
                            )}
                          </div>
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            data.level === 'low' ? 'bg-secondary-100 text-secondary-800' :
                            data.level === 'medium' ? 'bg-warning-100 text-warning-800' :
                            data.level === 'high' ? 'bg-danger-100 text-danger-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {data.level || (data.detected ? 'Detected' : 'Not Detected')}
                          </span>
                        </div>
                      ))}
                    </div>
                  </Card.Body>
                </Card>
              </div>
            </div>
          )}

          {!isAnalyzing && !analysisResult && (
            <Card>
              <Card.Body>
                <div className="text-center py-12">
                  <Map className="mx-auto h-12 w-12 text-gray-400" />
                  <h3 className="mt-2 text-sm font-medium text-gray-900">No analysis yet</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Enter a location and start your first satellite image analysis.
                  </p>
                </div>
              </Card.Body>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default Analysis;

