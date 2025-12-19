import React from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from 'react-query';
import { userEndpoints, reportEndpoints } from '../services/api';
import Card from '../components/UI/Card';
import Button from '../components/UI/Button';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import {
  BarChart3,
  Map,
  FileText,
  Plus,
  TrendingUp,
  Globe,
  Clock,
  AlertTriangle,
} from 'lucide-react';

const Dashboard = () => {
  const { data: dashboardData, isLoading: dashboardLoading } = useQuery(
    'dashboard',
    userEndpoints.getDashboard,
    {
      select: (response) => response.data.dashboard,
    }
  );

  const { data: statsData, isLoading: statsLoading } = useQuery(
    'reportStats',
    reportEndpoints.getStats,
    {
      select: (response) => response.data.stats,
    }
  );

  if (dashboardLoading || statsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  const { recentReports, stats, topLocations } = dashboardData || {};
  const { totalReports, completedReports, publicReports } = stats || {};

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="md:flex md:items-center md:justify-between">
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
            Dashboard
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Welcome back! Here's an overview of your satellite image analysis activities.
          </p>
        </div>
        <div className="mt-4 flex md:mt-0 md:ml-4">
          <Link to="/analysis">
            <Button>
              <Plus className="w-4 h-4 mr-2" />
              New Analysis
            </Button>
          </Link>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <Card.Body>
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <BarChart3 className="h-8 w-8 text-primary-600" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">
                    Total Reports
                  </dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {totalReports || 0}
                  </dd>
                </dl>
              </div>
            </div>
          </Card.Body>
        </Card>

        <Card>
          <Card.Body>
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <TrendingUp className="h-8 w-8 text-secondary-600" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">
                    Completed
                  </dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {completedReports || 0}
                  </dd>
                </dl>
              </div>
            </div>
          </Card.Body>
        </Card>

        <Card>
          <Card.Body>
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Globe className="h-8 w-8 text-warning-600" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">
                    Public Reports
                  </dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {publicReports || 0}
                  </dd>
                </dl>
              </div>
            </div>
          </Card.Body>
        </Card>

        <Card>
          <Card.Body>
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Map className="h-8 w-8 text-danger-600" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">
                    Locations Analyzed
                  </dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {topLocations?.length || 0}
                  </dd>
                </dl>
              </div>
            </div>
          </Card.Body>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Recent Reports */}
        <Card>
          <Card.Header>
            <div className="flex items-center justify-between">
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Recent Reports
              </h3>
              <Link
                to="/reports"
                className="text-sm font-medium text-primary-600 hover:text-primary-500"
              >
                View all
              </Link>
            </div>
          </Card.Header>
          <Card.Body>
            {recentReports && recentReports.length > 0 ? (
              <div className="space-y-4">
                {recentReports.map((report) => (
                  <div
                    key={report._id}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {report.title}
                      </p>
                      <p className="text-sm text-gray-500 truncate">
                        {report.location.name}
                      </p>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        report.status === 'completed'
                          ? 'bg-secondary-100 text-secondary-800'
                          : report.status === 'processing'
                          ? 'bg-warning-100 text-warning-800'
                          : 'bg-danger-100 text-danger-800'
                      }`}>
                        {report.status}
                      </span>
                      <Link
                        to={`/reports/${report._id}`}
                        className="text-primary-600 hover:text-primary-500"
                      >
                        <FileText className="h-4 w-4" />
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6">
                <FileText className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900">No reports yet</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Get started by creating your first analysis.
                </p>
                <div className="mt-6">
                  <Link to="/analysis">
                    <Button>
                      <Plus className="w-4 h-4 mr-2" />
                      Create Analysis
                    </Button>
                  </Link>
                </div>
              </div>
            )}
          </Card.Body>
        </Card>

        {/* Top Locations */}
        <Card>
          <Card.Header>
            <h3 className="text-lg leading-6 font-medium text-gray-900">
              Most Analyzed Locations
            </h3>
          </Card.Header>
          <Card.Body>
            {topLocations && topLocations.length > 0 ? (
              <div className="space-y-4">
                {topLocations.map((location, index) => (
                  <div
                    key={location._id}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                  >
                    <div className="flex items-center">
                      <div className="flex-shrink-0">
                        <div className="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center">
                          <span className="text-sm font-medium text-primary-600">
                            {index + 1}
                          </span>
                        </div>
                      </div>
                      <div className="ml-3">
                        <p className="text-sm font-medium text-gray-900">
                          {location._id}
                        </p>
                        <p className="text-sm text-gray-500">
                          {location.count} analysis{location.count !== 1 ? 'es' : ''}
                        </p>
                      </div>
                    </div>
                    <Map className="h-4 w-4 text-gray-400" />
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6">
                <Globe className="mx-auto h-12 w-12 text-gray-400" />
                <h3 className="mt-2 text-sm font-medium text-gray-900">No locations yet</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Start analyzing locations to see them here.
                </p>
              </div>
            )}
          </Card.Body>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <Card.Header>
          <h3 className="text-lg leading-6 font-medium text-gray-900">
            Quick Actions
          </h3>
        </Card.Header>
        <Card.Body>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Link to="/analysis" className="group">
              <div className="p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-md transition-all duration-200">
                <div className="flex items-center">
                  <Map className="h-8 w-8 text-primary-600 group-hover:text-primary-700" />
                  <div className="ml-3">
                    <p className="text-sm font-medium text-gray-900">New Analysis</p>
                    <p className="text-xs text-gray-500">Analyze a location</p>
                  </div>
                </div>
              </div>
            </Link>

            <Link to="/reports" className="group">
              <div className="p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-md transition-all duration-200">
                <div className="flex items-center">
                  <FileText className="h-8 w-8 text-secondary-600 group-hover:text-secondary-700" />
                  <div className="ml-3">
                    <p className="text-sm font-medium text-gray-900">View Reports</p>
                    <p className="text-xs text-gray-500">Browse all reports</p>
                  </div>
                </div>
              </div>
            </Link>

            <Link to="/profile" className="group">
              <div className="p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-md transition-all duration-200">
                <div className="flex items-center">
                  <BarChart3 className="h-8 w-8 text-warning-600 group-hover:text-warning-700" />
                  <div className="ml-3">
                    <p className="text-sm font-medium text-gray-900">Profile</p>
                    <p className="text-xs text-gray-500">Manage account</p>
                  </div>
                </div>
              </div>
            </Link>

            <a href="#" className="group">
              <div className="p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-md transition-all duration-200">
                <div className="flex items-center">
                  <AlertTriangle className="h-8 w-8 text-danger-600 group-hover:text-danger-700" />
                  <div className="ml-3">
                    <p className="text-sm font-medium text-gray-900">Help</p>
                    <p className="text-xs text-gray-500">Get support</p>
                  </div>
                </div>
              </div>
            </a>
          </div>
        </Card.Body>
      </Card>
    </div>
  );
};

export default Dashboard;

