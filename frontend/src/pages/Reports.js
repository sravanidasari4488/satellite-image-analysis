import React, { useState } from 'react';
import { useQuery } from 'react-query';
import { reportEndpoints } from '../services/api';
import Card from '../components/UI/Card';
import Button from '../components/UI/Button';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import {
  FileText,
  Download,
  Eye,
  Trash2,
  Calendar,
  MapPin,
  BarChart3,
  Filter,
} from 'lucide-react';

const Reports = () => {
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState('');

  const { data: reportsData, isLoading, refetch } = useQuery(
    ['reports', page, statusFilter],
    () => reportEndpoints.getAll({ page, limit: 10, status: statusFilter }),
    {
      select: (response) => response.data,
    }
  );

  const handleDownload = async (reportId, format) => {
    try {
      const endpoint = format === 'pdf' 
        ? reportEndpoints.downloadPDF 
        : reportEndpoints.downloadCSV;
      
      const response = await endpoint(reportId);
      const blob = new Blob([response.data]);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `report_${reportId}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download error:', error);
    }
  };

  const handleDelete = async (reportId) => {
    if (window.confirm('Are you sure you want to delete this report?')) {
      try {
        await reportEndpoints.delete(reportId);
        refetch();
      } catch (error) {
        console.error('Delete error:', error);
      }
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  const { reports, pagination } = reportsData || {};

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="md:flex md:items-center md:justify-between">
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
            Reports
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            View and manage your satellite image analysis reports.
          </p>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <Card.Body>
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <Filter className="h-5 w-5 text-gray-400 mr-2" />
              <label htmlFor="status-filter" className="text-sm font-medium text-gray-700">
                Status:
              </label>
            </div>
            <select
              id="status-filter"
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="">All</option>
              <option value="completed">Completed</option>
              <option value="processing">Processing</option>
              <option value="failed">Failed</option>
            </select>
          </div>
        </Card.Body>
      </Card>

      {/* Reports List */}
      {reports && reports.length > 0 ? (
        <div className="space-y-4">
          {reports.map((report) => (
            <Card key={report._id}>
              <Card.Body>
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center">
                      <h3 className="text-lg font-medium text-gray-900 truncate">
                        {report.title}
                      </h3>
                      <span className={`ml-3 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        report.status === 'completed'
                          ? 'bg-secondary-100 text-secondary-800'
                          : report.status === 'processing'
                          ? 'bg-warning-100 text-warning-800'
                          : 'bg-danger-100 text-danger-800'
                      }`}>
                        {report.status}
                      </span>
                    </div>
                    <div className="mt-1 flex items-center text-sm text-gray-500">
                      <MapPin className="h-4 w-4 mr-1" />
                      {report.location.name}
                    </div>
                    <div className="mt-1 flex items-center text-sm text-gray-500">
                      <Calendar className="h-4 w-4 mr-1" />
                      {new Date(report.createdAt).toLocaleDateString()}
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDownload(report._id, 'pdf')}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDownload(report._id, 'csv')}
                    >
                      <FileText className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDelete(report._id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </Card.Body>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <Card.Body>
            <div className="text-center py-12">
              <FileText className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No reports found</h3>
              <p className="mt-1 text-sm text-gray-500">
                Get started by creating your first analysis.
              </p>
            </div>
          </Card.Body>
        </Card>
      )}

      {/* Pagination */}
      {pagination && pagination.total > 1 && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-700">
            Showing {pagination.count} of {pagination.totalCount} reports
          </div>
          <div className="flex space-x-2">
            <Button
              variant="secondary"
              size="sm"
              disabled={pagination.current === 1}
              onClick={() => setPage(page - 1)}
            >
              Previous
            </Button>
            <Button
              variant="secondary"
              size="sm"
              disabled={pagination.current === pagination.total}
              onClick={() => setPage(page + 1)}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Reports;

