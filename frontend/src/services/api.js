import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Create axios instances
export const authAPI = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

authAPI.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle errors
const responseInterceptor = (response) => response;

const errorInterceptor = (error) => {
  if (error.response?.status === 401) {
    // Token expired or invalid
    localStorage.removeItem('token');
    delete api.defaults.headers.common['Authorization'];
    delete authAPI.defaults.headers.common['Authorization'];
    window.location.href = '/login';
  }
  return Promise.reject(error);
};

api.interceptors.response.use(responseInterceptor, errorInterceptor);
authAPI.interceptors.response.use(responseInterceptor, errorInterceptor);

// API endpoints
export const authEndpoints = {
  login: (data) => authAPI.post('/auth/login', data),
  register: (data) => authAPI.post('/auth/register', data),
  logout: () => authAPI.post('/auth/logout'),
  getProfile: () => authAPI.get('/auth/profile'),
  updateProfile: (data) => authAPI.put('/auth/profile', data),
  changePassword: (data) => authAPI.put('/auth/change-password', data),
};

export const satelliteEndpoints = {
  fetchImage: (data) => api.post('/satellite/fetch', data),
  analyze: (data) => api.post('/satellite/analyze', data),
  getHistory: (params) => api.get('/satellite/history', { params }),
};

export const weatherEndpoints = {
  getCurrent: (data) => api.post('/weather/current', data),
  getForecast: (data) => api.post('/weather/forecast', data),
  getAlerts: (data) => api.post('/weather/alerts', data),
  getHistorical: (data) => api.post('/weather/historical', data),
};

export const reportEndpoints = {
  getAll: (params) => api.get('/reports', { params }),
  getById: (id) => api.get(`/reports/${id}`),
  update: (id, data) => api.put(`/reports/${id}`, data),
  delete: (id) => api.delete(`/reports/${id}`),
  downloadPDF: (id) => api.get(`/reports/${id}/download/pdf`, { responseType: 'blob' }),
  downloadCSV: (id) => api.get(`/reports/${id}/download/csv`, { responseType: 'blob' }),
  getPublic: (params) => api.get('/reports/public/list', { params }),
  getStats: () => api.get('/reports/stats/overview'),
};

export const userEndpoints = {
  getProfile: () => api.get('/users/profile'),
  updateProfile: (data) => api.put('/users/profile', data),
  getDashboard: () => api.get('/users/dashboard'),
  deleteAccount: () => api.delete('/users/account'),
};

// Utility functions
export const downloadFile = (blob, filename) => {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const formatDate = (date) => {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const formatRelativeTime = (date) => {
  const now = new Date();
  const diffInSeconds = Math.floor((now - new Date(date)) / 1000);
  
  if (diffInSeconds < 60) return 'Just now';
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`;
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
  if (diffInSeconds < 2592000) return `${Math.floor(diffInSeconds / 86400)} days ago`;
  
  return formatDate(date);
};

export default api;

