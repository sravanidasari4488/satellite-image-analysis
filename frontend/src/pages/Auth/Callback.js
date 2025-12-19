import React, { useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import LoadingSpinner from '../../components/UI/LoadingSpinner';

const AuthCallback = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { checkAuthStatus } = useAuth();

  useEffect(() => {
    const handleCallback = async () => {
      try {
        const token = searchParams.get('token');
        
        if (token) {
          // Store the token
          localStorage.setItem('token', token);
          
          // Check auth status to get user data
          await checkAuthStatus();
          
          // Redirect to dashboard
          navigate('/dashboard', { replace: true });
        } else {
          // No token, redirect to login with error
          navigate('/login?error=oauth_failed', { replace: true });
        }
      } catch (error) {
        console.error('OAuth callback error:', error);
        navigate('/login?error=oauth_failed', { replace: true });
      }
    };

    handleCallback();
  }, [searchParams, navigate, checkAuthStatus]);

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <LoadingSpinner size="lg" />
        <p className="mt-4 text-gray-600">Completing sign in...</p>
      </div>
    </div>
  );
};

export default AuthCallback;









