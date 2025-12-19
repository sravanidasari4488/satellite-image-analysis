import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { useAuth } from '../contexts/AuthContext';
import Card from '../components/UI/Card';
import Button from '../components/UI/Button';
import Input from '../components/UI/Input';
import LoadingSpinner from '../components/UI/LoadingSpinner';
import {
  User,
  Mail,
  Settings,
  Key,
  Trash2,
  Save,
  Eye,
  EyeOff,
} from 'lucide-react';

const Profile = () => {
  const { user, updateProfile, changePassword, logout } = useAuth();
  const [activeTab, setActiveTab] = useState('profile');
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const [isChangingPassword, setIsChangingPassword] = useState(false);

  const {
    register: registerProfile,
    handleSubmit: handleSubmitProfile,
    formState: { errors: profileErrors },
  } = useForm({
    defaultValues: {
      name: user?.name || '',
      preferences: {
        units: user?.preferences?.units || 'metric',
        notifications: {
          email: user?.preferences?.notifications?.email || true,
          push: user?.preferences?.notifications?.push || false,
        },
      },
    },
  });

  const {
    register: registerPassword,
    handleSubmit: handleSubmitPassword,
    watch,
    formState: { errors: passwordErrors },
    reset: resetPassword,
  } = useForm();

  const newPassword = watch('newPassword');

  const onProfileSubmit = async (data) => {
    setIsUpdating(true);
    const result = await updateProfile(data);
    setIsUpdating(false);
    
    if (result.success) {
      // Profile updated successfully
    }
  };

  const onPasswordSubmit = async (data) => {
    setIsChangingPassword(true);
    const result = await changePassword(data.currentPassword, data.newPassword);
    setIsChangingPassword(false);
    
    if (result.success) {
      resetPassword();
    }
  };

  const handleDeleteAccount = async () => {
    if (window.confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
      if (window.confirm('This will permanently delete all your data. Type "DELETE" to confirm.')) {
        // Implement account deletion
        console.log('Account deletion requested');
      }
    }
  };

  const tabs = [
    { id: 'profile', name: 'Profile', icon: User },
    { id: 'password', name: 'Password', icon: Key },
    { id: 'settings', name: 'Settings', icon: Settings },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="md:flex md:items-center md:justify-between">
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
            Profile Settings
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Manage your account settings and preferences.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-4">
        {/* Sidebar */}
        <div className="lg:col-span-1">
          <Card>
            <Card.Body>
              <nav className="space-y-1">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`${
                      activeTab === tab.id
                        ? 'bg-primary-100 text-primary-700'
                        : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                    } group flex items-center px-3 py-2 text-sm font-medium rounded-md w-full text-left`}
                  >
                    <tab.icon
                      className={`${
                        activeTab === tab.id ? 'text-primary-500' : 'text-gray-400 group-hover:text-gray-500'
                      } mr-3 h-5 w-5`}
                    />
                    {tab.name}
                  </button>
                ))}
              </nav>
            </Card.Body>
          </Card>
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3">
          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <Card>
              <Card.Header>
                <h3 className="text-lg leading-6 font-medium text-gray-900">
                  Profile Information
                </h3>
              </Card.Header>
              <Card.Body>
                <form onSubmit={handleSubmitProfile(onProfileSubmit)} className="space-y-6">
                  <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                    <div>
                      <Input
                        label="Full Name"
                        type="text"
                        error={profileErrors.name?.message}
                        {...registerProfile('name', {
                          required: 'Name is required',
                          minLength: {
                            value: 2,
                            message: 'Name must be at least 2 characters',
                          },
                        })}
                      />
                    </div>
                    <div>
                      <Input
                        label="Email"
                        type="email"
                        value={user?.email || ''}
                        disabled
                        helperText="Email cannot be changed"
                      />
                    </div>
                  </div>

                  <div className="flex justify-end">
                    <Button
                      type="submit"
                      loading={isUpdating}
                      disabled={isUpdating}
                    >
                      <Save className="w-4 h-4 mr-2" />
                      Save Changes
                    </Button>
                  </div>
                </form>
              </Card.Body>
            </Card>
          )}

          {/* Password Tab */}
          {activeTab === 'password' && (
            <Card>
              <Card.Header>
                <h3 className="text-lg leading-6 font-medium text-gray-900">
                  Change Password
                </h3>
              </Card.Header>
              <Card.Body>
                <form onSubmit={handleSubmitPassword(onPasswordSubmit)} className="space-y-6">
                  <div>
                    <div className="relative">
                      <Input
                        label="Current Password"
                        type={showCurrentPassword ? 'text' : 'password'}
                        error={passwordErrors.currentPassword?.message}
                        {...registerPassword('currentPassword', {
                          required: 'Current password is required',
                        })}
                      />
                      <button
                        type="button"
                        className="absolute inset-y-0 right-0 pr-3 flex items-center"
                        onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                      >
                        {showCurrentPassword ? (
                          <EyeOff className="h-5 w-5 text-gray-400" />
                        ) : (
                          <Eye className="h-5 w-5 text-gray-400" />
                        )}
                      </button>
                    </div>
                  </div>

                  <div>
                    <div className="relative">
                      <Input
                        label="New Password"
                        type={showNewPassword ? 'text' : 'password'}
                        error={passwordErrors.newPassword?.message}
                        {...registerPassword('newPassword', {
                          required: 'New password is required',
                          minLength: {
                            value: 6,
                            message: 'Password must be at least 6 characters',
                          },
                        })}
                      />
                      <button
                        type="button"
                        className="absolute inset-y-0 right-0 pr-3 flex items-center"
                        onClick={() => setShowNewPassword(!showNewPassword)}
                      >
                        {showNewPassword ? (
                          <EyeOff className="h-5 w-5 text-gray-400" />
                        ) : (
                          <Eye className="h-5 w-5 text-gray-400" />
                        )}
                      </button>
                    </div>
                  </div>

                  <div>
                    <div className="relative">
                      <Input
                        label="Confirm New Password"
                        type={showConfirmPassword ? 'text' : 'password'}
                        error={passwordErrors.confirmPassword?.message}
                        {...registerPassword('confirmPassword', {
                          required: 'Please confirm your password',
                          validate: (value) =>
                            value === newPassword || 'Passwords do not match',
                        })}
                      />
                      <button
                        type="button"
                        className="absolute inset-y-0 right-0 pr-3 flex items-center"
                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      >
                        {showConfirmPassword ? (
                          <EyeOff className="h-5 w-5 text-gray-400" />
                        ) : (
                          <Eye className="h-5 w-5 text-gray-400" />
                        )}
                      </button>
                    </div>
                  </div>

                  <div className="flex justify-end">
                    <Button
                      type="submit"
                      loading={isChangingPassword}
                      disabled={isChangingPassword}
                    >
                      <Key className="w-4 h-4 mr-2" />
                      Change Password
                    </Button>
                  </div>
                </form>
              </Card.Body>
            </Card>
          )}

          {/* Settings Tab */}
          {activeTab === 'settings' && (
            <div className="space-y-6">
              <Card>
                <Card.Header>
                  <h3 className="text-lg leading-6 font-medium text-gray-900">
                    Preferences
                  </h3>
                </Card.Header>
                <Card.Body>
                  <form onSubmit={handleSubmitProfile(onProfileSubmit)} className="space-y-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Units
                      </label>
                      <select
                        {...registerProfile('preferences.units')}
                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                      >
                        <option value="metric">Metric (Celsius, km/h)</option>
                        <option value="imperial">Imperial (Fahrenheit, mph)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Notifications
                      </label>
                      <div className="space-y-2">
                        <div className="flex items-center">
                          <input
                            type="checkbox"
                            {...registerProfile('preferences.notifications.email')}
                            className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                          />
                          <label className="ml-2 text-sm text-gray-700">
                            Email notifications
                          </label>
                        </div>
                        <div className="flex items-center">
                          <input
                            type="checkbox"
                            {...registerProfile('preferences.notifications.push')}
                            className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                          />
                          <label className="ml-2 text-sm text-gray-700">
                            Push notifications
                          </label>
                        </div>
                      </div>
                    </div>

                    <div className="flex justify-end">
                      <Button
                        type="submit"
                        loading={isUpdating}
                        disabled={isUpdating}
                      >
                        <Save className="w-4 h-4 mr-2" />
                        Save Preferences
                      </Button>
                    </div>
                  </form>
                </Card.Body>
              </Card>

              <Card>
                <Card.Header>
                  <h3 className="text-lg leading-6 font-medium text-gray-900 text-danger-600">
                    Danger Zone
                  </h3>
                </Card.Header>
                <Card.Body>
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-sm font-medium text-gray-900">Delete Account</h4>
                      <p className="text-sm text-gray-500">
                        Permanently delete your account and all associated data. This action cannot be undone.
                      </p>
                    </div>
                    <Button
                      variant="danger"
                      onClick={handleDeleteAccount}
                    >
                      <Trash2 className="w-4 h-4 mr-2" />
                      Delete Account
                    </Button>
                  </div>
                </Card.Body>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Profile;

