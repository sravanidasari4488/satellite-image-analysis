import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import Button from '../components/UI/Button';
import Card from '../components/UI/Card';
import {
  Satellite,
  Map,
  BarChart3,
  Cloud,
  Shield,
  Download,
  ArrowRight,
  CheckCircle,
  Globe,
  Zap,
} from 'lucide-react';

const Home = () => {
  const { user } = useAuth();

  const features = [
    {
      icon: Satellite,
      title: 'Satellite Image Analysis',
      description: 'Analyze high-resolution satellite images using advanced AI models to detect land types, vegetation health, and environmental changes.',
    },
    {
      icon: Map,
      title: 'Location-Based Insights',
      description: 'Get detailed analysis for any location worldwide. Enter coordinates, city names, or postal codes to start your analysis.',
    },
    {
      icon: BarChart3,
      title: 'Comprehensive Reports',
      description: 'Generate detailed reports with land classification, weather data, risk assessments, and actionable insights.',
    },
    {
      icon: Cloud,
      title: 'Real-Time Weather',
      description: 'Integrate live weather data with satellite analysis to understand environmental conditions and their impact on the landscape.',
    },
    {
      icon: Shield,
      title: 'Risk Assessment',
      description: 'Identify flood-prone areas, drought risks, deforestation patterns, and other environmental hazards automatically.',
    },
    {
      icon: Download,
      title: 'Export & Share',
      description: 'Download reports as PDF or CSV files, share insights with your team, and track changes over time.',
    },
  ];

  const stats = [
    { label: 'Satellite Images Analyzed', value: '10,000+' },
    { label: 'Locations Covered', value: '150+ Countries' },
    { label: 'Accuracy Rate', value: '95%+' },
    { label: 'Active Users', value: '5,000+' },
  ];

  return (
    <div className="bg-white">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto">
          <div className="relative z-10 pb-8 sm:pb-16 md:pb-20 lg:max-w-2xl lg:w-full lg:pb-28 xl:pb-32">
            <main className="mt-10 mx-auto max-w-7xl px-4 sm:mt-12 sm:px-6 md:mt-16 lg:mt-20 lg:px-8 xl:mt-28">
              <div className="sm:text-center lg:text-left">
                <h1 className="text-4xl tracking-tight font-extrabold text-gray-900 sm:text-5xl md:text-6xl">
                  <span className="block xl:inline">Satellite Image</span>{' '}
                  <span className="block text-primary-600 xl:inline">Analysis Platform</span>
                </h1>
                <p className="mt-3 text-base text-gray-500 sm:mt-5 sm:text-lg sm:max-w-xl sm:mx-auto md:mt-5 md:text-xl lg:mx-0">
                  Analyze satellite images with AI-powered insights. Get comprehensive reports on land classification, 
                  weather conditions, and environmental risks for any location worldwide.
                </p>
                <div className="mt-5 sm:mt-8 sm:flex sm:justify-center lg:justify-start">
                  <div className="rounded-md shadow">
                    {user ? (
                      <Link to="/analysis">
                        <Button size="lg" className="w-full sm:w-auto">
                          Start Analysis
                          <ArrowRight className="ml-2 h-5 w-5" />
                        </Button>
                      </Link>
                    ) : (
                      <Link to="/register">
                        <Button size="lg" className="w-full sm:w-auto">
                          Get Started
                          <ArrowRight className="ml-2 h-5 w-5" />
                        </Button>
                      </Link>
                    )}
                  </div>
                  <div className="mt-3 sm:mt-0 sm:ml-3">
                    <Link to="/login">
                      <Button variant="secondary" size="lg" className="w-full sm:w-auto">
                        Sign In
                      </Button>
                    </Link>
                  </div>
                </div>
              </div>
            </main>
          </div>
        </div>
        <div className="lg:absolute lg:inset-y-0 lg:right-0 lg:w-1/2">
          <div className="h-56 w-full bg-gradient-to-r from-primary-400 to-secondary-400 sm:h-72 md:h-96 lg:w-full lg:h-full flex items-center justify-center">
            <Satellite className="h-32 w-32 text-white opacity-80" />
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="bg-primary-600">
        <div className="max-w-7xl mx-auto py-12 px-4 sm:py-16 sm:px-6 lg:px-8 lg:py-20">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-3xl font-extrabold text-white sm:text-4xl">
              Trusted by researchers and organizations worldwide
            </h2>
            <p className="mt-3 text-xl text-primary-200 sm:mt-4">
              Our platform processes thousands of satellite images daily with high accuracy and reliability.
            </p>
          </div>
          <div className="mt-10 pb-12 bg-primary-600 sm:pb-16">
            <div className="relative">
              <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="max-w-4xl mx-auto">
                  <dl className="rounded-lg bg-white shadow-lg sm:grid sm:grid-cols-4">
                    {stats.map((stat, index) => (
                      <div
                        key={stat.label}
                        className={`flex flex-col border-b border-gray-100 p-6 text-center sm:border-0 ${
                          index < stats.length - 1 ? 'sm:border-r' : ''
                        }`}
                      >
                        <dt className="order-2 mt-2 text-lg leading-6 font-medium text-gray-500">
                          {stat.label}
                        </dt>
                        <dd className="order-1 text-5xl font-extrabold text-primary-600">
                          {stat.value}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-12 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <h2 className="text-base text-primary-600 font-semibold tracking-wide uppercase">Features</h2>
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
              Everything you need for satellite analysis
            </p>
            <p className="mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
              Our comprehensive platform combines cutting-edge AI with real-time data to provide 
              actionable insights for environmental monitoring and research.
            </p>
          </div>

          <div className="mt-10">
            <div className="space-y-10 md:space-y-0 md:grid md:grid-cols-2 md:gap-x-8 md:gap-y-10 lg:grid-cols-3">
              {features.map((feature, index) => (
                <Card key={index} className="relative">
                  <Card.Body>
                    <div>
                      <span className="rounded-md inline-flex p-3 bg-primary-500 text-white">
                        <feature.icon className="h-6 w-6" />
                      </span>
                    </div>
                    <div className="mt-4">
                      <h3 className="text-lg leading-6 font-medium text-gray-900">
                        {feature.title}
                      </h3>
                      <p className="mt-2 text-base text-gray-500">
                        {feature.description}
                      </p>
                    </div>
                  </Card.Body>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* How It Works Section */}
      <div className="py-12 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <h2 className="text-base text-primary-600 font-semibold tracking-wide uppercase">How it works</h2>
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
              Get started in minutes
            </p>
          </div>

          <div className="mt-10">
            <div className="space-y-10 md:space-y-0 md:grid md:grid-cols-3 md:gap-x-8 md:gap-y-10">
              <div className="relative">
                <div className="flex items-center justify-center h-12 w-12 rounded-md bg-primary-500 text-white text-xl font-bold">
                  1
                </div>
                <h3 className="mt-4 text-lg leading-6 font-medium text-gray-900">
                  Enter Location
                </h3>
                <p className="mt-2 text-base text-gray-500">
                  Provide coordinates, city name, or postal code for the area you want to analyze.
                </p>
              </div>

              <div className="relative">
                <div className="flex items-center justify-center h-12 w-12 rounded-md bg-primary-500 text-white text-xl font-bold">
                  2
                </div>
                <h3 className="mt-4 text-lg leading-6 font-medium text-gray-900">
                  AI Analysis
                </h3>
                <p className="mt-2 text-base text-gray-500">
                  Our AI models analyze satellite images to classify land types and assess environmental conditions.
                </p>
              </div>

              <div className="relative">
                <div className="flex items-center justify-center h-12 w-12 rounded-md bg-primary-500 text-white text-xl font-bold">
                  3
                </div>
                <h3 className="mt-4 text-lg leading-6 font-medium text-gray-900">
                  Get Reports
                </h3>
                <p className="mt-2 text-base text-gray-500">
                  Download comprehensive reports with insights, visualizations, and actionable recommendations.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-primary-600">
        <div className="max-w-2xl mx-auto text-center py-16 px-4 sm:py-20 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-extrabold text-white sm:text-4xl">
            <span className="block">Ready to get started?</span>
            <span className="block">Start analyzing satellite images today.</span>
          </h2>
          <p className="mt-4 text-lg leading-6 text-primary-200">
            Join thousands of researchers and organizations using our platform for environmental monitoring and analysis.
          </p>
          {user ? (
            <Link to="/analysis" className="mt-8 w-full inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-primary-600 bg-white hover:bg-primary-50 sm:w-auto">
              Start Analysis
              <ArrowRight className="ml-2 h-5 w-5" />
            </Link>
          ) : (
            <Link to="/register" className="mt-8 w-full inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-primary-600 bg-white hover:bg-primary-50 sm:w-auto">
              Get Started
              <ArrowRight className="ml-2 h-5 w-5" />
            </Link>
          )}
        </div>
      </div>
    </div>
  );
};

export default Home;

