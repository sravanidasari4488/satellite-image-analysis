-- Supabase Database Setup for Geospatial Intelligence System
-- Run these commands in Supabase SQL Editor

-- 1. Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- 2. Create localities table
CREATE TABLE IF NOT EXISTS localities (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  city TEXT NOT NULL,
  name TEXT NOT NULL,
  geometry GEOMETRY(POLYGON, 4326),
  lat DOUBLE PRECISION NOT NULL,
  lon DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMP DEFAULT now(),
  UNIQUE(city, name)
);

-- Create index on geometry for spatial queries
CREATE INDEX IF NOT EXISTS idx_localities_geometry ON localities USING GIST (geometry);

-- Create index on city and name for faster lookups
CREATE INDEX IF NOT EXISTS idx_localities_city_name ON localities (city, name);

-- 3. Create landcover_cache table
CREATE TABLE IF NOT EXISTS landcover_cache (
  locality_id UUID PRIMARY KEY REFERENCES localities(id) ON DELETE CASCADE,
  dw_histogram JSONB NOT NULL,
  last_updated TIMESTAMP DEFAULT now(),
  satellite_source TEXT DEFAULT 'Dynamic World',
  satellite_date TEXT
);

-- Create index on last_updated for cache expiration queries
CREATE INDEX IF NOT EXISTS idx_landcover_cache_updated ON landcover_cache (last_updated);

-- 4. Function to check if cache is fresh (within 30 days)
CREATE OR REPLACE FUNCTION is_cache_fresh(cache_timestamp TIMESTAMP)
RETURNS BOOLEAN AS $$
BEGIN
  RETURN cache_timestamp > (NOW() - INTERVAL '30 days');
END;
$$ LANGUAGE plpgsql;

