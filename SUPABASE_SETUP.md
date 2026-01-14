# Supabase Setup Guide

This guide will help you set up Supabase as the backend database for caching satellite land-cover results.

## Prerequisites

1. A Supabase project (already created)
2. Supabase project URL and service role key

## Step 1: Enable PostGIS

1. Go to your Supabase project dashboard
2. Navigate to **SQL Editor**
3. Run the following command:

```sql
CREATE EXTENSION IF NOT EXISTS postgis;
```

## Step 2: Create Database Tables

Run the SQL script `supabase_setup.sql` in the Supabase SQL Editor. This will create:

- `localities` table: Stores locality polygons and metadata
- `landcover_cache` table: Stores cached landcover histograms
- Indexes for performance
- Helper function for cache freshness checks

## Step 3: Get Supabase Credentials

1. Go to your Supabase project dashboard
2. Navigate to **Settings** → **API**
3. Copy the following:
   - **Project URL** (e.g., `https://xxxxx.supabase.co`) - This is the "Project URL" field
   - **Service Role Key** - This is the **secret key**, NOT the publishable key!
     - Scroll down to find "Project API keys" section
     - Look for `service_role` key (it starts with `eyJ...` or similar)
     - This key has full database access and bypasses RLS
     - **Keep this secret!** Never expose it in client-side code

## Step 4: Configure Environment Variables

Add the following to your `.env` file:

```env
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=your_service_role_key_here
```

**Important**: Use the **service role key**, not the anon key. The service role key has full database access and is required for server-side operations.

## Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `supabase>=2.0.0` - Python Supabase client
- `psycopg2-binary>=2.9.9` - PostgreSQL adapter

## How It Works

### 1. Locality Storage

When localities are fetched from OpenStreetMap:
- Geometry is converted to GeoJSON format
- Stored in `localities` table with PostGIS `GEOMETRY(POLYGON, 4326)` type
- Supabase automatically uses `ST_GeomFromGeoJSON()` to store the geometry

### 2. Cache Lookup

When a user selects a locality:
- System checks `landcover_cache` table for cached data
- If cache exists and is fresh (< 30 days), returns cached histogram
- If cache is missing or stale, runs Earth Engine analysis

### 3. Cache Storage

After running Earth Engine analysis:
- Pixel histogram is stored in `landcover_cache.dw_histogram` (JSONB format)
- `last_updated` timestamp is set to current time
- Cache expires after 30 days

### 4. Cache Benefits

- **Faster responses**: Cached results return instantly
- **Cost savings**: Reduces Earth Engine API calls
- **Reliability**: Works even if Earth Engine is temporarily unavailable
- **Data persistence**: Results persist across server restarts

## Database Schema

### `localities` Table

```sql
CREATE TABLE localities (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  city TEXT NOT NULL,
  name TEXT NOT NULL,
  geometry GEOMETRY(POLYGON, 4326),
  lat DOUBLE PRECISION NOT NULL,
  lon DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMP DEFAULT now(),
  UNIQUE(city, name)
);
```

### `landcover_cache` Table

```sql
CREATE TABLE landcover_cache (
  locality_id UUID PRIMARY KEY REFERENCES localities(id) ON DELETE CASCADE,
  dw_histogram JSONB NOT NULL,
  last_updated TIMESTAMP DEFAULT now(),
  satellite_source TEXT DEFAULT 'ESA WorldCover',
  satellite_date TEXT
);
```

## Troubleshooting

### Issue: "Import supabase could not be resolved"

**Solution**: Install the package:
```bash
pip install supabase
```

### Issue: "Connection refused" or "Invalid API key"

**Solution**: 
1. Verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env`
2. Make sure you're using the **service role key**, not the anon key
3. Check that your Supabase project is active

### Issue: "relation 'localities' does not exist"

**Solution**: Run the SQL setup script in Supabase SQL Editor

### Issue: "PostGIS extension not found"

**Solution**: Enable PostGIS extension:
```sql
CREATE EXTENSION IF NOT EXISTS postgis;
```

## Testing

To verify the setup is working:

1. Fetch localities for a city (e.g., "Hyderabad")
2. Select a locality to analyze
3. Check Supabase dashboard → **Table Editor** → `localities` table
4. Check `landcover_cache` table after analysis completes

## Notes

- Cache is **optional**: System works without Supabase, but won't cache results
- Cache expiration: 30 days (configurable in code)
- Geometry storage: Uses PostGIS for spatial queries
- JSONB storage: Histograms stored as JSONB for efficient querying

