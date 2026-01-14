"""
Satellite-based Geospatial Intelligence System
Real-time land cover classification and climate risk assessment
"""

import os
import json
import numpy as np
import ee
import requests
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from opencage.geocoder import OpenCageGeocode
import warnings
warnings.filterwarnings('ignore')

# Supabase client (optional - only if credentials are provided)
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


@dataclass
class LandCoverResult:
    """Land cover classification results"""
    urban: float
    forest: float
    vegetation: float
    water: float
    bare_land: float
    total_pixels: int


@dataclass
class WeatherData:
    """Weather data from OpenWeather API"""
    temperature: float
    humidity: float
    precipitation: float
    wind_speed: float
    pressure: float
    coordinates: Tuple[float, float]


class GeocodingService:
    """Handle geocoding and city boundary fetching"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenCage API key is required")
        self.geocoder = OpenCageGeocode(api_key)
        self.nominatim_url = "https://nominatim.openstreetmap.org/search"
        self.nominatim_url = "https://nominatim.openstreetmap.org/search"
    
    def get_city_boundary_polygon(self, location: str) -> Tuple[ee.Geometry, BoundingBox, Tuple[float, float]]:
        """
        Fetch city administrative boundary polygon from OpenStreetMap
        
        Uses ONLY admin_level=8 boundaries (actual city boundaries like GHMC for Hyderabad).
        Rejects admin_level=6, admin_level=4, and place=region to ensure we get
        the actual city, not the entire district.
        
        Returns:
            Tuple of (Earth Engine polygon geometry, bounding box, center coordinates)
        """
        try:
            # First, geocode to get coordinates for search
            if ',' in location:
                # If coordinates provided, use them directly
                coords = location.split(',')
                lat, lon = float(coords[0].strip()), float(coords[1].strip())
                center = (lat, lon)
                query = location
            else:
                # Geocode city name to get coordinates
                results = self.geocoder.geocode(location)
                if not results:
                    raise ValueError(f"Location '{location}' not found")
                
                geometry = results[0]['geometry']
                lat, lon = geometry['lat'], geometry['lng']
                center = (lat, lon)
                query = location
            
            # Use Overpass API to get admin_level=8 boundaries (primary method)
            # This ensures we get actual city boundaries like GHMC for Hyderabad
            polygon, bbox = self._get_boundary_from_overpass(query, center, lat, lon)
            
            if polygon is not None and bbox is not None:
                # Preprocess polygon before returning
                polygon = EarthEngineService.preprocess_polygon(polygon)
                return polygon, bbox, center
            
            # Fallback to Nominatim if Overpass fails
            return self._get_boundary_from_nominatim(query, center, lat, lon)
            
        except Exception as e:
            raise ValueError(f"Failed to fetch city boundary: {str(e)}")
    
    def _get_boundary_from_nominatim(self, query: str, center: Tuple[float, float], lat: float, lon: float) -> Tuple[ee.Geometry, BoundingBox, Tuple[float, float]]:
        """Fallback: Try Nominatim API (less reliable for admin_level filtering)"""
        try:
            params = {
                'q': query,
                'format': 'geojson',
                'limit': 10,
                'polygon_geojson': 1,
                'addressdetails': 1,
                'extratags': 1,
                'namedetails': 1
            }
            
            response = requests.get(self.nominatim_url, params=params, timeout=15, 
                                   headers={'User-Agent': 'GeospatialIntelligenceSystem/1.0'})
                
            if response.status_code == 200:
                data = response.json()
                if data.get('features'):
                    # Look for admin_level=8 in extratags
                    for feature in data['features']:
                        geometry_data = feature.get('geometry')
                        properties = feature.get('properties', {})
                        extratags = properties.get('extratags', {})
                        
                        # Check admin_level - ONLY accept admin_level=8
                        admin_level = extratags.get('admin_level')
                        place = extratags.get('place', '').lower()
                        
                        # Reject admin_level 6, 4, and place=region
                        if admin_level in ['6', '4'] or place == 'region':
                            continue
                        
                        # ONLY accept admin_level=8
                        if admin_level != '8':
                            continue
                        
                        # Found admin_level=8 boundary
                        if geometry_data and geometry_data.get('type') == 'Polygon':
                            coordinates = geometry_data['coordinates'][0]
                            ee_coords = [[coord[0], coord[1]] for coord in coordinates]
                            polygon = ee.Geometry.Polygon(ee_coords)
                            
                            lons = [coord[0] for coord in coordinates]
                            lats = [coord[1] for coord in coordinates]
                            bbox = BoundingBox(
                                min_lon=min(lons),
                                min_lat=min(lats),
                                max_lon=max(lons),
                                max_lat=max(lats)
                            )
                            
                            polygon = EarthEngineService.preprocess_polygon(polygon)
                            return polygon, bbox, center
                        
                        elif geometry_data and geometry_data.get('type') == 'MultiPolygon':
                            multi_polygon_coords = []
                            all_lons = []
                            all_lats = []
                            
                            for polygon_coords in geometry_data['coordinates']:
                                outer_ring = polygon_coords[0]
                                multi_polygon_coords.append([[coord[0], coord[1]] for coord in outer_ring])
                                all_lons.extend([coord[0] for coord in outer_ring])
                                all_lats.extend([coord[1] for coord in outer_ring])
                            
                            polygon = ee.Geometry.MultiPolygon(multi_polygon_coords)
                            bbox = BoundingBox(
                                min_lon=min(all_lons),
                                min_lat=min(all_lats),
                                max_lon=max(all_lons),
                                max_lat=max(all_lats)
                            )
                            
                            polygon = EarthEngineService.preprocess_polygon(polygon)
                            return polygon, bbox, center
            
            # Ultimate fallback: use a larger bounding box (10km radius) to ensure we get Dynamic World data
            # This is better than a tiny box that might not have coverage
            radius_degrees = 0.09  # ~10km radius (0.09 degrees ≈ 10km)
            bbox = BoundingBox(
                min_lon=lon - radius_degrees,
                min_lat=lat - radius_degrees,
                max_lon=lon + radius_degrees,
                max_lat=lat + radius_degrees
            )
            polygon = ee.Geometry.Rectangle([
                bbox.min_lon, bbox.min_lat,
                bbox.max_lon, bbox.max_lat
            ])
            polygon = EarthEngineService.preprocess_polygon(polygon)
            return polygon, bbox, center
            
        except Exception as e:
            # Ultimate fallback: use a larger bounding box (10km radius)
            radius_degrees = 0.09  # ~10km radius
            bbox = BoundingBox(
                min_lon=lon - radius_degrees,
                min_lat=lat - radius_degrees,
                max_lon=lon + radius_degrees,
                max_lat=lat + radius_degrees
            )
            polygon = ee.Geometry.Rectangle([
                bbox.min_lon, bbox.min_lat,
                bbox.max_lon, bbox.max_lat
            ])
            polygon = EarthEngineService.preprocess_polygon(polygon)
            return polygon, bbox, center
    
    def _get_boundary_from_overpass(self, query: str, center: Tuple[float, float], lat: float, lon: float) -> Tuple[Optional[ee.Geometry], Optional[BoundingBox]]:
        """
        Fetch admin_level=8 administrative boundary from Overpass API
        
        ONLY uses admin_level=8 (actual city boundaries like GHMC).
        Rejects admin_level=6, admin_level=4, and place=region.
        
        Returns:
            Tuple of (polygon, bbox) or (None, None) if not found
        """
        try:
            overpass_url = "https://overpass-api.de/api/interpreter"
            
            # Search ONLY for admin_level=8 boundaries
            # Reject admin_level=6, admin_level=4, and place=region
            # Search by name and also by area around coordinates for better matching
            overpass_query = f"""
            [out:json][timeout:25];
            (
              relation["admin_level"="8"]["boundary"="administrative"]["name"~"{query}",i];
              relation["admin_level"="8"]["boundary"="administrative"](around:5000,{lat},{lon});
            );
            (._;>;);
            out geom;
            """
            
            response = requests.post(overpass_url, data=overpass_query, timeout=30,
                                   headers={'User-Agent': 'GeospatialIntelligenceSystem/1.0'})
            
            if response.status_code == 200:
                data = response.json()
                if data.get('elements'):
                    # Find admin_level=8 relation
                    for element in data['elements']:
                        if element.get('type') == 'relation':
                            tags = element.get('tags', {})
                            
                            # Verify it's admin_level=8 and reject others
                            admin_level = tags.get('admin_level')
                            place = tags.get('place', '').lower()
                            
                            # Reject admin_level 6, 4, and place=region
                            if admin_level in ['6', '4'] or place == 'region':
                                continue
                            
                            # ONLY accept admin_level=8
                            if admin_level != '8':
                                continue
                            
                            # Extract geometry from relation
                            # With 'out geom', Overpass returns geometry directly
                            # Try Nominatim lookup first (most reliable for getting full polygon)
                            osm_id = element.get('id')
                            if osm_id:
                                # Try to get polygon via Nominatim lookup with OSM ID
                                nominatim_lookup_url = f"https://nominatim.openstreetmap.org/lookup"
                                lookup_params = {
                                    'osm_ids': f"R{osm_id}",
                                    'format': 'geojson',
                                    'polygon_geojson': 1
                                }
                                
                                lookup_response = requests.get(nominatim_lookup_url, params=lookup_params, timeout=15,
                                                              headers={'User-Agent': 'GeospatialIntelligenceSystem/1.0'})
                                
                                if lookup_response.status_code == 200:
                                    lookup_data = lookup_response.json()
                                    if lookup_data.get('features'):
                                        feature = lookup_data['features'][0]
                                        geometry_data = feature.get('geometry')
                                        
                                        if geometry_data and geometry_data.get('type') == 'Polygon':
                                            coordinates = geometry_data['coordinates'][0]
                                            ee_coords = [[coord[0], coord[1]] for coord in coordinates]
                                            polygon = ee.Geometry.Polygon(ee_coords)
                                            
                                            lons = [coord[0] for coord in coordinates]
                                            lats = [coord[1] for coord in coordinates]
                                            bbox = BoundingBox(
                                                min_lon=min(lons),
                                                min_lat=min(lats),
                                                max_lon=max(lons),
                                                max_lat=max(lats)
                                            )
                                            
                                            return polygon, bbox
            
            # No admin_level=8 boundary found
            return None, None
             
        except Exception as e:
            # Return None to trigger fallback
            return None, None
    
    def get_city_center(self, city_name: str) -> Tuple[float, float]:
        """
        Get city center coordinates using OpenCage API
        
        Args:
            city_name: Name of the city
            
        Returns:
            Tuple of (lat, lon) coordinates
        """
        try:
            results = self.geocoder.geocode(city_name)
            if not results:
                raise ValueError(f"City '{city_name}' not found")
            
            geometry = results[0]['geometry']
            lat, lon = geometry['lat'], geometry['lng']
            return lat, lon
        except Exception as e:
            raise ValueError(f"Failed to get city center: {str(e)}")
    
    def get_localities(self, city_name: str, radius_km: int = 20) -> List[Dict]:
        """
        Fetch locality names and centers only (fast, no polygons)
        
        Uses radius-based Overpass query to get only:
        - name
        - center coordinates
        - place type
        
        Does NOT fetch full polygons at this stage.
        
        Args:
            city_name: Name of the city
            radius_km: Search radius in kilometers (default 20km)
            
        Returns:
            List of dictionaries with 'name', 'lat', 'lon', 'place_type', 'osm_id', 'osm_type'
            Example: [
                { 'name': 'Gachibowli', 'lat': 17.42, 'lon': 78.35, 'place_type': 'suburb', 'osm_id': 12345, 'osm_type': 'way' },
                ...
            ]
        """
        try:
            # Step 1: Get city center coordinates using OpenCage
            lat, lon = self.get_city_center(city_name)
            
            # Step 2: Use Overpass API with radius-based query
            # Only fetch names and centers, NOT full polygons
            overpass_servers = [
                "https://overpass-api.de/api/interpreter",
                "https://overpass.kumi.systems/api/interpreter",
                "https://overpass.openstreetmap.ru/api/interpreter"
            ]
            
            # Query pattern: radius-based UNION query
            # Includes: place=suburb|neighbourhood AND boundary=administrative with admin_level=9|10 (wards)
            radius_meters = radius_km * 1000
            overpass_query = f"""[out:json][timeout:20];
(
  node["place"~"suburb|neighbourhood"](around:{radius_meters},{lat},{lon});
  way["place"~"suburb|neighbourhood"](around:{radius_meters},{lat},{lon});
  relation["place"~"suburb|neighbourhood"](around:{radius_meters},{lat},{lon});
  relation["boundary"="administrative"]["admin_level"~"9|10"](around:{radius_meters},{lat},{lon});
);
out tags center;"""
            
            response = None
            last_error = None
            
            # Try each server until one works
            for server_url in overpass_servers:
                try:
                    response = requests.post(server_url, data=overpass_query, timeout=90,
                                           headers={'User-Agent': 'GeospatialIntelligenceSystem/1.0'})
                    if response.status_code == 200:
                        # Check if response has content
                        if response.text and response.text.strip():
                            break
                        else:
                            # Empty response, try next server
                            last_error = f"Server {server_url} returned empty response"
                            response = None
                            continue
                    elif response.status_code == 504:
                        # Timeout - try next server
                        last_error = f"Server {server_url} timed out (504)"
                        response = None
                        continue
                    elif response.status_code == 429:
                        # Rate limited - try next server
                        last_error = f"Server {server_url} rate limited (429)"
                        response = None
                        continue
                    else:
                        last_error = f"Server {server_url} returned status {response.status_code}: {response.text[:100] if response.text else 'No response body'}"
                        response = None
                        continue
                except requests.exceptions.Timeout:
                    last_error = f"Server {server_url} timed out"
                    response = None
                    continue
                except requests.exceptions.ConnectionError as e:
                    last_error = f"Server {server_url} connection error: {str(e)}"
                    response = None
                    continue
                except Exception as e:
                    last_error = f"Server {server_url} error: {str(e)}"
                    response = None
                    continue
            
            if response is None:
                raise RuntimeError(
                    f"All Overpass servers failed. Last error: {last_error}. "
                    f"Try reducing the search radius (currently {radius_km}km) or try again later."
                )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"Overpass API returned status {response.status_code}. "
                    f"Last error: {last_error}. "
                    f"Response: {response.text[:200] if response.text else 'No response body'}. "
                    f"Try reducing the search radius (currently {radius_km}km) or try again later."
                )
            
            # Parse response
            response_text = response.text.strip() if response.text else ""
            
            # Check if response is empty
            if not response_text:
                raise RuntimeError("Overpass API returned empty response. The server may be overloaded or the query timed out.")
            
            # Check if response is HTML/XML (error page) instead of JSON
            if response_text.startswith('<?xml') or response_text.startswith('<!DOCTYPE') or response_text.startswith('<html'):
                # Extract error message from HTML if possible
                import re
                error_match = re.search(r'<p[^>]*>(.*?)</p>', response_text, re.IGNORECASE | re.DOTALL)
                if error_match:
                    error_msg = error_match.group(1).strip()[:200]
                else:
                    # Try to find title or h1
                    title_match = re.search(r'<title[^>]*>(.*?)</title>', response_text, re.IGNORECASE | re.DOTALL)
                    if title_match:
                        error_msg = title_match.group(1).strip()[:200]
                    else:
                        error_msg = "Overpass API returned an HTML error page"
                
                raise RuntimeError(
                    f"Overpass API error: {error_msg}. "
                    f"The query may be too complex or the server is overloaded. "
                    f"Try reducing the search radius (currently {radius_km}km) or try again later."
                )
            
            # Try to parse as JSON
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                # Log the actual response for debugging
                error_msg = f"Failed to parse Overpass API response as JSON: {str(e)}"
                if response_text:
                    # Show first 200 chars of response for debugging
                    preview = response_text[:200] if len(response_text) > 200 else response_text
                    error_msg += f"\nResponse preview: {preview}"
                raise RuntimeError(error_msg)
            
            # Check for errors in Overpass response
            if 'remark' in data and 'runtime' not in data:
                raise RuntimeError(f"Overpass API error: {data.get('remark', 'Unknown error')}")
            
            # Check if response indicates an error
            if 'error' in data:
                error_info = data.get('error', {})
                if isinstance(error_info, dict):
                    error_msg = error_info.get('message', 'Unknown Overpass API error')
                else:
                    error_msg = str(error_info)
                raise RuntimeError(f"Overpass API error: {error_msg}")
            
            localities = []
            seen_names = set()  # Deduplicate by name
            
            if data.get('elements'):
                for element in data['elements']:
                    tags = element.get('tags', {})
                    
                    # Get locality name
                    name = tags.get('name') or tags.get('name:en') or tags.get('alt_name')
                    if not name:
                        continue
                    
                    # Deduplicate by name (case-insensitive)
                    name_lower = name.lower()
                    if name_lower in seen_names:
                        continue
                    seen_names.add(name_lower)
                    
                    # Get place type or boundary type
                    place_type = tags.get('place', 'unknown')
                    if place_type == 'unknown':
                        # Check if it's an administrative boundary
                        if tags.get('boundary') == 'administrative':
                            admin_level = tags.get('admin_level', '')
                            place_type = f'ward_{admin_level}' if admin_level else 'administrative'
                    
                    # Get center coordinates
                    center = element.get('center')
                    if center:
                        center_lat = center.get('lat')
                        center_lon = center.get('lon')
                    else:
                        # Fallback: use lat/lon if available
                        center_lat = element.get('lat')
                        center_lon = element.get('lon')
                    
                    if center_lat is None or center_lon is None:
                        continue  # Skip if no coordinates
                    
                    # Get OSM ID and type for later geometry fetching
                    osm_id = element.get('id')
                    osm_type = element.get('type')  # 'node', 'way', or 'relation'
                    
                    localities.append({
                        'name': name,
                        'lat': center_lat,
                        'lon': center_lon,
                        'place_type': place_type,
                        'osm_id': osm_id,
                        'osm_type': osm_type
                    })
            
            # Sort alphabetically by name
            localities.sort(key=lambda x: x['name'].lower())
            
            return localities
            
        except Exception as e:
            raise ValueError(f"Failed to fetch localities: {str(e)}")
    
    def get_locality_geometry(self, locality_name: str, lat: float, lon: float) -> Tuple[ee.Geometry, BoundingBox]:
        """
        Fetch full geometry for a single locality (called on-demand when user selects it)
        
        Queries Overpass AGAIN for the selected locality name and fetches full geometry.
        Supports relation (multipolygon) and way (polygon).
        
        Args:
            locality_name: Name of the locality
            lat: Latitude of locality center (for fallback)
            lon: Longitude of locality center (for fallback)
            
        Returns:
            Tuple of (polygon, bbox)
            If geometry unavailable, returns approximate 500m buffer around center
        """
        try:
            # Query Overpass AGAIN for the selected locality name
            overpass_servers = [
                "https://overpass-api.de/api/interpreter",
                "https://overpass.kumi.systems/api/interpreter",
                "https://overpass.openstreetmap.ru/api/interpreter"
            ]
            
            # Query pattern: search by name around center
            overpass_query = f"""[out:json][timeout:25];
(
  relation["name"="{locality_name}"](around:5000,{lat},{lon});
  way["name"="{locality_name}"](around:5000,{lat},{lon});
);
out geom;"""
            
            response = None
            last_error = None
            
            # Try each server until one works
            for server_url in overpass_servers:
                try:
                    response = requests.post(server_url, data=overpass_query, timeout=30,
                                            headers={'User-Agent': 'GeospatialIntelligenceSystem/1.0'})
                    if response.status_code == 200:
                        if response.text and response.text.strip():
                            break
                        else:
                            last_error = f"Server {server_url} returned empty response"
                            response = None
                            continue
                    elif response.status_code == 504:
                        last_error = f"Server {server_url} timed out (504)"
                        response = None
                        continue
                    elif response.status_code == 429:
                        last_error = f"Server {server_url} rate limited (429)"
                        response = None
                        continue
                    else:
                        last_error = f"Server {server_url} returned status {response.status_code}"
                        response = None
                        continue
                except requests.exceptions.Timeout:
                    last_error = f"Server {server_url} timed out"
                    response = None
                    continue
                except requests.exceptions.ConnectionError as e:
                    last_error = f"Server {server_url} connection error: {str(e)}"
                    response = None
                    continue
                except Exception as e:
                    last_error = f"Server {server_url} error: {str(e)}"
                    response = None
                    continue
            
            # If all servers failed, use fallback
            if response is None or response.status_code != 200:
                return self._create_fallback_geometry(lat, lon, "Locality geometry unavailable. Using approximate area.")
            
            # Parse response
            response_text = response.text.strip() if response.text else ""
            if not response_text:
                return self._create_fallback_geometry(lat, lon, "Locality geometry unavailable. Using approximate area.")
            
            # Check if response is HTML/XML (error page)
            if response_text.startswith('<?xml') or response_text.startswith('<!DOCTYPE') or response_text.startswith('<html'):
                return self._create_fallback_geometry(lat, lon, "Locality geometry unavailable. Using approximate area.")
            
            # Parse JSON
            try:
                data = response.json()
            except json.JSONDecodeError:
                return self._create_fallback_geometry(lat, lon, "Locality geometry unavailable. Using approximate area.")
            
            # Check for errors
            if 'remark' in data and 'runtime' not in data:
                return self._create_fallback_geometry(lat, lon, "Locality geometry unavailable. Using approximate area.")
            
            if 'error' in data:
                return self._create_fallback_geometry(lat, lon, "Locality geometry unavailable. Using approximate area.")
            
            # Process elements to extract geometry
            geometry = None
            bbox = None
            
            if data.get('elements'):
                for element in data['elements']:
                    element_type = element.get('type')
                    
                    if element_type == 'relation':
                        # Relation → multipolygon (use Nominatim lookup)
                        osm_id = element.get('id')
                        if osm_id:
                            geometry, bbox = self._get_geometry_from_nominatim_lookup(f"R{osm_id}")
                            if geometry is not None:
                                break  # Found valid geometry
                    
                    elif element_type == 'way':
                        # Way → polygon (extract geometry directly)
                        if 'geometry' in element:
                            coords = element['geometry']
                            if coords and len(coords) >= 3:
                                # Convert to Earth Engine format
                                ee_coords = [[point['lon'], point['lat']] for point in coords]
                                geometry = ee.Geometry.Polygon([ee_coords])
                                
                                # Calculate bounding box
                                lons = [point['lon'] for point in coords]
                                lats = [point['lat'] for point in coords]
                                bbox = BoundingBox(
                                    min_lon=min(lons),
                                    min_lat=min(lats),
                                    max_lon=max(lons),
                                    max_lat=max(lats)
                                )
                                break  # Found valid geometry
            
            # If geometry found, return it
            if geometry is not None and bbox is not None:
                return geometry, bbox
            
            # Fallback: 500m buffer around center
            return self._create_fallback_geometry(lat, lon, "Locality geometry unavailable. Using approximate area.")
            
        except Exception as e:
            # On any error, use fallback
            return self._create_fallback_geometry(lat, lon, f"Locality geometry unavailable. Using approximate area. Error: {str(e)}")
    
    def _create_fallback_geometry(self, lat: float, lon: float, message: str = None) -> Tuple[ee.Geometry, BoundingBox]:
        """
        Create fallback geometry: 500m buffer around center
        
        Args:
            lat: Latitude
            lon: Longitude
            message: Optional warning message (logged but not raised)
            
        Returns:
            Tuple of (polygon, bbox) - approximate 500m buffer
        """
        if message:
            # Log warning but don't crash
            print(f"Warning: {message}")
        
        # Create 500m buffer around center
        # 500m ≈ 0.0045 degrees at equator
        radius_degrees = 0.0045
        bbox = BoundingBox(
            min_lon=lon - radius_degrees,
            min_lat=lat - radius_degrees,
            max_lon=lon + radius_degrees,
            max_lat=lat + radius_degrees
        )
        geometry = ee.Geometry.Rectangle([
            bbox.min_lon, bbox.min_lat,
            bbox.max_lon, bbox.max_lat
        ])
        
        return geometry, bbox
    
    def _get_geometry_from_nominatim_lookup(self, osm_id: str) -> Tuple[Optional[ee.Geometry], Optional[BoundingBox]]:
        """
        Get geometry from Nominatim lookup using OSM ID
        
        Args:
            osm_id: OSM ID (e.g., "R123456" for relation, "W123456" for way)
            
        Returns:
            Tuple of (geometry, bbox) or (None, None) if not found
        """
        try:
            nominatim_lookup_url = "https://nominatim.openstreetmap.org/lookup"
            lookup_params = {
                'osm_ids': osm_id,
                'format': 'geojson',
                'polygon_geojson': 1
            }
            
            lookup_response = requests.get(nominatim_lookup_url, params=lookup_params, timeout=15,
                                          headers={'User-Agent': 'GeospatialIntelligenceSystem/1.0'})
            
            if lookup_response.status_code == 200:
                lookup_data = lookup_response.json()
                if lookup_data.get('features'):
                    feature = lookup_data['features'][0]
                    geometry_data = feature.get('geometry')
                    
                    if geometry_data and geometry_data.get('type') == 'Polygon':
                        coordinates = geometry_data['coordinates'][0]
                        ee_coords = [[coord[0], coord[1]] for coord in coordinates]
                        polygon = ee.Geometry.Polygon(ee_coords)
                        
                        lons = [coord[0] for coord in coordinates]
                        lats = [coord[1] for coord in coordinates]
                        bbox = BoundingBox(
                            min_lon=min(lons),
                            min_lat=min(lats),
                            max_lon=max(lons),
                            max_lat=max(lats)
                        )
                        
                        return polygon, bbox
                    
                    elif geometry_data and geometry_data.get('type') == 'MultiPolygon':
                        multi_polygon_coords = []
                        all_lons = []
                        all_lats = []
                        
                        for polygon_coords in geometry_data['coordinates']:
                            outer_ring = polygon_coords[0]
                            multi_polygon_coords.append([[coord[0], coord[1]] for coord in outer_ring])
                            all_lons.extend([coord[0] for coord in outer_ring])
                            all_lats.extend([coord[1] for coord in outer_ring])
                        
                        polygon = ee.Geometry.MultiPolygon(multi_polygon_coords)
                        bbox = BoundingBox(
                            min_lon=min(all_lons),
                            min_lat=min(all_lats),
                            max_lon=max(all_lons),
                            max_lat=max(all_lats)
                        )
                        
                        return polygon, bbox
            
            return None, None
        except Exception as e:
            return None, None


class EarthEngineService:
    """Handle Google Earth Engine operations"""
    
    @staticmethod
    def validate_geometry(geom: ee.Geometry) -> Tuple[bool, Optional[str]]:
        """
        Validate geometry for Dynamic World analysis
        
        Checks:
        - Area > 1 km² (1e6 square meters)
        - Geometry is valid
        
        Args:
            geom: Earth Engine geometry
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check area
            area = geom.area().getInfo()
            if area < 1e6:  # Less than 1 km²
                return False, f"Geometry area too small: {area:.0f} m² (minimum 1 km²)"
            
            # Check validity
            is_valid = geom.isValid().getInfo()
            if not is_valid:
                return False, "Geometry is invalid (self-intersections or other topology errors)"
            
            return True, None
            
        except Exception as e:
            return False, f"Geometry validation failed: {str(e)}"
    
    @staticmethod
    def create_centroid_buffer(lat: float, lon: float, radius_meters: int) -> ee.Geometry:
        """
        Create circular buffer around centroid
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_meters: Buffer radius in meters
            
        Returns:
            Earth Engine geometry (circular buffer)
        """
        point = ee.Geometry.Point(lon, lat)
        buffer_geom = point.buffer(radius_meters)
        return buffer_geom
    
    @staticmethod
    def preprocess_polygon(raw_polygon: ee.Geometry) -> ee.Geometry:
        """
        Preprocess OSM polygon before using in Google Earth Engine
        
        Production-grade preprocessing to reduce vertex count and fix topology:
        1. Simplify with 200m tolerance to reduce vertex count
        2. Buffer with 1 meter to fix invalid topology
        3. Transform to EPSG:4326 with scale 1
        
        Args:
            raw_polygon: Raw Earth Engine geometry from OSM
            
        Returns:
            Preprocessed Earth Engine geometry ready for Earth Engine operations
        """
        # Step 1: Simplify with 200m tolerance to reduce vertex count
        geometry = raw_polygon.simplify(maxError=200)
        
        # Step 2: Buffer with 1 meter to fix invalid topology (self-intersections, etc.)
        geometry = geometry.buffer(1)
        
        # Step 3: Transform to EPSG:4326 with scale 1
        geometry = geometry.transform('EPSG:4326', 1)
        
        return geometry
    
    @staticmethod
    def preprocess_locality_polygon(raw_polygon: ee.Geometry) -> ee.Geometry:
        """
        Preprocess locality polygon safely for Dynamic World analysis
        
        EXACT logic as specified:
        - For small polygons (< 1 km²): buffer(30) only, no simplify
        - For larger polygons: simplify(100) then buffer(30)
        - Always transform to EPSG:4326
        
        Args:
            raw_polygon: Raw Earth Engine geometry from OSM
            
        Returns:
            Preprocessed Earth Engine geometry ready for Dynamic World
        """
        # geom = ee.Geometry(localityPolygon)
        geom = ee.Geometry(raw_polygon)
        
        # DO NOT oversimplify small polygons
        # Check area: 1e6 square meters = 1 km²
        area = geom.area().getInfo()
        
        if area < 1e6:  # < 1 km²
            # Small polygon: expand slightly, no simplify
            geom = geom.buffer(30)
        else:
            # Larger polygon: simplify then buffer
            geom = geom.simplify(maxError=100)  # 100m tolerance (not 200m)
            geom = geom.buffer(30)
        
        # Transform to EPSG:4326
        geom = geom.transform('EPSG:4326', 1)
        
        return geom
    
    def __init__(self, project: Optional[str] = None):
        """
        Initialize Earth Engine service
        
        Args:
            project: Optional Google Cloud project ID. Can also be set via 
                    EARTHENGINE_PROJECT environment variable.
        """
        try:
            # Get project from parameter or environment variable
            if not project:
                import os
                project = os.getenv('EARTHENGINE_PROJECT')
            
            # Initialize Earth Engine
            try:
                if project:
                    ee.Initialize(project=project)
                else:
                    ee.Initialize()
            except Exception as init_error:
                error_msg = str(init_error).lower()
                
                # If no project found, provide helpful error
                if "no project found" in error_msg or "project" in error_msg:
                    raise RuntimeError(
                        f"Google Earth Engine requires a Google Cloud project.\n\n"
                        f"To fix this:\n"
                        f"1. Visit https://code.earthengine.google.com/ to see your projects\n"
                        f"2. Or set EARTHENGINE_PROJECT environment variable:\n"
                        f"   PowerShell: $env:EARTHENGINE_PROJECT='your-project-id'\n"
                        f"   Or add to .env file: EARTHENGINE_PROJECT=your-project-id\n\n"
                        f"Original error: {str(init_error)}"
                    )
                
                # If authentication needed, try to authenticate
                if "auth" in error_msg or "credential" in error_msg:
                    try:
                        ee.Authenticate()
                        if project:
                            ee.Initialize(project=project)
                        else:
                            ee.Initialize()
                    except Exception as auth_error:
                        raise RuntimeError(
                            f"Failed to authenticate Google Earth Engine.\n"
                            f"Please run: python authenticate_earth_engine.py\n"
                            f"Or: python -c \"import ee; ee.Authenticate()\"\n\n"
                            f"Error: {str(auth_error)}"
                        )
                else:
                    raise RuntimeError(
                        f"Failed to initialize Google Earth Engine: {str(init_error)}\n"
                        f"Please run: python authenticate_earth_engine.py"
                    )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Google Earth Engine: {str(e)}\n"
                f"Please run: python authenticate_earth_engine.py"
            )
    
    def get_dynamic_world_image(self, polygon: ee.Geometry, bbox: BoundingBox, 
                                start_date: str = None, end_date: str = None) -> Tuple[ee.Image, str]:
        """
        Fetch Google Dynamic World land cover image for locality polygon
        
        Exact pipeline as specified:
        1. Load: ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        2. Filter by geometry (not bbox)
        3. Filter to last 30 days
        4. Sort by system:time_start descending
        5. Select the first (most recent) image
        6. Select ONLY the 'label' band
        
        Args:
            polygon: Earth Engine polygon geometry (locality boundary, preprocessed)
            bbox: Bounding box (not used, kept for compatibility)
            start_date: Optional start date filter (defaults to 30 days ago)
            end_date: Optional end date filter (defaults to now)
        
        Returns:
            Tuple of (image with 'label' band, date_string): The Dynamic World image and its date
        """
        # Load Dynamic World ImageCollection
        dw_collection = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
        
        # .filterBounds(geom)
        collection = dw_collection.filterBounds(polygon)
        
        # DO NOT restrict to only 30 days
        # Take the most recent available image
        # .sort("system:time_start", false)
        collection = collection.sort('system:time_start', False)
        # .first()
        dw_image = collection.first()
        
        # Safety check: If dw is null → throw error "No Dynamic World image available"
        try:
            # Verify image exists by checking collection size
            collection_size = collection.size().getInfo()
            if collection_size == 0:
                raise RuntimeError("No Dynamic World image available")
            
            # Try to get image properties to verify it's valid
            try:
                _ = dw_image.get('system:time_start').getInfo()
            except:
                raise RuntimeError("No Dynamic World image available")
        except RuntimeError:
            raise
        except Exception:
            # If check fails, continue - might still be valid
            pass
        
        # Get the date: dw.get("system:time_start")
        try:
            date_info = dw_image.get('system:time_start').getInfo()
            if date_info:
                date_obj = datetime.fromtimestamp(date_info / 1000)
                date_string = date_obj.strftime('%Y-%m-%d')
            else:
                date_string = datetime.now().strftime('%Y-%m-%d')
        except:
            date_string = datetime.now().strftime('%Y-%m-%d')
        
        # SELECT + UNMASK (CRITICAL)
        # var labels = dw.select("label").unmask(-1)
        # Convert masked pixels to -1 so they can be counted and filtered out later
        labels = dw_image.select('label').unmask(-1)
        
        return labels, date_string
    
    def _create_2km_tiles(self, geometry: ee.Geometry, bbox: BoundingBox) -> List[ee.Geometry]:
        """
        Create 2km × 2km grid tiles from bounding box and intersect with geometry
        
        Args:
            geometry: Earth Engine geometry (city boundary)
            bbox: Bounding box of the geometry
            
        Returns:
            List of tile geometries (intersected with city boundary)
        """
        # Approximate conversion: 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        # For 2km tiles: 2/111 ≈ 0.018 degrees
        
        # Use average latitude for longitude conversion
        avg_lat = (bbox.min_lat + bbox.max_lat) / 2
        lat_degree_to_km = 111.0
        lon_degree_to_km = 111.0 * abs(np.cos(np.radians(avg_lat)))
        
        # 2km in degrees
        tile_size_lat = 2.0 / lat_degree_to_km  # ~0.018 degrees
        tile_size_lon = 2.0 / lon_degree_to_km  # ~0.018 degrees (adjusted for latitude)
        
        tiles = []
        current_lat = bbox.min_lat
        
        while current_lat < bbox.max_lat:
            current_lon = bbox.min_lon
            while current_lon < bbox.max_lon:
                # Create 2km × 2km tile
                tile_bbox = ee.Geometry.Rectangle([
                    current_lon,
                    current_lat,
                    min(current_lon + tile_size_lon, bbox.max_lon),
                    min(current_lat + tile_size_lat, bbox.max_lat)
                ])
                
                # Intersect tile with city geometry (only count pixels inside city boundary)
                tile = geometry.intersection(tile_bbox)
                tiles.append(tile)
                
                current_lon += tile_size_lon
            
            current_lat += tile_size_lat
        
        return tiles
    
    def _merge_histograms(self, histograms: List[Dict]) -> Dict[int, int]:
        """
        Merge multiple histogram dictionaries into one
        
        Args:
            histograms: List of histogram dictionaries
            
        Returns:
            Merged histogram dictionary
        """
        merged = {}
        for hist in histograms:
            if hist and 'label' in hist:
                label_hist = hist['label']
                for label_str, count in label_hist.items():
                    try:
                        label = int(float(label_str))
                        count_val = int(float(count))
                        merged[label] = merged.get(label, 0) + count_val
                    except (ValueError, TypeError):
                        continue
        return merged
    
    def count_pixels_by_class_direct(self, image: ee.Image, polygon: ee.Geometry, scale: int = 30) -> Dict[int, int]:
        """
        Count pixels per land cover class using direct reduceRegion
        
        EXACT pipeline as specified:
        1. Run reduceRegion with frequencyHistogram
        2. Post-process: Remove key "-1" from histogram (masked pixels)
        3. Use remaining counts
        
        Args:
            image: Earth Engine image with 'label' band (Dynamic World, already unmasked)
            polygon: Earth Engine polygon geometry (preprocessed)
            scale: Resolution in meters (default 30m)
            
        Returns:
            Dictionary mapping class labels to pixel counts
        """
        try:
            # Step 4: reduceRegion
            # var hist = labels.reduceRegion({
            #     reducer: ee.Reducer.frequencyHistogram(),
            #     geometry: geom,
            #     scale: 30,
            #     maxPixels: 1e13
            # });
            histogram = image.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=polygon,
                scale=scale,
                maxPixels=1e13,
                bestEffort=True  # Use bestEffort to handle edge cases
            )
            
            histogram_dict = histogram.getInfo()
            
            # Check if hist is empty
            if not histogram_dict or 'label' not in histogram_dict:
                # Try with expanded geometry
                expanded_geom = polygon.buffer(100)
                histogram = image.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=expanded_geom,
                    scale=scale,
                    maxPixels=1e13,
                    bestEffort=True
                )
                histogram_dict = histogram.getInfo()
                
                if not histogram_dict or 'label' not in histogram_dict:
                    # Return empty dict - let caller handle fallback
                    return {}
            
            label_histogram = histogram_dict.get('label', {})
            
            if not label_histogram or len(label_histogram) == 0:
                # Try with expanded geometry
                expanded_geom = polygon.buffer(100)
                histogram = image.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=expanded_geom,
                    scale=scale,
                    maxPixels=1e13,
                    bestEffort=True
                )
                histogram_dict = histogram.getInfo()
                
                if histogram_dict and 'label' in histogram_dict:
                    label_histogram = histogram_dict['label']
                else:
                    return {}
            
            # Step 5: Post-process - Remove key "-1" from histogram (masked pixels)
            # Convert to integer keys and values, filtering out masked pixels (-1)
            pixel_counts = {}
            total_pixels = 0
            masked_pixels = 0
            
            for label_str, count in label_histogram.items():
                try:
                    label = int(float(label_str))
                    count_val = int(float(count))
                    
                    # Remove masked pixels (key = -1)
                    if label == -1:
                        masked_pixels += count_val
                        continue
                    
                    # Only count valid Dynamic World labels (0-7)
                    if 0 <= label <= 7 and count_val > 0:
                        pixel_counts[label] = count_val
                        total_pixels += count_val
                except (ValueError, TypeError):
                    continue
            
            # If all pixels are masked, return empty dict
            if total_pixels == 0:
                return {}
            
            return pixel_counts
            
        except Exception as e:
            # Return empty dict on error - let caller handle fallback
            return {}
    
    def _count_pixels_tiled(self, image: ee.Image, polygon: ee.Geometry, scale: int = 30) -> Dict[int, int]:
        """
        Count pixels by subdividing geometry into 2km × 2km tiles
        
        Step 5: If hist is empty, subdivide geom into 2km × 2km tiles,
        run reduceRegion per tile, merge histograms
        
        Args:
            image: Earth Engine image with 'label' band
            polygon: Earth Engine polygon geometry (preprocessed)
            scale: Resolution in meters (default 30m)
            
        Returns:
            Dictionary mapping class labels to pixel counts
        """
        # Get bounding box of polygon
        try:
            bounds = polygon.bounds().getInfo()['coordinates'][0]
            min_lon, min_lat = bounds[0]
            max_lon, max_lat = bounds[2]
        except Exception as e:
            raise RuntimeError(f"Failed to get polygon bounds: {str(e)}")
        
        # Create 2km × 2km tiles
        # 1 degree latitude ≈ 111 km, so 2km ≈ 0.018 degrees
        avg_lat = (min_lat + max_lat) / 2
        lat_degree_to_km = 111.0
        lon_degree_to_km = 111.0 * abs(np.cos(np.radians(avg_lat)))
        
        tile_size_lat = 2.0 / lat_degree_to_km  # ~0.018 degrees
        tile_size_lon = 2.0 / lon_degree_to_km  # ~0.018 degrees
        
        all_pixel_counts = {}
        successful_tiles = 0
        failed_tiles = 0
        total_tiles = 0
        current_lat = min_lat
        
        while current_lat < max_lat:
            current_lon = min_lon
            while current_lon < max_lon:
                total_tiles += 1
                # Create tile
                try:
                    tile_bbox = ee.Geometry.Rectangle([
                        current_lon,
                        current_lat,
                        min(current_lon + tile_size_lon, max_lon),
                        min(current_lat + tile_size_lat, max_lat)
                    ])
                    
                    # Intersect tile with polygon
                    tile = polygon.intersection(tile_bbox)
                    
                    # Check if tile has area
                    try:
                        tile_area = tile.area().getInfo()
                        if tile_area < 1e-6:  # Skip very small tiles
                            current_lon += tile_size_lon
                            continue
                    except:
                        # If area check fails, try anyway
                        pass
                    
                    # Run reduceRegion on tile
                    # Use lower maxPixels and bestEffort for individual tiles
                    histogram = image.reduceRegion(
                        reducer=ee.Reducer.frequencyHistogram(),
                        geometry=tile,
                        scale=scale,
                        maxPixels=1e9,  # Lower for individual tiles
                        bestEffort=True,
                        tileScale=2  # Use tileScale for better performance
                    )
                    
                    histogram_dict = histogram.getInfo()
                    
                    if histogram_dict and 'label' in histogram_dict:
                        label_histogram = histogram_dict['label']
                        if label_histogram and len(label_histogram) > 0:
                            # Merge into all_pixel_counts
                            for label_str, count in label_histogram.items():
                                try:
                                    label = int(float(label_str))
                                    count_val = int(float(count))
                                    all_pixel_counts[label] = all_pixel_counts.get(label, 0) + count_val
                                except (ValueError, TypeError):
                                    continue
                            successful_tiles += 1
                        else:
                            failed_tiles += 1
                    else:
                        failed_tiles += 1
                except Exception as e:
                    # Skip failed tiles but log for debugging
                    failed_tiles += 1
                    continue
                
                current_lon += tile_size_lon
            
            current_lat += tile_size_lat
        
        if not all_pixel_counts:
            raise RuntimeError(
                f"Empty histogram after tiling. "
                f"Total tiles: {total_tiles}, Successful: {successful_tiles}, Failed: {failed_tiles}. "
                f"Dynamic World may not have data for this locality in the last 30 days. "
                f"Try expanding the date range or check if the locality geometry is valid."
            )
        
        return all_pixel_counts
    
    def count_pixels_by_class(self, image: ee.Image, polygon: ee.Geometry, bbox: BoundingBox, scale: int = 30) -> Dict[int, int]:
        """
        Count pixels per land cover class using tiled reduceRegion approach
        
        Always uses 2km × 2km tiles to prevent Earth Engine timeouts.
        Uses scale=30 for 9x faster processing while maintaining accuracy.
        
        Args:
            image: Earth Engine image with 'label' band
            polygon: Earth Engine polygon geometry (city boundary, preprocessed)
            bbox: Bounding box for area calculation
            scale: Resolution in meters (default 30m - 9x faster than 10m)
            
        Returns:
            Dictionary mapping class labels to pixel counts
        """
        # Always use tiled approach with 2km × 2km tiles
        # This prevents timeouts and is more reliable
        return self._count_pixels_tiled_2km(image, polygon, bbox, scale)
    
    def _count_pixels_tiled_2km(self, image: ee.Image, geometry: ee.Geometry, bbox: BoundingBox, scale: int) -> Dict[int, int]:
        """
        Count pixels by subdividing into 2km × 2km tiles and merging results
        
        Args:
            image: Earth Engine image with 'label' band
            geometry: Earth Engine polygon geometry (city boundary)
            bbox: Bounding box for area calculation
            scale: Resolution in meters (30m recommended)
            
        Returns:
            Dictionary mapping class labels to pixel counts
        """
        # Create 2km × 2km grid tiles
        tiles = self._create_2km_tiles(geometry, bbox)
        
        if not tiles:
            raise RuntimeError("Failed to create tiles from city geometry")
        
        # Validate that we have tiles
        if len(tiles) == 0:
            raise RuntimeError("No tiles created from city geometry")
        
        # For very small cities, 1 tile is acceptable - continue processing
        # The tile processing will handle empty/failed tiles gracefully
        
        # Process each tile
        tile_histograms = []
        successful_tiles = 0
        failed_tiles = 0
        skipped_tiles = 0
        
        for i, tile in enumerate(tiles):
            try:
                # Check if tile has any area (intersection with city boundary)
                tile_area = tile.area().getInfo()
                if tile_area < 1e-6:  # Skip very small tiles (< 1 square meter)
                    skipped_tiles += 1
                    continue
                
                # Clip dw.select("label") to tile
                tile_image = image.clip(tile)
                
                # Run reduceRegion with frequencyHistogram on this tile
                tile_histogram = tile_image.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=tile,
                    scale=scale,
                    maxPixels=1e8,
                    bestEffort=True,
                    tileScale=2  # Lower tileScale for individual tiles
                )
                
                tile_hist_dict = tile_histogram.getInfo()
                
                # If tile returns empty, skip it (do not fail the whole job)
                if tile_hist_dict and 'label' in tile_hist_dict and tile_hist_dict['label']:
                    if len(tile_hist_dict['label']) > 0:  # Ensure histogram is not empty
                        tile_histograms.append(tile_hist_dict)
                        successful_tiles += 1
                    else:
                        failed_tiles += 1  # Empty histogram, skip
                else:
                    failed_tiles += 1  # No data, skip
                    
            except Exception as e:
                # Skip tiles that fail, continue with others
                failed_tiles += 1
                continue
        
        # If any tile returns empty, skip it (already done above)
        # Only fail if ALL tiles are empty
        if not tile_histograms:
            raise RuntimeError(
                f"Failed to get pixel counts from any tiles. "
                f"Total tiles created: {len(tiles)}, "
                f"Successful: {successful_tiles}, "
                f"Failed/empty: {failed_tiles}, "
                f"Skipped (too small): {skipped_tiles}. "
                f"Dynamic World may not have data coverage for this location, "
                f"or the city boundary geometry may be invalid."
            )
        
        # Merge all tile histograms
        merged_counts = self._merge_histograms(tile_histograms)
        
        if not merged_counts:
            raise RuntimeError(
                "Merged histogram is empty. "
                "Dynamic World may not have valid classification data for this location."
            )
        
        return merged_counts


class LandCoverClassifier:
    """Land cover classification using Google Dynamic World model"""
    
    # Dynamic World label mapping
    # 0 = Water
    # 1 = Trees
    # 2 = Grass
    # 3 = Flooded vegetation
    # 4 = Crops
    # 5 = Shrub & scrub
    # 6 = Built area
    # 7 = Bare ground
    # 8 = Snow & ice
    
    DYNAMIC_WORLD_LABELS = {
        0: 'Water',
        1: 'Trees',
        2: 'Grass',
        3: 'Flooded vegetation',
        4: 'Crops',
        5: 'Shrub & scrub',
        6: 'Built area',
        7: 'Bare ground',
        8: 'Snow & ice'
    }
    
    def aggregate_classes(self, pixel_counts: Dict[int, int]) -> LandCoverResult:
        """
        Aggregate Dynamic World classes into our categories and calculate percentages
        
        Mapping:
        - Urban = 6 (Built area)
        - Forest = 1 (Trees)
        - Vegetation = 2 + 3 + 4 + 5 (Grass, Flooded vegetation, Crops, Shrub & scrub)
        - Water = 0 (Water)
        - Bare land = 7 (Bare ground)
        """
        if not pixel_counts:
            raise ValueError("No pixel counts provided")
        
        # Calculate total pixels
        total_pixels = sum(pixel_counts.values())
        
        if total_pixels == 0:
            raise ValueError("No pixels found in the specified area. The location may have no data.")
        
        # Aggregate Dynamic World classes
        urban_count = pixel_counts.get(6, 0)  # Built area
        forest_count = pixel_counts.get(1, 0)  # Trees
        vegetation_count = (
            pixel_counts.get(2, 0) +  # Grass
            pixel_counts.get(3, 0) +  # Flooded vegetation
            pixel_counts.get(4, 0) +  # Crops
            pixel_counts.get(5, 0)    # Shrub & scrub
        )
        water_count = pixel_counts.get(0, 0)  # Water
        bare_land_count = pixel_counts.get(7, 0)  # Bare ground
        
        # Calculate percentages
        return LandCoverResult(
            urban=(urban_count / total_pixels) * 100,
            forest=(forest_count / total_pixels) * 100,
            vegetation=(vegetation_count / total_pixels) * 100,
            water=(water_count / total_pixels) * 100,
            bare_land=(bare_land_count / total_pixels) * 100,
            total_pixels=total_pixels
        )


class DisasterService:
    """Handle real-time natural disaster data from government and satellite feeds"""
    
    def __init__(self, openweather_key: str):
        self.openweather_key = openweather_key
    
    def get_earthquakes(self, lat: float, lon: float, max_radius_km: int = 500, days: int = 7) -> List[Dict]:
        """
        Fetch live earthquake data from USGS
        
        Args:
            lat: Latitude of locality
            lon: Longitude of locality
            max_radius_km: Maximum radius in km (default 500km)
            days: Number of days to look back (default 7)
            
        Returns:
            List of earthquake dictionaries
        """
        try:
            from datetime import datetime, timedelta
            
            # Calculate start time (now - days)
            start_time = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%S')
            
            url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
            params = {
                'format': 'geojson',
                'latitude': lat,
                'longitude': lon,
                'maxradiuskm': max_radius_km,
                'starttime': start_time,
                'minmagnitude': 4.0  # Only significant earthquakes
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return []
            
            data = response.json()
            earthquakes = []
            
            if 'features' in data:
                for feature in data['features']:
                    props = feature.get('properties', {})
                    geom = feature.get('geometry', {})
                    coords = geom.get('coordinates', [])
                    
                    if len(coords) >= 2:
                        eq_lon, eq_lat = coords[0], coords[1]
                        
                        # Calculate distance from locality
                        distance_km = self._calculate_distance(lat, lon, eq_lat, eq_lon)
                        
                        # Filter: only show events within 300-500 km
                        if 300 <= distance_km <= 500:
                            # Parse time
                            time_ms = props.get('time', 0)
                            event_time = datetime.fromtimestamp(time_ms / 1000)
                            time_ago = self._format_time_ago(event_time)
                            
                            earthquakes.append({
                                'type': 'earthquake',
                                'title': f"Earthquake M{props.get('mag', 0):.1f}",
                                'severity': self._get_earthquake_severity(props.get('mag', 0)),
                                'distance_km': round(distance_km, 0),
                                'time': time_ago,
                                'source': 'USGS',
                                'magnitude': props.get('mag', 0),
                                'timestamp': event_time.isoformat()
                            })
            
            return earthquakes
            
        except Exception as e:
            # Return empty list on error, don't block the analysis
            return []
    
    def get_cyclones(self, lat: float, lon: float) -> List[Dict]:
        """
        Fetch live cyclone/storm data from NOAA
        
        Args:
            lat: Latitude of locality
            lon: Longitude of locality
            
        Returns:
            List of cyclone dictionaries
        """
        try:
            url = "https://www.nhc.noaa.gov/CurrentStorms.json"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return []
            
            data = response.json()
            cyclones = []
            
            # NOAA JSON structure may vary, try to parse it
            if isinstance(data, dict):
                storms = data.get('storms', []) or data.get('activeStorms', [])
                
                for storm in storms:
                    # Try to get storm position
                    storm_lat = storm.get('lat') or storm.get('latitude')
                    storm_lon = storm.get('lon') or storm.get('longitude')
                    
                    if storm_lat and storm_lon:
                        distance_km = self._calculate_distance(lat, lon, storm_lat, storm_lon)
                        
                        # Filter: only show storms within 500 km
                        if distance_km <= 500:
                            name = storm.get('name') or storm.get('stormName', 'Unnamed Storm')
                            category = storm.get('category') or storm.get('intensity', 'Unknown')
                            
                            # Try to get forecast time
                            forecast_time = storm.get('forecastTime') or storm.get('expectedLandfall')
                            time_str = self._format_cyclone_time(forecast_time)
                            
                            cyclones.append({
                                'type': 'cyclone',
                                'title': f"Cyclone {name}",
                                'severity': self._get_cyclone_severity(category),
                                'distance_km': round(distance_km, 0),
                                'time': time_str,
                                'source': 'NOAA',
                                'category': category
                            })
            
            return cyclones
            
        except Exception as e:
            # Return empty list on error
            return []
    
    def get_weather_alerts(self, lat: float, lon: float) -> List[Dict]:
        """
        Fetch weather alerts from OpenWeather OneCall API
        
        Args:
            lat: Latitude of locality
            lon: Longitude of locality
            
        Returns:
            List of weather alert dictionaries
        """
        try:
            # Try OneCall API 3.0 first (requires subscription)
            url = "https://api.openweathermap.org/data/3.0/onecall"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.openweather_key,
                'exclude': 'minutely,hourly,daily'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            # If 3.0 fails, try 2.5 (free tier, but may not have alerts)
            if response.status_code != 200:
                url = "https://api.openweathermap.org/data/2.5/onecall"
                response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            alerts = []
            
            if 'alerts' in data and data['alerts']:
                for alert in data['alerts']:
                    event = alert.get('event', '').lower()
                    description = alert.get('description', '')
                    
                    # Filter for relevant disaster types
                    if any(keyword in event for keyword in ['flood', 'rain', 'heat', 'storm', 'wind', 'tornado', 'warning']):
                        severity = self._get_alert_severity(alert.get('severity', ''))
                        
                        alerts.append({
                            'type': 'weather_alert',
                            'title': alert.get('event', 'Weather Alert'),
                            'severity': severity,
                            'distance_km': 0,  # Local alert
                            'time': 'Active',
                            'source': 'OpenWeather',
                            'description': description[:200] if description else 'Weather alert active'
                        })
            
            return alerts
            
        except Exception as e:
            # Return empty list on error (don't block analysis)
            return []
    
    def get_all_disasters(self, lat: float, lon: float) -> List[Dict]:
        """
        Fetch all disaster data in parallel
        
        Args:
            lat: Latitude of locality
            lon: Longitude of locality
            
        Returns:
            Combined list of all disasters
        """
        import concurrent.futures
        
        all_disasters = []
        
        # Fetch all disaster types in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_eq = executor.submit(self.get_earthquakes, lat, lon)
            future_cyc = executor.submit(self.get_cyclones, lat, lon)
            future_weather = executor.submit(self.get_weather_alerts, lat, lon)
            
            try:
                all_disasters.extend(future_eq.result(timeout=10))
            except:
                pass
            
            try:
                all_disasters.extend(future_cyc.result(timeout=10))
            except:
                pass
            
            try:
                all_disasters.extend(future_weather.result(timeout=10))
            except:
                pass
        
        return all_disasters
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km using Haversine formula"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth radius in km
        
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)
        
        a = sin(delta_lat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        return R * c
    
    def _format_time_ago(self, event_time: datetime) -> str:
        """Format time as 'X hours/days ago'"""
        now = datetime.now()
        diff = now - event_time
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds >= 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        else:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    
    def _format_cyclone_time(self, forecast_time) -> str:
        """Format cyclone forecast time"""
        if not forecast_time:
            return "Active"
        
        try:
            if isinstance(forecast_time, str):
                # Try to parse ISO format
                forecast_dt = datetime.fromisoformat(forecast_time.replace('Z', '+00:00'))
                now = datetime.now(forecast_dt.tzinfo) if forecast_dt.tzinfo else datetime.now()
                diff = forecast_dt - now
                
                if diff.total_seconds() > 0:
                    hours = int(diff.total_seconds() / 3600)
                    return f"Expected in {hours} hours"
                else:
                    return "Active"
        except:
            pass
        
        return "Active"
    
    def _get_earthquake_severity(self, magnitude: float) -> str:
        """Determine earthquake severity"""
        if magnitude >= 7.0:
            return "High"
        elif magnitude >= 5.5:
            return "Medium"
        else:
            return "Low"
    
    def _get_cyclone_severity(self, category: str) -> str:
        """Determine cyclone severity"""
        if isinstance(category, (int, float)):
            if category >= 3:
                return "High"
            elif category >= 1:
                return "Medium"
            else:
                return "Low"
        
        category_str = str(category).upper()
        if 'MAJOR' in category_str or 'CATEGORY 3' in category_str or 'CATEGORY 4' in category_str or 'CATEGORY 5' in category_str:
            return "High"
        elif 'CATEGORY 1' in category_str or 'CATEGORY 2' in category_str:
            return "Medium"
        else:
            return "Low"
    
    def _get_alert_severity(self, severity: str) -> str:
        """Determine weather alert severity"""
        severity_lower = str(severity).lower()
        if 'extreme' in severity_lower or 'severe' in severity_lower:
            return "High"
        elif 'moderate' in severity_lower:
            return "Medium"
        else:
            return "Low"


class SupabaseService:
    """Handle Supabase database operations for caching"""
    
    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        """
        Initialize Supabase client
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service role key (for server-side operations)
        """
        self.client: Optional[Client] = None
        if SUPABASE_AVAILABLE and supabase_url and supabase_key:
            try:
                self.client = create_client(supabase_url, supabase_key)
            except Exception as e:
                # Supabase not available, continue without caching
                self.client = None
    
    def is_available(self) -> bool:
        """Check if Supabase is configured and available"""
        return self.client is not None
    
    def insert_locality(self, city: str, name: str, geometry: ee.Geometry, lat: float, lon: float) -> Optional[str]:
        """
        Insert locality into Supabase database
        
        Args:
            city: City name
            name: Locality name
            geometry: Earth Engine geometry (will be converted to GeoJSON)
            lat: Latitude
            lon: Longitude
            
        Returns:
            Locality ID (UUID) if successful, None otherwise
        """
        if not self.client:
            return None
        
        try:
            # Convert Earth Engine geometry to GeoJSON
            geojson = self._ee_geometry_to_geojson(geometry)
            
            if not geojson:
                return None
            
            # Insert into localities table
            result = self.client.table('localities').insert({
                'city': city,
                'name': name,
                'geometry': geojson,  # Supabase will use ST_GeomFromGeoJSON()
                'lat': lat,
                'lon': lon
            }).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]['id']
            
            return None
            
        except Exception as e:
            # Fail silently - caching is optional
            return None
    
    def get_locality_id(self, city: str, name: str) -> Optional[str]:
        """
        Get locality ID from database
        
        Args:
            city: City name
            name: Locality name
            
        Returns:
            Locality ID if found, None otherwise
        """
        if not self.client:
            return None
        
        try:
            result = self.client.table('localities').select('id').eq('city', city).eq('name', name).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]['id']
            
            return None
            
        except Exception:
            return None
    
    def get_cached_landcover(self, locality_id: str) -> Optional[Dict]:
        """
        Get cached landcover data if it exists and is fresh (< 30 days)
        
        Args:
            locality_id: UUID of the locality
            
        Returns:
            Cached histogram data if fresh, None otherwise
        """
        if not self.client:
            return None
        
        try:
            result = self.client.table('landcover_cache').select('*').eq('locality_id', locality_id).execute()
            
            if result.data and len(result.data) > 0:
                cache_entry = result.data[0]
                last_updated = datetime.fromisoformat(cache_entry['last_updated'].replace('Z', '+00:00'))
                
                # Check if cache is fresh (within 30 days)
                age = datetime.now(last_updated.tzinfo) - last_updated
                if age.days < 30:
                    return {
                        'dw_histogram': cache_entry['dw_histogram'],
                        'satellite_source': cache_entry.get('satellite_source', 'Dynamic World'),
                        'satellite_date': cache_entry.get('satellite_date')
                    }
            
            return None
            
        except Exception:
            return None
    
    def save_landcover_cache(self, locality_id: str, dw_histogram: Dict, 
                            satellite_source: str = 'Dynamic World', 
                            satellite_date: Optional[str] = None):
        """
        Save landcover histogram to cache
        
        Args:
            locality_id: UUID of the locality
            dw_histogram: Pixel histogram dictionary
            satellite_source: Source of satellite data
            satellite_date: Date of satellite image
        """
        if not self.client:
            return
        
        try:
            # Upsert (insert or update) cache entry
            self.client.table('landcover_cache').upsert({
                'locality_id': locality_id,
                'dw_histogram': dw_histogram,
                'satellite_source': satellite_source,
                'satellite_date': satellite_date,
                'last_updated': datetime.now().isoformat()
            }).execute()
            
        except Exception:
            # Fail silently - caching is optional
            pass
    
    def _ee_geometry_to_geojson(self, geometry: ee.Geometry) -> Optional[Dict]:
        """
        Convert Earth Engine geometry to GeoJSON format
        
        Args:
            geometry: Earth Engine geometry
            
        Returns:
            GeoJSON dictionary or None if conversion fails
        """
        try:
            # Get geometry info from Earth Engine
            geom_info = geometry.getInfo()
            
            if geom_info.get('type') == 'Polygon':
                return {
                    'type': 'Polygon',
                    'coordinates': geom_info.get('coordinates', [])
                }
            elif geom_info.get('type') == 'MultiPolygon':
                return {
                    'type': 'MultiPolygon',
                    'coordinates': geom_info.get('coordinates', [])
                }
            
            return None
            
        except Exception:
            return None


class WeatherService:
    """Handle weather data from OpenWeather API"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenWeather API key is required")
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def get_weather_data(self, lat: float, lon: float) -> WeatherData:
        """Fetch current weather data for coordinates"""
        url = f"{self.base_url}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"OpenWeather API error: {response.status_code} - {response.text}")
        
        data = response.json()
        
        # Get precipitation from rain or snow if available
        precipitation = 0.0
        if 'rain' in data:
            precipitation = data['rain'].get('1h', 0.0)
        elif 'snow' in data:
            precipitation = data['snow'].get('1h', 0.0)
        
        return WeatherData(
            temperature=data['main']['temp'],
            humidity=data['main']['humidity'],
            precipitation=precipitation,
            wind_speed=data['wind']['speed'],
            pressure=data['main']['pressure'],
            coordinates=(lat, lon)
        )
    
    def get_forecast_data(self, lat: float, lon: float, days: int = 5) -> List[Dict]:
        """Fetch weather forecast data"""
        url = f"{self.base_url}/forecast"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"OpenWeather API forecast error: {response.status_code}")
        
        forecast = response.json()
        return forecast.get('list', [])[:days * 8]  # 8 forecasts per day


class ClimateRiskCalculator:
    """Calculate climate risks based on weather and land cover data"""
    
    @staticmethod
    def calculate_flood_risk(weather: WeatherData, land_cover: LandCoverResult, 
                           forecast: List[Dict] = None) -> str:
        """
        Calculate flood risk based on new rules:
        - if water% > 8 AND rainfall > 10mm → High
        - if water% > 5 → Medium
        - else Low
        """
        water_percent = land_cover.water
        rainfall = weather.precipitation
        
        if water_percent > 8 and rainfall > 10:
            return "High"
        elif water_percent > 5:
            return "Medium"
        else:
            return "Low"
    
    @staticmethod
    def calculate_heat_risk(weather: WeatherData, land_cover: LandCoverResult) -> str:
        """
        Calculate heat risk based on new rules:
        - if urban% > 50 AND temperature > 35°C → High
        - if urban% > 40 → Medium
        - else Low
        """
        urban_percent = land_cover.urban
        temperature = weather.temperature
        
        if urban_percent > 50 and temperature > 35:
            return "High"
        elif urban_percent > 40:
            return "Medium"
        else:
            return "Low"
    
    @staticmethod
    def calculate_drought_risk(weather: WeatherData, land_cover: LandCoverResult,
                              forecast: List[Dict] = None) -> str:
        """
        Calculate drought risk based on new rules:
        - if vegetation% < 20 AND rainfall < 5mm → High
        - if vegetation% < 30 → Medium
        - else Low
        """
        vegetation_percent = land_cover.vegetation
        rainfall = weather.precipitation
        
        if vegetation_percent < 20 and rainfall < 5:
            return "High"
        elif vegetation_percent < 30:
            return "Medium"
        else:
            return "Low"


class GeospatialIntelligenceSystem:
    """Main system orchestrating all components"""
    
    def __init__(self, opencage_key: str, openweather_key: str, earthengine_project: Optional[str] = None,
                 supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        self.geocoding = GeocodingService(opencage_key)
        self.ee_service = EarthEngineService(project=earthengine_project)
        self.classifier = LandCoverClassifier()
        self.weather = WeatherService(openweather_key)
        self.disaster = DisasterService(openweather_key)
        self.risk_calculator = ClimateRiskCalculator()
        self.supabase = SupabaseService(supabase_url, supabase_key)
    
    def analyze_locality(self, city_name: str, locality_name: str, 
                        locality_polygon: ee.Geometry, locality_bbox: BoundingBox,
                        start_date: str = None, end_date: str = None) -> Dict:
        """
        Analyze a specific locality using Google Dynamic World
        
        Fast production-grade pipeline:
        1. Use locality polygon (OSM)
        2. Simplify + buffer geometry
        3. Load Dynamic World (last 30 days)
        4. Clip to locality
        5. Run reduceRegion with frequencyHistogram at scale = 30
        6. Compute percentages
        7. Fetch weather data
        8. Compute risks
        
        Args:
            city_name: Name of the city
            locality_name: Name of the locality
            locality_polygon: Earth Engine polygon geometry for the locality
            locality_bbox: Bounding box for the locality
            start_date: Optional start date filter (defaults to 30 days ago)
            end_date: Optional end date filter (defaults to now)
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # Check Supabase cache first (if available)
            cached_data = None
            locality_id = None
            
            if self.supabase.is_available():
                locality_id = self.supabase.get_locality_id(city_name, locality_name)
                if locality_id:
                    cached_data = self.supabase.get_cached_landcover(locality_id)
            
            # If cache is fresh, use it instead of running Earth Engine
            if cached_data and cached_data.get('dw_histogram'):
                pixel_counts_raw = cached_data['dw_histogram']
                image_date = cached_data.get('satellite_date', '2021-01-01')
            else:
                # Get locality center for fallback
                center_lat = (locality_bbox.min_lat + locality_bbox.max_lat) / 2
                center_lon = (locality_bbox.min_lon + locality_bbox.max_lon) / 2
                
                # STEP 1: Try locality polygon
                geom = ee.Geometry(locality_polygon)
                is_valid, error_msg = EarthEngineService.validate_geometry(geom)
                
                pixel_counts_raw = None
                image_date = None
                buffer_radius_used = None
                
                if is_valid:
                    # Preprocess valid geometry
                    geom = EarthEngineService.preprocess_locality_polygon(geom)
                    
                    # Try Dynamic World with original polygon
                    try:
                        dw_labels, image_date = self.ee_service.get_dynamic_world_image(
                            geom, locality_bbox, None, None
                        )
                        
                        pixel_counts_raw = self.ee_service.count_pixels_by_class_direct(
                            dw_labels, geom, scale=30
                        )
                        
                        # Check if histogram is empty or has no valid pixels
                        if not pixel_counts_raw or len(pixel_counts_raw) == 0 or sum(pixel_counts_raw.values()) == 0:
                            pixel_counts_raw = None  # Trigger fallback
                    except RuntimeError as e:
                        # If it's a geometry error, try fallback
                        if "Empty histogram" in str(e) or "geometry" in str(e).lower():
                            pixel_counts_raw = None  # Trigger fallback
                        else:
                            # Re-raise other runtime errors (like "No Dynamic World image available")
                            raise
                    except Exception as e:
                        # For other exceptions, log and try fallback
                        print(f"Warning: Error processing original polygon: {str(e)}")
                        pixel_counts_raw = None  # Trigger fallback
                
                # STEP 2: If polygon invalid or empty histogram, fallback to centroid buffer
                # Use 2km buffer as default fallback (as specified)
                if pixel_counts_raw is None:
                    buffer_radii = [2000, 3000, 5000]  # Start with 2km, then 3km, max 5km
                    last_error = None
                    
                    for radius_meters in buffer_radii:
                        try:
                            # Create circular buffer around centroid
                            geom = EarthEngineService.create_centroid_buffer(
                                center_lat, center_lon, radius_meters
                            )
                            buffer_radius_used = radius_meters
                            
                            # Get Dynamic World image
                            # Create temporary bbox for buffer
                            buffer_bbox = BoundingBox(
                                min_lon=center_lon - (radius_meters / 111000),  # Approximate degrees
                                min_lat=center_lat - (radius_meters / 111000),
                                max_lon=center_lon + (radius_meters / 111000),
                                max_lat=center_lat + (radius_meters / 111000)
                            )
                            
                            # Check if Dynamic World has data for this location
                            try:
                                dw_labels, image_date = self.ee_service.get_dynamic_world_image(
                                    geom, buffer_bbox, None, None
                                )
                            except RuntimeError as e:
                                if "No Dynamic World image available" in str(e):
                                    # No Dynamic World data for this location - this is a data issue, not geometry
                                    raise RuntimeError(
                                        f"No Dynamic World satellite data available for locality '{locality_name}' "
                                        f"at coordinates ({center_lat:.4f}, {center_lon:.4f}). "
                                        f"Dynamic World may not have coverage for this location."
                                    )
                                raise
                            
                            # Run reduceRegion
                            pixel_counts_raw = self.ee_service.count_pixels_by_class_direct(
                                dw_labels, geom, scale=30
                            )
                            
                            # Check if histogram is empty or has no valid pixels
                            if pixel_counts_raw and len(pixel_counts_raw) > 0 and sum(pixel_counts_raw.values()) > 0:
                                # Success - we have valid pixel counts
                                break  # Exit loop
                            else:
                                pixel_counts_raw = None  # Try next buffer size
                                last_error = f"Empty histogram for {radius_meters}m buffer (no valid pixels found)"
                                
                        except RuntimeError as e:
                            # Re-raise RuntimeErrors (like "No Dynamic World image available")
                            if "No Dynamic World image available" in str(e) or "No Dynamic World satellite data" in str(e):
                                raise
                            # For geometry errors, try next buffer size
                            pixel_counts_raw = None
                            last_error = str(e)
                            continue
                        except Exception as e:
                            pixel_counts_raw = None  # Try next buffer size
                            last_error = str(e)
                            continue
                
                # STEP 5: If STILL empty after all fallbacks
                if pixel_counts_raw is None or sum(pixel_counts_raw.values()) == 0:
                    # Provide detailed error message
                    tried_what = []
                    if is_valid:
                        tried_what.append("original polygon")
                    if buffer_radius_used:
                        tried_what.append(f"buffers up to {buffer_radius_used}m")
                    
                    error_msg = (
                        f"Unable to compute land cover for locality '{locality_name}'. "
                        f"Tried: {', '.join(tried_what) if tried_what else 'centroid buffers'}. "
                    )
                    
                    # Check if we can verify Dynamic World has data
                    try:
                        # Quick check: try to get any Dynamic World image for this location
                        test_geom = EarthEngineService.create_centroid_buffer(center_lat, center_lon, 5000)
                        test_bbox = BoundingBox(
                            min_lon=center_lon - 0.045,
                            min_lat=center_lat - 0.045,
                            max_lon=center_lon + 0.045,
                            max_lat=center_lat + 0.045
                        )
                        try:
                            test_labels, _ = self.ee_service.get_dynamic_world_image(test_geom, test_bbox, None, None)
                            
                            # Try a direct reduceRegion test to see if we get any pixels
                            test_hist = test_labels.reduceRegion(
                                reducer=ee.Reducer.frequencyHistogram(),
                                geometry=test_geom,
                                scale=30,
                                maxPixels=1e13,
                                bestEffort=True
                            )
                            test_hist_dict = test_hist.getInfo()
                            
                            if test_hist_dict and 'label' in test_hist_dict:
                                test_label_hist = test_hist_dict['label']
                                # Count non-masked pixels
                                test_valid_pixels = sum(
                                    int(float(count)) 
                                    for label_str, count in test_label_hist.items() 
                                    if int(float(label_str)) != -1
                                )
                                
                                if test_valid_pixels > 0:
                                    error_msg += (
                                        f"Dynamic World has data ({test_valid_pixels} valid pixels found in test), "
                                        f"but reduceRegion returned empty results for the locality geometry. "
                                        f"This may indicate the locality geometry is outside the Dynamic World coverage area "
                                        f"or there's a geometry processing issue. "
                                        f"Try selecting a different locality or check the locality coordinates."
                                    )
                                else:
                                    error_msg += (
                                        "Dynamic World image exists but all pixels are masked (no data) for this location. "
                                        "This may indicate the area is outside Dynamic World coverage."
                                    )
                            else:
                                error_msg += (
                                    "Dynamic World has data for this location, but reduceRegion returned empty results. "
                                    "This may indicate a geometry processing issue."
                                )
                        except RuntimeError as e:
                            if "No Dynamic World image available" in str(e):
                                error_msg += (
                                    "Dynamic World does not have satellite data coverage for this location. "
                                    "Try a different locality or check if the coordinates are correct."
                                )
                            else:
                                error_msg += f"Error accessing Dynamic World: {str(e)}"
                    except Exception:
                        error_msg += (
                            "Unable to verify Dynamic World coverage. "
                            "The locality geometry may be invalid or outside Dynamic World coverage area."
                        )
                    
                    raise RuntimeError(error_msg)
                
                # Store in Supabase cache for future use
                if self.supabase.is_available() and locality_id:
                    self.supabase.save_landcover_cache(
                        locality_id,
                        pixel_counts_raw,
                        'Dynamic World',
                        image_date
                    )
            
            # Step 6: Aggregate Dynamic World classes and calculate percentages
            land_cover = self.classifier.aggregate_classes(pixel_counts_raw)
            
            # Get locality center for weather data and disasters
            center_lat = (locality_bbox.min_lat + locality_bbox.max_lat) / 2
            center_lon = (locality_bbox.min_lon + locality_bbox.max_lon) / 2
            
            # Fetch weather data and disasters in parallel (non-blocking)
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_weather = executor.submit(self.weather.get_weather_data, center_lat, center_lon)
                future_disasters = executor.submit(self.disaster.get_all_disasters, center_lat, center_lon)
                
                # Get weather data
                weather_data = future_weather.result()
                forecast_data = self.weather.get_forecast_data(center_lat, center_lon)
                
                # Get disaster data (non-blocking, won't delay if it fails)
                try:
                    disasters = future_disasters.result(timeout=15)
                except:
                    disasters = []
            
            # Compute risks with new rules
            flood_risk = self.risk_calculator.calculate_flood_risk(
                weather_data, land_cover, forecast_data
            )
            heat_risk = self.risk_calculator.calculate_heat_risk(
                weather_data, land_cover
            )
            drought_risk = self.risk_calculator.calculate_drought_risk(
                weather_data, land_cover, forecast_data
            )
            
            # Compile results (exact format as specified)
            result = {
                'locality': locality_name,
                'pixel_counts': pixel_counts_raw,  # Raw Dynamic World pixel counts
                'landcover_percentages': {  # Changed from 'percentages' to 'landcover_percentages' for frontend compatibility
                    'urban': round(land_cover.urban, 2),
                    'forest': round(land_cover.forest, 2),
                    'vegetation': round(land_cover.vegetation, 2),
                    'water': round(land_cover.water, 2),
                    'bare_land': round(land_cover.bare_land, 2)
                },
                'satellite_date': image_date,
                # Additional data for UI
                'city': city_name,
                'weather': {
                    'temperature': round(weather_data.temperature, 1),
                    'rainfall': round(weather_data.precipitation, 1),
                    'humidity': round(weather_data.humidity, 1),
                    'wind_speed': round(weather_data.wind_speed, 1),
                    'pressure': round(weather_data.pressure, 1)
                },
                'flood_risk': flood_risk,
                'heat_risk': heat_risk,
                'drought_risk': drought_risk,
                'satellite_source': 'Dynamic World',
                'disasters': disasters
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze locality: {str(e)}")
    
    def analyze_location(self, location: str, start_date: str = None, 
                        end_date: str = None) -> Dict:
        """
        Main analysis function using Google Dynamic World
        
        Returns JSON with:
        - city: Location name
        - coordinates: Center coordinates
        - pixel_counts: Raw pixel counts per Dynamic World class
        - percentages: Aggregated land cover percentages
        - date_of_satellite_image: Date of the Dynamic World image
        """
        try:
            # Step 1: Fetch city administrative boundary polygon from OpenStreetMap
            city_polygon, bbox, center = self.geocoding.get_city_boundary_polygon(location)
            center_lat, center_lon = center
            
            # Step 2: Get Dynamic World image for city polygon
            dw_image, image_date = self.ee_service.get_dynamic_world_image(city_polygon, bbox, start_date, end_date)
            
            # Step 3: Count pixels per class using tiled reduceRegion with frequencyHistogram (only inside city polygon)
            # Uses 2km × 2km tiles with scale=30 for 9x faster processing
            pixel_counts_raw = self.ee_service.count_pixels_by_class(dw_image, city_polygon, bbox, scale=30)
            
            # Step 4: Aggregate Dynamic World classes and calculate percentages
            land_cover = self.classifier.aggregate_classes(pixel_counts_raw)
            
            # Step 5: Get weather data
            weather_data = self.weather.get_weather_data(center_lat, center_lon)
            forecast_data = self.weather.get_forecast_data(center_lat, center_lon)
            
            # Step 6: Calculate climate risks
            flood_risk = self.risk_calculator.calculate_flood_risk(
                weather_data, land_cover, forecast_data
            )
            heat_risk = self.risk_calculator.calculate_heat_risk(
                weather_data, land_cover
            )
            drought_risk = self.risk_calculator.calculate_drought_risk(
                weather_data, land_cover, forecast_data
            )
            
            # Step 7: Compile results as JSON
            result = {
                'city': location,
                'coordinates': {
                    'lat': round(center_lat, 6),
                    'lon': round(center_lon, 6)
                },
                'pixel_counts': {
                    'water': pixel_counts_raw.get(0, 0),
                    'trees': pixel_counts_raw.get(1, 0),
                    'grass': pixel_counts_raw.get(2, 0),
                    'flooded_vegetation': pixel_counts_raw.get(3, 0),
                    'crops': pixel_counts_raw.get(4, 0),
                    'shrub_scrub': pixel_counts_raw.get(5, 0),
                    'built_area': pixel_counts_raw.get(6, 0),
                    'bare_ground': pixel_counts_raw.get(7, 0),
                    'snow_ice': pixel_counts_raw.get(8, 0)
                },
                'percentages': {
                    'urban': round(land_cover.urban, 2),
                    'forest': round(land_cover.forest, 2),
                    'vegetation': round(land_cover.vegetation, 2),
                    'water': round(land_cover.water, 2),
                    'bare_land': round(land_cover.bare_land, 2)
                },
                'date_of_satellite_image': image_date,
                'bounding_box': {
                    'min_lon': round(bbox.min_lon, 6),
                    'min_lat': round(bbox.min_lat, 6),
                    'max_lon': round(bbox.max_lon, 6),
                    'max_lat': round(bbox.max_lat, 6)
                },
                'weather': {
                    'temperature': round(weather_data.temperature, 2),
                    'humidity': round(weather_data.humidity, 2),
                    'precipitation': round(weather_data.precipitation, 2),
                    'wind_speed': round(weather_data.wind_speed, 2),
                    'pressure': round(weather_data.pressure, 2)
                },
                'climate_risks': {
                    'flood': round(flood_risk, 2),
                    'heat': round(heat_risk, 2),
                    'drought': round(drought_risk, 2)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {str(e)}")


def main():
    """Example usage"""
    import sys
    
    # Get API keys from environment variables
    opencage_key = os.getenv('OPENCAGE_API_KEY')
    openweather_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not opencage_key:
        raise ValueError("OPENCAGE_API_KEY environment variable not set")
    if not openweather_key:
        raise ValueError("OPENWEATHER_API_KEY environment variable not set")
    
    # Initialize system
    system = GeospatialIntelligenceSystem(opencage_key, openweather_key)
    
    # Get location from command line or use default
    location = sys.argv[1] if len(sys.argv) > 1 else "Paris, France"
    
    # Run analysis
    print(f"Analyzing location: {location}")
    result = system.analyze_location(location)
    
    # Output JSON
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


