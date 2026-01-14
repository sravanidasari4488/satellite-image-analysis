"""
Area of Interest (AOI) Handler
Fetches exact administrative boundary polygons from OpenStreetMap Nominatim
and clips Sentinel-2 imagery to the polygon.

Scientific Rationale:
- Administrative boundaries provide exact city extents, not approximations
- Polygon clipping ensures analysis only covers the city area
- Reduces edge effects and improves accuracy for city-level statistics
"""

import numpy as np
import requests
import json
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

try:
    from shapely.geometry import Polygon, Point, mapping
    from shapely.ops import transform
    import pyproj
    from functools import partial
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logger.warning("Shapely not available. Polygon operations will be limited.")

class AOIHandler:
    """
    Handles city boundary polygon fetching and image clipping
    
    Uses OpenCage for place identification, OpenStreetMap for boundaries
    """
    
    def __init__(self):
        self.nominatim_base = "https://nominatim.openstreetmap.org"
        self.overpass_base = "https://overpass-api.de/api/interpreter"
        self.opencage_base = "https://api.opencagedata.com/geocode/v1/json"
        self.request_delay = 1.0  # Rate limiting for OSM
    
    def geocode_with_opencage(self, location: str) -> Dict:
        """
        Use OpenCage only for place identification (coordinates, address)
        Does NOT fetch boundaries - that's done via OSM
        
        Args:
            location: City name or address
        
        Returns:
            {
                'coordinates': {'latitude': float, 'longitude': float},
                'address': str,
                'components': dict
            }
        """
        import os
        api_key = os.getenv('OPENCAGE_API_KEY')
        if not api_key:
            raise ValueError("OPENCAGE_API_KEY not set")
        
        try:
            response = requests.get(
                self.opencage_base,
                params={
                    'q': location,
                    'key': api_key,
                    'limit': 1,
                    'no_annotations': 1
                },
                timeout=10
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenCage API error: {response.status_code}")
            
            data = response.json()
            if not data.get('results'):
                raise Exception(f"Location not found: {location}")
            
            result = data['results'][0]
            geometry = result['geometry']
            
            return {
                'coordinates': {
                    'latitude': geometry['lat'],
                    'longitude': geometry['lng']
                },
                'address': result['formatted'],
                'components': result.get('components', {})
            }
        except Exception as e:
            logger.error(f"OpenCage geocoding failed: {e}")
            raise
    
    def fetch_city_polygon_osm(self, location: str, admin_level: int = 8) -> Dict:
        """
        Fetch exact administrative boundary polygon from OpenStreetMap
        
        Scientific Rationale:
        - OSM provides authoritative administrative boundaries
        - admin_level 8 = city/town level
        - Returns GeoJSON-compatible polygon coordinates
        
        Args:
            location: City name
            admin_level: Administrative level (8=city, 6=state, 4=country)
        
        Returns:
            {
                'polygon': [[lng, lat], ...],  # GeoJSON coordinates
                'bbox': [min_lng, min_lat, max_lng, max_lat],
                'area_km2': float,
                'source': 'osm_nominatim',
                'admin_level': int
            }
        """
        try:
            # Step 1: Search for place in Nominatim
            time.sleep(self.request_delay)  # Rate limiting
            search_url = f"{self.nominatim_base}/search"
            params = {
                'q': location,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1,
                'extratags': 1
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code != 200:
                raise Exception(f"OSM search failed: {response.status_code}")
            
            results = response.json()
            if not results:
                raise Exception(f"Location not found in OSM: {location}")
            
            place = results[0]
            osm_id = place.get('osm_id')
            osm_type = place.get('osm_type')  # 'relation', 'way', or 'node'
            
            if not osm_id:
                raise Exception("OSM ID not found")
            
            logger.info(f"Found OSM place: {place['display_name']} (type: {osm_type}, id: {osm_id})")
            
            # Step 2: Fetch boundary using Overpass API
            time.sleep(self.request_delay)
            
            # Try to get relation ID for administrative boundary
            # First, try to find admin boundary relation
            if osm_type == 'relation':
                query = f"""
                [out:json][timeout:25];
                (
                  relation({osm_id});
                );
                out geom;
                """
            elif osm_type == 'way':
                # For ways, try to get the way geometry
                query = f"""
                [out:json][timeout:25];
                (
                  way({osm_id});
                );
                out geom;
                """
            else:
                # For nodes, search for admin boundary relation
                # Use the place's bounding box to search
                bbox = place.get('boundingbox', [])
                if bbox:
                    min_lat, max_lat, min_lng, max_lng = map(float, bbox)
                    # Search for admin boundary relation in the area
                    query = f"""
                    [out:json][timeout:25];
                    (
                      relation["admin_level"="8"]["boundary"="administrative"]({min_lat},{min_lng},{max_lat},{max_lng});
                    );
                    out geom;
                    """
                else:
                    raise Exception(f"Cannot get polygon for node type: {location}")
            
            overpass_response = requests.post(
                self.overpass_base,
                data={'data': query},
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=30
            )
            
            if overpass_response.status_code != 200:
                raise Exception(f"Overpass API error: {overpass_response.status_code}")
            
            data = overpass_response.json()
            
            # Step 3: Parse polygon from Overpass response
            polygon_coords = self._parse_overpass_polygon(data, osm_type)
            
            if not polygon_coords:
                # Fallback to bounding box
                bbox = place.get('boundingbox', [])
                if bbox:
                    logger.warning(f"Using bounding box fallback for {location}")
                    return self._bbox_to_polygon(bbox)
                raise Exception("Could not extract polygon or bbox")
            
            # Step 4: Calculate bbox and area
            lngs = [coord[0] for coord in polygon_coords]
            lats = [coord[1] for coord in polygon_coords]
            
            bbox = [min(lngs), min(lats), max(lngs), max(lats)]
            area_km2 = self._calculate_polygon_area(polygon_coords)
            
            logger.info(f"Fetched polygon for {location}: {len(polygon_coords)} points, "
                       f"area: {area_km2:.2f} km²")
            
            return {
                'polygon': polygon_coords,
                'bbox': bbox,
                'area_km2': area_km2,
                'source': 'osm_nominatim',
                'admin_level': admin_level,
                'osm_id': osm_id,
                'osm_type': osm_type
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch OSM polygon: {e}")
            raise
    
    def _parse_overpass_polygon(self, data: Dict, osm_type: str) -> Optional[List[List[float]]]:
        """
        Parse polygon coordinates from Overpass API response
        
        Handles different OSM element types and their geometry structures
        """
        try:
            elements = data.get('elements', [])
            if not elements:
                return None
            
            # Find the main element (relation, way, or node)
            element = None
            for elem in elements:
                if elem.get('type') == osm_type:
                    element = elem
                    break
            
            if not element:
                element = elements[0]  # Fallback to first element
            
            if osm_type == 'relation':
                # Relations have members with geometries
                # Look for outer way (boundary)
                members = element.get('members', [])
                outer_ways = []
                
                for member in members:
                    if member.get('type') == 'way' and 'geometry' in member:
                        role = member.get('role', '')
                        if role == 'outer' or not outer_ways:  # Prefer outer, but use any if no outer
                            coords = member.get('geometry', [])
                            if coords and len(coords) > 2:
                                way_coords = [[point.get('lon', 0), point.get('lat', 0)] 
                                            for point in coords if 'lon' in point and 'lat' in point]
                                if way_coords:
                                    outer_ways.append(way_coords)
                
                # Combine outer ways into polygon (simplified - assumes single ring)
                if outer_ways:
                    # Use the longest way (main boundary)
                    longest_way = max(outer_ways, key=len)
                    # Close polygon if not closed
                    if longest_way[0] != longest_way[-1]:
                        longest_way.append(longest_way[0])
                    return longest_way
            
            elif osm_type == 'way':
                # Ways have direct geometry
                if 'geometry' in element:
                    coords = element.get('geometry', [])
                    if coords and len(coords) > 2:
                        way_coords = [[point.get('lon', 0), point.get('lat', 0)] 
                                    for point in coords if 'lon' in point and 'lat' in point]
                        if way_coords:
                            # Close polygon if not closed
                            if way_coords[0] != way_coords[-1]:
                                way_coords.append(way_coords[0])
                            return way_coords
            
            return None
        except Exception as e:
            logger.error(f"Error parsing Overpass polygon: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _bbox_to_polygon(self, bbox: List[str]) -> Dict:
        """Convert bounding box to polygon (fallback)"""
        min_lat, max_lat, min_lng, max_lng = map(float, bbox)
        
        polygon_coords = [
            [min_lng, min_lat],
            [max_lng, min_lat],
            [max_lng, max_lat],
            [min_lng, max_lat],
            [min_lng, min_lat]  # Close polygon
        ]
        
        area_km2 = self._calculate_polygon_area(polygon_coords)
        
        return {
            'polygon': polygon_coords,
            'bbox': [min_lng, min_lat, max_lng, max_lat],
            'area_km2': area_km2,
            'source': 'osm_bbox_fallback',
            'admin_level': None
        }
    
    def _calculate_polygon_area(self, coords: List[List[float]]) -> float:
        """
        Calculate polygon area in km² using spherical approximation
        
        Uses Shoelace formula with Earth radius correction
        """
        if not coords or len(coords) < 3:
            return 0.0
        
        # Earth radius in km
        R = 6371.0
        
        # Close polygon if not closed
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        
        area = 0.0
        n = len(coords) - 1
        
        for i in range(n):
            lat1, lng1 = coords[i][1], coords[i][0]
            lat2, lng2 = coords[i+1][1], coords[i+1][0]
            
            # Convert to radians
            lat1_rad = np.radians(lat1)
            lat2_rad = np.radians(lat2)
            d_lng = np.radians(lng2 - lng1)
            
            # Spherical excess formula
            area += d_lng * (2 + np.sin(lat1_rad) + np.sin(lat2_rad))
        
        area = abs(area * R * R / 2.0)
        return area
    
    def clip_image_to_polygon(self, image: np.ndarray,
                             polygon_coords: List[List[float]],
                             image_bbox: List[float],
                             image_crs: str = 'EPSG:4326') -> Tuple[np.ndarray, np.ndarray]:
        """
        Clip satellite image to exact polygon boundary
        
        Scientific Rationale:
        - Ensures analysis only covers city area
        - Reduces edge effects from surrounding areas
        - Provides accurate area calculations
        
        Args:
            image: Full satellite image array (H, W, C)
            polygon_coords: Polygon coordinates [[lng, lat], ...]
            image_bbox: [min_lng, min_lat, max_lng, max_lat] of image
            image_crs: Coordinate reference system
        
        Returns:
            (clipped_image, mask) where mask is boolean array
        """
        height, width = image.shape[:2]
        mask = np.ones((height, width), dtype=bool)
        
        # Convert polygon coords to pixel coordinates
        min_lng, min_lat, max_lng, max_lat = image_bbox
        
        # Simple linear mapping (assumes image covers bbox exactly)
        def coord_to_pixel(lng: float, lat: float) -> Tuple[int, int]:
            x = int((lng - min_lng) / (max_lng - min_lng) * width)
            y = int((lat - min_lat) / (max_lat - min_lat) * height)
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)
            return x, y
        
        # Create polygon mask using point-in-polygon test
        # For each pixel, check if it's inside polygon
        for y in range(height):
            for x in range(width):
                # Convert pixel to lat/lng
                lng = min_lng + (x / width) * (max_lng - min_lng)
                lat = min_lat + (y / height) * (max_lat - min_lat)
                
                # Point-in-polygon test (ray casting algorithm)
                inside = self._point_in_polygon(lng, lat, polygon_coords)
                mask[y, x] = inside
        
        # Apply mask to image
        clipped = image.copy()
        if len(image.shape) == 3:
            mask_3d = np.stack([mask] * image.shape[2], axis=2)
            clipped[~mask_3d] = 0
        else:
            clipped[~mask] = 0
        
        logger.info(f"Clipped image: {np.sum(mask)}/{mask.size} pixels inside polygon "
                   f"({100*np.sum(mask)/mask.size:.1f}%)")
        
        return clipped, mask
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[List[float]]) -> bool:
        """
        Ray casting algorithm for point-in-polygon test
        
        Returns True if point (x, y) is inside polygon
        """
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

