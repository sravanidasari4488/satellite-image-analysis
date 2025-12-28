"""
Multi-Factor Risk Assessment Models
Implements scientifically-validated flood and drought risk calculations

Scientific Rationale:
- Single-factor risk models are unreliable
- Multi-factor models capture complex interactions
- Normalized scores allow comparison across cities
- Weighted combination based on scientific literature
"""

import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FloodRiskModel:
    """
    Multi-factor flood risk assessment
    
    Formula:
    FloodScore = 0.4 × RainfallIndex + 0.3 × UrbanIndex + 
                 0.2 × LowElevationIndex + 0.1 × WaterProximityIndex
    
    Weights based on:
    - Rainfall: Primary driver (40%)
    - Urbanization: Impervious surfaces increase runoff (30%)
    - Elevation: Low-lying areas flood more (20%)
    - Water proximity: Existing water bodies increase risk (10%)
    """
    
    def __init__(self):
        # Normalization parameters (based on typical ranges)
        self.rainfall_max_mm = 100.0  # 100mm in 7 days = high risk
        self.elevation_variance_max = 50.0  # Low variance = flood-prone
        self.ndwi_density_max = 0.5  # High NDWI density = water nearby
    
    def calculate_flood_risk(self, 
                             rainfall_7d: float,
                             rainfall_30d: float,
                             urban_percentage: float,
                             elevation_variance: Optional[float] = None,
                             ndwi_density: Optional[float] = None) -> Dict:
        """
        Calculate normalized flood risk score
        
        Args:
            rainfall_7d: 7-day rainfall accumulation (mm)
            rainfall_30d: 30-day rainfall accumulation (mm)
            urban_percentage: Percentage of urban/impervious surface (0-100)
            elevation_variance: Elevation variance (lower = more flood-prone)
            ndwi_density: Mean NDWI value (higher = more water nearby)
        
        Returns:
            {
                'level': 'low' | 'moderate' | 'high',
                'score': float (0-1),
                'components': {...},
                'metadata': {...}
            }
        """
        # Normalize inputs to [0, 1] range
        # Use 7-day rainfall (more relevant for flooding)
        rainfall_index = min(rainfall_7d / self.rainfall_max_mm, 1.0)
        
        # Urban percentage already 0-100
        urban_index = min(urban_percentage / 100.0, 1.0)
        
        # Elevation: lower variance = higher risk (inverse relationship)
        if elevation_variance is not None:
            # Normalize: variance of 0 = max risk (1.0), variance of max = min risk (0.0)
            elevation_index = 1.0 - min(elevation_variance / self.elevation_variance_max, 1.0)
        else:
            elevation_index = 0.5  # Neutral if not available
        
        # Water proximity: higher NDWI = more water nearby = higher risk
        if ndwi_density is not None:
            water_proximity = min(ndwi_density / self.ndwi_density_max, 1.0)
        else:
            water_proximity = 0.5  # Neutral if not available
        
        # Weighted combination
        flood_score = (
            0.4 * rainfall_index +
            0.3 * urban_index +
            0.2 * elevation_index +
            0.1 * water_proximity
        )
        
        # Ensure score is in [0, 1]
        flood_score = max(0.0, min(1.0, flood_score))
        
        # Map to risk levels using percentiles
        if flood_score >= 0.7:
            level = 'high'
        elif flood_score >= 0.4:
            level = 'moderate'
        else:
            level = 'low'
        
        return {
            'level': level,
            'score': float(flood_score),
            'components': {
                'rainfall_contribution': float(0.4 * rainfall_index),
                'urban_contribution': float(0.3 * urban_index),
                'elevation_contribution': float(0.2 * elevation_index),
                'water_proximity_contribution': float(0.1 * water_proximity)
            },
            'metadata': {
                'rainfall_7d_mm': rainfall_7d,
                'rainfall_30d_mm': rainfall_30d,
                'urban_percentage': urban_percentage,
                'elevation_variance': elevation_variance,
                'ndwi_density': ndwi_density
            }
        }


class DroughtRiskModel:
    """
    Multi-factor drought risk assessment
    
    Uses long-term indicators:
    - 3-6 month rainfall anomaly (deviation from normal)
    - NDVI trend (declining = drought stress)
    - Water body surface area change (shrinking = drought)
    
    Scientific Rationale:
    - Drought is a long-term phenomenon (not 7-day)
    - Multiple indicators provide robust assessment
    - Trend analysis captures ongoing conditions
    """
    
    def __init__(self):
        # Normalization parameters
        self.rainfall_anomaly_max_pct = 50.0  # -50% = severe drought
        self.ndvi_trend_max = 0.2  # -0.2 NDVI change = significant
        self.water_area_change_max = 0.3  # -30% water loss = severe
    
    def calculate_drought_risk(self,
                              rainfall_anomaly_3m: Optional[float] = None,
                              rainfall_anomaly_6m: Optional[float] = None,
                              ndvi_trend: Optional[float] = None,
                              water_area_change: Optional[float] = None) -> Dict:
        """
        Calculate drought risk using long-term indicators
        
        Args:
            rainfall_anomaly_3m: 3-month rainfall anomaly (% deviation from normal)
            rainfall_anomaly_6m: 6-month rainfall anomaly (% deviation from normal)
            ndvi_trend: Change in NDVI over time (negative = declining vegetation)
            water_area_change: Change in water body area (negative = shrinking)
        
        Returns:
            {
                'level': 'low' | 'moderate' | 'high',
                'score': float (0-1),
                'components': {...},
                'metadata': {...}
            }
        """
        # Normalize inputs
        # Use 6-month anomaly if available, else 3-month
        if rainfall_anomaly_6m is not None:
            anomaly_norm = min(abs(rainfall_anomaly_6m) / self.rainfall_anomaly_max_pct, 1.0)
            anomaly_3m_norm = min(abs(rainfall_anomaly_3m or 0) / self.rainfall_anomaly_max_pct, 1.0)
        elif rainfall_anomaly_3m is not None:
            anomaly_norm = min(abs(rainfall_anomaly_3m) / self.rainfall_anomaly_max_pct, 1.0)
            anomaly_3m_norm = anomaly_norm
        else:
            anomaly_norm = 0.5  # Neutral if not available
            anomaly_3m_norm = 0.5
        
        # NDVI trend: negative = declining = drought
        if ndvi_trend is not None:
            ndvi_decline = max(-ndvi_trend, 0)  # Only negative trends matter
            ndvi_norm = min(ndvi_decline / self.ndvi_trend_max, 1.0)
        else:
            ndvi_norm = 0.5  # Neutral if not available
        
        # Water area change: negative = shrinking = drought
        if water_area_change is not None:
            water_loss = max(-water_area_change, 0)  # Only losses matter
            water_norm = min(water_loss / self.water_area_change_max, 1.0)
        else:
            water_norm = 0.5  # Neutral if not available
        
        # Weighted combination
        # 6-month anomaly gets more weight (longer-term indicator)
        drought_score = (
            0.3 * anomaly_norm +      # 6-month anomaly
            0.2 * anomaly_3m_norm +   # 3-month anomaly
            0.3 * ndvi_norm +         # Vegetation decline
            0.2 * water_norm          # Water loss
        )
        
        # Ensure score is in [0, 1]
        drought_score = max(0.0, min(1.0, drought_score))
        
        # Map to risk levels
        if drought_score >= 0.7:
            level = 'high'
        elif drought_score >= 0.4:
            level = 'moderate'
        else:
            level = 'low'
        
        return {
            'level': level,
            'score': float(drought_score),
            'components': {
                'rainfall_anomaly_6m': float(anomaly_norm),
                'rainfall_anomaly_3m': float(anomaly_3m_norm),
                'ndvi_decline': float(ndvi_norm),
                'water_loss': float(water_norm)
            },
            'metadata': {
                'rainfall_anomaly_3m_pct': rainfall_anomaly_3m,
                'rainfall_anomaly_6m_pct': rainfall_anomaly_6m,
                'ndvi_trend': ndvi_trend,
                'water_area_change': water_area_change
            }
        }






