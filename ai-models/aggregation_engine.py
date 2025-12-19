"""
Aggregation Engine
Statistical aggregation of patch-level predictions to city-level statistics

Scientific Rationale:
- Aggregates patch predictions to city-level percentages
- Converts pixel counts to area (km²) using known pixel resolution
- Ensures percentages sum to 100% (±1% tolerance)
- Provides confidence metrics based on prediction consistency
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AggregationEngine:
    """
    Aggregates patch-level predictions to city-level statistics
    """
    
    def __init__(self, pixel_resolution_m: float = 10.0):
        """
        Args:
            pixel_resolution_m: Sentinel-2 pixel resolution in meters (default: 10m)
        """
        self.pixel_resolution_m = pixel_resolution_m
        self.pixel_area_km2 = (pixel_resolution_m / 1000.0) ** 2  # Convert to km²
    
    def aggregate_landcover(self, 
                           all_predictions: List[Dict[str, float]],
                           image_shape: Tuple[int, int],
                           mask: Optional[np.ndarray] = None,
                           tile_size: int = 64,
                           stride: int = 32) -> Dict[str, Dict[str, float]]:
        """
        Aggregate patch predictions to city-level land cover statistics
        
        Scientific Rationale:
        - Each tile represents a fixed area (tile_size² pixels)
        - Weighted by tile probabilities (softmax averaging)
        - Accounts for overlap by using effective area per tile
        - Converts to percentages ensuring sum = 100%
        
        Args:
            all_predictions: List of high-level class probability dicts
            image_shape: (height, width) of full image
            mask: Optional boolean mask (True = valid pixels)
            tile_size: Size of tiles used
            stride: Stride used in tiling
        
        Returns:
            {
                'Vegetation': {'percentage': float, 'area_km2': float},
                'Water': {...},
                'Urban': {...},
                'Agricultural': {...},
                'Barren': {...}
            }
        """
        if not all_predictions:
            return self._empty_landcover()
        
        # Calculate effective area per tile (accounting for overlap)
        # Overlapping tiles contribute less area
        overlap_ratio = stride / tile_size
        tile_area_pixels = tile_size * tile_size * overlap_ratio
        tile_area_km2 = tile_area_pixels * self.pixel_area_km2
        
        # Aggregate probabilities weighted by tile area
        total_area_km2 = 0.0
        class_areas_km2 = {
            'Vegetation': 0.0,
            'Water': 0.0,
            'Urban': 0.0,
            'Agricultural': 0.0,
            'Barren': 0.0
        }
        
        for pred in all_predictions:
            for cls, prob in pred.items():
                if cls in class_areas_km2:
                    # Weight by probability and tile area
                    class_areas_km2[cls] += prob * tile_area_km2
            total_area_km2 += tile_area_km2
        
        # If mask provided, calculate actual city area
        if mask is not None:
            valid_pixels = np.sum(mask)
            actual_area_km2 = valid_pixels * self.pixel_area_km2
            # Use actual area for normalization
            total_area_km2 = actual_area_km2
        
        # Convert to percentages
        if total_area_km2 > 0:
            percentages = {
                cls: (area_km2 / total_area_km2) * 100.0
                for cls, area_km2 in class_areas_km2.items()
            }
        else:
            percentages = {cls: 0.0 for cls in class_areas_km2.keys()}
        
        # Normalize to ensure sum = 100% (±1% tolerance)
        total_pct = sum(percentages.values())
        if abs(total_pct - 100.0) > 1.0:
            logger.warning(f"Percentages sum to {total_pct:.2f}%, normalizing to 100%")
            if total_pct > 0:
                percentages = {cls: (pct / total_pct) * 100.0 
                             for cls, pct in percentages.items()}
            else:
                # Fallback: uniform distribution
                percentages = {cls: 100.0 / len(percentages) 
                             for cls in percentages.keys()}
        
        # Round to 2 decimal places
        percentages = {cls: round(pct, 2) for cls, pct in percentages.items()}
        class_areas_km2 = {cls: round(area, 2) for cls, area in class_areas_km2.items()}
        
        # Build result dictionary
        result = {}
        for cls in class_areas_km2.keys():
            result[cls] = {
                'percentage': percentages[cls],
                'area_km2': class_areas_km2[cls]
            }
        
        logger.info(f"Aggregated land cover: {percentages}")
        logger.info(f"Total area: {total_area_km2:.2f} km²")
        
        return result
    
    def calculate_confidence(self, 
                           all_predictions: List[Dict[str, float]]) -> float:
        """
        Calculate confidence score based on prediction consistency
        
        Scientific Rationale:
        - High variance in predictions = low confidence
        - Consistent predictions = high confidence
        - Uses entropy-based measure
        
        Args:
            all_predictions: List of prediction dictionaries
        
        Returns:
            Confidence score (0-1)
        """
        if not all_predictions:
            return 0.0
        
        # Calculate mean probabilities
        mean_probs = {}
        for pred in all_predictions:
            for cls, prob in pred.items():
                mean_probs[cls] = mean_probs.get(cls, 0.0) + prob
        
        n = len(all_predictions)
        mean_probs = {cls: prob / n for cls, prob in mean_probs.items()}
        
        # Calculate entropy (lower entropy = higher confidence)
        entropy = 0.0
        for prob in mean_probs.values():
            if prob > 0:
                entropy -= prob * np.log2(prob + 1e-10)
        
        # Normalize entropy to [0, 1] confidence
        # Max entropy for 5 classes = log2(5) ≈ 2.32
        max_entropy = np.log2(len(mean_probs))
        normalized_entropy = entropy / max_entropy
        
        # Confidence = 1 - normalized entropy
        confidence = 1.0 - normalized_entropy
        
        return float(max(0.0, min(1.0, confidence)))
    
    def _empty_landcover(self) -> Dict[str, Dict[str, float]]:
        """Return empty land cover structure"""
        classes = ['Vegetation', 'Water', 'Urban', 'Agricultural', 'Barren']
        return {
            cls: {'percentage': 0.0, 'area_km2': 0.0}
            for cls in classes
        }


