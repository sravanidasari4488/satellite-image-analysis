"""
Spectral Index Validation Layer
Validates CNN predictions using spectral indices and performs hybrid fusion

Scientific Rationale:
- Spectral indices provide physical constraints on land cover
- CNN predictions can be validated against these constraints
- Hybrid fusion improves robustness by combining ML and physics-based methods
- Down-weighting conflicting predictions reduces errors
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class IndexValidator:
    """
    Validates CNN predictions with spectral indices
    
    Uses NDVI, NDWI, NDBI, and BSI to verify predictions
    """
    
    def __init__(self):
        # Thresholds based on scientific literature
        self.index_thresholds = {
            'ndvi_vegetation': 0.3,    # NDVI > 0.3 indicates vegetation
            'ndwi_water': 0.3,         # NDWI > 0.3 indicates water
            'ndbi_urban': 0.1,         # NDBI > 0.1 indicates built-up
            'bsi_barren': 0.2,         # BSI > 0.2 indicates bare soil
            'ndvi_agricultural_min': 0.2,  # Agricultural NDVI range
            'ndvi_agricultural_max': 0.4
        }
        
        # Fusion weights: how much to trust indices vs CNN
        self.fusion_weights = {
            'agreement_boost': 1.2,    # Boost confidence when indices agree
            'conflict_penalty': 0.5,   # Reduce confidence when indices conflict
            'neutral_weight': 1.0      # No change when indices neutral
        }
    
    def calculate_bsi(self, red: np.ndarray, blue: np.ndarray, 
                     nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """
        Calculate Bare Soil Index (BSI)
        
        Formula: BSI = ((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))
        
        Higher BSI indicates bare soil/barren land
        """
        numerator = (swir + red) - (nir + blue)
        denominator = (swir + red) + (nir + blue) + 1e-10
        bsi = numerator / denominator
        return np.clip(bsi, -1, 1)
    
    def validate_with_indices(self, cnn_probs: Dict[str, float],
                             indices: Dict[str, np.ndarray],
                             patch_coords: Optional[Tuple[int, int]] = None) -> Dict[str, float]:
        """
        Validate CNN prediction using spectral indices (hybrid fusion)
        
        Scientific Rationale:
        - Spectral indices provide physical constraints
        - If CNN strongly predicts a class but indices disagree, reduce confidence
        - If CNN and indices agree, boost confidence
        - This hybrid approach combines ML and physics-based methods
        
        Args:
            cnn_probs: CNN prediction probabilities (high-level classes)
            indices: Spectral indices for the patch {'ndvi': array, 'ndwi': array, ...}
            patch_coords: Optional patch coordinates for logging
        
        Returns:
            Adjusted probabilities with index validation
        """
        # Get mean indices for patch
        mean_ndvi = float(np.mean(indices.get('ndvi', np.array([0]))))
        mean_ndwi = float(np.mean(indices.get('ndwi', np.array([0]))))
        mean_ndbi = float(np.mean(indices.get('ndbi', np.array([0]))))
        mean_bsi = float(np.mean(indices.get('bsi', np.array([0]))))
        
        # Calculate index-based signals (0-1 scale)
        index_signals = {
            'Vegetation': 1.0 if mean_ndvi > self.index_thresholds['ndvi_vegetation'] else 0.0,
            'Water': 1.0 if mean_ndwi > self.index_thresholds['ndwi_water'] else 0.0,
            'Urban': 1.0 if mean_ndbi > self.index_thresholds['ndbi_urban'] else 0.0,
            'Barren': 1.0 if mean_bsi > self.index_thresholds['bsi_barren'] else 0.0,
            'Agricultural': 1.0 if (self.index_thresholds['ndvi_agricultural_min'] < 
                                   mean_ndvi < self.index_thresholds['ndvi_agricultural_max']) else 0.0
        }
        
        # Hybrid fusion: Adjust CNN probabilities based on index agreement
        adjusted_probs = {}
        for cls in cnn_probs:
            cnn_prob = cnn_probs[cls]
            index_signal = index_signals.get(cls, 0.5)  # Neutral if no index available
            
            # Determine fusion weight
            if cnn_prob > 0.5:
                # CNN strongly predicts this class
                if index_signal > 0.7:
                    # Indices agree → boost confidence
                    weight = self.fusion_weights['agreement_boost']
                elif index_signal < 0.3:
                    # Indices strongly disagree → reduce confidence
                    weight = self.fusion_weights['conflict_penalty']
                else:
                    # Neutral agreement
                    weight = self.fusion_weights['neutral_weight']
            else:
                # CNN doesn't strongly predict this class
                weight = self.fusion_weights['neutral_weight']
            
            adjusted_probs[cls] = cnn_prob * weight
        
        # Normalize to ensure probabilities sum to ~1.0
        total = sum(adjusted_probs.values())
        if total > 0:
            adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}
        else:
            # Fallback: return original probabilities
            adjusted_probs = cnn_probs
        
        # Log if significant adjustment occurred
        if patch_coords:
            max_original = max(cnn_probs.values())
            max_adjusted = max(adjusted_probs.values())
            if abs(max_original - max_adjusted) > 0.2:
                logger.debug(f"Patch {patch_coords}: Significant adjustment "
                           f"({max_original:.3f} → {max_adjusted:.3f})")
        
        return adjusted_probs
    
    def get_index_info(self) -> Dict:
        """Get information about index thresholds"""
        return {
            'thresholds': self.index_thresholds,
            'fusion_weights': self.fusion_weights
        }


