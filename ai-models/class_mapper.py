"""
Class Mapping Layer
Maps EuroSAT 10-class predictions to high-level 5-class land cover

Scientific Rationale:
- EuroSAT classes are too granular for city-level statistics
- Aggregation improves statistical robustness
- High-level classes align with common land-use categories
- Softmax averaging preserves uncertainty information
"""

from typing import Dict, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ClassMapper:
    """
    Maps EuroSAT classes to high-level land cover categories
    
    Mapping Rules:
    - Forest, HerbaceousVegetation → Vegetation
    - River, SeaLake → Water
    - Residential, Industrial, Highway → Urban
    - AnnualCrop, PermanentCrop, Pasture → Agricultural
    - Low confidence / remaining → Barren
    """
    
    def __init__(self):
        # Define mapping from EuroSAT to high-level classes
        self.eurosat_to_highlevel = {
            'Forest': 'Vegetation',
            'HerbaceousVegetation': 'Vegetation',
            'River': 'Water',
            'SeaLake': 'Water',
            'Residential': 'Urban',
            'Industrial': 'Urban',
            'Highway': 'Urban',
            'AnnualCrop': 'Agricultural',
            'PermanentCrop': 'Agricultural',
            'Pasture': 'Agricultural'
        }
        
        self.highlevel_classes = [
            'Vegetation',
            'Water',
            'Urban',
            'Agricultural',
            'Barren'
        ]
        
        # Confidence threshold for assigning to high-level class
        # Below this, assign to Barren (uncertain)
        self.confidence_threshold = 0.3
    
    def map_patch_prediction(self, eurosat_probs: Dict[str, float], 
                            confidence_threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Map EuroSAT probabilities to high-level classes using softmax averaging
        
        Scientific Rationale:
        - Preserves uncertainty from CNN predictions
        - Aggregates related classes (e.g., all vegetation types)
        - Handles low-confidence predictions gracefully
        
        Args:
            eurosat_probs: Dictionary of EuroSAT class probabilities
            confidence_threshold: Override default threshold
        
        Returns:
            Dictionary with high-level class probabilities (sums to ~1.0)
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Initialize high-level probabilities
        highlevel_probs = {cls: 0.0 for cls in self.highlevel_classes}
        
        # Aggregate probabilities by mapping
        for eurosat_class, prob in eurosat_probs.items():
            highlevel_class = self.eurosat_to_highlevel.get(eurosat_class, 'Barren')
            highlevel_probs[highlevel_class] += prob
        
        # Normalize to ensure sum ≈ 1.0
        total = sum(highlevel_probs.values())
        if total > 0:
            highlevel_probs = {k: v / total for k, v in highlevel_probs.items()}
        else:
            # All zeros → assign to Barren
            highlevel_probs = {cls: 0.0 if cls != 'Barren' else 1.0 
                             for cls in self.highlevel_classes}
            return highlevel_probs
        
        # Handle low-confidence predictions → assign to Barren
        max_prob = max(highlevel_probs.values())
        if max_prob < confidence_threshold:
            # Low confidence across all classes → uncertain → Barren
            logger.debug(f"Low confidence prediction (max={max_prob:.3f}), assigning to Barren")
            highlevel_probs = {cls: 0.0 if cls != 'Barren' else 1.0 
                             for cls in self.highlevel_classes}
        
        return highlevel_probs
    
    def aggregate_predictions(self, all_predictions: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate multiple patch predictions using softmax averaging
        
        Scientific Rationale:
        - Averages probabilities across all patches
        - Preserves uncertainty information
        - More robust than hard voting
        
        Args:
            all_predictions: List of high-level probability dictionaries
        
        Returns:
            Averaged high-level class probabilities (sums to 1.0)
        """
        if not all_predictions:
            return {cls: 0.0 for cls in self.highlevel_classes}
        
        # Sum all probabilities
        aggregated = {cls: 0.0 for cls in self.highlevel_classes}
        for pred in all_predictions:
            for cls, prob in pred.items():
                if cls in aggregated:
                    aggregated[cls] += prob
        
        # Average
        n = len(all_predictions)
        aggregated = {cls: prob / n for cls, prob in aggregated.items()}
        
        # Normalize (should already be ~1.0, but ensure)
        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v / total for k, v in aggregated.items()}
        else:
            # Fallback: uniform distribution
            aggregated = {cls: 1.0 / len(self.highlevel_classes) 
                         for cls in self.highlevel_classes}
        
        return aggregated
    
    def get_mapping_info(self) -> Dict:
        """Get information about the class mapping"""
        return {
            'eurosat_classes': list(self.eurosat_to_highlevel.keys()),
            'highlevel_classes': self.highlevel_classes,
            'mapping': self.eurosat_to_highlevel,
            'confidence_threshold': self.confidence_threshold
        }


