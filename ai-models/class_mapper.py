# #Code to be updated in ai-models
# #class_mapper.py
# """
# Class Mapping Layer
# Maps EuroSAT 10-class predictions to high-level 5-class land cover
# Scientific Rationale:
# - EuroSAT classes are too granular for city-level statistics
# - Aggregation improves statistical robustness
# - High-level classes align with common land-use categories
# - Softmax averaging preserves uncertainty information
# """
# from typing import Dict, List, Optional
# import numpy as np
# import logging
# logger = logging.getLogger(__name__)
# class ClassMapper:
#     """
#     Maps EuroSAT classes to high-level land cover categories

#     Mapping Rules:
#     - Forest, HerbaceousVegetation → Vegetation
#     - River, SeaLake → Water
#     - Residential, Industrial, Highway → Urban
#     - AnnualCrop, PermanentCrop, Pasture → Agricultural
#     - Low confidence / remaining → Barren
#     """

#     def __init__(self):
#         # Define mapping from EuroSAT to high-level classes
#         self.eurosat_to_highlevel = {
#             'Forest': 'Vegetation',
#             'HerbaceousVegetation': 'Vegetation',
#             'River': 'Water',
#             'SeaLake': 'Water',
#             'Residential': 'Urban',
#             'Industrial': 'Urban',
#             'Highway': 'Urban',
#             'AnnualCrop': 'Agricultural',
#             'PermanentCrop': 'Agricultural',
#             'Pasture': 'Agricultural'
#         }

#         self.highlevel_classes = [
#             'Vegetation',
#             'Water',
#             'Urban',
#             'Agricultural',
#             'Barren'
#         ]

#         # Confidence threshold for assigning to high-level class
#         # Below this, assign to Barren only if the max probability is extremely low
#         self.confidence_threshold = 0.15

#     def map_patch_prediction(self, eurosat_probs: Dict[str, float],
#                             confidence_threshold: Optional[float] = None) -> Dict[str, float]:
#         """
#         Map EuroSAT probabilities to high-level classes using softmax averaging

#         Scientific Rationale:
#         - Preserves uncertainty from CNN predictions
#         - Aggregates related classes (e.g., all vegetation types)
#         - Handles low-confidence predictions gracefully

#         Args:
#             eurosat_probs: Dictionary of EuroSAT class probabilities
#             confidence_threshold: Override default threshold

#         Returns:
#             Dictionary with high-level class probabilities (sums to ~1.0)
#         """
#         if confidence_threshold is None:
#             confidence_threshold = self.confidence_threshold

#         # Initialize high-level probabilities
#         highlevel_probs = {cls: 0.0 for cls in self.highlevel_classes}

#         # Aggregate probabilities by mapping
#         for eurosat_class, prob in eurosat_probs.items():
#             highlevel_class = self.eurosat_to_highlevel.get(eurosat_class, 'Barren')
#             highlevel_probs[highlevel_class] += prob

#         # Normalize to ensure sum ≈ 1.0
#         total = sum(highlevel_probs.values())
#         if total > 0:
#             highlevel_probs = {k: v / total for k, v in highlevel_probs.items()}
#         else:
#             # All zeros → assign to Barren
#             highlevel_probs = {cls: 0.0 if cls != 'Barren' else 1.0
#                              for cls in self.highlevel_classes}
#             return highlevel_probs

#         # Handle low-confidence predictions
#         max_prob = max(highlevel_probs.values())
#         if max_prob < confidence_threshold:
#             # If extremely low (< 0.05), it's probably noise -> Barren
#             if max_prob < 0.05:
#                 logger.debug(f"Extremely low confidence prediction (max={max_prob:.3f}), assigning to Barren")
#                 highlevel_probs = {cls: 0.0 if cls != 'Barren' else 1.0
#                                  for cls in self.highlevel_classes}
#             else:
#                 # If between 0.05 and 0.15, keep the distribution but boost Barren slightly
#                 # instead of nuking the whole distribution
#                 # This preserves some information about what the model *thinks* is there
#                 logger.debug(f"Low confidence prediction (max={max_prob:.3f}), boosting Barren slightly")
#                 highlevel_probs['Barren'] = (highlevel_probs['Barren'] + 0.1) / 1.1
#                 # Re-normalize
#                 total = sum(highlevel_probs.values())
#                 highlevel_probs = {k: v / total for k, v in highlevel_probs.items()}

#         return highlevel_probs

#     def aggregate_predictions(self, all_predictions: List[Dict[str, float]]) -> Dict[str, float]:
#         """
#         Aggregate multiple patch predictions using softmax averaging

#         Scientific Rationale:
#         - Averages probabilities across all patches
#         - Preserves uncertainty information
#         - More robust than hard voting

#         Args:
#             all_predictions: List of high-level probability dictionaries

#         Returns:
#             Averaged high-level class probabilities (sums to 1.0)
#         """
#         if not all_predictions:
#             return {cls: 0.0 for cls in self.highlevel_classes}

#         # Sum all probabilities
#         aggregated = {cls: 0.0 for cls in self.highlevel_classes}
#         for pred in all_predictions:
#             for cls, prob in pred.items():
#                 if cls in aggregated:
#                     aggregated[cls] += prob

#         # Average
#         n = len(all_predictions)
#         aggregated = {cls: prob / n for cls, prob in aggregated.items()}

#         # Normalize (should already be ~1.0, but ensure)
#         total = sum(aggregated.values())
#         if total > 0:
#             aggregated = {k: v / total for k, v in aggregated.items()}
#         else:
#             # Fallback: uniform distribution
#             aggregated = {cls: 1.0 / len(self.highlevel_classes)
#                          for cls in self.highlevel_classes}

#         return aggregated

#     def get_mapping_info(self) -> Dict:
#         """Get information about the class mapping"""
#         return {
#             'eurosat_classes': list(self.eurosat_to_highlevel.keys()),
#             'highlevel_classes': self.highlevel_classes,
#             'mapping': self.eurosat_to_highlevel,
#             'confidence_threshold': self.confidence_threshold
#         }

"""
Advanced Dynamic Class Mapping & Uncertainty Layer
-------------------------------------------------
Orchestrates the translation of granular CNN outputs (e.g., EuroSAT) into 
flexible, high-level land cover hierarchies.

Scientific Rationale:
- Hierarchical Aggregation: Reduces class noise by grouping spectrally similar categories.
- Aleatoric Uncertainty: Uses Shannon Entropy to identify patches where the model 
  is fundamentally "confused" between multiple classes.
- Softmax Averaging: Preserves the probabilistic manifold of the underlying model, 
  preventing information loss inherent in "hard" class assignment.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import logging

# Configure standardized logging
logger = logging.getLogger(__name__)

class ClassMapper:
    """
    Dynamic Mapper for Satellite Image Classification.
    
    This class handles the mapping between a specific model's output neurons 
    and the application's required land-use categories.
    """

    def __init__(self, 
                 custom_mapping: Optional[Dict[str, str]] = None,
                 fallback_class: str = 'Barren'):
        """
        Args:
            custom_mapping: Optional dict to override the EuroSAT -> High Level mapping.
            fallback_class: The class to assign when the model is highly uncertain.
        """
        # Default high-fidelity mapping schema
        self.mapping_schema = custom_mapping or {
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

        self.fallback_class = fallback_class
        
        # Dynamically determine output classes based on schema
        self.highlevel_classes = sorted(list(set(self.mapping_schema.values())))
        if self.fallback_class not in self.highlevel_classes:
            self.highlevel_classes.append(self.fallback_class)

        # Scientific Calibration Constants
        self.ENTROPY_THRESHOLD = 1.5  # Bits: High entropy means high confusion
        self.CONFIDENCE_FLOOR = 0.15   # Minimum probability for a "hit"

    def map_patch_prediction(self, 
                            eurosat_probs: Dict[str, float],
                            confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Maps a single patch's probability distribution to the higher hierarchy.
        
        Logic:
        1. Aggregate probabilities based on the mapping schema.
        2. Calculate Shannon Entropy to measure classification reliability.
        3. Apply Bayesian fallback if entropy is too high or max prob is too low.
        """
        threshold = confidence_threshold or self.CONFIDENCE_FLOOR
        
        # Initialize high-level probability vector
        highlevel_probs = {cls: 0.0 for cls in self.highlevel_classes}

        # Step 1: Probabilistic Summation
        for in_class, prob in eurosat_probs.items():
            target_class = self.mapping_schema.get(in_class, self.fallback_class)
            highlevel_probs[target_class] += prob

        # Convert to numpy for vectorized math
        prob_values = np.array([highlevel_probs[c] for c in self.highlevel_classes])
        
        # Step 2: Uncertainty Quantification (Shannon Entropy)
        # H(x) = -sum(p * log2(p))
        # Indicates how "spread out" the model's prediction is.
        non_zero_p = prob_values[prob_values > 0]
        entropy = -np.sum(non_zero_p * np.log2(non_zero_p))
        
        max_prob = np.max(prob_values)
        top_class = self.highlevel_classes[np.argmax(prob_values)]

        # Step 3: Reliability Logic
        # If entropy is high, the model is seeing "noise" or a mix of classes.
        is_unreliable = (max_prob < threshold) or (entropy > self.ENTROPY_THRESHOLD)

        if is_unreliable:
            logger.debug(f"Uncertain patch: Max Prob {max_prob:.2f}, Entropy {entropy:.2f}")
            # Bias towards fallback class without destroying the distribution
            highlevel_probs[self.fallback_class] = (highlevel_probs[self.fallback_class] + 0.2) / 1.2
            # Re-normalize
            total = sum(highlevel_probs.values())
            highlevel_probs = {k: v / total for k, v in highlevel_probs.items()}

        return {
            'probabilities': highlevel_probs,
            'top_class': top_class,
            'entropy': float(entropy),
            'confidence': float(max_prob),
            'is_reliable': not is_unreliable
        }

    def aggregate_predictions(self, 
                             all_patch_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregates multiple patch results (e.g. from an entire city) into a 
        final statistical summary.
        
        Uses "Soft Voting" (Mean of Probabilities) which is scientifically 
        superior to "Hard Voting" (counting winners) as it preserves the 
        model's internal uncertainty across the spatial extent.
        """
        if not all_patch_results:
            return {cls: 0.0 for cls in self.highlevel_classes}

        # Vectorized accumulation
        n_patches = len(all_patch_results)
        aggregated = {cls: 0.0 for cls in self.highlevel_classes}

        for patch_prob_dict in all_patch_results:
            for cls, prob in patch_prob_dict.items():
                if cls in aggregated:
                    aggregated[cls] += prob

        # Compute mean and ensure normalization
        final_distribution = {cls: (prob / n_patches) for cls, prob in aggregated.items()}
        
        # Final safety normalization (correction for floating point drift)
        total_sum = sum(final_distribution.values())
        if total_sum > 0:
            final_distribution = {k: v / total_sum for k, v in final_distribution.items()}

        return final_distribution

    def update_mapping(self, new_mapping: Dict[str, str]):
        """
        Allows the app to dynamically change the mapping rules at runtime
        (e.g., if a user wants to group Urban and Agricultural into 'Developed').
        """
        self.mapping_schema.update(new_mapping)
        self.highlevel_classes = sorted(list(set(self.mapping_schema.values())))
        if self.fallback_class not in self.highlevel_classes:
            self.highlevel_classes.append(self.fallback_class)
        logger.info(f"Mapping updated. New target classes: {self.highlevel_classes}")

    def get_mapping_info(self) -> Dict[str, Any]:
        """Returns metadata for API responses and audit trails."""
        return {
            'input_classes_count': len(self.mapping_schema),
            'output_hierarchy': self.highlevel_classes,
            'entropy_threshold': self.ENTROPY_THRESHOLD,
            'fallback_logic': self.fallback_class,
            'schema_definition': self.mapping_schema
        }



