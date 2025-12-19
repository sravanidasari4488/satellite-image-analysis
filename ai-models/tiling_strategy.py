"""
Sliding Window Tiling Strategy
Implements configurable tiling for patch-level CNN classification

Scientific Rationale:
- EuroSAT model trained on 64x64 patches, not full images
- Sliding window ensures complete coverage with overlap
- Overlap reduces edge effects and improves statistical robustness
- Configurable stride allows balance between accuracy and speed
"""

import numpy as np
from typing import List, Tuple, Iterator, Optional
import logging

logger = logging.getLogger(__name__)

class SlidingWindowTiler:
    """
    Generates 64x64 patches with configurable stride
    
    Default stride=32 provides 50% overlap, improving accuracy
    """
    
    def __init__(self, tile_size: int = 64, stride: int = 32):
        """
        Args:
            tile_size: Size of each tile (64x64 for EuroSAT model)
            stride: Step size between tiles (32 = 50% overlap, 64 = no overlap)
        """
        if tile_size <= 0 or stride <= 0:
            raise ValueError("tile_size and stride must be positive")
        if stride > tile_size:
            raise ValueError("stride should not exceed tile_size")
        
        self.tile_size = tile_size
        self.stride = stride
        self.overlap_ratio = 1.0 - (stride / tile_size)
        
        logger.info(f"Initialized tiler: {tile_size}x{tile_size} tiles, "
                   f"stride={stride} ({self.overlap_ratio*100:.0f}% overlap)")
    
    def generate_tiles(self, image: np.ndarray, 
                       mask: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Generate tiles with sliding window
        
        Args:
            image: Image array (H, W, C) or (H, W)
            mask: Optional boolean mask (True = valid pixels)
        
        Yields:
            (tile_array, (row, col)) - Tile and its top-left position
        """
        if len(image.shape) == 2:
            height, width = image.shape
            channels = 1
        else:
            height, width, channels = image.shape
        
        total_tiles = 0
        valid_tiles = 0
        
        for row in range(0, height - self.tile_size + 1, self.stride):
            for col in range(0, width - self.tile_size + 1, self.stride):
                total_tiles += 1
                
                # Extract tile
                if channels == 1:
                    tile = image[row:row+self.tile_size, col:col+self.tile_size]
                else:
                    tile = image[row:row+self.tile_size, col:col+self.tile_size, :]
                
                # Check if tile is valid (not mostly masked/zero)
                if mask is not None:
                    tile_mask = mask[row:row+self.tile_size, col:col+self.tile_size]
                    valid_ratio = np.sum(tile_mask) / tile_mask.size
                    
                    # Skip tiles with < 50% valid pixels
                    if valid_ratio < 0.5:
                        continue
                
                # Skip tiles that are mostly zeros (outside polygon)
                if np.sum(tile == 0) / tile.size > 0.5:
                    continue
                
                valid_tiles += 1
                yield tile, (row, col)
        
        logger.info(f"Generated {valid_tiles}/{total_tiles} valid tiles "
                   f"({100*valid_tiles/total_tiles:.1f}% valid)")
    
    def get_total_tiles(self, image_shape: Tuple[int, int]) -> int:
        """
        Calculate total number of tiles that would be generated
        
        Args:
            image_shape: (height, width)
        
        Returns:
            Total number of tiles
        """
        height, width = image_shape[:2]
        rows = max(0, (height - self.tile_size) // self.stride + 1)
        cols = max(0, (width - self.tile_size) // self.stride + 1)
        return rows * cols
    
    def get_tile_coverage(self, image_shape: Tuple[int, int]) -> float:
        """
        Calculate coverage ratio (accounting for overlap)
        
        Returns:
            Coverage ratio (1.0 = complete coverage)
        """
        total_pixels = image_shape[0] * image_shape[1]
        tile_pixels = self.tile_size * self.tile_size
        num_tiles = self.get_total_tiles(image_shape)
        
        # Account for overlap
        unique_pixels = num_tiles * (self.stride * self.stride)
        coverage = unique_pixels / total_pixels
        
        return min(coverage, 1.0)


