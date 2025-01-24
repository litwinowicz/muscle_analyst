import numpy as np
from typing import Tuple, Dict
from numpy.typing import NDArray

def calculate_volume(segmentation_data: np.ndarray, voxel_dims: Tuple[float, ...]) -> float:
    """
    Calculate volume of a binary segmentation mask.
    """
    voxel_volume = np.prod(voxel_dims)
    return np.sum(segmentation_data > 0) * voxel_volume

def find_segmentation_bounds(seg_data: np.ndarray) -> np.ndarray:
    """
    Find the slices where segmentation exists along Z axis.
    Returns array of indices where segmentation is present.
    """
    nonzero_z = np.nonzero(np.any(seg_data > 0, axis=(0, 1)))[0]
    if len(nonzero_z) == 0:
        raise ValueError("No segmentation found in the volume!")
    return nonzero_z



def calculate_fat_fraction(
    fat_data: NDArray[np.float64],
    water_data: NDArray[np.float64],
    seg_data: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], Dict[str, float]]:
    """
    Calculate fat fraction from fat and water images using segmentation mask
    
    Args:
        fat_data: Fat image data array
        water_data: Water image data array
        seg_data: Segmentation mask array
    
    Returns:
        Tuple containing:
            - NDArray[np.float64]: Calculated fat fraction array
            - Dict[str, float]: Dictionary containing mean, median, and std of fat fraction
    
    Raises:
        ValueError: If input arrays don't have the same shape
    """
    # Input validation
    if not (fat_data.shape == water_data.shape == seg_data.shape):
        raise ValueError("All input arrays must have the same shape")
    
    # Create binary mask from segmentation
    binary_mask = seg_data > 0
    
    # Apply mask to fat and water data
    masked_fat = fat_data * binary_mask
    masked_water = water_data * binary_mask
    
    # Calculate fat fraction with epsilon to prevent division by zero
    epsilon = 1e-10
    denominator = masked_fat + masked_water + epsilon
    fat_fraction = np.divide(masked_fat, denominator)
    
    # Calculate statistics on masked fat fraction
    masked_ff = fat_fraction[binary_mask]
    stats = {
        'mean_fat_fraction': float(np.mean(masked_ff)),
        'median_fat_fraction': float(np.median(masked_ff)),
        'std_fat_fraction': float(np.std(masked_ff))
    }
    
    return fat_fraction, stats