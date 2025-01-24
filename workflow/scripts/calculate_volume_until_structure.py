import nibabel as nib
import numpy as np
import json
from typing import Optional, Tuple, Dict, Union
from numpy.typing import NDArray
from helper_functions import calculate_volume, find_segmentation_bounds, calculate_fat_fraction


def process_muscle_segmentation(
    muscle_segmentation_path: str,
    output_path: str,
    start_seg_path: Optional[str] = None,
    end_seg_path: Optional[str] = None,
    single_ref_seg_path: Optional[str] = None,
    ip_image_path: Optional[str] = None,
    oop_image_path: Optional[str] = None,
    fat_fraction_output_path: Optional[str] = None
) -> Dict[str, Union[tuple, float, int, Dict[str, float]]]:
    """
    Process and crop a muscle segmentation based on reference segmentations.
    Optionally calculates fat fraction if in-phase (IP) and out-of-phase (OOP) images are provided.
    Supports two modes of operation:
    1. Single reference mode: Crops from the bottom of reference segmentation to the end of volume
    2. Two-point mode: Crops between the top of start segmentation and bottom of end segmentation

    Parameters
    ----------
    muscle_segmentation_path : str
        Path to the input muscle segmentation NIfTI file to be cropped
    output_path : str
        Path where the cropped muscle segmentation will be saved
    start_seg_path : Optional[str]
        Path to the starting boundary segmentation for two-point mode
    end_seg_path : Optional[str]
        Path to the ending boundary segmentation for two-point mode
    single_ref_seg_path : Optional[str]
        Path to the reference segmentation for single reference mode
    ip_image_path : Optional[str]
        Path to the in-phase image for fat fraction calculation
    oop_image_path : Optional[str]
        Path to the out-of-phase image for fat fraction calculation
    fat_fraction_output_path : Optional[str]
        Path where the fat fraction map will be saved as NIfTI file

    Returns
    -------
    Dict[str, Union[tuple, float, int, Dict[str, float]]]
        Dictionary containing:
        - original_volume_cm3: Original volume in cm³
        - cropped_volume_cm3: Cropped volume in cm³
        - fat_fraction_stats: Dictionary with fat fraction statistics (if IP/OOP images provided)
            - mean: Mean fat fraction
            - median: Median fat fraction
            - std: Standard deviation of fat fraction

    Examples
    --------
    # Single reference mode with fat fraction calculation
    result = process_muscle_segmentation(
        muscle_segmentation_path="muscle.nii.gz",
        output_path="cropped_muscle.nii.gz",
        single_ref_seg_path="L3.nii.gz",
        ip_image_path="ip.nii.gz",
        oop_image_path="oop.nii.gz",
        fat_fraction_output_path="fat_fraction.nii.gz"
    )

    Raises
    ------
    ValueError
        If no segmentation is found in the reference volumes
        If input volumes have different dimensions
        If end point is before start point in two-point mode
        If neither single_ref_seg_path nor both start_seg_path and end_seg_path are provided
        If only one of ip_image_path or oop_image_path is provided
    """
    muscle_seg_nii = nib.load(muscle_segmentation_path)
    muscle_seg_data = muscle_seg_nii.get_fdata()
    voxel_dims = muscle_seg_nii.header.get_zooms()

    # Load and validate IP/OOP images if provided
    calculate_ff = False
    if ip_image_path is not None or oop_image_path is not None:
        if ip_image_path is None or oop_image_path is None:
            raise ValueError("Both ip_image_path and oop_image_path must be provided for fat fraction calculation")
        
        ip_nii = nib.load(ip_image_path)
        oop_nii = nib.load(oop_image_path)
        ip_data = ip_nii.get_fdata()
        oop_data = oop_nii.get_fdata()
        
        if not (ip_data.shape == oop_data.shape == muscle_seg_data.shape):
            raise ValueError("IP, OOP, and muscle segmentation must have the same dimensions!")
        
        calculate_ff = True
    
    if single_ref_seg_path is not None:
        ref_seg_data = nib.load(single_ref_seg_path).get_fdata()
        if ref_seg_data.shape != muscle_seg_data.shape:
            raise ValueError("Reference and muscle segmentations must have the same dimensions!")
        
        ref_bounds = find_segmentation_bounds(ref_seg_data)
        start_point = ref_bounds[0]  # First slice of reference segmentation
        end_point = muscle_seg_data.shape[2]  # Go to the end of the volume
        
    elif start_seg_path is not None and end_seg_path is not None:
        start_seg_data = nib.load(start_seg_path).get_fdata()
        end_seg_data = nib.load(end_seg_path).get_fdata()
        
        if not (start_seg_data.shape == end_seg_data.shape == muscle_seg_data.shape):
            raise ValueError("All segmentations must have the same dimensions!")
        start_bounds = find_segmentation_bounds(start_seg_data)
        start_point = start_bounds[0]
        end_bounds = find_segmentation_bounds(end_seg_data)
        end_point = end_bounds[-1]
        
        if end_point < start_point:
            raise ValueError(
                f"End point (slice {end_point}) is before start point (slice {start_point}). "
                "Check that your start and end segmentations are in the correct order."
            )
    else:
        raise ValueError(
            "Must provide either single_ref_seg_path for single-point cropping "
            "or both start_seg_path and end_seg_path for two-point cropping"
        )
    
    # add 1 to end_point to include it
    cropped_muscle_seg = muscle_seg_data[:, :, start_point:end_point + 1]
    
    # Calculate volumes
    original_volume = calculate_volume(muscle_seg_data, voxel_dims)
    cropped_volume = calculate_volume(cropped_muscle_seg, voxel_dims)
    
    # Calculate fat fraction if images were provided
    result_dict = {
        'original_volume_cm3': original_volume / 1000,
        'cropped_volume_cm3': cropped_volume / 1000
    }
    
    if calculate_ff:
        cropped_ip = ip_data[:, :, start_point:end_point + 1]
        cropped_oop = oop_data[:, :, start_point:end_point + 1]
        
        ff_map, ff_stats = calculate_fat_fraction(
            cropped_ip,
            cropped_oop,
            cropped_muscle_seg
        )
        result_dict['fat_fraction_stats'] = ff_stats
        
    # Calculate updated affine matrix for cropped volumes
    original_affine = muscle_seg_nii.affine.copy()
    new_affine = original_affine.copy()
    new_affine[:3, 3] = original_affine[:3, 3] + (original_affine[:3, 2] * start_point)
    
    # Save cropped segmentation
    cropped_nii = nib.Nifti1Image(
        cropped_muscle_seg,
        new_affine,
        muscle_seg_nii.header
    )
    cropped_nii.header.set_data_shape(cropped_muscle_seg.shape)
    nib.save(cropped_nii, output_path)
    
    # Save fat fraction map if output path is provided
    if calculate_ff and fat_fraction_output_path:
        ff_nii = nib.Nifti1Image(
            ff_map,
            new_affine,  # Using the same updated affine as the cropped segmentation
            muscle_seg_nii.header
        )
        ff_nii.header.set_data_shape(ff_map.shape)
        nib.save(ff_nii, fat_fraction_output_path)
    
    return result_dict


if hasattr(snakemake.input, "single_segmentation"):
    # Set up base parameters
    params = {
        "muscle_segmentation_path": snakemake.input["muscle_segmentation"],
        "output_path": snakemake.output["cropped_muscle"]
    }
    
    # Add reference segmentation
    params["single_ref_seg_path"] = snakemake.input["single_ref_seg_path"]
    
    # Add IP/OOP parameters and fat fraction output if both images are present
    if hasattr(snakemake.input, "ip_image") and hasattr(snakemake.input, "oop_image"):
        params["ip_image_path"] = snakemake.input["ip_image"]
        params["oop_image_path"] = snakemake.input["oop_image"]
        params["fat_fraction_output_path"] = snakemake.output["fat_fraction"]
    
    result = process_muscle_segmentation(**params)
    with open(snakemake.output["muscle_volume"], "w") as f:
        json.dump(result, f)
        
if hasattr(snakemake.input, "start_segmentation") and hasattr(snakemake.input, "end_segmentation"):
    # Set up base parameters
    params = {
        "muscle_segmentation_path": snakemake.input["muscle_segmentation"],
        "output_path": snakemake.output["cropped_muscle"],
        "start_seg_path": snakemake.input["start_segmentation"],
        "end_seg_path": snakemake.input["end_segmentation"]
    }
    
    # Add IP/OOP parameters and fat fraction output if both images are present
    if hasattr(snakemake.input, "ip_image") and hasattr(snakemake.input, "oop_image"):
        params["ip_image_path"] = snakemake.input["ip_image"]
        params["oop_image_path"] = snakemake.input["oop_image"]
        params["fat_fraction_output_path"] = snakemake.output["fat_fraction"]
    
    result = process_muscle_segmentation(**params)
    with open(snakemake.output["muscle_volume"], "w") as f:
        json.dump(result, f)