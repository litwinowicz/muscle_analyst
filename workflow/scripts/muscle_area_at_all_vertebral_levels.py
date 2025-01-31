import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from helper_functions import reorient_to_ras
import csv
VERTEBRAL_LEVELS = snakemake.params["vertebral_levels"]


def get_vertebral_name(level_number):
    """Translate numeric level to vertebral name."""
    # Adjust for 0-based indexing
    idx = int(level_number) - 1
    if 0 <= idx < len(VERTEBRAL_LEVELS):
        return VERTEBRAL_LEVELS[idx]
    return f"Unknown_{level_number}"

def load_nifti(file_path):
    """Load a NIFTI file and return the data array and affine matrix."""
    img = nib.load(file_path)
    img = reorient_to_ras(img)
    return img.get_fdata(), img.affine, img.header


def get_slice_levels(mask_data, level):
    """Get 5 evenly spaced slice indices for a given vertebral level."""
    # Find all slice indices for this level
    level_indices = np.where(np.any(mask_data == level, axis=(0, 1)))[0]
    
    if len(level_indices) == 0:
        return []
    
    if len(level_indices) < 5:
        # If less than 5 slices, repeat some indices
        indices = np.linspace(level_indices[0], level_indices[-1], 5, dtype=int)
        return [level_indices[0]] * (5 - len(level_indices)) + list(level_indices)
    else:
        # Get 5 evenly spaced indices
        return list(np.linspace(level_indices[0], level_indices[-1], 5, dtype=int))

def calculate_muscle_area(muscle_mask, slice_idx, voxel_dims):
    """Calculate the muscle area for a given slice in cm²."""
    # Calculate area taking into account voxel dimensions
    pixel_area = voxel_dims[0] * voxel_dims[1]  # mm²
    slice_area_mm2 = np.sum(muscle_mask[:, :, slice_idx]) * pixel_area
    # Convert to cm²
    slice_area_cm2 = slice_area_mm2 / 100.0
    return slice_area_cm2

def prepare_slice_for_display(slice_data):
    """Prepare a slice for display in RAS orientation."""
    # Flip and rotate the slice to match RAS orientation
    # First transpose, then flip left-right, then flip up-down to correct orientation
    return np.flip(np.flip(slice_data.T, axis=1), axis=0)

def save_slice_visualization(volume_data, muscle_mask, slice_idx, area, level, slice_num, image_output_dir, voxel_dims):
    """Save a PNG visualization of the slice with muscle overlay."""
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Prepare slices in RAS orientation
    volume_slice = prepare_slice_for_display(volume_data[:, :, slice_idx])
    mask_slice = prepare_slice_for_display(muscle_mask[:, :, slice_idx])
    
    # Calculate aspect ratio from voxel dimensions
    # voxel_dims[0] is x (R-L), voxel_dims[1] is y (A-P)
    aspect_ratio = voxel_dims[1] / voxel_dims[0]
    
    # Create axes
    ax = plt.gca()
    
    # Plot the original volume slice with correct aspect ratio
    plt.imshow(volume_slice, cmap='gray', aspect=aspect_ratio)
    
    # Create a red mask overlay
    mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    plt.imshow(mask_overlay, alpha=0.7, cmap='Greens', aspect=aspect_ratio)
    
    # Display reversed slice number (5 to 1 instead of 1 to 5)
    reversed_slice_num = 6 - slice_num
    vertebral_name = get_vertebral_name(level)
    plt.title(f'Vertebral Level {vertebral_name}, Slice {reversed_slice_num}/5\nMuscle Area: {area:.2f} cm²')
    
    # Remove axes
    plt.axis('off')
    
    
    # Save the figure with vertebral name and reversed slice number in filename
    output_path = os.path.join(image_output_dir, f'level_{vertebral_name}_slice_{reversed_slice_num}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150, pad_inches=0.1)
    plt.close()

def save_results_to_csv(results, csv_path):
    """Save results to a CSV file."""
    
    # Sort results by vertebral level and slice number
    sorted_results = sorted(results, key=lambda x: (x['level'], x['slice_num']))
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['subject_id', 'vertebral_level', 'slice', 'area']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in sorted_results:
            writer.writerow({
                'subject_id': snakemake.wildcards["subject_id"],
                'vertebral_level': get_vertebral_name(result['level']),
                'slice': result['slice_num'],
                'area': f"{result['area']:.2f}"
            })

def main(volume_path, muscle_mask_path, vertebral_mask_path, image_output_dir, csv_output):
    """Main function to process the NIFTI files and generate visualizations."""
    # Create output directory if it doesn't exist
    Path(image_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load all NIFTI files
    volume_data, _, volume_header = load_nifti(volume_path)
    muscle_mask_data, _, _ = load_nifti(muscle_mask_path)
    vertebral_mask_data, _, _ = load_nifti(vertebral_mask_path)
    
    # Get voxel dimensions (x, y, z) in mm
    voxel_dims = volume_header.get_zooms()
    
    # Find unique vertebral levels
    levels = np.unique(vertebral_mask_data)
    levels = levels[levels != 0]  # Remove background level if present
    
    results = []
    
    # Process each vertebral level
    for level in levels:
        slice_indices = get_slice_levels(vertebral_mask_data, level)
        
        for slice_num, slice_idx in enumerate(slice_indices, 1):
            # Calculate muscle area
            area = calculate_muscle_area(muscle_mask_data, slice_idx, voxel_dims)
            
            # Save visualization
            save_slice_visualization(volume_data, muscle_mask_data, slice_idx, 
                                  area, level, slice_num, image_output_dir, voxel_dims)
            
            # Store results with reversed slice number
            results.append({
                'level': level,
                'slice_num': 6 - slice_num,  # Reverse the numbering
                'slice_idx': slice_idx,
                'area': area
            })
    
    # Save results to CSV
    save_results_to_csv(results, csv_output)
    
    return results


volume_path = snakemake.input["volume"]
muscle_mask_path = snakemake.input["muscle_mask"]
vertebral_mask_path = snakemake.input["vertebral_mask"]
output_dir = snakemake.output["image_output_dir"]
csv_output = snakemake.output["csv_output"]
results = main(volume_path, muscle_mask_path, vertebral_mask_path, output_dir, csv_output)

