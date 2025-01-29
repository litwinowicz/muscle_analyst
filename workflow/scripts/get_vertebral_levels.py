import nibabel as nib
import numpy as np
from helper_functions import reorient_to_ras

def combine_nifti_files(vertebrae_path, curvature_path, output_path, backward_check=10):
    """
    Combine vertebral segmentation and spinal curvature NIFTI files.
    When spinal curvature intersects a vertebra, set the entire axial slice
    at that level to the vertebra's label value. If no intersection is found,
    check up to specified number of voxels backward.
    
    Args:
        vertebrae_path (str): Path to vertebral segmentation NIFTI file
        curvature_path (str): Path to spinal curvature NIFTI file
        output_path (str): Path where combined NIFTI file will be saved
        backward_check (int): Number of voxels to check backward for intersection
    """
    # Load NIFTI files
    vertebrae_img = nib.load(vertebrae_path)
    curvature_img = nib.load(curvature_path)
    
    vertebrae_img = reorient_to_ras(vertebrae_img)
    curvature_img = reorient_to_ras(curvature_img)
    
    # Get data arrays
    vertebrae_data = vertebrae_img.get_fdata()
    curvature_data = curvature_img.get_fdata()
    
    # Create output array with same shape as vertebrae data
    output_data = np.zeros_like(vertebrae_data)
    
    def check_intersection(vertebra_mask, curvature_slice, y_start, y_end):
        """
        Helper function to check for intersection between vertebra and curvature
        in a range of Y coordinates
        """
        for y in range(y_start, y_end):
            if y >= 0 and y < curvature_data.shape[1]:  # Ensure we're within bounds
                # Create a modified vertebra mask shifted to position y
                x_indices = np.where(np.any(vertebra_mask, axis=1))[0]
                if len(x_indices) == 0:
                    continue
                    
                shifted_mask = np.zeros_like(vertebra_mask)
                for x in x_indices:
                    shifted_mask[x, y] = True
                
                # Check if there's an intersection at this position
                if np.any(curvature_slice[shifted_mask] > 0):
                    return True
        return False
    
    # Iterate through axial slices
    for z in range(vertebrae_data.shape[2]):
        # Check if curvature exists in this slice
        if np.any(curvature_data[:, :, z] > 0):
            # Get unique vertebrae labels in this slice (excluding 0)
            slice_labels = np.unique(vertebrae_data[:, :, z])
            slice_labels = slice_labels[slice_labels != 0]
            
            # For each vertebra label in the slice
            for label in slice_labels:
                # Get mask for current vertebra
                vertebra_mask = vertebrae_data[:, :, z] == label
                
                # First check current position
                if np.any(curvature_data[:, :, z][vertebra_mask] > 0):
                    output_data[:, :, z] = label
                    break
                
                # If no intersection found, check backward (anterior in RAS)
                else:
                    # Get Y coordinates where vertebra is present

                    y_coords = np.where(np.any(vertebra_mask, axis=0))[0]
                    if len(y_coords) > 0:
                        min_y = np.min(y_coords)
                        # Check up to backward_check voxels anterior to the vertebra
                        if check_intersection(
                            vertebra_mask,
                            curvature_data[:, :, z],
                            min_y,
                            min_y + backward_check
                        ):
                            output_data[:, :, z] = label
                            break
    
    # Create new NIFTI image with combined data
    combined_img = nib.Nifti1Image(output_data, vertebrae_img.affine)
    
    # Reorient to RAS
    combined_img = reorient_to_ras(combined_img)
    
    # Save the combined image
    nib.save(combined_img, output_path)
    
    return combined_img


combined_img = combine_nifti_files(snakemake.input["combined_vertebral_levels"], snakemake.input["spine_curvature"], snakemake.output[0], 0)