import nibabel as nib
import numpy as np
from pathlib import Path

def process_vertebrae(level_files_and_labels, output_path):
    """
    Combine vertebral level segmentations into a single file with specified labels for each level.
    
    Args:
        level_files_and_labels: List of tuples [(file_path, label), ...]
        output_path: Path where the combined file will be saved
    """
    # Load the first level file to get dimensions and header info
    first_file, _ = level_files_and_labels[0]
    first_img = nib.load(first_file)
    combined_data = np.zeros_like(first_img.get_fdata(), dtype=np.uint16)
    
    # Process each level
    for file_path, label in level_files_and_labels:
        level_img = nib.load(file_path)
        level_data = level_img.get_fdata()
        # Assign label to all non-zero voxels in the level segmentation
        level_mask = level_data > 0
        combined_data[level_mask] = label
        print(f"Processed {Path(file_path).name} with label {label}")
    
    # Create and save the combined image
    combined_img = nib.Nifti1Image(combined_data, first_img.affine, first_img.header)
    combined_img.to_filename(output_path)
    print(f"Saved combined file to {output_path}")


levels_dir = snakemake.input["vertebral_levels_segmentations"][0]
level_files_and_labels = [(levels_dir + f"/vertebrae_{level}.nii.gz", label) for (level, label) in snakemake.params["label_level_pairs"]]
output_path = snakemake.output[0]
    
process_vertebrae(level_files_and_labels, output_path)